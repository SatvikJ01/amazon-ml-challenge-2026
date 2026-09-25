"""Country-agnostic text normalisation for business names and addresses.

Design principle
----------------
The test set contains **France**, which never appears in training.  Every
normalisation rule here is therefore either (a) a universal string operation, or
(b) *learned from the data at run time*.  There are **no hard-coded geography
tables** -- no US state list, no Indian state list, no French department list.

That is not only a compliance choice (the rules forbid external data lookup and
geocoding services); it is the *correct modelling* choice.  The observed noise

    "... Salt Lake City, UT"        <->  "... Salt Lake City, Utah"
    "... Visakhapatnam, AP"         <->  "... Visakhapatnam, Andhra Pradesh"
    "... Lille, Hauts-de-France"    <->  "... Lille, Nord"

is the same phenomenon in three countries.  A hard-coded map solves two of them
and silently fails on the third.  IDF weighting solves all three at once:
region-like tokens are high-document-frequency and therefore contribute almost
nothing to the score, while the house number and street name -- which *are*
preserved across sources -- carry it.

What is actually done
---------------------
1. Unicode folding.  ``unidecode`` transliterates Devanagari / Telugu / Kannada /
   Bengali names and strips French accents in one step, so
   ``महाराष्ट्र -> maharashtra`` and ``Président -> President``.  Applied only to
   the ~10 % of rows that contain non-ASCII, because it is a Python-level loop.
2. Case folding and punctuation stripping.  Handles ``[LLC]``, ``--``, ``<<``,
   ``&`` vs ``and``, ``S.A.S`` vs ``SAS``.
3. Structural token extraction:
   * ``digits``   -- the numeric tokens of an address (house / building number).
     These survive reordering and abbreviation and are the single most
     discriminative address signal.
   * ``alpha``    -- alphabetic tokens, order-independent (addresses are
     routinely reordered between sources).
4. Literal ``NULL`` placeholders -- which the generator injects into addresses --
   are removed, as are single-character noise tokens.

Everything is vectorised over pandas string columns; a full pass over all 26 M
records runs in a few minutes.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

import numpy as np
import pandas as pd
from unidecode import unidecode

__all__ = [
    "fold_unicode",
    "normalize_text",
    "tokenize",
    "split_tokens",
    "normalize_frame",
    "NULL_TOKENS",
]

# Placeholder strings the generator injects; they carry no identity information
# and would otherwise look like strong shared evidence between two records.
NULL_TOKENS = frozenset({"null", "na", "nan", "none", "nil", "unknown", "n a"})

_NON_ASCII = re.compile(r"[^\x00-\x7F]")
_PUNCT = re.compile(r"[^0-9a-z]+")
_MULTISPACE = re.compile(r"\s+")


def fold_unicode(s: pd.Series) -> pd.Series:
    """Transliterate to ASCII, touching only the rows that need it.

    ``unidecode`` is a per-string Python call (~1-2 us).  Running it over 26 M
    rows unconditionally costs a minute of pure overhead for the ~90 % of rows
    that are already ASCII, so we mask first.
    """
    s = s.fillna("")
    mask = s.str.contains(_NON_ASCII, regex=True, na=False)
    if not mask.any():
        return s
    out = s.copy()
    out.loc[mask] = [unidecode(x) for x in s.loc[mask]]
    return out


def normalize_text(s: pd.Series) -> pd.Series:
    """Lowercase, ASCII-fold, and reduce to space-separated alphanumeric tokens.

    ``&`` becomes a space rather than ``and``: the two sides of that variation
    ("Thermal & Fils" vs "Thermal and Fils") then agree on the tokens that
    matter, and the connector itself is near-zero IDF either way.
    """
    s = fold_unicode(s).str.lower()
    s = s.str.replace(_PUNCT, " ", regex=True)
    s = s.str.replace(_MULTISPACE, " ", regex=True).str.strip()
    return s


def split_tokens(norm: str) -> tuple[list[str], list[str]]:
    """Split one normalised string into (alphabetic tokens, numeric tokens).

    Mixed alphanumeric tokens such as ``20-52/2`` normalise to ``20 52 2`` and so
    arrive here already split.  Tokens like ``7th`` keep their digit and are
    classed as alphabetic, which is right: ``7th main`` is a street name, not a
    house number.
    """
    alpha: list[str] = []
    digits: list[str] = []
    for t in norm.split():
        if t in NULL_TOKENS or len(t) < 2:
            continue
        if t.isdigit():
            digits.append(t)
        else:
            alpha.append(t)
    return alpha, digits


def tokenize(norm: pd.Series) -> pd.Series:
    """Vectorised wrapper returning a Series of ``(alpha, digits)`` tuples."""
    return pd.Series([split_tokens(x) for x in norm], index=norm.index)


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalised name/address columns to a source frame.

    Returns a new frame with ``name_norm`` and ``addr_norm`` alongside the
    original ``code`` / ``country`` columns.  The raw text is dropped: downstream
    stages work from the normalised form plus a re-read of the raw text only
    where an exact-character feature is needed.
    """
    out = pd.DataFrame(
        {
            "code": df["code"].to_numpy(),
            "country": df["country"].to_numpy(),
            "name_norm": normalize_text(df["business_name"]),
            "addr_norm": normalize_text(df["business_address"]),
        }
    )
    return out


def learn_stop_tokens(
    token_counts: pd.Series, n_docs: int, max_df_ratio: float = 0.02
) -> set[str]:
    """Derive a stop-token set from document frequency instead of hard-coding it.

    Legal suffixes (``ltd``, ``llc``, ``pvt``, ``sarl``, ``sas``, ``inc``),
    street words (``road``, ``rue``, ``street``) and region names all float to
    the top of the document-frequency ranking *in whichever country they belong
    to*.  Learning the list per country block means France gets the same
    treatment as the US without anyone writing down a French word list.
    """
    return set(token_counts[token_counts > max_df_ratio * n_docs].index)


# --------------------------------------------------------------------------- #
# Consonant skeleton: transliteration-robust token key
# --------------------------------------------------------------------------- #
# Motivation (reports/blocking_misses_*.tsv): 76 % of blocking misses in the India
# block are names that arrived in an Indic script and were transliterated back to
# Latin phonetically, e.g. "shiv praaprttiis praiveett limittedd" for
# "shiv properties private limited".  Token equality fails on every word.
#
# The skeleton maps both spellings to the same key by keeping only the
# consonant frame: merge digraphs, merge voiced/unvoiced pairs, drop vowels and
# 'h', collapse repeats.  "praaprttiis" and "properties" -> "prprts";
# "limittedd" and "limited" -> "lmtt"; "dillii" and "delhi" -> "tl".  It is a
# purely orthographic rule, not a lookup table, so it applies unchanged to
# French names (accent- and typo-damaged) in the unseen France block.
_DIGRAPH = re.compile(r"ph|bh|dh|th|kh|gh|sh|ch|ck")
_DIGRAPH_MAP = {"ph": "f", "bh": "b", "dh": "d", "th": "t", "kh": "k",
                "gh": "g", "sh": "s", "ch": "k", "ck": "k"}
_SKEL_TRANS = str.maketrans({
    "c": "k", "q": "k", "x": "k", "z": "s", "w": "f", "v": "f",
    "d": "t", "g": "k", "j": "k", "b": "p",
    "a": None, "e": None, "i": None, "o": None, "u": None, "y": None, "h": None,
})
_REPEAT = re.compile(r"(.)\1+")


@lru_cache(maxsize=2**21)
def skeleton_token(tok: str) -> str:
    """Consonant skeleton of one lowercase ASCII token ('' if nothing is left).
    Numeric tokens are returned unchanged.  Cached: tokens repeat massively
    across records, and the uncached regex was ~48 % of feature time."""
    if tok.isdigit():
        return tok
    s = _DIGRAPH.sub(lambda m: _DIGRAPH_MAP[m.group()], tok)
    s = s.translate(_SKEL_TRANS)
    return _REPEAT.sub(r"\1", s)


def skeleton_tokens(norm: str) -> list[str]:
    """Skeletons of the alphabetic tokens of a normalised string (len >= 2)."""
    out = []
    for t in norm.split():
        if t.isdigit() or t in NULL_TOKENS:
            continue
        k = skeleton_token(t)
        if len(k) >= 2:
            out.append(k)
    return out
