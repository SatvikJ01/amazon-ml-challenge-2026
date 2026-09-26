"""E034: name-token substitution and shared-address features (stage 3).

Remaining error patterns after E033 (holdout sample, 2026-09-26):

* near-copies with one distinctive name word replaced ("Imperial Care" ->
  "Imperial Cole", "Logistics" -> "Global", "Private" -> "Public"), while true
  records mostly *add* generator filler words ("Services", "Partners", "LLC",
  "Shri", "M/s") or corrupt characters;
* several S1 businesses at one address, so address agreement alone is ambiguous.

Filler words are learned without labels, per country and split: a token's
over-representation in S2/S3 names relative to S1 names,
``noise(t) = log((df_T(t)+1)/(N_T+1)) - log((df_S1(t)+1)/(N_S1+1))``.
The generator adds filler to S2/S3 records, so filler scores high; a replaced
distinctive word comes from the ordinary business vocabulary and scores ~0.

Columns
-------
nm_c_extra / nm_q_miss   candidate tokens with no S1 counterpart / S1 tokens with no
                         candidate counterpart (exact, consonant-skeleton or fuzzy
                         >= 75 token matches removed first)
nm_c_extra_sub           ... extra tokens that are neither filler (noise < 0.5) nor
                         frequent words such as legal suffixes (S1-IDF >= 3)
nm_c_extra_minnoise      lowest filler score among the extra tokens (NaN if none)
nm_q_miss_idf            S1-IDF mass of the missing S1 tokens
nm_c_extra_idf           S1-IDF mass of the extra candidate tokens
s1_addr_n                S1 records of the country sharing this S1's address key
c_addr_s1_n              S1 records sharing the candidate's address key
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .normalize import NULL_TOKENS, skeleton_token

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
EXTRA2_COLS = ["nm_c_extra", "nm_q_miss", "nm_c_extra_sub", "nm_c_extra_minnoise", "nm_q_miss_idf",
               "nm_c_extra_idf", "s1_addr_n", "c_addr_s1_n"]
NOISE_T, IDF_T = 0.5, 3.0


def _addr_key(addr: str) -> str:
    """First numeric token + the two longest alphabetic tokens (sorted)."""
    toks = [t for t in addr.split() if t not in NULL_TOKENS]
    num = next((t.lstrip("0") or "0" for t in toks if t.isdigit()), "")
    alpha = sorted({t for t in toks if t.isalpha() and len(t) >= 3}, key=lambda t: (-len(t), t))[:2]
    return num + "|" + " ".join(sorted(alpha)) if (num or alpha) else ""


class NameStats:
    """Per-country, per-split token statistics (label-free)."""

    def __init__(self, split: str, country: str):
        s1n = pd.read_parquet(PROCESSED / f"{split}_s1_{country}.parquet", columns=["name_norm", "addr_norm"])
        df1: Counter = Counter()
        for x in s1n["name_norm"]:
            df1.update(set(x.split()))
        dft: Counter = Counter()
        nt = 0
        for s in (2, 3):
            for x in pd.read_parquet(PROCESSED / f"{split}_s{s}_{country}.parquet", columns=["name_norm"])["name_norm"]:
                dft.update(set(x.split()))
                nt += 1
        n1 = len(s1n)
        l1, lt = np.log(n1 + 1), np.log(nt + 1)
        self.noise = {t: float(np.log(c + 1) - lt - np.log(df1.get(t, 0) + 1) + l1) for t, c in dft.items()}
        self.idf = {t: float(l1 - np.log(c + 1)) for t, c in df1.items()}
        self.idf_default = float(l1)
        self.addr_count = Counter(_addr_key(a) for a in s1n["addr_norm"])
        self.addr_count.pop("", None)


def _residual(q: set, c: set) -> tuple[set, set]:
    eq, ec = q - c, c - q
    if eq and ec:
        sq = {}
        for t in eq:
            sq.setdefault(skeleton_token(t), t)
        for t in list(ec):
            k = skeleton_token(t)
            if k and k in sq and sq[k] in eq:
                eq.discard(sq[k]); ec.discard(t)
    if eq and ec:
        for t in list(ec):
            best = max(eq, key=lambda u: fuzz.ratio(t, u), default=None)
            if best is not None and fuzz.ratio(t, best) >= 75:
                eq.discard(best); ec.discard(t)
                if not eq:
                    break
    return eq, ec


def extra_features2(q_name: np.ndarray, c_name: np.ndarray, q_addr: np.ndarray, c_addr: np.ndarray,
                    st: NameStats) -> pd.DataFrame:
    n = len(q_name)
    cx = np.zeros(n, np.float32); qm = np.zeros(n, np.float32); cs = np.zeros(n, np.float32)
    mn = np.full(n, np.nan, np.float32); qi = np.zeros(n, np.float32); ci = np.zeros(n, np.float32)
    noise, idf, dflt = st.noise, st.idf, st.idf_default
    cache: dict = {}
    for i in range(n):
        key = (q_name[i], c_name[i])
        r = cache.get(key)
        if r is None:
            eq, ec = _residual(set(q_name[i].split()), set(c_name[i].split()))
            ec = sorted(ec)
            nz = [noise.get(t, 0.0) for t in ec]
            r = (len(ec), len(eq), sum(1 for t, v in zip(ec, nz) if v < NOISE_T and idf.get(t, dflt) >= IDF_T),
                 min(nz) if nz else np.nan,
                 sum(idf.get(t, dflt) for t in eq), sum(idf.get(t, dflt) for t in ec))
            if len(cache) < 2_000_000:
                cache[key] = r
        cx[i], qm[i], cs[i], mn[i], qi[i], ci[i] = r
    ac = st.addr_count
    ka = pd.Series(q_addr).map(lambda a: ac.get(_addr_key(a), 0)).to_numpy(np.float32)
    kc = pd.Series(c_addr).map(lambda a: ac.get(_addr_key(a), 0)).to_numpy(np.float32)
    return pd.DataFrame({"nm_c_extra": cx, "nm_q_miss": qm, "nm_c_extra_sub": cs, "nm_c_extra_minnoise": mn,
                         "nm_q_miss_idf": qi, "nm_c_extra_idf": ci, "s1_addr_n": ka, "c_addr_s1_n": kc})
