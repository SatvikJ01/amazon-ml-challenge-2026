"""E041: hard anti-match flags (triage item #2), on top of the E032/E039 stage-3 set.

Two explicit distractor signals, as proposed:

is_neighbor_number       the first house numbers of S1 and candidate are close but different:
                         both pure digits with 0 < |x - y| <= 4, or the same digits with a
                         different / added letter suffix (``12A`` vs ``12B``, ``12`` vs ``12A``).
                         NaN when either side has no number.
mismatch_token_max_idf   largest S1-IDF among name tokens present on exactly one side (symmetric
                         set difference of the normalised name tokens); 0 when the token sets agree.

Existing features already expose most of this (``nd_absdiff``, ``anum_first_rel``,
``nm_c_extra_idf``); E041 measures whether the explicit forms add anything.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .extra_features import _numtoks
from .extra_features2 import NameStats

EXTRA4_COLS = ["is_neighbor_number", "mismatch_token_max_idf"]
_DIG = re.compile(r"^(\d+)([a-z]*)$")


def _first(addr: str):
    t = _numtoks(addr)
    return t[0][1] if t else None


def _neighbor(a: str | None, b: str | None) -> float:
    if a is None or b is None:
        return np.nan
    if a == b:
        return 0.0
    ma, mb = _DIG.match(a), _DIG.match(b)
    if ma and mb:
        da, sa = ma.groups()
        db, sb = mb.groups()
        if da == db and sa != sb:
            return 1.0
        if not sa and not sb and len(da) < 12 and len(db) < 12:
            return float(0 < abs(int(da) - int(db)) <= 4)
    return 0.0


def anti_match_features(q_name: np.ndarray, c_name: np.ndarray, q_addr: np.ndarray, c_addr: np.ndarray,
                        st: NameStats) -> pd.DataFrame:
    n = len(q_name)
    qf = pd.Series(q_addr).map(_first).to_numpy()
    cf = pd.Series(c_addr).map(_first).to_numpy()
    nb = np.fromiter((_neighbor(a, b) for a, b in zip(qf, cf)), np.float32, n)
    idf, dflt = st.idf, st.idf_default
    cache: dict = {}
    mx = np.zeros(n, np.float32)
    for i in range(n):
        key = (q_name[i], c_name[i])
        v = cache.get(key)
        if v is None:
            d = set(q_name[i].split()) ^ set(c_name[i].split())
            v = max((idf.get(t, dflt) for t in d), default=0.0)
            if len(cache) < 2_000_000:
                cache[key] = v
        mx[i] = v
    return pd.DataFrame({"is_neighbor_number": nb, "mismatch_token_max_idf": mx})
