"""E033: alphanumeric house-number agreement and within-entity number consensus.

Two patterns dominate the stage-2 holdout errors (error sample, 2026-09-26):

* **False positives** are near-copies of a true record whose house / unit number
  was replaced, often by an alphanumeric token (``93c``, ``226a``, ``a214``,
  ``7a``, ``77th``).  The digit-only features in ``features.py`` never see such
  tokens, so "same name, same street, different number" scores like a match.
* **Missed true matches** often carry a *corrupted* number (``671`` vs S1 ``471``,
  ``60`` vs ``603``), but the corruption is shared by several records of the same
  cluster, whereas a near-copy's number appears once.  So the support of a
  candidate's number among the entity's other candidates separates the two.

All features are comparisons (no record identities).  They are computed after
stage 2, because the consensus features weight candidates by their stage-2
probability ``p2`` (cross-fitted on train, model scores on test; NaN -> 0).

Columns
-------
anum_n_q, anum_n_c     numeric tokens (containing a digit) on each side
anum_first_rel         first numeric tokens: 0 missing, 1 equal, 2 truncated /
                       zero-padded, 3 same digits with other letters, 4 one edit,
                       5 conflict
anum_q_miss            S1 numeric tokens with no compatible candidate token
anum_c_extra           candidate numeric tokens with no compatible S1 token
anum_c_alpha_extra     ... of which contain a letter
nsup_first_p / _n      support of the candidate's first number among the entity's
                       other candidates (p2-weighted sum / count)
nsup_extra_p / _n      support of the candidate's unexplained numbers (min over
                       those numbers); NaN when every number is explained
nsup_s1_p              support of the S1 numbers in the candidate pool (min over
                       S1 numbers, entity level)
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

from .normalize import NULL_TOKENS

_ORD = re.compile(r"^(\d+)(?:st|nd|rd|th)$")
_NONDIGIT = re.compile(r"\D")
EXTRA_COLS = ["anum_n_q", "anum_n_c", "anum_first_rel", "anum_q_miss", "anum_c_extra", "anum_c_alpha_extra",
              "nsup_first_p", "nsup_first_n", "nsup_extra_p", "nsup_extra_n", "nsup_s1_p"]


def _numtoks(addr: str) -> tuple:
    """((raw, canonical), ...) for every token containing a digit.  ``raw`` keeps
    leading zeros (``002`` is ``1002`` with its first digit dropped); ``canonical``
    strips them and ordinal suffixes (``0040`` -> ``40``, ``77th`` -> ``77``)."""
    out = []
    for t in addr.split():
        if t in NULL_TOKENS or not any(ch.isdigit() for ch in t):
            continue
        m = _ORD.match(t)
        raw = m.group(1) if m else t
        can = raw.lstrip("0")
        if not can or not can[0].isdigit():
            can = "0" + can if raw.startswith("0") else can
        out.append((raw, can or "0"))
    return tuple(out)


def _rel(x: tuple, y: tuple) -> int:
    (xr, xc), (yr, yc) = x, y
    if xc == yc:
        return 1
    if xc.isdigit() and yc.isdigit():
        for a, b in ((xr, yr), (xc, yc)):
            if abs(len(a) - len(b)) == 1:
                lo, hi = (a, b) if len(a) < len(b) else (b, a)
                if hi.startswith(lo) or hi.endswith(lo):
                    return 2
    elif _NONDIGIT.sub("", xc) == _NONDIGIT.sub("", yc):
        return 3
    if abs(len(xc) - len(yc)) <= 1 and Levenshtein.distance(xc, yc) <= 1:
        return 4
    return 5


def _map_unique(arr: np.ndarray, fn) -> np.ndarray:
    codes, uniques = pd.factorize(arr)
    vals = np.empty(len(uniques), dtype=object)
    for i, u in enumerate(uniques):
        vals[i] = fn(u)
    return vals[codes]


def extra_features(s1: np.ndarray, p2: np.ndarray, q_addr: np.ndarray, c_addr: np.ndarray) -> pd.DataFrame:
    """Features for aligned pair arrays (any row order)."""
    n = len(s1)
    qt = _map_unique(q_addr, _numtoks)
    ct = _map_unique(c_addr, _numtoks)
    p2 = np.nan_to_num(np.asarray(p2, np.float64), nan=0.0)
    nq = np.fromiter((len(x) for x in qt), np.float32, n)
    nc = np.fromiter((len(x) for x in ct), np.float32, n)
    first = np.zeros(n, np.float32)
    qmiss = np.full(n, np.nan, np.float32)
    cextra = np.full(n, np.nan, np.float32)
    calpha = np.full(n, np.nan, np.float32)
    extra_sets = np.empty(n, dtype=object)
    for i in range(n):
        q, c = qt[i], ct[i]
        if not q or not c:
            extra_sets[i] = ()
            continue
        first[i] = _rel(q[0], c[0])
        ok_c = [False] * len(c)
        miss = 0
        for x in q:
            hit = False
            for j, y in enumerate(c):
                if _rel(x, y) <= 3:
                    hit = True
                    ok_c[j] = True
            miss += not hit
        ex = tuple(c[j][1] for j in range(len(c)) if not ok_c[j])
        qmiss[i], cextra[i] = miss, len(ex)
        calpha[i] = sum(1 for t in ex if not t.isdigit())
        extra_sets[i] = ex

    # Within-entity consensus: per entity, p2-weighted and plain counts of the
    # candidates containing each canonical number.
    sup_first_p = np.zeros(n, np.float32); sup_first_n = np.zeros(n, np.float32)
    sup_ex_p = np.full(n, np.nan, np.float32); sup_ex_n = np.full(n, np.nan, np.float32)
    sup_s1 = np.full(n, np.nan, np.float32)
    order = np.argsort(s1, kind="stable")
    so = s1[order]
    bounds = np.r_[np.flatnonzero(np.r_[True, so[1:] != so[:-1]]), n]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        rows = order[lo:hi]
        tp: dict = {}; tn: dict = {}
        for r in rows:
            for t in {y for _, y in ct[r]}:
                tp[t] = tp.get(t, 0.0) + p2[r]
                tn[t] = tn.get(t, 0) + 1
        qs = {y for _, y in qt[rows[0]]}
        s1sup = min((tp.get(t, 0.0) for t in qs), default=np.nan)
        for r in rows:
            sup_s1[r] = s1sup
            c = ct[r]
            if c:
                f = c[0][1]
                sup_first_p[r] = tp[f] - p2[r]
                sup_first_n[r] = tn[f] - 1
            ex = extra_sets[r]
            if ex:
                sup_ex_p[r] = min(tp[t] - p2[r] for t in ex)
                sup_ex_n[r] = min(tn[t] - 1 for t in ex)
    return pd.DataFrame({
        "anum_n_q": nq, "anum_n_c": nc, "anum_first_rel": first, "anum_q_miss": qmiss,
        "anum_c_extra": cextra, "anum_c_alpha_extra": calpha,
        "nsup_first_p": sup_first_p, "nsup_first_n": sup_first_n,
        "nsup_extra_p": sup_ex_p, "nsup_extra_n": sup_ex_n, "nsup_s1_p": sup_s1,
    })
