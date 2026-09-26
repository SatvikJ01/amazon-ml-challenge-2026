"""E035: house-number distance, empty-address context and name duplication (stage 3).

Remaining errors after E034 (holdout sample, 2026-09-26):

* near-copies carry a *neighbouring* house number (``1002->1004``, ``629->631``,
  ``6325->6327``, ``C-44->C-49``) -- small differences, often the same parity --
  while corrupted true numbers change a digit anywhere (``471->671``, ``49->29``);
* most remaining misses are empty-address candidates whose name matches, but the
  generator also makes empty-address name copies as distractors, so the decision
  depends on the entity's other candidates;
* a candidate name that recurs across the pool is weaker evidence.

Columns
-------
nd_absdiff, nd_logdiff   |x - y| of the first S1 / candidate numbers (pure digits, when
                         they differ); NaN otherwise
nd_same_parity           1 if that difference is even
nd_first_diff_pos        index (from the left) of the first differing digit, same-length
                         numbers only; NaN otherwise
ent_n_empty              empty-address candidates of the entity
ent_n_empty_sim          ... whose name token-set similarity to S1 is >= 90
empty_sim_rank           rank of this candidate's name similarity among the entity's
                         empty-address candidates (NaN for non-empty addresses)
c_name_ent_n             candidates of the entity with exactly this normalised name
c_name_pool_n            S2/S3 records of the country with exactly this normalised name
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
EXTRA3_COLS = ["nd_absdiff", "nd_logdiff", "nd_same_parity", "nd_first_diff_pos", "ent_n_empty", "ent_n_empty_sim",
               "empty_sim_rank", "c_name_ent_n", "c_name_pool_n"]


def _first_digits(addr: str) -> str:
    for t in addr.split():
        if t.isdigit():
            return t.lstrip("0") or "0"
    return ""


class PoolNames:
    """Exact normalised-name counts over the S2/S3 pools of one country (label-free)."""

    def __init__(self, split: str, country: str):
        c: Counter = Counter()
        for s in (2, 3):
            c.update(pd.read_parquet(PROCESSED / f"{split}_s{s}_{country}.parquet", columns=["name_norm"])["name_norm"])
        self.count = c


def extra_features3(s1: np.ndarray, q_addr: np.ndarray, c_addr: np.ndarray, c_name: np.ndarray,
                    name_tset: np.ndarray, addr_empty_c: np.ndarray, pool: PoolNames) -> pd.DataFrame:
    n = len(s1)
    qf = pd.Series(q_addr).map(_first_digits).to_numpy()
    cf = pd.Series(c_addr).map(_first_digits).to_numpy()
    absd = np.full(n, np.nan, np.float32); par = np.full(n, np.nan, np.float32); pos = np.full(n, np.nan, np.float32)
    for i in range(n):
        a, b = qf[i], cf[i]
        if a and b and a != b and len(a) < 12 and len(b) < 12:
            d = abs(int(a) - int(b))
            absd[i] = d
            par[i] = float(d % 2 == 0)
            if len(a) == len(b):
                pos[i] = next(k for k in range(len(a)) if a[k] != b[k])
    df = pd.DataFrame({"s1": s1, "e": np.asarray(addr_empty_c, np.float32),
                       "sim": np.asarray(name_tset, np.float32), "nm": c_name})
    df["es"] = ((df["e"] > 0) & (df["sim"] >= 90)).astype(np.float32)
    g = df.groupby("s1", sort=False)
    ent_e = g["e"].transform("sum").to_numpy(np.float32)
    ent_es = g["es"].transform("sum").to_numpy(np.float32)
    simr = df["sim"].where(df["e"] > 0)
    rank = simr.groupby(df["s1"]).rank(ascending=False, method="min").to_numpy(np.float32)
    ent_nm = df.groupby(["s1", "nm"], sort=False)["e"].transform("size").to_numpy(np.float32)
    pc = pool.count
    pool_n = pd.Series(c_name).map(lambda x: pc.get(x, 0)).to_numpy(np.float32)
    return pd.DataFrame({"nd_absdiff": absd, "nd_logdiff": np.log1p(absd), "nd_same_parity": par,
                         "nd_first_diff_pos": pos, "ent_n_empty": ent_e, "ent_n_empty_sim": ent_es,
                         "empty_sim_rank": rank, "c_name_ent_n": ent_nm, "c_name_pool_n": pool_n})
