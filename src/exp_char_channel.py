"""E018 pilot: a character n-gram retrieval channel for near-miss spellings.

Residual blocking misses after forward + reverse retrieval (India S2) are 71 %
zero-name-token-overlap pairs: phonetic transliterations whose consonant
skeleton is off by one letter (``lf ntstrk`` vs ``lf ntstrs``) and concatenated
domain forms (``exportprivatedelhi com``).  Exact-term keys cannot bridge a
one-character difference; character n-grams can.

The channel indexes, per record:
    g:<3-gram>  of the no-space consonant skeleton of the name
    h:<4-gram>  of the no-space normalised name
    d:<num>     canonical (leading-zero-stripped) address numbers, as anchors
and is queried on its own (its own IDF, its own top-k), then unioned with the
existing forward candidates.  Measured on a sample of S1 queries against the
full target pool.

Usage: python -m src.exp_char_channel --country India --source 2 --n-queries 20000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .blocking import BlockingConfig, RareTermIndex, search_topk
from .candidates import cand_path
from .ids import OFFSET
from .normalize import skeleton_tokens

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
INTERIM = ROOT / "data" / "interim"
REPORTS = ROOT / "reports"


def char_blobs(name_norm: np.ndarray, addr_norm: np.ndarray) -> list[str]:
    out = []
    for name, addr in zip(name_norm, addr_norm):
        sk = "".join(skeleton_tokens(name))
        ns = name.replace(" ", "")
        terms = ["g:" + sk[i:i + 3] for i in range(len(sk) - 2)]
        terms += ["h:" + ns[i:i + 4] for i in range(len(ns) - 3)]
        terms += ["d:" + (t.lstrip("0") or "0") for t in addr.split() if t.isdigit()]
        out.append(" ".join(terms))
    return out


class CharIndex(RareTermIndex):
    """RareTermIndex over character n-gram blobs instead of word blobs."""

    def __init__(self, name_norm, addr_norm, cfg, chunk: int = 500_000):
        import gc
        import scipy.sparse as sp
        self.cfg = cfg
        from .blocking import _hasher
        self.hasher = _hasher(cfg)
        t0 = time.time()
        parts = [self.hasher.transform(char_blobs(name_norm[i:i + chunk], addr_norm[i:i + chunk])).tocsr()
                 for i in range(0, len(name_norm), chunk)]
        X = sp.vstack(parts, format="csr"); del parts; gc.collect()
        self.n_targets = X.shape[0]
        df = np.bincount(X.indices, minlength=cfg.n_features).astype(np.int64)
        self.df = df
        keep = (df > 0) & (df <= cfg.max_df_abs)
        idf = np.zeros(cfg.n_features, np.float32)
        idf[keep] = np.log((self.n_targets + 1.0) / (df[keep] + 1.0)) + 1.0
        self.idf = idf
        self._weight_inplace(X)
        self.BT = X.T.tocsr(); del X; gc.collect()
        self.build_sec = time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="India")
    ap.add_argument("--source", type=int, default=2)
    ap.add_argument("--n-queries", type=int, default=20_000)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--max-df", type=int, default=5_000)
    args = ap.parse_args()
    t0 = time.time()
    cfg = BlockingConfig(topk=args.topk, max_df_abs=args.max_df, nnz_budget=20_000_000)

    q = pd.read_parquet(PROCESSED / f"train_s1_{args.country}.parquet", columns=["code", "name_norm", "addr_norm"])
    q = q.sample(min(args.n_queries, len(q)), random_state=0)
    t = pd.read_parquet(PROCESSED / f"train_s{args.source}_{args.country}.parquet",
                        columns=["code", "name_norm", "addr_norm"])
    tcodes = t["code"].to_numpy()
    index = CharIndex(t["name_norm"].to_numpy(), t["addr_norm"].to_numpy(), cfg)
    del t
    idx, _ = search_topk(index, char_blobs(q["name_norm"].to_numpy(), q["addr_norm"].to_numpy()), label="char")
    qcodes = q["code"].to_numpy()

    ps1, pm = np.load(INTERIM / "gt_pair_s1.npy"), np.load(INTERIM / "gt_pair_m.npy")
    k = ((pm // OFFSET) == args.source) & np.isin(ps1, qcodes)
    ps1, pm = ps1[k], pm[k]
    key = lambda a, b: (a % OFFSET) * OFFSET + (b % OFFSET)
    tk = key(ps1, pm)
    fwd = pd.read_parquet(cand_path("trnall", args.country, args.source), columns=["s1", "cand"])
    fwd = fwd[np.isin(fwd["s1"].to_numpy(), qcodes)]
    f_hit = np.isin(tk, key(fwd["s1"].to_numpy(), fwd["cand"].to_numpy()))
    rows = np.broadcast_to(np.arange(qcodes.size)[:, None], idx.shape)
    res = {"forward_recall": float(f_hit.mean()), "n_true": int(tk.size), "by_k": {}}
    print(f"forward top-30 recall {f_hit.mean():.4f}  ({tk.size:,} true pairs)")
    for kk in (1, 3, 5, 10, 20):
        v = (idx[:, :kk] >= 0)
        ck = key(qcodes[rows[:, :kk][v]], tcodes[idx[:, :kk][v]])
        c_hit = np.isin(tk, ck)
        res["by_k"][kk] = {"char_alone": float(c_hit.mean()), "union": float((f_hit | c_hit).mean())}
        print(f"  char top{kk:<2}: alone {c_hit.mean():.4f}   forward ∪ char {(f_hit | c_hit).mean():.4f}")
    res["elapsed_sec"] = round(time.time() - t0, 1)
    (REPORTS / f"exp_char_{args.country}_S{args.source}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
