"""E012: does reverse retrieval (target -> top-k S1) recover blocking misses?

Forward blocking gives each S1 entity its top-30 targets.  For generic names an
S1 entity can have more than 30 near-identical targets, so a true match falls
outside its list.  Because every target has exactly one owner, retrieving from
the target side (each S2/S3 record finds its top-k S1 entities) is the natural
complement.  This script measures, on one train country block and source:

* pair recall of forward top-30 alone
* pair recall of forward ∪ reverse top-k, and the number of extra pairs
* the same with canonical (leading-zero-stripped) digit terms

Usage: python -m src.exp_reverse_blocking --country India --source 2 --k 3
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .blocking import BlockingConfig, RareTermIndex, build_blobs, search_topk
from .candidates import cand_path
from .ids import OFFSET

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
INTERIM = ROOT / "data" / "interim"
REPORTS = ROOT / "reports"


def pair_keys(s1: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """Unique int key per (s1, cand): S1 numbers < 1e10 and target numbers < 1e10."""
    return (s1 % OFFSET).astype(np.int64) * OFFSET + (cand % OFFSET).astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="India")
    ap.add_argument("--source", type=int, default=2)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    t0 = time.time()
    cfg = BlockingConfig(topk=args.k)

    s1 = pd.read_parquet(PROCESSED / f"train_s1_{args.country}.parquet", columns=["code", "name_norm", "addr_norm"])
    s1codes = s1["code"].to_numpy()
    index = RareTermIndex(s1["name_norm"].to_numpy(), s1["addr_norm"].to_numpy(), cfg)
    del s1
    gc.collect()

    t = pd.read_parquet(PROCESSED / f"train_s{args.source}_{args.country}.parquet",
                        columns=["code", "name_norm", "addr_norm"])
    tcodes = t["code"].to_numpy()
    tn, ta = t["name_norm"].to_numpy(), t["addr_norm"].to_numpy()
    del t
    rev_s1, rev_c, rev_rank = [], [], []
    for b0 in range(0, tcodes.size, 250_000):
        b1 = min(b0 + 250_000, tcodes.size)
        idx, sc = search_topk(index, build_blobs(tn[b0:b1], ta[b0:b1], cfg), label=f"rev [{b0}:{b1}]")
        valid = idx >= 0
        rows = np.broadcast_to(np.arange(b0, b1)[:, None], idx.shape)
        rev_s1.append(s1codes[idx[valid]])
        rev_c.append(tcodes[rows[valid]])
        rev_rank.append(np.broadcast_to(np.arange(idx.shape[1]), idx.shape)[valid])
    del index
    rev_s1 = np.concatenate(rev_s1); rev_c = np.concatenate(rev_c); rev_rank = np.concatenate(rev_rank)

    fwd = pd.read_parquet(cand_path("trnall", args.country, args.source), columns=["s1", "cand"])
    fkeys = pair_keys(fwd["s1"].to_numpy(), fwd["cand"].to_numpy())
    del fwd

    ps1 = np.load(INTERIM / "gt_pair_s1.npy"); pm = np.load(INTERIM / "gt_pair_m.npy")
    keep = ((pm // OFFSET) == args.source) & np.isin(ps1, s1codes)
    tkeys = pair_keys(ps1[keep], pm[keep])
    n_true = tkeys.size

    fset_hit = np.isin(tkeys, fkeys)
    res = {"country": args.country, "source": args.source, "k": args.k, "n_true": int(n_true),
           "forward_recall": float(fset_hit.mean()), "forward_pairs": int(fkeys.size), "by_k": {}}
    for k in range(1, args.k + 1):
        rk = pair_keys(rev_s1[rev_rank < k], rev_c[rev_rank < k])
        union_hit = fset_hit | np.isin(tkeys, rk)
        extra = int((~np.isin(rk, fkeys)).sum())
        res["by_k"][k] = {"union_recall": float(union_hit.mean()), "extra_pairs": extra,
                          "extra_per_entity": extra / s1codes.size}
        print(f"  reverse top{k}: recall {fset_hit.mean():.4f} -> {union_hit.mean():.4f}  "
              f"(+{extra:,} pairs, {extra / s1codes.size:.1f}/entity)", flush=True)
    # Residual misses after forward ∪ reverse top-3, for categorisation.
    rk3 = pair_keys(rev_s1[rev_rank < 3], rev_c[rev_rank < 3])
    miss = ~(fset_hit | np.isin(tkeys, rk3))
    ms1, mc = ps1[keep][miss], pm[keep][miss]
    pd.DataFrame({"s1": ms1, "cand": mc}).to_parquet(REPORTS / f"residual_misses_{args.country}_S{args.source}.parquet")
    res["residual_misses_k3"] = int(miss.sum())
    res["elapsed_sec"] = round(time.time() - t0, 1)
    (REPORTS / f"exp_reverse_{args.country}_S{args.source}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
