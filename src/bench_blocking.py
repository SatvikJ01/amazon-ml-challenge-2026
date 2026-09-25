"""Measure blocking recall and cost on one real country block.

Runs a sample of query S1 entities against the **full** target pool for that
country and source.  Shrinking the pool would inflate top-k recall by removing
the distractors that actually compete, so the pool is never subsampled.

Usage (always under a memory cap, see README):
    systemd-run --user --scope -p MemoryMax=5G -p MemorySwapMax=0 \
        .venv/bin/python -m src.bench_blocking --country India --source 2
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .blocking import BlockingConfig, RareTermIndex, build_blobs, search_topk
from .ids import OFFSET
from .normalize import normalize_text

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
REPORTS = ROOT / "reports"


def load_block(split: str, source: int, country: str) -> pd.DataFrame:
    """Load one (split, source, country) slice with normalised text only."""
    df = pd.read_parquet(
        INTERIM / f"{split}_s{source}.parquet",
        filters=[("country", "==", country)],
    )
    out = pd.DataFrame(
        {
            "code": df["code"].to_numpy(),
            "name_norm": normalize_text(df["business_name"]).to_numpy(),
            "addr_norm": normalize_text(df["business_address"]).to_numpy(),
        }
    )
    del df
    gc.collect()
    return out


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="India")
    ap.add_argument("--source", type=int, default=2)
    ap.add_argument("--n-queries", type=int, default=20_000)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--max-df", type=int, default=2_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--canon-digits", action="store_true")
    args = ap.parse_args()

    cfg = BlockingConfig(topk=args.topk, max_df_abs=args.max_df, canon_digits=args.canon_digits)
    t0 = time.time()

    q = load_block("train", 1, args.country)
    rng = np.random.default_rng(args.seed)
    q = q.iloc[np.sort(rng.choice(len(q), size=min(args.n_queries, len(q)), replace=False))]
    t = load_block("train", args.source, args.country)
    print(f"loaded {len(q):,} queries, {len(t):,} targets  ({time.time()-t0:.0f}s, rss {rss_gb():.2f} GB)")

    # Ground truth for these queries, restricted to the target source.
    ps1 = np.load(INTERIM / "gt_pair_s1.npy")
    pm = np.load(INTERIM / "gt_pair_m.npy")
    keep = ((pm // OFFSET) == args.source) & np.isin(ps1, q["code"].to_numpy())
    ps1, pm = ps1[keep], pm[keep]
    n_true = int(pm.size)

    tb = time.time()
    t_blobs = build_blobs(t["name_norm"].to_numpy(), t["addr_norm"].to_numpy(), cfg)
    q_blobs = build_blobs(q["name_norm"].to_numpy(), q["addr_norm"].to_numpy(), cfg)
    tcodes = t["code"].to_numpy()
    qcodes = q["code"].to_numpy()
    t_text = t[["name_norm", "addr_norm"]].reset_index(drop=True)
    q_text = q[["name_norm", "addr_norm"]].reset_index(drop=True)
    del t, q
    gc.collect()
    print(f"blobs in {time.time()-tb:.0f}s  (rss {rss_gb():.2f} GB)")

    index = RareTermIndex(t_text["name_norm"].to_numpy(), t_text["addr_norm"].to_numpy(), cfg)
    del t_blobs
    gc.collect()
    print(f"index built in {index.build_sec:.0f}s  (rss {rss_gb():.2f} GB)")

    idx, sc = search_topk(index, q_blobs, label=f"{args.country}-S{args.source}")
    print(f"search done  (rss {rss_gb():.2f} GB)")

    # --- recall vs depth ---
    # Map every true pair to its target position, then find its rank in the query's list.
    pos_of = pd.Series(np.arange(tcodes.size), index=tcodes)
    qpos_of = pd.Series(np.arange(qcodes.size), index=qcodes)
    tp = pos_of.reindex(pm).to_numpy()
    qp = qpos_of.reindex(ps1).to_numpy()
    rows = idx[qp]                                     # (n_true, topk)
    hit = rows == tp[:, None]
    rank = np.where(hit.any(axis=1), hit.argmax(axis=1), 10**6)

    n_empty = int((idx[:, 0] < 0).sum())

    # Dump missed pairs (not within top-k) with both sides' text for inspection.
    miss = rank >= cfg.topk
    md = pd.DataFrame({
        "s1": ps1[miss], "m": pm[miss],
        "s1_name": q_text["name_norm"].to_numpy()[qp[miss]],
        "s1_addr": q_text["addr_norm"].to_numpy()[qp[miss]],
        "m_name": t_text["name_norm"].to_numpy()[tp[miss]],
        "m_addr": t_text["addr_norm"].to_numpy()[tp[miss]],
        "top1_score": sc[qp[miss], 0],
    })
    md.to_csv(REPORTS / f"blocking_misses_{args.country}_S{args.source}_df{args.max_df}{args.tag}.tsv",
              sep="\t", index=False)
    curve = {}
    for k in (1, 3, 5, 10, 15, 20, 25, 30, 40, 50):
        if k <= cfg.topk:
            curve[k] = float((rank < k).mean())
    for k, r in curve.items():
        print(f"  recall@{k:<3d} {r:.4f}")

    # Score separation of true vs non-true neighbours at rank 0.
    true_mask = np.zeros_like(idx, dtype=bool)
    true_mask[qp[rank < cfg.topk], rank[rank < cfg.topk]] = True
    out = {
        "country": args.country,
        "target_source": args.source,
        "n_queries": int(qcodes.size),
        "n_targets": int(tcodes.size),
        "n_true_pairs": n_true,
        "config": {"topk": cfg.topk, "max_df_abs": cfg.max_df_abs, "nnz_budget": cfg.nnz_budget},
        "recall_at_k": curve,
        "queries_with_no_candidates": n_empty,
        "mean_true_score": float(sc[true_mask].mean()) if true_mask.any() else None,
        "mean_false_score": float(sc[(~true_mask) & (idx >= 0)].mean()),
        "peak_rss_gb": round(rss_gb(), 2),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / f"bench_blocking_{args.country}_S{args.source}_df{args.max_df}{args.tag}.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("queries_with_no_candidates", "mean_true_score",
                                           "mean_false_score", "peak_rss_gb", "elapsed_sec")}))


if __name__ == "__main__":
    main()
