"""Stage 1 of the matching cascade: a cheap pruning model.

Stage 1 sees only features that exist before any string comparison -- retrieval
score and rank, target source, the competition features, and retrieval-score
context over the full depth-30 list.  It drops candidates that are near-certain
non-matches, so the expensive string features (stage 2) run on ~25 % of pairs.

E012 measured the trade-off on the holdout: at p1 >= 0.001 it keeps 25.1 % of
pairs and 99.89 % of true pairs, costing -0.00004 F0.5 (noise level).

Training is cross-fitted over the same 5 entity folds used everywhere else, so
the training survivors for stage 2 are selected with out-of-fold probabilities,
exactly as test pairs will be.  A final model on all rows is used at test time.

Usage: python -m src.stage1 --tag trn2 --out E013_stage1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .train import entity_fold

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
EXPERIMENTS = ROOT / "experiments"

STAGE1_FEATURES = ["blk_score", "blk_rank", "src", "comp_best_other", "comp_margin", "comp_is_top",
                   "comp_n", "blk_score_gap", "blk_score_rank", "blk_score_src_gap"]
# With the reverse channel (``--reverse``) stage 1 also sees reverse-retrieval evidence.
STAGE1_FEATURES_REV = STAGE1_FEATURES + ["rev_score", "rev_rank", "in_fwd", "in_rev", "ret_score",
                                         "ret_score_gap", "ret_score_rank"]
STAGE1_THRESHOLD = 1e-3
PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, num_threads=12,
              verbose=-1, seed=0)


def add_stage1_context(df: pd.DataFrame) -> pd.DataFrame:
    """Retrieval-score context over the full candidate list (stage-1 inputs)."""
    g = df.groupby("s1", sort=False)["blk_score"]
    df["blk_score_gap"] = (g.transform("max") - df["blk_score"]).astype(np.float32)
    df["blk_score_rank"] = g.rank(ascending=False, method="min").astype(np.float32)
    df["blk_score_src_gap"] = (df.groupby(["s1", "src"], sort=False)["blk_score"].transform("max")
                               - df["blk_score"]).astype(np.float32)
    if "ret_score" in df.columns:
        r = df.groupby("s1", sort=False)["ret_score"]
        df["ret_score_gap"] = (r.transform("max") - df["ret_score"]).astype(np.float32)
        df["ret_score_rank"] = r.rank(ascending=False, method="min").astype(np.float32)
    return df


def _fit(X, y, es_mask, w=None):
    w = np.ones(len(y), np.float32) if w is None else w
    return lgb.train(PARAMS, lgb.Dataset(X[~es_mask], y[~es_mask], weight=w[~es_mask]), 1000,
                     valid_sets=[lgb.Dataset(X[es_mask], y[es_mask], weight=w[es_mask])],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def _sample(idx: np.ndarray, y: np.ndarray, neg_frac: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """All positives + a ``neg_frac`` sample of negatives, negatives weighted 1/neg_frac
    so predicted probabilities stay calibrated."""
    if neg_frac >= 1.0:
        return idx, np.ones(idx.size, np.float32)
    pos = idx[y[idx] == 1]
    neg = idx[y[idx] == 0]
    neg = neg[rng.random(neg.size) < neg_frac]
    sel = np.sort(np.concatenate([pos, neg]))
    w = np.where(y[sel] == 1, 1.0, 1.0 / neg_frac).astype(np.float32)
    return sel, w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="trn2")
    ap.add_argument("--out", default="E013_stage1")
    ap.add_argument("--countries", nargs="*", default=["India", "US"])
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--v3", action="store_true", help="multi-channel (forward/reverse/key) feature set")
    ap.add_argument("--neg-frac", type=float, default=1.0, help="negative sampling rate for training folds")
    args = ap.parse_args()
    if args.v3:
        from .v3 import STAGE1_V3
        feat_list = STAGE1_V3
    else:
        feat_list = STAGE1_FEATURES_REV if args.reverse else STAGE1_FEATURES
    out = EXPERIMENTS / args.out
    out.mkdir(parents=True, exist_ok=True)

    # Column-wise load into a pre-allocated float32 matrix (a frame concat of
    # 24.5 M rows would copy everything and exceed the memory cap).
    import gc
    import pyarrow.parquet as pq
    paths = [PROCESSED / f"feats_{args.tag}_{c}.parquet" for c in args.countries]
    sizes = [pq.ParquetFile(p).metadata.num_rows for p in paths]
    n = sum(sizes)
    X = np.empty((n, len(feat_list)), np.float32)
    y = np.empty(n, np.int8); s1 = np.empty(n, np.int64); cand = np.empty(n, np.int64)
    off = 0
    for p, sz in zip(paths, sizes):
        for j, col in enumerate(feat_list):
            X[off:off + sz, j] = pq.read_table(p, columns=[col]).column(col).to_numpy()
        y[off:off + sz] = pq.read_table(p, columns=["label"]).column("label").to_numpy()
        s1[off:off + sz] = pq.read_table(p, columns=["s1"]).column("s1").to_numpy()
        cand[off:off + sz] = pq.read_table(p, columns=["cand"]).column("cand").to_numpy()
        off += sz
    gc.collect()
    rng = np.random.default_rng(0)
    fold = entity_fold(s1, 5)
    es = entity_fold(s1 * 7 + 3, 10) == 0
    print(f"stage 1: {n:,} pairs x {len(feat_list)} features", flush=True)

    oof = np.zeros(n, np.float32)
    for k in range(5):
        sel, w = _sample(np.flatnonzero(fold != k), y, args.neg_frac, rng)
        m = _fit(X[sel], y[sel], es[sel], w)
        te = np.flatnonzero(fold == k)
        oof[te] = m.predict(X[te], num_iteration=m.best_iteration)
        m.save_model(str(out / f"fold{k}.txt"), num_iteration=m.best_iteration)
        del sel, w
    sel, w = _sample(np.arange(len(y)), y, args.neg_frac, rng)
    full = _fit(X[sel], y[sel], es[sel], w)
    full.save_model(str(out / "model.txt"), num_iteration=full.best_iteration)

    keep = oof >= STAGE1_THRESHOLD
    rep = {"threshold": STAGE1_THRESHOLD, "pairs_kept": float(keep.mean()),
           "pos_recall": float(keep[y == 1].mean()), "features": feat_list}
    pd.DataFrame({"s1": s1, "cand": cand, "p1": oof}).to_parquet(out / "oof_p1.parquet", index=False)
    (out / "report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
