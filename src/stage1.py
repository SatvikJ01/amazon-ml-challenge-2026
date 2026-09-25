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
    return df


def _fit(X, y, es_mask):
    return lgb.train(PARAMS, lgb.Dataset(X[~es_mask], y[~es_mask]), 1000,
                     valid_sets=[lgb.Dataset(X[es_mask], y[es_mask])],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="trn2")
    ap.add_argument("--out", default="E013_stage1")
    ap.add_argument("--countries", nargs="*", default=["India", "US"])
    args = ap.parse_args()
    out = EXPERIMENTS / args.out
    out.mkdir(parents=True, exist_ok=True)

    df = pd.concat([pd.read_parquet(PROCESSED / f"feats_{args.tag}_{c}.parquet",
                                    columns=["s1", "cand", "label"] + STAGE1_FEATURES)
                    for c in args.countries], ignore_index=True)
    X = df[STAGE1_FEATURES].to_numpy(np.float32)
    y = df["label"].to_numpy()
    s1 = df["s1"].to_numpy()
    fold = entity_fold(s1, 5)
    es = entity_fold(s1 * 7 + 3, 10) == 0

    oof = np.zeros(len(df), np.float32)
    for k in range(5):
        tr = fold != k
        m = _fit(X[tr], y[tr], es[tr])
        oof[~tr] = m.predict(X[~tr], num_iteration=m.best_iteration)
        m.save_model(str(out / f"fold{k}.txt"), num_iteration=m.best_iteration)
    full = _fit(X, y, es)
    full.save_model(str(out / "model.txt"), num_iteration=full.best_iteration)

    keep = oof >= STAGE1_THRESHOLD
    rep = {"threshold": STAGE1_THRESHOLD, "pairs_kept": float(keep.mean()),
           "pos_recall": float(keep[y == 1].mean()), "features": STAGE1_FEATURES}
    pd.DataFrame({"s1": s1, "cand": df["cand"].to_numpy(), "p1": oof}).to_parquet(out / "oof_p1.parquet", index=False)
    (out / "report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
