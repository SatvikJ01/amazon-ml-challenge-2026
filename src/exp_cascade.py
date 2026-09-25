"""E012: can a cheap stage-1 model prune candidates before the expensive
string features, without losing matches?

Stage-1 features exist *before* any string comparison: retrieval score/rank,
target source, competition features and retrieval-score context.  We train it on
the same entity folds as E011 and report, on the holdout, (a) the fraction of true
pairs that survive and the fraction of all pairs kept, per threshold, and (b) the
exact F0.5 cost of pruning, by applying the stage-1 mask to E011's holdout
predictions (a lower bound: a stage-2 model retrained on survivors can only
adapt to the pruned distribution).

Usage: python -m src.exp_cascade
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .decision import threshold_sets
from .metrics import evaluate
from .train import as_sets, entity_fold, truth_for

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
EXPERIMENTS = ROOT / "experiments"

STAGE1 = ["blk_score", "blk_rank", "src", "comp_best_other", "comp_margin", "comp_is_top", "comp_n",
          "blk_score_gap", "blk_score_rank", "blk_score_src_gap"]


def main() -> None:
    df = pd.concat([pd.read_parquet(PROCESSED / f"feats_trn2_{c}.parquet", columns=["s1", "cand", "label"] + STAGE1)
                    for c in ("India", "US")], ignore_index=True)
    fold = entity_fold(df["s1"].to_numpy(), 5)
    hold = fold == 0
    inner = entity_fold(df["s1"].to_numpy() * 7 + 3, 10) == 0
    tr, es = (~hold) & (~inner), (~hold) & inner
    X = df[STAGE1].to_numpy(np.float32); y = df["label"].to_numpy()
    params = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, num_threads=12, verbose=-1, seed=0)
    m = lgb.train(params, lgb.Dataset(X[tr], y[tr]), 1000, valid_sets=[lgb.Dataset(X[es], y[es])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    p1 = m.predict(X[hold], num_iteration=m.best_iteration)
    yh = y[hold]
    out = {"best_iter": m.best_iteration, "thresholds": {}}
    ev = pd.read_parquet(EXPERIMENTS / "E011_lgb_v2feats" / "holdout_pred.parquet")
    hk = df.loc[hold, ["s1", "cand"]].reset_index(drop=True)
    hk["p1"] = p1
    ev = ev.merge(hk, on=["s1", "cand"], how="left")
    sampled = np.load(PROCESSED / "entities_trn2.npy")
    ents = sampled[entity_fold(sampled, 5) == 0]
    truth = truth_for(ents)
    s1, cand, prob = ev["s1"].to_numpy(), ev["cand"].to_numpy(), ev["prob"].to_numpy()
    base = evaluate(as_sets(threshold_sets(s1, cand, prob, 0.7), ents), truth)
    out["base_f05"] = base
    for t in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
        keep = p1 >= t
        rec = float(keep[yh == 1].mean())
        surv = float(keep.mean())
        kmask = ev["p1"].to_numpy() >= t
        f = evaluate(as_sets(threshold_sets(s1[kmask], cand[kmask], prob[kmask], 0.7), ents), truth)
        out["thresholds"][t] = {"pos_recall": rec, "pairs_kept": surv, "f05_after_prune": f, "delta": f - base}
        print(f"thr {t:<7} pos_recall {rec:.5f}  pairs_kept {surv:.4f}  F0.5 {f:.5f} (Δ {f - base:+.5f})", flush=True)
    m.save_model(str(EXPERIMENTS / "E012_stage1_model.txt"))
    (EXPERIMENTS / "E012_stage1.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
