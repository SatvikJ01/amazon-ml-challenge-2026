"""E031: self-training for an unseen country (France proxy = India under LOCO).

France has no labels, and a model trained on one country transfers poorly to
another (E014: US -> India 0.884 vs 0.955 in-distribution).  Self-training adapts
the matcher to the target country using only its *unlabelled* candidate pairs:

1. train on the source country (labels);
2. score the target country; pseudo-label confident pairs
   (p >= ``HI`` -> 1, p <= ``LO`` -> 0), everything else unused;
3. retrain on source labels + target pseudo-labels (target weight ``w``);
4. evaluate on the target holdout fold with its *true* labels.

No target ground truth enters training.  If the procedure closes a meaningful
part of the LOCO gap here, the same code is applied to France at test time.

Usage: python -m src.selftrain --tag v3c --source US --target India
"""
from __future__ import annotations

import argparse
import os
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .decision import select_sets
from .inference import enforce_exclusivity
from .metrics import evaluate, evaluate_by_group
from .train import LGB_PARAMS, NON_FEATURES, as_sets, entity_fold, truth_for

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
EXPERIMENTS = ROOT / "experiments"
HI, LO = 0.98, 0.02


XGB = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device=os.environ.get("XGB_DEVICE", "cuda"),
           learning_rate=0.05, max_depth=9, min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
           reg_lambda=1.0, max_bin=128, seed=0)


class _Fit:
    """GPU XGBoost wrapper with the LightGBM-like interface used below."""
    def __init__(self, b):
        self.b, self.best_iteration = b, b.best_iteration
    def predict(self, X, num_iteration=None):
        import xgboost as xgb
        return self.b.predict(xgb.DMatrix(X), iteration_range=(0, self.best_iteration + 1))


def fit(X, y, w, es):
    import xgboost as xgb
    dtr = xgb.QuantileDMatrix(X[~es], y[~es], weight=w[~es], max_bin=128)
    dva = xgb.QuantileDMatrix(X[es], y[es], weight=w[es], ref=dtr, max_bin=128)
    return _Fit(xgb.train(XGB, dtr, 3000, evals=[(dva, "es")], early_stopping_rounds=100, verbose_eval=False))


def score(tgt, prob, ents, truth):
    s1, cand = tgt["s1"].to_numpy(), tgt["cand"].to_numpy()
    m = enforce_exclusivity(s1, cand, prob)
    pred = as_sets(select_sets(s1[m], cand[m], prob[m])[0], ents)
    nt = {e: ("0" if not truth[e] else "1" if len(truth[e]) == 1 else "2+") for e in ents}
    return evaluate(pred, truth), evaluate_by_group(pred, truth, nt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v3c")
    ap.add_argument("--source", default="US")
    ap.add_argument("--target", default="India")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--w", type=float, default=1.0)
    args = ap.parse_args()
    out = EXPERIMENTS / f"E031_selftrain_{args.source}2{args.target}"
    out.mkdir(parents=True, exist_ok=True)
    import pyarrow.parquet as pq

    def load(country):
        path = PROCESSED / f"feats_{args.tag}_{country}.parquet"
        cols = [c for c in pq.read_schema(path).names if c not in NON_FEATURES]
        n = pq.ParquetFile(path).metadata.num_rows
        X = np.empty((n, len(cols)), np.float32)
        for j, c in enumerate(cols):
            X[:, j] = pq.read_table(path, columns=[c]).column(c).to_numpy()
        meta = pq.read_table(path, columns=["s1", "cand", "label"]).to_pandas()
        return X, meta, cols

    Xs, src, feats = load(args.source)
    Xt, tgt, feats_t = load(args.target)
    assert feats == feats_t
    ys = src["label"].to_numpy().astype(np.float32)
    es_s = entity_fold(src["s1"].to_numpy() * 7 + 3, 10) == 0
    ents = np.unique(tgt["s1"].to_numpy())
    ents = ents[entity_fold(ents, 5) == 0]
    truth = truth_for(ents)
    hold = np.isin(tgt["s1"].to_numpy(), ents)

    m = fit(Xs, ys, np.ones(len(ys), np.float32), es_s)
    p = m.predict(Xt, num_iteration=m.best_iteration)
    f0, b0 = score(tgt[hold], p[hold], ents, truth)
    rep = {"loco_f05": f0, "loco_by_n_true": b0, "rounds": []}
    print(f"LOCO {args.source}->{args.target}: F0.5 {f0:.4f}  singletons {b0['0']['f05']:.4f}", flush=True)
    es_t = entity_fold(tgt["s1"].to_numpy() * 7 + 3, 10) == 0
    for r in range(args.rounds):
        conf = (p >= HI) | (p <= LO)
        yt = (p >= HI).astype(np.float32)
        X = np.concatenate([Xs, Xt[conf]]); y = np.concatenate([ys, yt[conf]])
        w = np.concatenate([np.ones(len(ys), np.float32), np.full(int(conf.sum()), args.w, np.float32)])
        es = np.concatenate([es_s, es_t[conf]])
        m = fit(X, y, w, es)
        p = m.predict(Xt, num_iteration=m.best_iteration)
        f, b = score(tgt[hold], p[hold], ents, truth)
        rep["rounds"].append({"round": r + 1, "pseudo_pairs": int(conf.sum()), "pseudo_pos": int(yt[conf].sum()),
                              "f05": f, "by_n_true": b})
        print(f"round {r + 1}: pseudo {int(conf.sum()):,} ({int(yt[conf].sum()):,} pos)  F0.5 {f:.4f}  "
              f"singletons {b['0']['f05']:.4f}", flush=True)
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=float))


if __name__ == "__main__":
    main()
