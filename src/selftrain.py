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


def fit(X, y, w, es):
    return lgb.train(LGB_PARAMS, lgb.Dataset(X[~es], y[~es], weight=w[~es]), 3000,
                     valid_sets=[lgb.Dataset(X[es], y[es], weight=w[es])],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


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
    src = pd.read_parquet(PROCESSED / f"feats_{args.tag}_{args.source}.parquet")
    tgt = pd.read_parquet(PROCESSED / f"feats_{args.tag}_{args.target}.parquet")
    feats = [c for c in src.columns if c not in NON_FEATURES]
    Xs, ys = src[feats].to_numpy(np.float32), src["label"].to_numpy().astype(np.float32)
    Xt = tgt[feats].to_numpy(np.float32)
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
