"""Train the pairwise matcher and evaluate it with the competition metric.

Validation protocol
-------------------
* Folds are assigned **per Source-1 entity** (hash of the id), so all candidate
  pairs of an entity sit in the same fold.  Pairs of one entity are highly
  correlated; splitting them across folds would leak.
* Early stopping uses an inner 10 % entity split of the *training* folds, never
  the holdout, so the holdout score is not tuned on itself.
* The holdout is scored with ``src.metrics.evaluate`` against the **full**
  ground truth of every holdout entity, including true matches the blocker never
  retrieved.  That makes the local number directly comparable to the
  leaderboard, blocking losses included.
* Optional ``--train-countries`` / ``--eval-countries`` give the
  leave-one-country-out stress test used as the proxy for unseen France.

Usage:
    python -m src.train --tag trn --exp E003
    python -m src.train --tag trn --exp E004_loco_us2in --train-countries US --eval-countries India
"""

from __future__ import annotations

import argparse
import os
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .decision import select_sets, threshold_sets
from .metrics import candidate_recall_ceiling, evaluate, evaluate_by_group, micro_prf

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
INTERIM = ROOT / "data" / "interim"
EXPERIMENTS = ROOT / "experiments"

NON_FEATURES = {"s1", "cand", "label", "country", "fold"}

LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.05,
    num_leaves=127,
    min_data_in_leaf=200,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    max_bin=127,
    num_threads=12,
    verbose=-1,
    seed=0,
)


def entity_fold(codes: np.ndarray, n_folds: int) -> np.ndarray:
    return (pd.util.hash_array(codes.astype(np.int64)) % np.uint64(n_folds)).astype(np.int8)


def load_feats(tag: str, countries: list[str]) -> pd.DataFrame:
    parts = []
    for c in countries:
        df = pd.read_parquet(PROCESSED / f"feats_{tag}_{c}.parquet")
        df["country"] = c
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def truth_for(entities: np.ndarray) -> dict:
    """Full ground-truth sets (codes) for the given S1 entities, empty for singletons."""
    ps1 = np.load(INTERIM / "gt_pair_s1.npy")
    pm = np.load(INTERIM / "gt_pair_m.npy")
    keep = np.isin(ps1, entities)
    ps1, pm = ps1[keep], pm[keep]
    order = np.argsort(ps1, kind="stable")
    ps1, pm = ps1[order], pm[order]
    uniq, start = np.unique(ps1, return_index=True)
    bounds = np.append(start, ps1.size)
    truth = {int(e): set() for e in entities}
    for i, e in enumerate(uniq):
        truth[int(e)] = set(pm[bounds[i]:bounds[i + 1]].tolist())
    return truth


def as_sets(d: dict, entities: np.ndarray) -> dict:
    return {int(e): set(np.asarray(d.get(int(e), ())).tolist()) for e in entities}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--countries", nargs="*", default=["India", "US"])
    ap.add_argument("--train-countries", nargs="*", default=None)
    ap.add_argument("--eval-countries", nargs="*", default=None)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--holdout-fold", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--drop-features", nargs="*", default=[])
    ap.add_argument("--train-frac", type=float, default=1.0, help="fraction of training entities (holdout unchanged)")
    ap.add_argument("--model", choices=["lgb", "xgb"], default="lgb",
                    help="xgb = XGBoost on the GPU (second model family for ensembling)")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = EXPERIMENTS / args.exp
    out_dir.mkdir(parents=True, exist_ok=True)

    tr_c = args.train_countries or args.countries
    ev_c = args.eval_countries or args.countries
    loco = set(tr_c) != set(ev_c)

    # Memory-lean loading: one country frame at a time, sliced straight into
    # pre-allocated float32 matrices (the naive version held ~3 copies).
    import pyarrow.parquet as pq
    paths = {c: PROCESSED / f"feats_{args.tag}_{c}.parquet" for c in set(tr_c) | set(ev_c)}
    schema_cols = pq.read_schema(next(iter(paths.values()))).names
    feats = [c for c in schema_cols if c not in NON_FEATURES and c not in args.drop_features]

    def masks_for(c):
        s1 = pq.read_table(paths[c], columns=["s1"]).column("s1").to_numpy()
        fold = entity_fold(s1, args.n_folds)
        hold = fold == args.holdout_fold
        inner = entity_fold(s1 * 7 + 3, 10) == 0          # 10 % early-stop split
        tr = (~hold) & (~inner) if c in tr_c else np.zeros(s1.size, bool)
        if args.train_frac < 1.0:        # learning curve: subsample training entities only
            tr &= entity_fold(s1 * 13 + 5, 1000) < int(args.train_frac * 1000)
        es = (~hold) & inner if c in tr_c else np.zeros(s1.size, bool)
        ev = hold if c in ev_c else np.zeros(s1.size, bool)
        return tr, es, ev

    masks = {c: masks_for(c) for c in paths}
    n_tr = sum(int(m[0].sum()) for m in masks.values())
    n_es = sum(int(m[1].sum()) for m in masks.values())
    X_tr = np.empty((n_tr, len(feats)), np.float32); y_tr = np.empty(n_tr, np.float32)
    X_es = np.empty((n_es, len(feats)), np.float32); y_es = np.empty(n_es, np.float32)
    ev_parts = []
    i_tr = i_es = 0
    for c, (mtr, mes, mev) in masks.items():
        # Column-by-column reads: never hold a whole country frame plus a copy.
        k_tr, k_es, k_ev = int(mtr.sum()), int(mes.sum()), int(mev.sum())
        X_ev = np.empty((k_ev, len(feats)), np.float32)
        for j, col in enumerate(feats):
            v = pq.read_table(paths[c], columns=[col]).column(col).to_numpy().astype(np.float32, copy=False)
            X_tr[i_tr:i_tr + k_tr, j] = v[mtr]
            X_es[i_es:i_es + k_es, j] = v[mes]
            X_ev[:, j] = v[mev]
            del v
        y = pq.read_table(paths[c], columns=["label"]).column("label").to_numpy().astype(np.float32)
        y_tr[i_tr:i_tr + k_tr] = y[mtr]
        y_es[i_es:i_es + k_es] = y[mes]
        i_tr += k_tr
        i_es += k_es
        if k_ev:
            e = pq.read_table(paths[c], columns=["s1", "cand", "src", "label"]).to_pandas()
            e = e[mev].reset_index(drop=True)
            e["country"] = c
            ev_parts.append((e, X_ev))
        del y
    print(f"train pairs {n_tr:,}  early-stop pairs {n_es:,}  features {len(feats)}  "
          f"pos_rate {y_tr.mean():.4f}", flush=True)

    if args.model == "xgb":
        import xgboost as xgb
        params = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device=os.environ.get("XGB_DEVICE", "cuda"),
                      learning_rate=0.05, max_depth=9, min_child_weight=5, subsample=0.8,
                      colsample_bytree=0.8, reg_lambda=1.0, max_bin=128, seed=0)
        dtr = xgb.QuantileDMatrix(X_tr, y_tr, max_bin=128, feature_names=feats)
        dva = xgb.QuantileDMatrix(X_es, y_es, ref=dtr, max_bin=128, feature_names=feats)
        booster = xgb.train(params, dtr, num_boost_round=args.rounds, evals=[(dva, "es")],
                            early_stopping_rounds=100, verbose_eval=200)
        del dtr, dva, X_tr, y_tr, X_es, y_es
        booster.save_model(str(out_dir / "model.json"))

        class _M:  # minimal adapter so the evaluation code below is shared
            best_iteration = booster.best_iteration
            def predict(self, X, num_iteration=None):
                return booster.predict(xgb.DMatrix(X, feature_names=feats),
                                       iteration_range=(0, booster.best_iteration + 1))
            def feature_importance(self, kind):
                sc = booster.get_score(importance_type="gain")
                return np.array([sc.get(f, 0.0) for f in feats])
        model = _M()
    else:
        dtr = lgb.Dataset(X_tr, y_tr, feature_name=feats, free_raw_data=True)
        dva = lgb.Dataset(X_es, y_es, reference=dtr)
        model = lgb.train(
            LGB_PARAMS, dtr, num_boost_round=args.rounds, valid_sets=[dva],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)],
        )
        del dtr, dva, X_tr, y_tr, X_es, y_es
        model.save_model(str(out_dir / "model.txt"))

    ev = pd.concat([e for e, _ in ev_parts], ignore_index=True)
    ev["prob"] = np.concatenate([model.predict(X, num_iteration=model.best_iteration)
                                 for _, X in ev_parts]).astype(np.float32)
    del ev_parts
    ev.to_parquet(out_dir / "holdout_pred.parquet", index=False)

    # Evaluation entities: every holdout entity of the eval countries, including
    # entities that got no candidates at all (they are still scored).
    sampled = np.load(PROCESSED / f"entities_{args.tag}.npy")
    country_of = {}
    for c in ev_c:
        codes = pd.read_parquet(PROCESSED / f"train_s1_{c}.parquet", columns=["code"])["code"].to_numpy()
        codes = codes[np.isin(codes, sampled)]
        for x in codes:
            country_of[int(x)] = c
    ents = np.array(sorted(country_of), dtype=np.int64)
    ents = ents[entity_fold(ents, args.n_folds) == args.holdout_fold]
    truth = truth_for(ents)

    s1, cand, prob = ev["s1"].to_numpy(), ev["cand"].to_numpy(), ev["prob"].to_numpy()
    cands_sets = as_sets(pd.Series(cand).groupby(s1).apply(np.array).to_dict(), ents)
    ceiling = candidate_recall_ceiling(cands_sets, truth)

    results = {}
    for thr in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        pred = as_sets(threshold_sets(s1, cand, prob, thr), ents)
        results[f"thr_{thr}"] = evaluate(pred, truth)
    best_thr = max((k for k in results if k.startswith("thr_")), key=results.get)

    for missed in (0.0, 0.25, 0.5):
        sets, _, _ = select_sets(s1, cand, prob, missed=missed)
        results[f"expF_m{missed}"] = evaluate(as_sets(sets, ents), truth)
    best_rule = max(results, key=results.get)

    if best_rule.startswith("thr_"):
        pred = as_sets(threshold_sets(s1, cand, prob, float(best_rule[4:])), ents)
    else:
        sets, _, _ = select_sets(s1, cand, prob, missed=float(best_rule.split("_m")[1]))
        pred = as_sets(sets, ents)

    groups_country = {e: country_of[e] for e in ents}
    ntrue_bucket = {e: ("0" if len(truth[e]) == 0 else "1" if len(truth[e]) == 1 else
                        "2-3" if len(truth[e]) <= 3 else "4-6" if len(truth[e]) <= 6 else "7+") for e in ents}

    report = {
        "exp": args.exp,
        "tag": args.tag,
        "train_countries": tr_c,
        "eval_countries": ev_c,
        "loco": loco,
        "n_features": len(feats),
        "best_iteration": model.best_iteration,
        "n_eval_entities": int(ents.size),
        "blocking_ceiling": ceiling,
        "decision_rules": results,
        "best_rule": best_rule,
        "best_f05": results[best_rule],
        "by_country": evaluate_by_group(pred, truth, groups_country),
        "by_n_true": evaluate_by_group(pred, truth, ntrue_bucket),
        "micro": micro_prf(pred, truth),
        "pair_auc_note": "see holdout_pred.parquet",
        "elapsed_sec": round(time.time() - t0, 1),
    }
    imp = pd.Series(model.feature_importance("gain"), index=feats).sort_values(ascending=False)
    imp.to_csv(out_dir / "feature_importance.csv")
    report["top_features"] = imp.head(15).round(0).to_dict()
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=float))
    (out_dir / "features.json").write_text(json.dumps(feats))

    print(json.dumps({k: report[k] for k in ("best_rule", "best_f05", "decision_rules",
                                             "by_country", "by_n_true")}, indent=2, default=float))
    print(f"ceiling max_f05={ceiling['max_f05']:.4f} pair_recall={ceiling['pair_recall']:.4f} "
          f"mean_cands={ceiling['mean_candidates_per_entity']:.1f}  ({report['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
