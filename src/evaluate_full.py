"""Full-density evaluation on one training country block.

Scores **every** S1 entity of a train country with a saved model and evaluates
decision rules with and without the hard exclusivity constraint.  This is the
only faithful way to measure exclusivity: it acts through competition between
entities, so it is invisible on a sparse holdout but decisive on the test set,
where every entity is present.

Use a model trained on the *other* country (``src.train --train-countries US``)
so every scored entity is out-of-sample.  The result then doubles as the
leave-one-country-out stress test, our proxy for the unseen France block.

Usage:
    python -m src.evaluate_full --exp E011_loco_us --country India --depth 30
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .decision import select_sets, threshold_sets
from .inference import enforce_exclusivity, score_split
from .metrics import evaluate, evaluate_by_group, micro_prf
from .train import as_sets, truth_for

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
EXPERIMENTS = ROOT / "experiments"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--country", required=True)
    ap.add_argument("--depth", type=int, default=30)
    ap.add_argument("--chunk-entities", type=int, default=40_000)
    ap.add_argument("--stage1", default=None, help="stage-1 experiment dir (cascade)")
    ap.add_argument("--reverse", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = EXPERIMENTS / args.exp
    score_path = out_dir / f"full_scores_{args.country}_d{args.depth}.parquet"
    if score_path.exists():
        sc = pd.read_parquet(score_path)
    else:
        s1m = str(EXPERIMENTS / args.stage1 / "model.txt") if args.stage1 else None
        sc = score_split(args.exp, "train", "trnall", args.depth, args.chunk_entities, [args.country],
                         stage1_model=s1m, use_reverse=args.reverse)
        sc.to_parquet(score_path, index=False)
    print(f"scored {len(sc):,} pairs ({time.time() - t0:.0f}s)", flush=True)

    ents = pd.read_parquet(PROCESSED / f"train_s1_{args.country}.parquet", columns=["code"])["code"].to_numpy()
    truth = truth_for(ents)
    s1, cand, prob = sc["s1"].to_numpy(), sc["cand"].to_numpy(), sc["prob"].to_numpy()
    excl = enforce_exclusivity(s1, cand, prob)

    res = {}
    for label, mask in (("plain", np.ones(s1.size, bool)), ("exclusive", excl)):
        a, b, c = s1[mask], cand[mask], prob[mask]
        for thr in (0.3, 0.4, 0.5, 0.6):
            res[f"{label}_thr_{thr}"] = evaluate(as_sets(threshold_sets(a, b, c, thr), ents), truth)
        for missed in (0.0, 0.25, 0.5):
            sets, _, _ = select_sets(a, b, c, missed=missed)
            res[f"{label}_expF_m{missed}"] = evaluate(as_sets(sets, ents), truth)
        print({k: round(v, 5) for k, v in res.items() if k.startswith(label)}, flush=True)

    best = max(res, key=res.get)
    label, rule = best.split("_", 1)
    mask = excl if label == "exclusive" else np.ones(s1.size, bool)
    a, b, c = s1[mask], cand[mask], prob[mask]
    if rule.startswith("thr_"):
        pred = as_sets(threshold_sets(a, b, c, float(rule[4:])), ents)
    else:
        pred = as_sets(select_sets(a, b, c, missed=float(rule.split("_m")[1]))[0], ents)
    bucket = {e: ("0" if not truth[e] else "1" if len(truth[e]) == 1 else
                  "2-3" if len(truth[e]) <= 3 else "4-6" if len(truth[e]) <= 6 else "7+") for e in truth}
    report = {
        "exp": args.exp, "country": args.country, "depth": args.depth,
        "n_entities": int(ents.size), "rules": res, "best": best, "best_f05": res[best],
        "by_n_true": evaluate_by_group(pred, truth, bucket), "micro": micro_prf(pred, truth),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out_dir / f"full_eval_{args.country}_d{args.depth}.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps({k: report[k] for k in ("best", "best_f05", "by_n_true")}, indent=2, default=float))


if __name__ == "__main__":
    main()
