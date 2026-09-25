"""Score an experiment's holdout predictions on a subset of holdout entities.

Used to compare runs trained on different entity samples on the *same* entities
(e.g. v3's 300k-entity run evaluated on the original 30,007 holdout entities).

Usage: python -m src.eval_subset --exp E030_stage3 --entities data/processed/entities_trn.npy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .decision import select_sets, threshold_sets
from .inference import enforce_exclusivity
from .metrics import evaluate, evaluate_by_group, micro_prf
from .train import as_sets, entity_fold, truth_for

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--entities", default="data/processed/entities_trn.npy")
    args = ap.parse_args()
    ev = pd.read_parquet(ROOT / "experiments" / args.exp / "holdout_pred.parquet")
    ents = np.load(ROOT / args.entities)
    ents = ents[entity_fold(ents, 5) == 0]
    ev = ev[np.isin(ev["s1"].to_numpy(), ents)]
    truth = truth_for(ents)
    s1, cand, prob = ev["s1"].to_numpy(), ev["cand"].to_numpy(), ev["prob"].to_numpy()
    res = {}
    for thr in (0.6, 0.7):
        res[f"thr_{thr}"] = evaluate(as_sets(threshold_sets(s1, cand, prob, thr), ents), truth)
    res["expF"] = evaluate(as_sets(select_sets(s1, cand, prob)[0], ents), truth)
    m = enforce_exclusivity(s1, cand, prob)
    pred = as_sets(select_sets(s1[m], cand[m], prob[m])[0], ents)
    res["expF_excl"] = evaluate(pred, truth)
    cmap = dict(zip(ev["s1"].to_numpy(), ev["country"].to_numpy()))
    nt = {e: ("0" if not truth[e] else "1" if len(truth[e]) == 1 else "2+") for e in ents}
    out = {"exp": args.exp, "n_entities": int(ents.size), "rules": res,
           "by_country": evaluate_by_group(pred, truth, {e: cmap.get(e, "none") for e in ents}),
           "by_n_true": evaluate_by_group(pred, truth, nt), "micro": micro_prf(pred, truth)}
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
