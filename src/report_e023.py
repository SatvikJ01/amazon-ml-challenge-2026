"""Summarise E023 (predicted-anchor retrieval) against the rollback baselines.

Labelled metrics on the standard holdout (fold 0, 30,007 entities, full ground
truth) for E013 (stage 2), E021p (stage 3, sibling features), E023A (anchor
evidence on existing candidates) and E023B (+ anchor-retrieved candidates).

France has no ground truth, so only label-free anchor statistics are reported
for it (test split), next to the same statistics for US / India (train sample).
Writes reports/e023_summary.json and prints a compact table.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
PROCESSED = ROOT / "data" / "processed"

RUNS = {"E013 (stage 2, rollback)": "E013_cascade_stage2",
        "E021p (stage 3 sibling)": "E021p_collective_on_E013_stage3",
        "E023A (+anchor feats)": "E023A_anchor_feats_only",
        "E023B (+anchor candidates)": "E023B_anchor_union"}


def labelled() -> list[dict]:
    rows = []
    for name, d in RUNS.items():
        p = EXP / d / "report.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        c, mi, nt = r["blocking_ceiling"], r["micro"], r["by_n_true"]
        rows.append({
            "run": name, "rule": r["best_rule"], "F0.5": r["best_f05"],
            "cand_recall": c["pair_recall"], "oracle_F0.5": c["max_f05"], "cands/entity": c["mean_candidates_per_entity"],
            "TP": mi["tp"], "FP": mi["fp"], "FN": mi["fn"], "micro_P": mi["micro_precision"], "micro_R": mi["micro_recall"],
            "singleton_F0.5": nt["0"]["f05"], "1match_F0.5": nt["1"]["f05"],
            "US_F0.5": r["by_country"]["US"]["f05"], "India_F0.5": r["by_country"]["India"]["f05"],
            "US_P/R": (r["by_country"]["US"]["precision"], r["by_country"]["US"]["recall"]),
            "India_P/R": (r["by_country"]["India"]["precision"], r["by_country"]["India"]["recall"]),
        })
    return rows


def anchor_stats(split: str, country: str) -> dict:
    """Label-free: anchors per entity and new candidates per entity from anchor retrieval."""
    from .anchor_pass import load_anchors
    anc = load_anchors(split, country)
    hits = pd.read_parquet(EXP / "E023_anchor" / f"hits_{split}_{country}.parquet")
    if split == "train":
        surv = pd.read_parquet(PROCESSED / f"feats_trn2c_{country}.parquet", columns=["s1", "cand"])
    else:
        d = EXP / "E013_cascade_stage2" / "test_scores_d30_casc"
        surv = pd.concat([pd.read_parquet(p, columns=["s1", "cand"]) for p in sorted(d.glob(f"{country}_p*.parquet"))])
    n_ent = surv["s1"].nunique()
    m = hits.merge(surv.assign(_o=1), on=["s1", "cand"], how="left")
    new = int(m["_o"].isna().sum())
    per = anc.groupby("s1").size()
    return {"split": split, "country": country, "entities": int(n_ent),
            "entities_with_anchor": float(per.size / n_ent), "anchors_per_entity": float(len(anc) / n_ent),
            "survivors_per_entity": float(len(surv) / n_ent),
            "anchor_hits_per_entity": float(len(hits) / n_ent), "new_cands_per_entity": float(new / n_ent)}


def main() -> None:
    lab = labelled()
    df = pd.DataFrame(lab).set_index("run")
    pd.set_option("display.width", 250)
    cols = ["rule", "F0.5", "cand_recall", "oracle_F0.5", "cands/entity", "FP", "FN", "singleton_F0.5",
            "1match_F0.5", "US_F0.5", "India_F0.5"]
    print(df[cols].round(4).to_string())
    stats = []
    for split, c in (("train", "India"), ("train", "US"), ("test", "France")):
        try:
            stats.append(anchor_stats(split, c))
        except FileNotFoundError as e:
            print("missing:", e)
    print(pd.DataFrame(stats).set_index(["split", "country"]).round(3).to_string())
    (ROOT / "reports" / "e023_summary.json").write_text(json.dumps({"labelled": lab, "label_free": stats}, indent=2, default=float))


if __name__ == "__main__":
    main()
