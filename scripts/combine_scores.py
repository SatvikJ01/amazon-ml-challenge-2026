"""Build a scores directory for the write step by combining per-country correction rules.

    python scripts/combine_scores.py --run E039_test --raw scores_l2b_in --corrected scores_l2b \
        --rule India=full US=down France=raw --out scores_l2b_usdown

rule per country:  full = corrected prob;  raw = stage-3 prob;  down = min(raw, corrected) (the correction
may only lower a probability);  up = max(raw, corrected).  The raw directory must hold the stage-3 probs for
every country it is used for (scores_l2in / scores_l2b_in keep US and France at prob3).  Pairs are aligned on
(s1, cand) and checked for identical coverage.
Leaderboard evidence (2026-09-27): E044 correction full on India +0.00073 LB; full on US -0.00062 LB (its
upward moves admit test-set distractors); holdout: US down-only +0.00061 per US entity.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="E039_test")
    ap.add_argument("--raw", required=True)
    ap.add_argument("--corrected", required=True)
    ap.add_argument("--rule", nargs="+", required=True, help="Country=full|raw|down|up")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run = ROOT / "experiments" / a.run
    rules = dict(r.split("=") for r in a.rule)
    out = run / a.out
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted((run / a.raw).glob("*_p*.parquet")):
        country = f.name.split("_p")[0]
        rule = rules.get(country, "raw")
        raw = pd.read_parquet(f)
        if rule == "raw":
            res = raw
        else:
            cor = pd.read_parquet(run / a.corrected / f.name)
            m = raw.merge(cor[["s1", "cand", "prob"]].rename(columns={"prob": "pc"}), on=["s1", "cand"], how="left")
            assert len(m) == len(raw) == len(cor) and not m.pc.isna().any(), f"coverage mismatch in {f.name}"
            pr, pc = m.prob.to_numpy(), m.pc.to_numpy()
            p = {"full": pc, "down": np.minimum(pr, pc), "up": np.maximum(pr, pc)}[rule]
            res = m.drop(columns=["pc"]).assign(prob=p.astype(np.float32))
            print(f"{f.name}: rule {rule}: lowered {(p < pr).sum():,}, raised {(p > pr).sum():,} of {len(p):,}", flush=True)
        res.to_parquet(out / f.name, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
