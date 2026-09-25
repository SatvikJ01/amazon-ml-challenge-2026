"""Test inference for the light stage 3 (collective / sibling-aware) model.

Reuses the checkpointed stage-2 test scores (``prob`` = p2 for every stage-1
survivor) instead of recomputing the 63 string features.  For each country:

1. retrieval features (blk_score, blk_rank) and competition features are
   recomputed from the forward candidate files exactly as in training --
   aggregates over the full candidate table, materialised only for survivors;
2. sibling features from candidate texts (``collective.sibling_features``);
3. stage-3 probabilities from the light model; checkpoint per country.

The final step applies the decision rule (expected-F0.5 + exclusivity), writes
both files, runs the streamed checks and the official validator, and archives.

Usage:
    python -m src.stage3_infer score --stage2 E013_cascade_stage2 --stage3 E021L_stage3_light --country India
    python -m src.stage3_infer write --stage2 E013_cascade_stage2 --stage3 E021L_stage3_light --sub-id day1_s3
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .build_features import competition_features
from .candidates import cand_path, countries_for
from .collective import _texts, sibling_features
from .ids import OFFSET

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"


def stage2_scores(stage2: str, country: str) -> pd.DataFrame:
    d = EXPERIMENTS / stage2 / "test_scores_d30_casc"
    parts = sorted(d.glob(f"{country}_p*.parquet"))
    assert parts, f"no stage-2 checkpoints for {country} in {d}"
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def attach_retrieval(sc: pd.DataFrame, country: str, source: int, tag: str = "testall") -> pd.DataFrame:
    """blk_score, blk_rank and competition features for the survivor rows of one source."""
    fwd = pd.read_parquet(cand_path(tag, country, source), columns=["s1", "cand", "blk_score", "blk_rank"])
    sub = sc[sc["src"] == source].reset_index(drop=True)
    s1u = np.unique(fwd["s1"].to_numpy())
    key = lambda df: (np.searchsorted(s1u, df["s1"].to_numpy()).astype(np.int64) << 34) | (df["cand"].to_numpy() % OFFSET)
    fk, sk = key(fwd), key(sub)
    order = np.argsort(fk, kind="stable")
    pos = np.searchsorted(fk[order], sk)
    rows = order[np.minimum(pos, fk.size - 1)]
    assert np.all(fk[rows] == sk), "survivor pair missing from forward candidates"
    comp = competition_features(fwd["s1"].to_numpy(), fwd["cand"].to_numpy(), fwd["blk_score"].to_numpy(), rows)
    out = pd.concat([sub, comp], axis=1)
    out["blk_score"] = fwd["blk_score"].to_numpy()[rows]
    out["blk_rank"] = fwd["blk_rank"].to_numpy()[rows]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["score", "write"])
    ap.add_argument("--stage2", default="E013_cascade_stage2")
    ap.add_argument("--stage3", default="E021L_stage3_light")
    ap.add_argument("--country", default=None)
    ap.add_argument("--sub-id", default=None)
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    t0 = time.time()
    ck = EXPERIMENTS / args.stage3 / "test_scores"
    ck.mkdir(parents=True, exist_ok=True)

    if args.step == "score":
        c = args.country
        sc = stage2_scores(args.stage2, c)
        df = pd.concat([attach_retrieval(sc, c, s) for s in (2, 3)], ignore_index=True)
        del sc
        gc.collect()
        df = df.rename(columns={"prob": "p2"})
        cc = df["cand"].to_numpy()
        tx = _texts("test", c, cc)
        sf = sibling_features(df["s1"].to_numpy(), cc, df["p2"].to_numpy(),
                              tx.loc[cc, "name_norm"].to_numpy(), tx.loc[cc, "addr_norm"].to_numpy())
        del tx
        X = pd.concat([df.drop(columns="p2"), sf], axis=1)
        feats = json.loads((EXPERIMENTS / args.stage3 / "features.json").read_text())
        model = lgb.Booster(model_file=str(EXPERIMENTS / args.stage3 / "model.txt"))
        out = pd.DataFrame({"s1": X["s1"].to_numpy(), "cand": X["cand"].to_numpy(), "src": X["src"].to_numpy(),
                            "prob": model.predict(X[feats].to_numpy(np.float32)).astype(np.float32),
                            "p2": X["p2"].to_numpy()})
        out.to_parquet(ck / f"{c}.parquet", index=False)
        print(f"{c}: stage-3 scored {len(out):,} pairs ({time.time() - t0:.0f}s)", flush=True)
        return

    from .decision import select_sets
    from .inference import enforce_exclusivity
    from .submission import (CAND_HEADER, MATCH_HEADER, OUTPUT, archive, run_official_validator,
                             test_s1_ids, verify_file, write_submission)
    sc = pd.concat([pd.read_parquet(ck / f"{c}.parquet") for c in countries_for("test")], ignore_index=True)
    s1, cand, prob = sc["s1"].to_numpy(), sc["cand"].to_numpy(), sc["prob"].to_numpy()
    live = prob >= 0.01
    a, b, p = s1[live], cand[live], prob[live]
    m = enforce_exclusivity(a, b, p)
    sets, _, _ = select_sets(a[m], b[m], p[m])
    ms1 = np.concatenate([np.full(len(v), k, np.int64) for k, v in sets.items() if len(v)])
    mc = np.concatenate([np.asarray(v, np.int64) for v in sets.values() if len(v)])
    stats = write_submission(ms1, mc, s1, cand)
    required = set(test_s1_ids().tolist())
    errs = (verify_file(OUTPUT / "matching_results.tsv", MATCH_HEADER, required)
            + verify_file(OUTPUT / "candidate_pairs.tsv", CAND_HEADER, required))
    if errs:
        raise SystemExit("format verification FAILED:\n" + "\n".join(errs))
    code, text = run_official_validator()
    print(text)
    if code != 0:
        raise SystemExit("official validator FAILED -- not archiving")
    rep = json.loads((EXPERIMENTS / args.stage3 / "report.json").read_text())
    meta = dict(vars(args), stats=stats, decision="expF + exclusivity", model_report=rep.get("best_f05"),
                validator="PASS (official on matching; streamed checks on both)",
                elapsed_sec=round(time.time() - t0, 1))
    print(json.dumps(stats, indent=2))
    print(f"archived -> {archive(args.sub_id, meta)}")


if __name__ == "__main__":
    main()
