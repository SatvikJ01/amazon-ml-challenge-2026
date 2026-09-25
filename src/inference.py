"""End-to-end test inference: candidates -> features -> model -> decision -> files.

Steps (each deterministic, no notebook state):
1. read test candidates ``cands_{tag}_{country}.parquet`` (``src.candidates``)
2. featurise in entity chunks with the *same* code path as training
   (``build_features.featurize_country``), score with the saved LightGBM
   model, keep only ``(s1, cand, src, prob)``
3. optional hard exclusivity: an S2/S3 record keeps at most one owner (the
   highest-probability S1), because the ground truth is a strict many-to-one
   assignment (0 shared records in 7.6 M training pairs)
4. per-entity set selection (expected-F0.5 prefix or threshold)
5. ``submission.write_submission`` -> strict checks + official validator ->
   archive under ``submissions/<sub_id>/``

Usage:
    python -m src.inference --exp E010 --sub-id day1_s1 --depth 20 --rule expF --missed 0.25
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

from .build_features import featurize_country
from .candidates import countries_for
from .decision import select_sets, threshold_sets
from .submission import (CAND_HEADER, MATCH_HEADER, OUTPUT, archive, run_official_validator,
                         test_s1_ids, verify_file, write_submission)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
EXPERIMENTS = ROOT / "experiments"


def enforce_exclusivity(s1: np.ndarray, cand: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """Boolean mask keeping, for every candidate record, only its best claimant."""
    order = np.lexsort((-prob, cand))
    first = np.r_[True, cand[order][1:] != cand[order][:-1]]
    keep = np.zeros(cand.size, bool)
    keep[order[first]] = True
    return keep


def score_split(exp: str, split: str, tag: str, depth: int, chunk: int,
                countries: list[str] | None = None, keep: np.ndarray | None = None,
                part_entities: int = 200_000, stage1_model: str | None = None,
                ckpt_dir: Path | None = None) -> pd.DataFrame:
    """Score every candidate pair of the given countries with a saved model.

    Entities are processed in parts of ``part_entities`` through the ``keep`` path
    of ``featurize_country`` (competition aggregates still use the full table, so
    features equal a single pass).  With ``ckpt_dir`` each (country, part) is saved
    as it finishes and skipped on rerun, so an interruption loses at most one part.
    """
    model = lgb.Booster(model_file=str(EXPERIMENTS / exp / "model.txt"))
    feats = json.loads((EXPERIMENTS / exp / "features.json").read_text())
    out = []
    for country in countries or countries_for(split):
        t0 = time.time()
        codes = pd.read_parquet(PROCESSED / f"{split}_s1_{country}.parquet", columns=["code"])["code"].to_numpy()
        if keep is not None:
            codes = codes[np.isin(codes, keep)]
        n_parts = max(1, int(np.ceil(codes.size / part_entities)))
        for pi, part in enumerate(np.array_split(codes, n_parts)):
            ck = ckpt_dir / f"{country}_p{pi}of{n_parts}.parquet" if ckpt_dir else None
            if ck is not None and ck.exists():
                out.append(pd.read_parquet(ck))
                print(f"  {country} part {pi}: checkpoint reused", flush=True)
                continue
            parts, n = [], 0
            for df in featurize_country(split, tag, country, depth, part, chunk,
                                        stage1_model=stage1_model):
                parts.append(pd.DataFrame({
                    "s1": df["s1"].to_numpy(), "cand": df["cand"].to_numpy(),
                    "src": df["src"].to_numpy(),
                    "prob": model.predict(df[feats]).astype(np.float32),
                }))
                n += len(df)
                print(f"  {country} part {pi}/{n_parts}: scored {n:,} pairs ({time.time() - t0:.0f}s)", flush=True)
                del df
                gc.collect()
            res = pd.concat(parts, ignore_index=True)
            if ck is not None:
                tmp = ck.with_suffix(".partial")
                res.to_parquet(tmp, index=False)
                tmp.replace(ck)
            out.append(res)
            del parts
            gc.collect()
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="experiment dir holding model.txt / features.json")
    ap.add_argument("--sub-id", required=True)
    ap.add_argument("--tag", default="testall")
    ap.add_argument("--depth", type=int, default=30)
    ap.add_argument("--chunk-entities", type=int, default=40_000)
    ap.add_argument("--rule", choices=["expF", "thr"], default="expF")
    ap.add_argument("--missed", type=float, default=0.0)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--exclusive", action="store_true")
    ap.add_argument("--reuse-scores", action="store_true",
                    help="reuse experiments/<exp>/test_scores.parquet if present")
    ap.add_argument("--stage1", default=None, help="stage-1 experiment dir (cascade)")
    ap.add_argument("--countries", nargs="*", default=None,
                    help="score only these countries (checkpoints), then exit without writing")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    t0 = time.time()
    s1m = str(EXPERIMENTS / args.stage1 / "model.txt") if args.stage1 else None
    ckpt = EXPERIMENTS / args.exp / f"test_scores_d{args.depth}{'_casc' if args.stage1 else ''}"
    ckpt.mkdir(parents=True, exist_ok=True)
    if args.countries:
        # Scoring-only mode: one country per process keeps the heap fresh.
        score_split(args.exp, "test", args.tag, args.depth, args.chunk_entities,
                    countries=args.countries, stage1_model=s1m, ckpt_dir=ckpt)
        print(f"scored {args.countries} -> {ckpt}")
        return
    sc = score_split(args.exp, "test", args.tag, args.depth, args.chunk_entities,
                     stage1_model=s1m, ckpt_dir=ckpt)
    assert sc["s1"].nunique() <= 1_732_544
    s1, cand, prob = sc["s1"].to_numpy(), sc["cand"].to_numpy(), sc["prob"].to_numpy()
    del sc

    # candidate_pairs.tsv = every pair the final model scored (stage-1 survivors
    # under the cascade).  Decision works on plausible pairs only; pairs below
    # 1 % can never be chosen by any rule we use.
    live = prob >= 0.01
    s1_d, cand_d, prob_d = s1[live], cand[live], prob[live]
    if args.exclusive:
        m = enforce_exclusivity(s1_d, cand_d, prob_d)
        s1_d, cand_d, prob_d = s1_d[m], cand_d[m], prob_d[m]
    if args.rule == "expF":
        sets, _, _ = select_sets(s1_d, cand_d, prob_d, missed=args.missed)
    else:
        sets = threshold_sets(s1_d, cand_d, prob_d, args.thr)
    ms1 = np.concatenate([np.full(len(v), k, np.int64) for k, v in sets.items() if len(v)] or [np.empty(0, np.int64)])
    mc = np.concatenate([np.asarray(v, np.int64) for v in sets.values() if len(v)] or [np.empty(0, np.int64)])
    del sets, s1_d, cand_d, prob_d

    stats = write_submission(ms1, mc, s1, cand)
    required = set(test_s1_ids().tolist())
    errs = (verify_file(OUTPUT / "matching_results.tsv", MATCH_HEADER, required)
            + verify_file(OUTPUT / "candidate_pairs.tsv", CAND_HEADER, required))
    if errs:
        raise SystemExit("format verification FAILED:\n" + "\n".join(errs))
    code, out = run_official_validator()
    print(out)
    if code != 0:
        raise SystemExit("official validator FAILED -- not archiving")

    meta = dict(vars(args), stats=stats, validator="PASS (official on matching; streamed checks on both)",
                model_report=json.loads((EXPERIMENTS / args.exp / "report.json").read_text()).get("best_f05"),
                elapsed_sec=round(time.time() - t0, 1))
    dest = archive(args.sub_id, meta)
    print(json.dumps(stats, indent=2))
    print(f"archived -> {dest}")


def _group(s1: np.ndarray, cand: np.ndarray):
    order = np.argsort(s1, kind="stable")
    s1s, cs = s1[order], cand[order]
    uniq, start = np.unique(s1s, return_index=True)
    bounds = np.append(start, s1s.size)
    return uniq, [cs[bounds[i]:bounds[i + 1]] for i in range(uniq.size)]


if __name__ == "__main__":
    main()
