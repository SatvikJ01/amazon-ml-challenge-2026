"""v3 test inference: two passes with a checkpoint after every entity part.

Mirrors the training path exactly (``src/v3.py`` -> ``collective`` -> ``anchor_pass``):

pass1  --country C   per part of <= 200k entities:
         union candidates (forward ∪ reverse ∪ key) -> stage 1 (p1 >= 1e-3)
         -> string + context features on survivors -> stage 2 -> p2
         saved: pass1_feats/{C}_p{i}.parquet (all features + p2)
                pass1_anchors/{C}_p{i}of{n}.parquet (s1, cand, prob=p2)
anchors --country C  confident matches (p2 >= 0.9) query the test pools (top-3)
pass2  --country C   per part: + anchor-only pairs (full features), context and
         sibling features recomputed on the union (p2 = NaN for new pairs),
         anchor features -> stage 3 -> scores/{C}_p{i}.parquet
write  --sub-id ID   expected-F0.5 + exclusivity -> both files -> streamed checks
         + official validator -> submissions/ID/

Usage (one capped process per step/country):
    python -m src.infer_v3 pass1 --country France
    python -m src.infer_v3 anchors --country France
    python -m src.infer_v3 pass2 --country France
    python -m src.infer_v3 write --sub-id day2_s1
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

from . import anchor_pass
from .candidates import countries_for
from .collective import _texts, sibling_features
from .features import add_context_features
from .make_stage2 import CONTEXT_PREFIXES
from .v3 import STAGE1_THRESHOLD, STAGE1_V3, stage1_frame, string_features

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
EXPERIMENTS = ROOT / "experiments"
SIB_COLS = anchor_pass.SIB_COLS


def _booster(exp: str) -> tuple[lgb.Booster, list[str]]:
    return (lgb.Booster(model_file=str(EXPERIMENTS / exp / "model.txt")),
            json.loads((EXPERIMENTS / exp / "features.json").read_text()))


SPLIT, TAG = "test", "testall"


def parts_of(country: str, part_entities: int = 200_000) -> list[np.ndarray]:
    codes = pd.read_parquet(PROCESSED / f"{SPLIT}_s1_{country}.parquet", columns=["code"])["code"].to_numpy()
    return np.array_split(codes, max(1, int(np.ceil(codes.size / part_entities))))


def pass1(country: str, run: Path, stage1: str, stage2: str) -> None:
    s1m = lgb.Booster(model_file=str(EXPERIMENTS / stage1 / "model.txt"))
    s2m, s2f = _booster(stage2)
    (run / "pass1_feats").mkdir(parents=True, exist_ok=True)
    (run / "pass1_anchors").mkdir(parents=True, exist_ok=True)
    parts = parts_of(country)
    for i, part in enumerate(parts):
        fp = run / "pass1_feats" / f"{country}_p{i}.parquet"
        if fp.exists():
            continue
        t0 = time.time()
        df = stage1_frame(SPLIT, TAG, country, part, with_labels=False)
        p1 = s1m.predict(df[STAGE1_V3].to_numpy(np.float32))
        df = df[p1 >= STAGE1_THRESHOLD].reset_index(drop=True)
        del p1
        gc.collect()
        chunks = []
        for c in string_features(SPLIT, country, df):
            c["p2"] = s2m.predict(c[s2f].to_numpy(np.float32)).astype(np.float32)
            chunks.append(c)
        out = pd.concat(chunks, ignore_index=True)
        del chunks, df
        out.to_parquet(fp.with_suffix(".partial"), index=False, compression="zstd")
        fp.with_suffix(".partial").replace(fp)
        out[["s1", "cand"]].assign(prob=out["p2"]).to_parquet(
            run / "pass1_anchors" / f"{country}_p{i}of{len(parts)}.parquet", index=False)
        print(f"  pass1 {country} part {i + 1}/{len(parts)}: {len(out):,} survivor pairs ({time.time() - t0:.0f}s)", flush=True)
        del out
        gc.collect()


def anchors(country: str, run: Path) -> None:
    anchor_pass.CFG["test_scores"] = str((run / "pass1_anchors").relative_to(EXPERIMENTS))
    h = anchor_pass.retrieve(SPLIT, country)
    h.to_parquet(run / f"hits_test_{country}.parquet", index=False)
    print(f"  anchors {country}: {len(h):,} anchor-retrieved pairs", flush=True)


def pass2(country: str, run: Path, stage3: str) -> None:
    s3m, s3f = _booster(stage3)
    (run / "scores").mkdir(parents=True, exist_ok=True)
    hits_all = pd.read_parquet(run / f"hits_test_{country}.parquet")
    for i in range(len(parts_of(country))):
        sp = run / "scores" / f"{country}_p{i}.parquet"
        if sp.exists():
            continue
        t0 = time.time()
        old = pd.read_parquet(run / "pass1_feats" / f"{country}_p{i}.parquet")
        hits = hits_all[np.isin(hits_all["s1"].to_numpy(), old["s1"].unique())].reset_index(drop=True)
        m = hits.merge(old[["s1", "cand"]].assign(_old=1), on=["s1", "cand"], how="left")
        new = hits[m["_old"].isna().to_numpy()].reset_index(drop=True)
        del m
        new = anchor_pass._new_pair_features(country, new, split=SPLIT, tag=TAG, with_labels=False)
        old = old.merge(hits, on=["s1", "cand"], how="left")
        old["anc_hits"] = old["anc_hits"].fillna(0).astype(np.float32)
        old["from_anchor_only"] = np.float32(0)
        df = pd.concat([old, new], ignore_index=True)
        del old, new, hits
        gc.collect()
        p2 = df["p2"].to_numpy().astype(np.float32)
        df = df.drop(columns=[c for c in df.columns if c.startswith(CONTEXT_PREFIXES)] + [c for c in SIB_COLS if c in df.columns])
        df = add_context_features(df)
        cc = df["cand"].to_numpy()
        tx = _texts(SPLIT, country, cc)
        sf = sibling_features(df["s1"].to_numpy(), cc, p2, tx.loc[cc, "name_norm"].to_numpy(), tx.loc[cc, "addr_norm"].to_numpy())
        del tx
        df = pd.concat([df.reset_index(drop=True), sf], axis=1)
        for c in s3f:
            if c not in df.columns:
                df[c] = np.nan
        prob = s3m.predict(df[s3f].to_numpy(np.float32)).astype(np.float32)
        out = pd.DataFrame({"s1": df["s1"].to_numpy(), "cand": df["cand"].to_numpy(), "src": df["src"].to_numpy(), "prob": prob})
        out.to_parquet(sp, index=False)
        print(f"  pass2 {country} part {i}: {len(out):,} pairs scored ({time.time() - t0:.0f}s)", flush=True)
        del df, out
        gc.collect()


def write(run: Path, sub_id: str, note: str) -> None:
    from .decision import select_sets
    from .inference import enforce_exclusivity
    from .submission import (CAND_HEADER, MATCH_HEADER, OUTPUT, archive, run_official_validator,
                             test_s1_ids, verify_file, write_submission)
    t0 = time.time()
    sc = pd.concat([pd.read_parquet(p) for p in sorted((run / "scores").glob("*.parquet"))], ignore_index=True)
    s1, cand, prob = sc["s1"].to_numpy(), sc["cand"].to_numpy(), sc["prob"].to_numpy()
    live = prob >= 0.01
    a, b, p = s1[live], cand[live], prob[live]
    k = enforce_exclusivity(a, b, p)
    sets, _, _ = select_sets(a[k], b[k], p[k])
    ms1 = np.concatenate([np.full(len(v), e, np.int64) for e, v in sets.items() if len(v)])
    mc = np.concatenate([np.asarray(v, np.int64) for v in sets.values() if len(v)])
    stats = write_submission(ms1, mc, s1, cand)
    req = set(test_s1_ids().tolist())
    errs = verify_file(OUTPUT / "matching_results.tsv", MATCH_HEADER, req) + verify_file(OUTPUT / "candidate_pairs.tsv", CAND_HEADER, req)
    if errs:
        raise SystemExit("format verification FAILED:\n" + "\n".join(errs))
    code, text = run_official_validator()
    print(text)
    if code != 0:
        raise SystemExit("official validator FAILED -- not archiving")
    meta = {"pipeline": "v3", "run": str(run.relative_to(ROOT)), "decision": "expF + exclusivity", "note": note,
            "stats": stats, "validator": "PASS", "elapsed_sec": round(time.time() - t0, 1)}
    print(json.dumps(stats, indent=2))
    print(f"archived -> {archive(sub_id, meta)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["pass1", "anchors", "pass2", "write"])
    ap.add_argument("--country", default=None)
    ap.add_argument("--run", default="E030_test")
    ap.add_argument("--stage1", default="E030_stage1")
    ap.add_argument("--stage2", default="E030_stage2")
    ap.add_argument("--stage3", default="E030_stage3")
    ap.add_argument("--sub-id", default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--split", default="test", help="data version (test or testT)")
    ap.add_argument("--tag", default="testall", help="candidate-file tag")
    args = ap.parse_args()
    global SPLIT, TAG
    SPLIT, TAG = args.split, args.tag
    run = EXPERIMENTS / args.run
    run.mkdir(parents=True, exist_ok=True)
    if args.step == "pass1":
        pass1(args.country, run, args.stage1, args.stage2)
    elif args.step == "anchors":
        anchors(args.country, run)
    elif args.step == "pass2":
        pass2(args.country, run, args.stage3)
    else:
        write(run, args.sub_id, args.note)


if __name__ == "__main__":
    main()
