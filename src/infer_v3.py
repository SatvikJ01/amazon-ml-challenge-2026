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
import os
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
from .extra_features import EXTRA_COLS, extra_features
from .extra_features2 import EXTRA2_COLS, NameStats, extra_features2
from .extra_features3 import EXTRA3_COLS, PoolNames, extra_features3
from .stage4 import stage4_features
from .features import add_context_features
from .make_stage2 import CONTEXT_PREFIXES
from .v3 import STAGE1_THRESHOLD, STAGE1_V3, stage1_frame, string_features

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
EXPERIMENTS = ROOT / "experiments"
SIB_COLS = anchor_pass.SIB_COLS


class _XGB:
    """XGBoost booster with LightGBM's ``predict(X)`` interface (best iteration)."""
    def __init__(self, path: Path, feats: list[str]):
        import xgboost as xgb
        self.feats = feats
        self.xgb, self.b = xgb, xgb.Booster()
        self.b.load_model(str(path))
        self.b.set_param({"device": os.environ.get("XGB_DEVICE", "cuda")})

    def predict(self, X):
        return self.b.predict(self.xgb.DMatrix(X, feature_names=self.feats),
                              iteration_range=(0, self.b.best_iteration + 1))


def _booster(exp: str):
    d = EXPERIMENTS / exp
    feats = json.loads((d / "features.json").read_text())
    if (d / "model.txt").exists():
        return lgb.Booster(model_file=str(d / "model.txt")), feats
    return _XGB(d / "model.json", feats), feats


SPLIT, TAG = "test", "testall"
PASS2_CHUNKS = 3


def parts_of(country: str, part_entities: int = 200_000) -> list[np.ndarray]:
    codes = pd.read_parquet(PROCESSED / f"{SPLIT}_s1_{country}.parquet", columns=["code"])["code"].to_numpy()
    return np.array_split(codes, max(1, int(np.ceil(codes.size / part_entities))))


def pass1(country: str, run: Path, stage1: str, stage2: str, only_part: int | None = None) -> None:
    """Without ``only_part``: run every missing part in its *own child process*
    (memory crept up across parts inside one long process and got OOM-killed on
    the 2nd-3rd part).  With ``only_part``: compute exactly that part."""
    import os
    import subprocess
    import sys
    if only_part is None:
        n = len(parts_of(country))
        for i in range(n):
            if (run / "pass1_feats" / f"{country}_p{i}.parquet").exists():
                continue
            cmd = [sys.executable, "-m", "src.infer_v3", "pass1", "--country", country, "--part", str(i),
                   "--run", run.name, "--stage1", stage1, "--stage2", stage2, "--split", SPLIT, "--tag", TAG]
            rc = subprocess.run(cmd, cwd=ROOT).returncode
            if rc != 0:
                raise SystemExit(f"pass1 {country} part {i} failed with code {rc}")
        return
    s1m = lgb.Booster(model_file=str(EXPERIMENTS / stage1 / "model.txt"))
    f1 = EXPERIMENTS / stage1 / "features.json"
    s1f = json.loads(f1.read_text()) if f1.exists() else STAGE1_V3
    s2m, s2f = _booster(stage2)
    (run / "pass1_feats").mkdir(parents=True, exist_ok=True)
    (run / "pass1_anchors").mkdir(parents=True, exist_ok=True)
    parts = parts_of(country)
    for i, part in enumerate(parts):
        fp = run / "pass1_feats" / f"{country}_p{i}.parquet"
        if fp.exists() or i != only_part:
            continue
        t0 = time.time()
        df = stage1_frame(SPLIT, TAG, country, part, with_labels=False)
        p1 = s1m.predict(df[s1f].to_numpy(np.float32))
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


def rescore(country: str, run: Path, stage2x: str) -> None:
    """Recompute p2 in the pass-1 files with the GPU XGBoost stage-2 model, so test p2
    comes from the same model family as the cross-fitted training p2.  Streams each
    part in row batches (a whole India part plus copies exceeded 5.5 GB)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import xgboost as xgb
    booster = xgb.Booster(); booster.load_model(str(EXPERIMENTS / stage2x / "model.json"))
    booster.set_param({"device": os.environ.get("XGB_DEVICE", "cuda")})
    feats = json.loads((EXPERIMENTS / stage2x / "features.json").read_text())
    parts = sorted((run / "pass1_feats").glob(f"{country}_p*.parquet"))
    for fp in parts:
        tmp, w, anc, tot, n = fp.with_suffix(".partial"), None, [], 0.0, 0
        for batch in pq.ParquetFile(fp).iter_batches(batch_size=500_000):
            t = pa.Table.from_batches([batch])
            X = np.column_stack([t.column(f).to_numpy(zero_copy_only=False).astype(np.float32) for f in feats])
            p2 = booster.predict(xgb.DMatrix(X, feature_names=feats),
                                 iteration_range=(0, booster.best_iteration + 1)).astype(np.float32)
            del X
            t = t.set_column(t.schema.get_field_index("p2"), pa.field("p2", pa.float32()), pa.array(p2))
            w = w or pq.ParquetWriter(tmp, t.schema, compression="zstd")
            w.write_table(t)
            anc.append(pd.DataFrame({"s1": t.column("s1").to_numpy(), "cand": t.column("cand").to_numpy(), "prob": p2}))
            tot, n = tot + float(p2.sum()), n + len(p2)
        w.close()
        tmp.replace(fp)
        i = int(fp.stem.split("_p")[1])
        pd.concat(anc, ignore_index=True).to_parquet(run / "pass1_anchors" / f"{country}_p{i}of{len(parts)}.parquet", index=False)
        print(f"  rescore {fp.name}: {n:,} pairs, mean p2 {tot / n:.4f}", flush=True)


def anchors(country: str, run: Path) -> None:
    anchor_pass.CFG["test_scores"] = str((run / "pass1_anchors").relative_to(EXPERIMENTS))
    h = anchor_pass.retrieve(SPLIT, country)
    h.to_parquet(run / f"hits_test_{country}.parquet", index=False)
    print(f"  anchors {country}: {len(h):,} anchor-retrieved pairs", flush=True)


def pass2(country: str, run: Path, stage3: str, scores: str = "scores", stage4: str | None = None,
          only_part: int | None = None) -> None:
    s3m, s3f = _booster(stage3)
    s4m, s4f = _booster(stage4) if stage4 else (None, None)
    need_extra = any(c in s3f for c in EXTRA_COLS)
    need_extra2 = any(c in s3f for c in EXTRA2_COLS)
    need_extra3 = any(c in s3f for c in EXTRA3_COLS)
    pool = PoolNames(SPLIT, country) if need_extra3 else None
    if need_extra or need_extra2 or need_extra3 or stage4:
        s1_txt = pd.read_parquet(PROCESSED / f"{SPLIT}_s1_{country}.parquet", columns=["code", "name_norm", "addr_norm"]).set_index("code")
    nstats = NameStats(SPLIT, country) if need_extra2 else None
    (run / scores).mkdir(parents=True, exist_ok=True)
    hits_all = pd.read_parquet(run / f"hits_test_{country}.parquet")
    for i in range(len(parts_of(country))):
        sp = run / scores / f"{country}_p{i}.parquet"
        if sp.exists() or (only_part is not None and i != only_part):
            continue
        t0 = time.time()
        fp = run / "pass1_feats" / f"{country}_p{i}.parquet"
        import pyarrow.parquet as pq
        ents = np.unique(pq.read_table(fp, columns=["s1"]).column("s1").to_numpy())
        outs = []
        # Entity chunks: every feature is computed within an entity, so chunking is exact;
        # it keeps India parts (8 M anchor pairs) under the memory cap.
        for chunk in np.array_split(ents, PASS2_CHUNKS):
            old = pq.read_table(fp, filters=[("s1", "in", chunk.tolist())]).to_pandas()
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
            if need_extra or need_extra2 or need_extra3:
                q = s1_txt.reindex(df["s1"].to_numpy())
                qa, ca = q["addr_norm"].to_numpy(), tx.loc[cc, "addr_norm"].to_numpy()
                if need_extra:
                    sf = pd.concat([sf, extra_features(df["s1"].to_numpy(), p2, qa, ca)], axis=1)
                if need_extra2:
                    sf = pd.concat([sf, extra_features2(q["name_norm"].to_numpy(), tx.loc[cc, "name_norm"].to_numpy(),
                                                        qa, ca, nstats)], axis=1)
                if need_extra3:
                    sf = pd.concat([sf, extra_features3(df["s1"].to_numpy(), qa, ca, tx.loc[cc, "name_norm"].to_numpy(),
                                                        df["name_tset"].to_numpy(), df["addr_empty_c"].to_numpy(), pool)], axis=1)
                del q
            df = pd.concat([df.reset_index(drop=True), sf], axis=1)
            for c in s3f:
                if c not in df.columns:
                    df[c] = np.nan
            prob = s3m.predict(df[s3f].to_numpy(np.float32)).astype(np.float32)
            if s4m is not None:      # stage 4: collective features recomputed from stage-3 probabilities
                s1a = df["s1"].to_numpy()
                f4 = stage4_features(s1a, cc, prob, tx.loc[cc, "name_norm"].to_numpy(), tx.loc[cc, "addr_norm"].to_numpy(),
                                     s1_txt["addr_norm"].reindex(s1a).to_numpy())
                df = pd.concat([df, f4], axis=1)
                for c in s4f:
                    if c not in df.columns:
                        df[c] = np.nan
                prob = s4m.predict(df[s4f].to_numpy(np.float32)).astype(np.float32)
            del tx
            outs.append(pd.DataFrame({"s1": df["s1"].to_numpy(), "cand": df["cand"].to_numpy(),
                                      "src": df["src"].to_numpy(), "prob": prob}))
            del df
            gc.collect()
        out = pd.concat(outs, ignore_index=True)
        out.to_parquet(sp, index=False)
        print(f"  pass2 {country} part {i}: {len(out):,} pairs scored ({time.time() - t0:.0f}s)", flush=True)
        del out, outs
        gc.collect()


def write(run: Path, sub_id: str, note: str, from_pass1: bool = False, scores: str = "scores") -> None:
    from .decision import select_sets
    from .inference import enforce_exclusivity
    from .submission import (CAND_HEADER, MATCH_HEADER, OUTPUT, archive, run_official_validator,
                             test_s1_ids, verify_file, write_submission)
    t0 = time.time()
    if from_pass1:   # fallback: stage-2 probabilities from pass 1 (no anchor pass)
        sc = pd.concat([pd.read_parquet(p).rename(columns={"prob": "prob"})
                        for p in sorted((run / "pass1_anchors").glob("*.parquet"))], ignore_index=True)
    else:
        sc = pd.concat([pd.read_parquet(p) for p in sorted((run / scores).glob("*.parquet"))], ignore_index=True)
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
    meta = {"pipeline": "v3" + (" (stage 2 only)" if from_pass1 else " (stage 3)"), "run": str(run.relative_to(ROOT)),
            "decision": "expF + exclusivity", "note": note,
            "stats": stats, "validator": "PASS", "elapsed_sec": round(time.time() - t0, 1)}
    print(json.dumps(stats, indent=2))
    print(f"archived -> {archive(sub_id, meta)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["pass1", "rescore", "anchors", "pass2", "write"])
    ap.add_argument("--stage2x", default="E030_stage2x", help="rescore: GPU XGBoost stage-2 model")
    ap.add_argument("--country", default=None)
    ap.add_argument("--run", default="E030_test")
    ap.add_argument("--stage1", default="E030_stage1")
    ap.add_argument("--stage2", default="E030_stage2")
    ap.add_argument("--stage3", default="E030_stage3")
    ap.add_argument("--stage4", default=None, help="pass2: optional stage-4 model")
    ap.add_argument("--sub-id", default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--split", default="test", help="data version (test or testT)")
    ap.add_argument("--from-pass1", action="store_true", help="write from stage-2 scores (fallback)")
    ap.add_argument("--part", type=int, default=None, help="pass1/pass2: compute only this part")
    ap.add_argument("--tag", default="testall", help="candidate-file tag")
    ap.add_argument("--scores", default="scores", help="pass2/write: scores sub-directory of the run")
    args = ap.parse_args()
    global SPLIT, TAG
    SPLIT, TAG = args.split, args.tag
    run = EXPERIMENTS / args.run
    run.mkdir(parents=True, exist_ok=True)
    if args.step == "pass1":
        pass1(args.country, run, args.stage1, args.stage2, args.part)
    elif args.step == "rescore":
        rescore(args.country, run, args.stage2x)
    elif args.step == "anchors":
        anchors(args.country, run)
    elif args.step == "pass2":
        pass2(args.country, run, args.stage3, args.scores, args.stage4, args.part)
    else:
        write(run, args.sub_id, args.note, args.from_pass1, args.scores)


if __name__ == "__main__":
    main()
