"""Apply the level-2 RESIDUAL model to the saved pass-2 stage-3 inputs of a test run.

Streams experiments/<run>/scores_feats/{country}_p{i}_c{k}.parquet (s1, cand, src, prob3 + the 114
E039 stage-3 features; chunks are entity-disjoint), computes the level-2 features with src/level2.py
(split testT tables built once per country from the FULL split data), and sets
    prob = sigmoid(logit(prob3) + residual raw score)   for rows with prob3 >= 1e-3 (others keep prob3)
for the countries in --countries (default India US; every other country keeps prob3).  Writes
experiments/<run>/scores_l2/{country}_p{i}.parquet (s1, cand, src, prob), one file per pass-2 part
(its chunks concatenated in chunk order), then prints the gate statistics per country.

Only parts whose pass-2 output scores/{country}_p{i}.parquet exists (= all 4 chunks written) are
processed; existing outputs are skipped unless --overwrite (so the script can be re-run to resume).

    python scripts/apply_l2.py --run E039_test --countries India US
    python -m src.infer_v3 write --run E039_test --split testT --tag tstT --scores scores_l2 --sub-id <id>

Smoke test of one chunk (writes only --smoke-out, applies the residual whatever the country):
    python scripts/apply_l2.py --smoke experiments/E039_test/scores_feats/France_p0_c0.parquet --smoke-out /tmp/x.parquet
"""
from __future__ import annotations

import argparse
import gc
import os
import json
import re
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import level2 as L2  # noqa: E402

EXP = ROOT / "experiments"
DATA_ROOT = Path(os.environ.get("L2_ROOT", ROOT))     # same data root as level2.Tables
NEED = ["s1", "cand", "src", "prob3", "name_tset", "addr_tset", "addr_empty_c"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


class Residual:
    def __init__(self, model_dir: Path, threads: int):
        import lightgbm as lgb
        self.feats114 = json.loads((EXP / "E039_stage3" / "features.json").read_text())
        self.s3 = lgb.Booster(model_file=str(EXP / "E039_stage3" / "model.txt"))
        self.cols = json.loads((model_dir / "features.json").read_text())
        self.variant = next((v for v in ("ALL_nc", "ALL") if L2.model_columns(self.feats114, v) == self.cols), None)
        if self.variant is None:
            raise SystemExit(f"{model_dir}/features.json does not match level2.model_columns(ALL_nc|ALL)")
        self.booster = lgb.Booster(model_file=str(model_dir / "model.txt"))
        if self.booster.num_feature() != len(self.cols):
            raise SystemExit("model / features.json column count mismatch")
        self.threads = threads
        self.preflight_done = False
        ref = model_dir / "ref_X5000.npy"
        if ref.exists():     # the EC2-trained model reproduces its own raw scores on this machine
            d = np.abs(self.booster.predict(np.load(ref), raw_score=True, num_threads=threads)
                       - np.load(model_dir / "ref_raw5000.npy")).max()
            log(f"model check: {model_dir.name} variant {self.variant}, {len(self.cols)} features, "
                f"{self.booster.num_trees()} trees, reference raw-score max diff {d:.2e}")
            if d > 1e-6:
                raise SystemExit("reference raw scores not reproduced")

    def preflight(self, pf: pq.ParquetFile, n: int) -> None:
        """Stage-3 model on the saved 114 features must reproduce prob3 (sample of n rows)."""
        t = pf.read_row_group(0, columns=self.feats114 + ["prob3"])
        k = min(n, t.num_rows)
        X = np.column_stack([t.column(c).to_numpy()[:k].astype(np.float32) for c in self.feats114])
        d = np.abs(self.s3.predict(X, num_threads=self.threads) - t.column("prob3").to_numpy()[:k]).max()
        log(f"PRE-FLIGHT: E039 stage-3 on saved features vs prob3 ({k:,} rows): max |diff| = {d:.2e}")
        if d > 1e-5:
            raise SystemExit("PRE-FLIGHT FAILED: saved stage-3 features do not reproduce prob3")
        self.preflight_done = True

    def chunk(self, path: Path, T, gate: dict | None, preflight_rows: int) -> pd.DataFrame:
        pf = pq.ParquetFile(path)
        if not self.preflight_done:
            self.preflight(pf, preflight_rows)
        base = pf.read(columns=NEED).to_pandas()
        prob3 = base["prob3"].to_numpy().astype(np.float32)
        t0 = time.time()
        F = L2.features(base.rename(columns={"prob3": "prob"}), T, as_dict=True)
        t1 = time.time()
        idx = np.flatnonzero(prob3 >= L2.SUB)
        X = np.empty((len(idx), len(self.cols)), np.float32)
        rest = []
        for j, c in enumerate(self.cols):
            if c == "prob":
                X[:, j] = prob3[idx]
            elif c in F:
                X[:, j] = F[c][idx]
            else:
                rest.append((j, c))
        del F
        gc.collect()
        for a in range(0, len(rest), 30):          # saved stage-3 features, 30 columns at a time
            part = rest[a:a + 30]
            t = pf.read(columns=[c for _, c in part])
            for j, c in part:
                X[:, j] = t.column(c).to_numpy()[idx]
            del t
        raw = self.booster.predict(X, raw_score=True, num_threads=self.threads)
        del X
        pa.default_memory_pool().release_unused()
        newp = prob3.astype(np.float64)
        newp[idx] = 1 / (1 + np.exp(-(L2.logit(prob3[idx]) + raw)))
        if gate is not None:
            gate["dl"].append((L2.logit(newp[idx]) - L2.logit(prob3[idx])).astype(np.float32))
        log(f"    {path.name}: {len(base):,} rows, {len(idx):,} rescored, features {t1 - t0:.0f}s, "
            f"total {time.time() - t0:.0f}s, peak RSS {rss_gb():.2f} GB")
        return pd.DataFrame({"s1": base["s1"].to_numpy(), "cand": base["cand"].to_numpy(),
                             "src": np.rint(base["src"].to_numpy()).astype(np.int8), "prob": newp.astype(np.float32),
                             "_prob3": prob3})


def chunks_by_part(d: Path) -> dict:
    out = {}
    for f in d.glob("*_p*_c*.parquet"):
        mm = re.fullmatch(r"(.+)_p(\d+)_c(\d+)\.parquet", f.name)
        if mm:
            out.setdefault((mm.group(1), int(mm.group(2))), []).append((int(mm.group(3)), f))
    return {k: [f for _, f in sorted(v)] for k, v in sorted(out.items())}


def gates(country: str, run: Path, out_dir: Path, in_dir: Path, dl: list) -> None:
    """Planner gate statistics (planner/gates.log): entities whose selected set changes, delta-logit."""
    from src.decision import select_sets
    from src.inference import enforce_exclusivity
    parts = sorted(out_dir.glob(f"{country}_p*.parquet"))
    A, B, ents = [], [], 0
    dl = []                     # always from the written files (covers parts skipped in this run)
    recompute = True
    for sp in parts:
        i = int(re.fullmatch(rf"{country}_p(\d+)\.parquet", sp.name).group(1))
        new = pd.read_parquet(sp, columns=["s1", "cand", "prob"])
        old = pd.concat([pd.read_parquet(f, columns=["s1", "cand", "prob3"]) for f in chunks_by_part(in_dir)[(country, i)]],
                        ignore_index=True)
        if len(old) != len(new) or not (old["cand"].to_numpy() == new["cand"].to_numpy()).all():
            log(f"  GATE WARNING {sp.name}: row alignment with the chunks failed")
            return
        ents += new["s1"].nunique()
        po, pn = old["prob3"].to_numpy(), new["prob"].to_numpy()
        if recompute:
            m = po >= L2.SUB
            dl.append((L2.logit(pn[m]) - L2.logit(po[m])).astype(np.float32))
        lo, ln = po >= 0.01, pn >= 0.01          # the write step's live filter
        A.append((new["s1"].to_numpy()[lo], new["cand"].to_numpy()[lo], po[lo]))
        B.append((new["s1"].to_numpy()[ln], new["cand"].to_numpy()[ln], pn[ln]))
        del new, old

    def sel(L):
        s, c, p = (np.concatenate([x[j] for x in L]) for j in range(3))
        p = p.astype(np.float64)
        k = enforce_exclusivity(s, c, p)
        sets, _, _ = select_sets(s[k], c[k], p[k])
        return set((int(e), int(x)) for e, xs in sets.items() for x in np.asarray(xs).tolist())
    SA, SB = sel(A), sel(B)
    add, rem = SB - SA, SA - SB
    ch = len(set(e for e, _ in add) | set(e for e, _ in rem))
    d = np.concatenate(dl) if dl else np.zeros(1, np.float32)
    log(f"GATES {country}: entities {ents:,} changed {ch:,} ({1000 * ch / max(ents, 1):.2f} per 1000) "
        f"added {len(add):,} removed {len(rem):,} | selected pairs {len(SA):,} -> {len(SB):,}")
    log(f"GATES {country}: delta-logit on rows prob3>=1e-3 (n={len(d):,}): mean {d.mean():.4f} sd {d.std():.4f} "
        f"p1 {np.percentile(d, 1):.3f} p99 {np.percentile(d, 99):.3f} |d|>1 share {(np.abs(d) > 1).mean():.4f}")
    log("GATES reference (holdout, planner/gates.log): India 13.97/1000, US 11.08/1000 entities changed; "
        "delta-logit mean -0.080 sd 0.756 p1 -2.280 p99 1.650 |d|>1 share 0.162")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="E039_test")
    ap.add_argument("--split", default="testT", help="data version for the level-2 tables (testT)")
    ap.add_argument("--countries", nargs="+", default=["India", "US"], help="countries that get the residual")
    ap.add_argument("--process", nargs="+", default=None,
                    help="countries processed in THIS invocation (default: all found); e.g. one process per country")
    ap.add_argument("--model-dir", default=str(EXP / "L2_ALL"))
    ap.add_argument("--in-dir", default="scores_feats")
    ap.add_argument("--out-dir", default="scores_l2")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--preflight-rows", type=int, default=20000)
    ap.add_argument("--no-gates", action="store_true")
    ap.add_argument("--smoke", default=None, help="process ONLY this chunk file (residual applied whatever the country)")
    ap.add_argument("--smoke-out", default=None)
    args = ap.parse_args()
    t00 = time.time()
    R = Residual(Path(args.model_dir), args.threads)
    if args.smoke:
        f = Path(args.smoke)
        country = re.fullmatch(r"(.+)_p\d+_c\d+\.parquet", f.name).group(1)
        T = L2.Tables(args.split, country, log=log)
        log(f"tables built, peak RSS {rss_gb():.2f} GB")
        g = {"dl": []}
        out = R.chunk(f, T, g, args.preflight_rows)
        d = np.concatenate(g["dl"])
        log(f"SMOKE {f.name}: delta-logit mean {d.mean():.4f} sd {d.std():.4f} |d|>1 share {(np.abs(d) > 1).mean():.4f}; "
            f"rows prob>=0.5: {(pd.read_parquet(f, columns=['prob3']).prob3.to_numpy() >= 0.5).sum():,} -> {(out.prob.to_numpy() >= 0.5).sum():,}")
        if args.smoke_out:
            out.drop(columns=["_prob3"]).to_parquet(args.smoke_out, index=False)
        log(f"done {time.time() - t00:.0f}s, peak RSS {rss_gb():.2f} GB")
        return
    run = EXP / args.run
    in_dir, out_dir = run / args.in_dir, run / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = chunks_by_part(in_dir)
    countries = sorted({c for c, _ in groups})
    log(f"run {args.run}: parts {len(groups)} in {in_dir.name} ({', '.join(countries)}); residual for {args.countries}")
    for country in countries:
        if args.process and country not in args.process:
            continue
        todo = []
        for (c, i), files in groups.items():
            if c != country:
                continue
            if not (run / "scores" / f"{country}_p{i}.parquet").exists():
                log(f"  SKIP {country} part {i}: pass 2 not finished (scores/{country}_p{i}.parquet missing)")
                continue
            if len(files) != 4:
                log(f"  WARNING {country} part {i}: {len(files)} chunk files (expected 4)")
            todo.append((i, files))
        T = None
        g = {"dl": []}
        for i, files in todo:
            sp = out_dir / f"{country}_p{i}.parquet"
            if sp.exists() and not args.overwrite:
                log(f"  {sp.name} exists, skipped")
                continue
            t0 = time.time()
            if country in args.countries:
                if T is None:
                    T = L2.Tables(args.split, country, log=log)
                    log(f"  tables {country} ready, peak RSS {rss_gb():.2f} GB")
                out = pd.concat([R.chunk(f, T, g, args.preflight_rows) for f in files], ignore_index=True)
            else:
                out = []
                for f in files:
                    t = pd.read_parquet(f, columns=["s1", "cand", "src", "prob3"])
                    out.append(pd.DataFrame({"s1": t["s1"].to_numpy(), "cand": t["cand"].to_numpy(),
                                             "src": np.rint(t["src"].to_numpy()).astype(np.int8),
                                             "prob": t["prob3"].to_numpy().astype(np.float32),
                                             "_prob3": t["prob3"].to_numpy().astype(np.float32)}))
                out = pd.concat(out, ignore_index=True)
            ref = pd.read_parquet(run / "scores" / f"{country}_p{i}.parquet", columns=["s1", "cand", "prob"])
            same = len(ref) == len(out) and (ref["cand"].to_numpy() == out["cand"].to_numpy()).all()
            note = "rows aligned with scores/" if same else "WARNING: rows NOT aligned with scores/"
            if same:
                dmax = float(np.abs(ref["prob"].to_numpy() - out["_prob3"].to_numpy()).max())
                note += f", prob3 vs scores/ prob max diff {dmax:.2e}"
                if dmax > 1e-6:
                    note += " (WARNING: scores/ is not the plain stage-3 prob)"
            del ref
            out = out.drop(columns=["_prob3"])
            tmp = sp.with_suffix(".partial")
            out.to_parquet(tmp, index=False)
            tmp.replace(sp)
            log(f"  wrote {sp.name}: {len(out):,} rows ({time.time() - t0:.0f}s; {note})")
            del out
            gc.collect()
        del T
        gc.collect()
        if country in args.countries and not args.no_gates:
            gates(country, run, out_dir, in_dir, g["dl"])
    # completeness: every pass-2 part (ceil(#S1 / 200k) per country, as infer_v3.parts_of) must be in out_dir
    ok_all = True
    for country in countries:
        n_s1 = pq.ParquetFile(DATA_ROOT / "data" / "processed" / f"{args.split}_s1_{country}.parquet").metadata.num_rows
        exp = max(1, int(np.ceil(n_s1 / 200_000)))
        have = sorted(int(re.fullmatch(rf"{country}_p(\d+)\.parquet", f.name).group(1)) for f in out_dir.glob(f"{country}_p*.parquet"))
        missing = sorted(set(range(exp)) - set(have))
        ok_all &= not missing
        log(f"  {country}: {len(have)}/{exp} parts in {out_dir.name}" + (f"  MISSING parts {missing}" if missing else "  complete"))
    for c in args.countries:
        if c not in countries:
            log(f"  WARNING: --countries {c} has no chunk files in {in_dir}")
    log(("ALL PARTS COMPLETE -> ready for: python -m src.infer_v3 write --run " + args.run + " --split testT --tag tstT "
         "--scores " + args.out_dir + " --sub-id <id>") if ok_all else "INCOMPLETE: do NOT run the write step yet (re-run this script when pass 2 has finished)")
    log(f"done {time.time() - t00:.0f}s, peak RSS {rss_gb():.2f} GB -> {out_dir}")


if __name__ == "__main__":
    main()
