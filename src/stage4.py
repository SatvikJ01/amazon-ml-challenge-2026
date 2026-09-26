"""E036: second collective round (stage 4).

Stage 3 already uses sibling / number-consensus features, but they are weighted
by stage-2 probabilities.  Stage-3 probabilities are much sharper (holdout 0.980 vs
0.972), so recomputing the same collective features from cross-fitted stage-3
probabilities ``p3`` gives every candidate better evidence about its siblings.

Features added on top of the stage-3 table (``s4_`` prefix = computed from p3):
  p3, s4_rank          stage-3 probability and its rank in the entity
  s4_ent_sum           sum of p3 over the entity (expected number of matches)
  s4_best_other        best p3 among the entity's other candidates
  s4_<sibling>         sibling features (anchors = p3 >= 0.9)
  s4_nsup_*            number-consensus features weighted by p3

Steps:
  python -m src.collective oof --tag v3c_ancz --exp E035_stage3 --out E036_oof --model xgb
  python -m src.stage4 build --country India --tag v3c_ancz --out-tag v3c_s4
  python -m src.train --tag v3c_s4 --exp E036_stage4
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .collective import _texts, sibling_features
from .extra_features import extra_features

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
EXPERIMENTS = ROOT / "experiments"
SIB = ["n_anchors", "anchor_p_mean", "sib_name", "sib_addr", "sib_all", "sib_num_eq", "is_anchor"]
NSUP = ["nsup_first_p", "nsup_extra_p", "nsup_s1_p"]


def stage4_features(s1: np.ndarray, cand: np.ndarray, p3: np.ndarray, c_name: np.ndarray, c_addr: np.ndarray,
                    q_addr: np.ndarray) -> pd.DataFrame:
    p3 = np.asarray(p3, np.float32)
    sf = sibling_features(s1, cand, p3, c_name, c_addr)
    ef = extra_features(s1, p3, q_addr, c_addr)
    d = pd.DataFrame({"s1": s1, "p3": p3})
    g = d.groupby("s1", sort=False)["p3"]
    ent_sum = g.transform("sum").to_numpy(np.float32)
    mx = g.transform("max").to_numpy(np.float32)
    # best other: the max, unless this row is the max, then the second max
    second = d.assign(r=g.rank(ascending=False, method="first")).query("r == 2").set_index("s1")["p3"]
    sec = second.reindex(s1).fillna(0).to_numpy(np.float32)
    best_other = np.where(p3 >= mx, sec, mx).astype(np.float32)
    out = pd.DataFrame({"p3": p3, "s4_rank": g.rank(ascending=False, method="min").to_numpy(np.float32),
                        "s4_ent_sum": ent_sum, "s4_best_other": best_other})
    for c in SIB:
        out[f"s4_{c}"] = sf[c].to_numpy()
    for c in NSUP:
        out[f"s4_{c}"] = ef[c].to_numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build"])
    ap.add_argument("--country", required=True)
    ap.add_argument("--tag", default="v3c_ancz")
    ap.add_argument("--out-tag", default="v3c_s4")
    ap.add_argument("--oof", default="E036_oof")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()
    t0 = time.time()
    t = pq.read_table(PROCESSED / f"feats_{args.tag}_{args.country}.parquet")
    s1, cand = t.column("s1").to_numpy(), t.column("cand").to_numpy()
    oof = pd.read_parquet(EXPERIMENTS / args.oof / "oof_p2.parquet", columns=["s1", "cand", "p2"])
    p3 = pd.DataFrame({"s1": s1, "cand": cand}).merge(oof, on=["s1", "cand"], how="left")["p2"].to_numpy(np.float32)
    del oof
    assert not np.isnan(p3).any(), "missing OOF stage-3 probabilities"
    q = pd.read_parquet(PROCESSED / f"{args.split}_s1_{args.country}.parquet", columns=["code", "addr_norm"]).set_index("code")
    tx = _texts(args.split, args.country, cand).reindex(cand)
    f = stage4_features(s1, cand, p3, tx["name_norm"].to_numpy(), tx["addr_norm"].to_numpy(), q["addr_norm"].reindex(s1).to_numpy())
    del q, tx
    for c in f.columns:
        t = t.append_column(c, pa.array(f[c].to_numpy()))
    pq.write_table(t, PROCESSED / f"feats_{args.out_tag}_{args.country}.parquet", compression="zstd")
    shutil.copy(PROCESSED / f"entities_{args.tag}.npy", PROCESSED / f"entities_{args.out_tag}.npy")
    print(f"{args.country}: {t.num_rows:,} pairs -> feats_{args.out_tag}_{args.country} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
