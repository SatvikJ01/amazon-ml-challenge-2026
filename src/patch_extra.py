"""Append the E033 extra features (``src/extra_features.py``) to a stage-3 training
table, written under a new tag; the source table is never modified.

Usage: python -m src.patch_extra --country India --tag v3c_anc --out-tag v3c_ancx
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

from .collective import _texts
from .extra_features import EXTRA_COLS, extra_features

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", required=True)
    ap.add_argument("--tag", default="v3c_anc")
    ap.add_argument("--out-tag", default="v3c_ancx")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()
    t0 = time.time()
    t = pq.read_table(PROCESSED / f"feats_{args.tag}_{args.country}.parquet")
    s1 = t.column("s1").to_numpy()
    cand = t.column("cand").to_numpy()
    p2 = t.column("p2").to_numpy(zero_copy_only=False)
    q = pd.read_parquet(PROCESSED / f"{args.split}_s1_{args.country}.parquet", columns=["code", "addr_norm"])
    q_addr = q.set_index("code")["addr_norm"].reindex(s1).to_numpy()
    del q
    c_addr = _texts(args.split, args.country, cand)["addr_norm"].reindex(cand).to_numpy()
    f = extra_features(s1, p2, q_addr, c_addr)
    for c in EXTRA_COLS:
        t = t.append_column(c, pa.array(f[c].to_numpy()))
    pq.write_table(t, PROCESSED / f"feats_{args.out_tag}_{args.country}.parquet", compression="zstd")
    shutil.copy(PROCESSED / f"entities_{args.tag}.npy", PROCESSED / f"entities_{args.out_tag}.npy")
    if "label" in t.column_names:
        y = t.column("label").to_numpy()
        for c in ("anum_first_rel", "anum_c_extra", "anum_c_alpha_extra"):
            v = f[c].to_numpy()
            print(f"  {c}: " + "  ".join(f"{k:g}: pos {np.mean(y[v == k]):.3f} (n={int((v == k).sum()):,})"
                                         for k in np.unique(v[~np.isnan(v)])[:6]), flush=True)
    print(f"{args.country}: {t.num_rows:,} pairs -> feats_{args.out_tag}_{args.country} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
