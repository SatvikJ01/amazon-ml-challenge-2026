"""Append extra stage-3 features to a training table, written under a new tag; the
source table is never modified.  Sets: ``num`` = E033 (``src/extra_features.py``),
``name`` = E034 (``src/extra_features2.py``), ``ctx`` = E035 (``src/extra_features3.py``).

Usage: python -m src.patch_extra --country India --tag v3c_anc --out-tag v3c_ancx --sets num
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
from .extra_features2 import EXTRA2_COLS, NameStats, extra_features2
from .extra_features3 import EXTRA3_COLS, PoolNames, extra_features3
from .extra_features4 import EXTRA4_COLS, anti_match_features

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", required=True)
    ap.add_argument("--tag", default="v3c_anc")
    ap.add_argument("--out-tag", default="v3c_ancx")
    ap.add_argument("--split", default="train")
    ap.add_argument("--sets", default="num", help="comma list: num,name,ctx,anti")
    args = ap.parse_args()
    t0 = time.time()
    t = pq.read_table(PROCESSED / f"feats_{args.tag}_{args.country}.parquet")
    s1 = t.column("s1").to_numpy()
    cand = t.column("cand").to_numpy()
    p2 = t.column("p2").to_numpy(zero_copy_only=False)
    sets = args.sets.split(",")
    q = pd.read_parquet(PROCESSED / f"{args.split}_s1_{args.country}.parquet", columns=["code", "name_norm", "addr_norm"]).set_index("code")
    q = q.reindex(s1)
    tx = _texts(args.split, args.country, cand).reindex(cand)
    q_addr, c_addr = q["addr_norm"].to_numpy(), tx["addr_norm"].to_numpy()
    cols = []
    if "num" in sets:
        f = extra_features(s1, p2, q_addr, c_addr)
        cols += [(c, f[c].to_numpy()) for c in EXTRA_COLS if c not in t.column_names]
    if "name" in sets:
        f = extra_features2(q["name_norm"].to_numpy(), tx["name_norm"].to_numpy(), q_addr, c_addr,
                            NameStats(args.split, args.country))
        cols += [(c, f[c].to_numpy()) for c in EXTRA2_COLS]
    if "ctx" in sets:
        f = extra_features3(s1, q_addr, c_addr, tx["name_norm"].to_numpy(), t.column("name_tset").to_numpy(),
                            t.column("addr_empty_c").to_numpy(), PoolNames(args.split, args.country))
        cols += [(c, f[c].to_numpy()) for c in EXTRA3_COLS]
    if "anti" in sets:
        f = anti_match_features(q["name_norm"].to_numpy(), tx["name_norm"].to_numpy(), q_addr, c_addr,
                                NameStats(args.split, args.country))
        cols += [(c, f[c].to_numpy()) for c in EXTRA4_COLS]
    del q, tx
    for c, v in cols:
        t = t.append_column(c, pa.array(v))
    pq.write_table(t, PROCESSED / f"feats_{args.out_tag}_{args.country}.parquet", compression="zstd")
    shutil.copy(PROCESSED / f"entities_{args.tag}.npy", PROCESSED / f"entities_{args.out_tag}.npy")
    if "label" in t.column_names:
        y = t.column("label").to_numpy()
        for c in [x for x in ("anum_c_extra", "nm_c_extra_sub", "nm_q_miss") if x in t.column_names]:
            v = t.column(c).to_numpy()
            print(f"  {c}: " + "  ".join(f"{k:g}: pos {np.mean(y[v == k]):.3f} (n={int((v == k).sum()):,})"
                                         for k in np.unique(v[~np.isnan(v)])[:6]), flush=True)
    print(f"{args.country}: {t.num_rows:,} pairs -> feats_{args.out_tag}_{args.country} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
