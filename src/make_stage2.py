"""Build the stage-2 training set: stage-1 survivors with survivor-level context.

Filters the full feature files by the out-of-fold stage-1 probability, drops the
context columns (computed over the full depth-30 list) and recomputes them over
the survivors -- exactly what ``featurize_country`` does at test time.

Usage: python -m src.make_stage2 --tag trn2 --stage1 E013_stage1 --out-tag trn2c
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .features import add_context_features
from .stage1 import STAGE1_THRESHOLD

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
EXPERIMENTS = ROOT / "experiments"

CONTEXT_PREFIXES = ("blk_score_gap", "blk_score_rank", "name_tset_", "addr_tset_", "name_idf_u_",
                    "addr_idf_u_", "blk_score_src_gap", "n_cands")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="trn2")
    ap.add_argument("--stage1", default="E013_stage1")
    ap.add_argument("--out-tag", default="trn2c")
    ap.add_argument("--countries", nargs="*", default=["India", "US"])
    args = ap.parse_args()

    oof = pd.read_parquet(EXPERIMENTS / args.stage1 / "oof_p1.parquet")
    for c in args.countries:
        import pyarrow as pa
        import pyarrow.parquet as pq
        path = PROCESSED / f"feats_{args.tag}_{c}.parquet"
        # Survivor mask from the key columns only; the full frame is filtered at
        # the Arrow level so only survivors are ever materialised in pandas.
        keys = pq.read_table(path, columns=["s1", "cand"]).to_pandas()
        p1 = keys.merge(oof, on=["s1", "cand"], how="left")["p1"].to_numpy()
        assert not np.isnan(p1).any(), "missing stage-1 OOF probabilities"
        mask = pa.array(p1 >= STAGE1_THRESHOLD)
        del keys, p1
        df = pq.read_table(path).filter(mask).to_pandas()
        ctx = [col for col in df.columns if col.startswith(CONTEXT_PREFIXES)]
        df = add_context_features(df.drop(columns=ctx).reset_index(drop=True))
        for col in df.columns:
            if df[col].dtype == np.float64:
                df[col] = df[col].astype(np.float32)
        df.to_parquet(PROCESSED / f"feats_{args.out_tag}_{c}.parquet", index=False, compression="zstd")
        print(f"{c}: {len(df):,} survivor pairs, pos_rate {df['label'].mean():.4f}", flush=True)
        del df
    shutil.copy(PROCESSED / f"entities_{args.tag}.npy", PROCESSED / f"entities_{args.out_tag}.npy")


if __name__ == "__main__":
    main()
