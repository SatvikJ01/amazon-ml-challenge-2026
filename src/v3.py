"""v3 pipeline: multi-channel candidates -> cheap stage 1 -> string features on survivors.

Candidates per (country, source) = forward top-30 ∪ reverse top-3 ∪ exact-key
channel (``src/key_channel.py``).  Every pair carries per-channel evidence
(scores, ranks, membership flags) and competition features computed from the
*full* forward table, so a pair's features never depend on which entities are
being processed.

Stage 1 runs on cheap features only (no string comparisons), cross-fitted; the
expensive string features are computed for stage-1 survivors only.  That is what
makes a 2x larger training sample affordable.

Steps:
    s1data --split train --country C --entities data/processed/entities_v3.npy
        -> feats_v3s1_{C}.parquet : every candidate pair of the entity set with
           stage-1 features (+ label on train)
    s2data --country C --stage1 E030_stage1
        -> feats_v3c_{C}.parquet  : OOF-p1 survivors with string + context features

``featurize`` (used by inference) runs both on the fly for one part of entities.
"""
from __future__ import annotations

import argparse
import gc
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .build_features import label_pairs
from .candidates import cand_path, rev_path
from .features import TokenStats, add_context_features, pair_features
from .ids import OFFSET
from .key_channel import key_path
from .make_stage2 import CONTEXT_PREFIXES

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
EXPERIMENTS = ROOT / "experiments"

STAGE1_V3 = ["blk_score", "blk_rank", "src", "comp_best_other", "comp_margin", "comp_is_top", "comp_n",
             "blk_score_gap", "blk_score_rank", "blk_score_src_gap", "rev_score", "rev_rank", "in_fwd",
             "in_rev", "in_key", "key_hits", "key_minfreq", "ret_score", "ret_score_gap", "ret_score_rank"]
STAGE1_THRESHOLD = 1e-3
# v4: extra retrieval channels (src/channels.py, src/embed_channel.py) unioned when listed here.
EXTRA_CHANNELS = [c for c in os.environ.get("V4_CHANNELS", "").split(",") if c]


def stage1_features() -> list[str]:
    return STAGE1_V3 + [f for ch in EXTRA_CHANNELS for f in (f"{ch}_score", f"{ch}_rank", f"in_{ch}")]


def _read(path: Path, cols: list[str], keep: np.ndarray | None) -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=cols)
    if keep is not None:
        t = t.filter(pc.is_in(t.column("s1"), value_set=pa.array(keep)))
    return t.to_pandas()


def union_source(tag: str, country: str, source: int, keep: np.ndarray | None) -> pd.DataFrame:
    """All channel candidates of one source for the ``keep`` entities, with
    channel evidence and competition features (aggregates over the full forward table)."""
    fa = pd.read_parquet(cand_path(tag, country, source), columns=["s1", "cand", "blk_score", "blk_rank"])
    so = np.lexsort((-fa["blk_score"].to_numpy(), fa["cand"].to_numpy()))
    cs = fa["cand"].to_numpy()[so]
    st = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
    uc, cnt = cs[st], np.diff(np.r_[st, cs.size])
    top1 = fa["blk_score"].to_numpy()[so[st]]
    top1_s1 = fa["s1"].to_numpy()[so[st]]
    top2 = np.where(cnt > 1, fa["blk_score"].to_numpy()[so[np.minimum(st + 1, cs.size - 1)]], 0.0)
    del so, cs, st
    f = fa if keep is None else fa[np.isin(fa["s1"].to_numpy(), keep)]
    del fa
    gc.collect()
    r = _read(rev_path(tag, country, source), ["s1", "cand", "rev_score", "rev_rank"], keep)
    k = _read(key_path(tag, country, source), ["s1", "cand", "key_hits", "key_minfreq"], keep)
    from .channels import channel_path
    ex = [(ch, _read(channel_path(tag, country, source, ch), ["s1", "cand", "score", "rank"], keep)
           .rename(columns={"score": f"{ch}_score", "rank": f"{ch}_rank"})) for ch in EXTRA_CHANNELS]

    allp = pd.concat([f[["s1", "cand"]], r[["s1", "cand"]], k[["s1", "cand"]]] + [e[["s1", "cand"]] for _, e in ex],
                     ignore_index=True)
    s1u = np.unique(allp["s1"].to_numpy())
    kk = lambda d: (np.searchsorted(s1u, d["s1"].to_numpy()).astype(np.int64) << 34) | (d["cand"].to_numpy() % OFFSET)
    uk, first = np.unique(kk(allp), return_index=True)
    u = allp.iloc[first].reset_index(drop=True)
    del allp

    def attach(src_df, cols, fill):
        pos = np.searchsorted(uk, kk(src_df))
        for c, fv in zip(cols, fill):
            arr = np.full(uk.size, fv, dtype=np.float32 if isinstance(fv, float) else np.int16)
            arr[pos] = src_df[c].to_numpy()
            u[c] = arr
        flag = np.zeros(uk.size, np.int8); flag[pos] = 1
        return flag

    u["in_fwd"] = attach(f, ["blk_score", "blk_rank"], [np.nan, 99])
    u["in_rev"] = attach(r, ["rev_score", "rev_rank"], [np.nan, 9])
    u["in_key"] = attach(k, ["key_hits", "key_minfreq"], [0.0, np.nan])
    for ch, e in ex:
        u[f"in_{ch}"] = attach(e, [f"{ch}_score", f"{ch}_rank"], [np.nan, 99])
    u["src"] = np.int8(source)
    u["ret_score"] = u["blk_score"].fillna(u["rev_score"]).astype(np.float32)

    pos = np.minimum(np.searchsorted(uc, u["cand"].to_numpy()), uc.size - 1)
    known = uc[pos] == u["cand"].to_numpy()
    is_top = known & (top1_s1[pos] == u["s1"].to_numpy())
    best_other = np.where(known, np.where(is_top, top2[pos], top1[pos]), 0.0).astype(np.float32)
    u["comp_best_other"] = best_other
    u["comp_margin"] = (u["blk_score"].to_numpy() - best_other).astype(np.float32)
    u["comp_is_top"] = is_top.astype(np.int8)
    u["comp_n"] = np.where(known, cnt[pos], 0).astype(np.int16)
    return u


def stage1_frame(split: str, tag: str, country: str, keep: np.ndarray | None, with_labels: bool) -> pd.DataFrame:
    from .stage1 import add_stage1_context
    df = pd.concat([union_source(tag, country, s, keep) for s in (2, 3)], ignore_index=True)
    df = df.sort_values(["s1", "src"], kind="stable").reset_index(drop=True)
    df = add_stage1_context(df)
    if with_labels:
        df["label"] = label_pairs(df["s1"].to_numpy(), df["cand"].to_numpy())
    return df


def _texts_full(split: str, country: str, codes: np.ndarray, source_ids=(2, 3)) -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    want = pa.array(np.unique(codes))
    parts = []
    for s in source_ids:
        t = pq.read_table(PROCESSED / f"{split}_s{s}_{country}.parquet", columns=["code", "name_norm", "addr_norm", "nonascii"])
        parts.append(t.filter(pc.is_in(t.column("code"), value_set=want)).to_pandas())
    return pd.concat(parts, ignore_index=True).set_index("code")


def string_features(split: str, country: str, df: pd.DataFrame, chunk_entities: int = 40_000):
    """Yield df chunks (whole entities) with pairwise string + context features."""
    s1t = pd.read_parquet(PROCESSED / f"{split}_s1_{country}.parquet")
    stats = TokenStats(s1t["name_norm"].to_numpy(), s1t["addr_norm"].to_numpy())
    s1t = s1t[np.isin(s1t["code"].to_numpy(), df["s1"].unique())].set_index("code")
    ct = _texts_full(split, country, df["cand"].to_numpy())
    df = df.sort_values(["s1", "src"], kind="stable").reset_index(drop=True)
    s1a = df["s1"].to_numpy()
    b = np.flatnonzero(np.r_[True, s1a[1:] != s1a[:-1], True])
    for e0 in range(0, b.size - 1, chunk_entities):
        c = df.iloc[b[e0]: b[min(e0 + chunk_entities, b.size - 1)]].reset_index(drop=True)
        q, m = s1t.loc[c["s1"].to_numpy()], ct.loc[c["cand"].to_numpy()]
        pf = pair_features(q["name_norm"].to_numpy(), q["addr_norm"].to_numpy(), m["name_norm"].to_numpy(),
                           m["addr_norm"].to_numpy(), m["nonascii"].to_numpy(), stats)
        c = c.drop(columns=[x for x in c.columns if x.startswith(CONTEXT_PREFIXES)])
        c = add_context_features(pd.concat([c, pf], axis=1))
        for col in c.columns:
            if c[col].dtype == np.float64:
                c[col] = c[col].astype(np.float32)
        yield c


def main() -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["s1data", "s2data"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--tag", default="trnall")
    ap.add_argument("--country", required=True)
    ap.add_argument("--entities", default="data/processed/entities_v3.npy")
    ap.add_argument("--stage1", default="E030_stage1")
    ap.add_argument("--prefix", default="v3", help="output tag prefix (feats_{prefix}s1 / feats_{prefix}c)")
    args = ap.parse_args()
    t0 = time.time()
    if args.step == "s1data":
        keep = np.load(args.entities)
        df = stage1_frame(args.split, args.tag, args.country, keep, with_labels=True)
        df.to_parquet(PROCESSED / f"feats_{args.prefix}s1_{args.country}.parquet", index=False, compression="zstd")
        print(f"{args.country}: {len(df):,} union pairs for {df['s1'].nunique():,} entities "
              f"({len(df) / df['s1'].nunique():.1f}/entity), pos_rate {df['label'].mean():.4f}, "
              f"label coverage {df['label'].sum():,} ({time.time() - t0:.0f}s)", flush=True)
        return
    oof = pd.read_parquet(EXPERIMENTS / args.stage1 / "oof_p1.parquet")
    path = PROCESSED / f"feats_{args.prefix}s1_{args.country}.parquet"
    keys = pq.read_table(path, columns=["s1", "cand"]).to_pandas()
    p1 = keys.merge(oof, on=["s1", "cand"], how="left")["p1"].to_numpy()
    assert not np.isnan(p1).any()
    df = pq.read_table(path).filter(pa.array(p1 >= STAGE1_THRESHOLD)).to_pandas()
    del keys, p1
    out = PROCESSED / f"feats_{args.prefix}c_{args.country}.parquet"
    tmp = out.with_suffix(".partial")
    w = None
    n = 0
    for c in string_features(args.split, args.country, df):
        t = pa.Table.from_pandas(c, preserve_index=False)
        w = w or pq.ParquetWriter(tmp, t.schema, compression="zstd")
        w.write_table(t)
        n += len(c)
    w.close()
    tmp.replace(out)
    shutil.copy(args.entities, PROCESSED / f"entities_{args.prefix}c.npy")
    print(f"{args.country}: {n:,} survivor pairs with string features ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
