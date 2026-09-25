"""Attach pairwise features (and labels, for train) to a candidate file.

Input:  data/processed/cands_{tag}_{country}.parquet
Output: data/processed/feats_{tag}_{country}.parquet

Candidates can be truncated to the first ``--depth`` ranks per (entity, source)
before feature computation; the depth used here defines ``candidate_pairs.tsv``.

Work is chunked by entity so the peak memory is set by ``--chunk-entities``, not
by the size of the block.  Token statistics (IDF, frequent-token set for the
"core name") are learned from that country's S1 records of the same split, which
are available at test time too -- no label information is involved.

Usage:
    python -m src.build_features --split train --tag trn --depth 20
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .candidates import cand_path, countries_for, read_cands
from .features import TokenStats, add_context_features, pair_features
from .ids import OFFSET

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
INTERIM = ROOT / "data" / "interim"


def load_texts(split: str, country: str):
    """Code-indexed text lookups for S1 and for S2+S3 of one country."""
    s1 = pd.read_parquet(PROCESSED / f"{split}_s1_{country}.parquet")
    t = pd.concat(
        [pd.read_parquet(PROCESSED / f"{split}_s{s}_{country}.parquet") for s in (2, 3)],
        ignore_index=True,
    )
    return s1.set_index("code"), t.set_index("code")


def label_pairs(s1: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """1 if (s1, cand) is a ground-truth pair.  Uses the exclusivity property:
    every S2/S3 record has at most one owner, so a cand -> owner lookup is exact."""
    ps1 = np.load(INTERIM / "gt_pair_s1.npy")
    pm = np.load(INTERIM / "gt_pair_m.npy")
    owner = pd.Series(ps1, index=pm)
    got = owner.reindex(cand).to_numpy()
    return (got == s1).astype(np.int8)


def competition_features(s1: np.ndarray, cand: np.ndarray, score: np.ndarray,
                         rows: np.ndarray | None = None) -> pd.DataFrame:
    """Per-pair features describing how contested the candidate is.

    Aggregates are taken over the candidate table of **all** S1 entities in the
    country block (train and test alike), so they encode the exclusivity
    structure of the data -- every S2/S3 record has exactly one owner -- without
    using labels.  ``rows`` selects which pairs to return features for; the
    aggregates still use every row.

        comp_best_other   best retrieval score this candidate gets from any
                          *other* S1 entity (0 if none)
        comp_margin       own score minus comp_best_other
        comp_is_top       1 if this S1 entity is the candidate's best claimant
        comp_n            number of S1 entities that retrieved this candidate
    """
    order = np.lexsort((-score, cand))
    cs = cand[order]
    first = np.r_[True, cs[1:] != cs[:-1]]
    starts = np.flatnonzero(first)
    uc = cs[starts]
    del cs, first
    counts = np.diff(np.r_[starts, order.size])
    top1 = score[order[starts]]
    top1_s1 = s1[order[starts]]
    second = np.minimum(starts + 1, order.size - 1)
    top2 = np.where(counts > 1, score[order[second]], 0.0).astype(np.float32)
    del order, second, starts

    if rows is None:
        rows = np.arange(s1.size)
    rc, rs, rsc = cand[rows], s1[rows], score[rows]
    pos = np.searchsorted(uc, rc)
    is_top = rs == top1_s1[pos]
    best_other = np.where(is_top, top2[pos], top1[pos]).astype(np.float32)
    return pd.DataFrame({
        "comp_best_other": best_other,
        "comp_margin": (rsc - best_other).astype(np.float32),
        "comp_is_top": is_top.astype(np.int8),
        "comp_n": counts[pos].astype(np.int16),
    })


def _load_source_cands(tag: str, country: str, source: int, depth: int,
                       keep: np.ndarray | None) -> pd.DataFrame:
    """Candidates of one target source with competition features attached,
    restricted to the ``keep`` entities.  S2 and S3 ids are disjoint, so the
    per-candidate aggregates can be computed one source at a time, which halves
    the peak memory of the sort."""
    df = pd.read_parquet(cand_path(tag, country, source))
    if depth <= int(df["blk_rank"].max()):
        df = df[df["blk_rank"] < depth].reset_index(drop=True)
    rows = np.flatnonzero(np.isin(df["s1"].to_numpy(), keep)) if keep is not None else None
    comp = competition_features(df["s1"].to_numpy(), df["cand"].to_numpy(),
                                df["blk_score"].to_numpy(), rows)
    if rows is not None:
        df = df.iloc[rows].reset_index(drop=True)
    out = pd.concat([df, comp], axis=1)
    del df, comp, rows
    gc.collect()
    return out


def _load_texts_for(split: str, country: str, s1_codes: np.ndarray, cand_codes: np.ndarray):
    """Text lookups restricted to the codes actually needed, plus token stats
    learned from the country's full S1 file."""
    s1 = pd.read_parquet(PROCESSED / f"{split}_s1_{country}.parquet")
    stats = TokenStats(s1["name_norm"].to_numpy(), s1["addr_norm"].to_numpy())
    s1 = s1[np.isin(s1["code"].to_numpy(), s1_codes)].set_index("code")
    parts = []
    for src in (2, 3):
        t = pd.read_parquet(PROCESSED / f"{split}_s{src}_{country}.parquet")
        parts.append(t[np.isin(t["code"].to_numpy(), cand_codes)])
        del t
        gc.collect()
    tgt = pd.concat(parts, ignore_index=True).set_index("code")
    return s1, tgt, stats


def featurize_country(split: str, tag: str, country: str, depth: int,
                      keep: np.ndarray | None = None, chunk_entities: int = 40_000,
                      with_labels: bool = False, stage1_model: str | None = None):
    """Yield feature frames for one country block, ``chunk_entities`` at a time.

    Shared by training (frames are written to disk) and inference (each frame is
    scored and dropped), so both paths compute identical features.
    """
    cands = pd.concat([_load_source_cands(tag, country, s, depth, keep) for s in (2, 3)],
                      ignore_index=True)
    cands = cands.sort_values(["s1", "src", "blk_rank"], kind="stable").reset_index(drop=True)
    if stage1_model is not None:
        import lightgbm as lgb
        from .stage1 import STAGE1_FEATURES, STAGE1_THRESHOLD, add_stage1_context
        cands = add_stage1_context(cands)
        p1 = lgb.Booster(model_file=stage1_model).predict(cands[STAGE1_FEATURES].to_numpy(np.float32))
        cands = cands[p1 >= STAGE1_THRESHOLD].drop(
            columns=["blk_score_gap", "blk_score_rank", "blk_score_src_gap"]).reset_index(drop=True)
        del p1
        gc.collect()
    s1_txt, t_txt, stats = _load_texts_for(split, country, np.unique(cands["s1"].to_numpy()),
                                           np.unique(cands["cand"].to_numpy()))

    s1_all = cands["s1"].to_numpy()
    bounds = np.flatnonzero(np.r_[True, s1_all[1:] != s1_all[:-1], True])   # entity starts + end
    n_ent = bounds.size - 1
    for e0 in range(0, n_ent, chunk_entities):
        lo, hi = bounds[e0], bounds[min(e0 + chunk_entities, n_ent)]
        c = cands.iloc[lo:hi].reset_index(drop=True)
        q = s1_txt.loc[c["s1"].to_numpy()]
        m = t_txt.loc[c["cand"].to_numpy()]
        feats = pair_features(
            q["name_norm"].to_numpy(), q["addr_norm"].to_numpy(),
            m["name_norm"].to_numpy(), m["addr_norm"].to_numpy(),
            m["nonascii"].to_numpy(), stats,
        )
        df = pd.concat([c, feats], axis=1)
        df["src"] = df["src"].astype(np.int8)
        df = add_context_features(df)
        if with_labels:
            df["label"] = label_pairs(df["s1"].to_numpy(), df["cand"].to_numpy())
        for col in df.columns:
            if df[col].dtype == np.float64:
                df[col] = df[col].astype(np.float32)
        yield df
        del q, m, feats, c, df
        gc.collect()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--depth", type=int, default=20)
    ap.add_argument("--chunk-entities", type=int, default=40_000)
    ap.add_argument("--countries", nargs="*", default=None)
    ap.add_argument("--entities", default=None,
                    help="npy of S1 codes to featurise (default: all in the candidate file)")
    ap.add_argument("--out-tag", default=None, help="output tag (default: --tag)")
    args = ap.parse_args()
    keep = np.load(args.entities) if args.entities else None
    out_tag = args.out_tag or args.tag

    for country in args.countries or countries_for(args.split):
        t0 = time.time()
        if not all(cand_path(args.tag, country, s).exists() for s in (2, 3)):
            print(f"skip {country}: candidates missing")
            continue
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = PROCESSED / f"feats_{out_tag}_{country}.parquet"
        tmp = path.with_suffix(".partial")
        writer, n_rows, n_pos, ents = None, 0, 0, 0
        try:
            # Stream chunks to disk: concatenating them in RAM would push the
            # US block over the memory cap.
            for df in featurize_country(args.split, args.tag, country, args.depth, keep,
                                        args.chunk_entities, with_labels=(args.split == "train")):
                table = pa.Table.from_pandas(df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
                writer.write_table(table)
                n_rows += len(df)
                ents += df["s1"].nunique()
                n_pos += int(df["label"].sum()) if "label" in df else 0
                print(f"  {country}: {n_rows:,} pairs ({time.time() - t0:.0f}s)", flush=True)
                del df, table
                gc.collect()
        finally:
            if writer is not None:
                writer.close()
        tmp.replace(path)
        print(f"== {country}: {n_rows:,} pairs for {ents:,} entities, "
              f"pos_rate={n_pos / max(n_rows, 1):.4f}, {time.time() - t0:.0f}s -> {path.name}", flush=True)
        gc.collect()

if __name__ == "__main__":
    main()
