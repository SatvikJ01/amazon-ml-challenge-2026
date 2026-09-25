"""Generate candidate pairs for a set of Source-1 entities.

For each country block and each target source (2, 3) this builds a
``RareTermIndex`` over the **full** target pool of that country and retrieves the
top-k neighbours of every query entity.  Output is one parquet per country:

    data/processed/cands_{tag}_{country}_s{source}.parquet
        s1         encoded S1 id
        cand       encoded S2/S3 id
        src        2 or 3
        blk_score  retrieval cosine
        blk_rank   0-based rank within (s1, src)

Countries are discovered from the processed files of the split, so France is
handled for test without any special casing.

Usage:
    python -m src.candidates --split test --tag test --topk 40
    python -m src.candidates --split train --tag trn --topk 40 --sample 150000
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .blocking import BlockingConfig, RareTermIndex, build_blobs, search_topk

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def countries_for(split: str) -> list[str]:
    names = sorted(p.stem.split("_s1_", 1)[1] for p in PROCESSED.glob(f"{split}_s1_*.parquet"))
    if not names:
        raise FileNotFoundError(f"no processed S1 files for split={split}; run src.prep first")
    return names


def sample_entities(split: str, n: int, seed: int) -> np.ndarray:
    """Uniform random sample of S1 codes across all countries of the split."""
    codes = np.concatenate([
        pd.read_parquet(PROCESSED / f"{split}_s1_{c}.parquet", columns=["code"])["code"].to_numpy()
        for c in countries_for(split)
    ])
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(codes, size=min(n, codes.size), replace=False))


def cand_path(tag: str, country: str, source: int) -> Path:
    return PROCESSED / f"cands_{tag}_{country}_s{source}.parquet"


def read_cands(tag: str, country: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Candidates of both target sources for one country, as one frame."""
    return pd.concat([pd.read_parquet(cand_path(tag, country, s), columns=columns) for s in (2, 3)],
                     ignore_index=True)


def rev_path(tag: str, country: str, source: int) -> Path:
    return PROCESSED / f"cands_{tag}_{country}_s{source}_rev.parquet"


def run_reverse(split: str, country: str, source: int, cfg: BlockingConfig, out_path: Path,
                query_batch: int = 250_000) -> tuple[int, int]:
    """Reverse channel: every target record of (country, source) retrieves its
    top-k S1 entities.  Every target has at most one owner, so for generic names
    the target side is where a match is findable when the S1 entity has more than
    30 lookalike targets (E017b: India S2 recall .9464 -> .9583 at k = 3).

    Output columns: s1, cand, src, rev_score, rev_rank.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    s1 = pd.read_parquet(PROCESSED / f"{split}_s1_{country}.parquet", columns=["code", "name_norm", "addr_norm"])
    s1codes = s1["code"].to_numpy()
    index = RareTermIndex(s1["name_norm"].to_numpy(), s1["addr_norm"].to_numpy(), cfg)
    del s1
    gc.collect()
    t = pd.read_parquet(PROCESSED / f"{split}_s{source}_{country}.parquet", columns=["code", "name_norm", "addr_norm"])
    tcodes = t["code"].to_numpy()
    tn, ta = t["name_norm"].to_numpy(), t["addr_norm"].to_numpy()
    del t
    writer, n_pairs = None, 0
    tmp = out_path.with_suffix(".partial")
    try:
        for b0 in range(0, tcodes.size, query_batch):
            b1 = min(b0 + query_batch, tcodes.size)
            idx, sc = search_topk(index, build_blobs(tn[b0:b1], ta[b0:b1], cfg),
                                  label=f"rev {split}-{country}-S{source} [{b0}:{b1}]")
            valid = idx >= 0
            rows = np.broadcast_to(np.arange(b0, b1)[:, None], idx.shape)
            part = pd.DataFrame({
                "s1": s1codes[idx[valid]],
                "cand": tcodes[rows[valid]],
                "src": np.full(int(valid.sum()), source, dtype=np.int8),
                "rev_score": sc[valid],
                "rev_rank": np.broadcast_to(np.arange(idx.shape[1], dtype=np.int16), idx.shape)[valid],
            })
            table = pa.Table.from_pandas(part, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table)
            n_pairs += len(part)
            del idx, sc, part, table
            gc.collect()
    finally:
        if writer is not None:
            writer.close()
    tmp.replace(out_path)
    return n_pairs, int(tcodes.size)


def run_source(split: str, country: str, source: int, cfg: BlockingConfig,
               keep: np.ndarray | None, out_path: Path, query_batch: int = 200_000) -> tuple[int, int]:
    """Block one (country, source) and stream the pairs to ``out_path``.

    Query blobs are built per batch and results are appended batch-by-batch
    through a ParquetWriter, so memory does not grow with the number of queries.
    Returns (n_pairs, n_queries).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    q = pd.read_parquet(PROCESSED / f"{split}_s1_{country}.parquet",
                        columns=["code", "name_norm", "addr_norm"])
    if keep is not None:
        q = q[np.isin(q["code"].to_numpy(), keep)]
    qcodes = q["code"].to_numpy()
    qn, qa = q["name_norm"].to_numpy(), q["addr_norm"].to_numpy()
    del q

    t = pd.read_parquet(PROCESSED / f"{split}_s{source}_{country}.parquet",
                        columns=["code", "name_norm", "addr_norm"])
    tcodes = t["code"].to_numpy()
    index = RareTermIndex(t["name_norm"].to_numpy(), t["addr_norm"].to_numpy(), cfg)
    del t
    gc.collect()

    writer = None
    n_pairs = 0
    tmp = out_path.with_suffix(".partial")
    try:
        for b0 in range(0, qcodes.size, query_batch):
            b1 = min(b0 + query_batch, qcodes.size)
            blobs = build_blobs(qn[b0:b1], qa[b0:b1], cfg)
            idx, sc = search_topk(index, blobs, label=f"{split}-{country}-S{source} [{b0}:{b1}]")
            del blobs
            k = idx.shape[1]
            valid = idx >= 0
            rows = np.broadcast_to(np.arange(b0, b1)[:, None], idx.shape)
            ranks = np.broadcast_to(np.arange(k, dtype=np.int16), idx.shape)
            part = pd.DataFrame({
                "s1": qcodes[rows[valid]],
                "cand": tcodes[idx[valid]],
                "src": np.full(int(valid.sum()), source, dtype=np.int8),
                "blk_score": sc[valid],
                "blk_rank": ranks[valid],
            })
            table = pa.Table.from_pandas(part, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table)
            n_pairs += len(part)
            del idx, sc, part, table
            gc.collect()
    finally:
        if writer is not None:
            writer.close()
    tmp.replace(out_path)            # only complete files get the final name
    return n_pairs, int(qcodes.size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--max-df", type=int, default=2_000)
    ap.add_argument("--sample", type=int, default=0, help="sample N S1 entities (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--countries", nargs="*", default=None)
    ap.add_argument("--sources", nargs="*", type=int, default=[2, 3])
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--reverse", action="store_true", help="run the reverse channel (target -> top-k S1)")
    args = ap.parse_args()

    cfg = BlockingConfig(topk=args.topk, max_df_abs=args.max_df)
    keep = sample_entities(args.split, args.sample, args.seed) if args.sample else None
    if keep is not None:
        np.save(PROCESSED / f"entities_{args.tag}.npy", keep)

    for country in args.countries or countries_for(args.split):
        for source in args.sources:
            path = rev_path(args.tag, country, source) if args.reverse else cand_path(args.tag, country, source)
            if args.skip_existing and path.exists():
                print(f"== {path.name} exists, skipping", flush=True)
                continue
            t0 = time.time()
            if args.reverse:
                n_pairs, n_q = run_reverse(args.split, country, source, cfg, path)
            else:
                n_pairs, n_q = run_source(args.split, country, source, cfg, keep, path)
            print(f"== {country} S{source}: {n_pairs:,} pairs for {n_q:,} entities "
                  f"({n_pairs / max(n_q, 1):.1f}/entity) in {time.time() - t0:.0f}s -> {path.name}", flush=True)
            gc.collect()


if __name__ == "__main__":
    main()
