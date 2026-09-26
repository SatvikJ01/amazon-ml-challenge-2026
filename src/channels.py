"""Extra candidate-generation channels (v4), production version of the E037 pilot.

Each channel writes ``cands_{tag}_{country}_s{source}_{channel}.parquet`` with
``s1, cand, score (float32), rank (int16)`` for every S1 record of the split, and is
unioned with forward / reverse / key candidates in ``v3.union_source`` when listed
in the ``V4_CHANNELS`` environment variable (e.g. ``V4_CHANNELS=name,caddr``).

Channels (all sparse, CPU):
  name   rare-term IDF cosine, S1 name vs target names only
  addr   rare-term IDF cosine, S1 address vs target addresses only (numbers and
         number+street bigrams are terms -> structured address retrieval)
  cname  character 3-gram TF-IDF on names   (typo / transliteration robust)
  caddr  character 3-gram TF-IDF on addresses
  bm25   BM25 over the forward channel's name+address terms
``embed`` (multilingual-e5-small, dense) lives in ``src/embed_channel.py`` and writes
the same format.

Memory: one target index per process; queries in batches of 200k, results streamed.

Usage: python -m src.channels --split train --tag trnall --country India --source 2 --channel name --topk 10
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .blocking import BlockingConfig, RareTermIndex, _hasher, build_blobs, search_topk
from .normalize import NULL_TOKENS

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
CHANNELS = ("name", "addr", "cname", "caddr", "bm25", "embed")


def channel_path(tag: str, country: str, source: int, channel: str) -> Path:
    return PROCESSED / f"cands_{tag}_{country}_s{source}_{channel}.parquet"


def char_blobs(texts: np.ndarray) -> list[str]:
    out = []
    for x in texts:
        g = set()
        for t in x.split():
            if len(t) < 2 or t in NULL_TOKENS:
                continue
            w = f"#{t}#"
            g.update(w[i:i + 3] for i in range(len(w) - 2))
        out.append(" ".join("c:" + x for x in g))
    return out


class ChannelIndex(RareTermIndex):
    """RareTermIndex over arbitrary term blobs, cosine or BM25 weighting."""

    def __init__(self, blob_fn, texts: np.ndarray, cfg: BlockingConfig, mode: str = "cosine", chunk: int = 500_000):
        import scipy.sparse as sp
        t0 = time.time()
        self.cfg, self.hasher, self.mode, self.blob_fn = cfg, _hasher(cfg), mode, blob_fn
        parts = [self.hasher.transform(blob_fn(texts[i:i + chunk])).tocsr() for i in range(0, len(texts), chunk)]
        X = sp.vstack(parts, format="csr") if len(parts) > 1 else parts[0]
        del parts
        self.n_targets = X.shape[0]
        df = np.bincount(X.indices, minlength=cfg.n_features).astype(np.int64)
        self.df = df
        keep = (df > 0) & (df <= cfg.max_df_abs)
        self.idf = np.zeros(cfg.n_features, np.float32)
        self.idf[keep] = (np.log((self.n_targets - df[keep] + 0.5) / (df[keep] + 0.5) + 1.0) if mode == "bm25"
                          else np.log((self.n_targets + 1.0) / (df[keep] + 1.0)) + 1.0)
        if mode == "bm25":
            X.data = self.idf[X.indices]
            X.eliminate_zeros()
            ln = np.diff(X.indptr).astype(np.float32)
            k1, b = 1.2, 0.75
            X.data *= np.repeat((k1 + 1) / (1 + k1 * (1 - b + b * ln / max(ln.mean(), 1.0))),
                                np.diff(X.indptr)).astype(np.float32)
        else:
            self._weight_inplace(X)
        self.BT = X.T.tocsr()
        del X
        gc.collect()
        self.build_sec = time.time() - t0

    def query_matrix(self, query_blobs):
        A = self.hasher.transform(query_blobs).tocsr()
        if self.mode == "bm25":
            A.data = (self.idf[A.indices] > 0).astype(np.float32)
            A.eliminate_zeros()
            return A
        self._weight_inplace(A)
        return A


def make_index(channel: str, tn: np.ndarray, ta: np.ndarray, topk: int):
    """(index, query-blob function(qn, qa)) for one sparse channel."""
    cfg = BlockingConfig(topk=topk)
    e = np.full(len(tn), "", dtype=object)
    if channel == "name":
        return RareTermIndex(tn, e, cfg), lambda qn, qa: build_blobs(qn, np.full(len(qn), "", object), cfg)
    if channel == "addr":
        return RareTermIndex(e, ta, cfg), lambda qn, qa: build_blobs(np.full(len(qa), "", object), qa, cfg)
    if channel in ("cname", "caddr"):
        cfg = BlockingConfig(topk=topk, max_df_abs=3000)
        idx = ChannelIndex(char_blobs, tn if channel == "cname" else ta, cfg)
        return idx, (lambda qn, qa: char_blobs(qn)) if channel == "cname" else (lambda qn, qa: char_blobs(qa))
    if channel == "bm25":
        idx = ChannelIndex(lambda x: build_blobs(x[:, 0], x[:, 1], cfg), np.stack([tn, ta], axis=1), cfg, mode="bm25")
        return idx, lambda qn, qa: build_blobs(qn, qa, cfg)
    raise ValueError(channel)


def run(split: str, tag: str, country: str, source: int, channel: str, topk: int,
        keep: np.ndarray | None = None, batch: int = 200_000) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    t0 = time.time()
    q = pd.read_parquet(PROCESSED / f"{split}_s1_{country}.parquet", columns=["code", "name_norm", "addr_norm"])
    if keep is not None:
        q = q[np.isin(q["code"].to_numpy(), keep)]
    t = pd.read_parquet(PROCESSED / f"{split}_s{source}_{country}.parquet", columns=["code", "name_norm", "addr_norm"])
    tcodes = t["code"].to_numpy()
    index, qfn = make_index(channel, t["name_norm"].to_numpy(), t["addr_norm"].to_numpy(), topk)
    del t
    gc.collect()
    out = channel_path(tag, country, source, channel)
    tmp = out.with_suffix(".partial")
    w, n = None, 0
    qc, qn, qa = q["code"].to_numpy(), q["name_norm"].to_numpy(), q["addr_norm"].to_numpy()
    for b0 in range(0, len(qc), batch):
        b1 = min(b0 + batch, len(qc))
        idx, sc = search_topk(index, qfn(qn[b0:b1], qa[b0:b1]), label=f"{split}-{country}-S{source}-{channel}")
        valid = idx >= 0
        rows = np.broadcast_to(np.arange(b0, b1)[:, None], idx.shape)
        tb = pa.table({"s1": qc[rows[valid]], "cand": tcodes[idx[valid]], "score": sc[valid].astype(np.float32),
                       "rank": np.broadcast_to(np.arange(idx.shape[1], dtype=np.int16), idx.shape)[valid]})
        w = w or pq.ParquetWriter(tmp, tb.schema, compression="zstd")
        w.write_table(tb)
        n += tb.num_rows
    w.close()
    tmp.replace(out)
    print(f"{split} {country} S{source} {channel}: {n:,} pairs for {len(qc):,} S1 ({time.time() - t0:.0f}s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--country", required=True)
    ap.add_argument("--source", type=int, required=True)
    ap.add_argument("--channel", required=True, choices=CHANNELS[:-1])
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--entities", default=None, help="optional .npy of S1 codes to query")
    a = ap.parse_args()
    run(a.split, a.tag, a.country, a.source, a.channel, a.topk, np.load(a.entities) if a.entities else None)


if __name__ == "__main__":
    main()
