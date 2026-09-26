"""Dense retrieval channel (v4, experimental): multilingual-e5-small embeddings.

``intfloat/multilingual-e5-small`` (MIT licence, 118 M parameters) encodes
"<name>, <address>" for S1 queries and target records (e5 convention: the
"query: " prefix on both sides for symmetric matching).  Embeddings are L2-
normalised; an IVF inner-product index (FAISS) per (country, source) pool returns
the top-k neighbours, written in the same format as ``src/channels.py``:
``cands_{tag}_{country}_s{source}_embed.parquet`` (s1, cand, score, rank).

Cost: encoding dominates (~12 M records for train+test pools).  On a GPU (T4 /
A10G, fp16) this is ~20-40 min; on CPU it is several hours, so run it on a GPU
worker.  Encodings are cached per (split, country, source) under
``data/processed/emb_*.npy`` (float16) and reused for train and test queries.

Usage: python -m src.embed_channel --split train --tag trnall --country India --source 2 --topk 10
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODEL = "intfloat/multilingual-e5-small"


def _texts(path: Path) -> tuple[np.ndarray, list[str]]:
    # raw-ish text keeps native script and accents for the multilingual model
    d = pd.read_parquet(path, columns=["code", "name_norm", "addr_norm"])
    return d["code"].to_numpy(), [f"query: {n}, {a}" for n, a in zip(d["name_norm"], d["addr_norm"])]


def encode(path: Path, model, batch: int = 512) -> tuple[np.ndarray, np.ndarray]:
    cache = PROCESSED / f"emb_{path.stem}.npy"
    codes, texts = _texts(path)
    if cache.exists():
        return codes, np.load(cache)
    t0 = time.time()
    e = model.encode(texts, batch_size=batch, normalize_embeddings=True, convert_to_numpy=True,
                     show_progress_bar=False).astype(np.float16)
    np.save(cache, e)
    print(f"  encoded {path.name}: {len(texts):,} in {time.time() - t0:.0f}s", flush=True)
    return codes, e


def main() -> None:
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--country", required=True)
    ap.add_argument("--source", type=int, required=True)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--nprobe", type=int, default=32)
    a = ap.parse_args()
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL, device=dev)
    if dev == "cuda":
        model.half()
    model.max_seq_length = 64
    qc, qe = encode(PROCESSED / f"{a.split}_s1_{a.country}.parquet", model)
    tc, te = encode(PROCESSED / f"{a.split}_s{a.source}_{a.country}.parquet", model)
    d = te.shape[1]
    nlist = int(min(16384, max(256, 4 * np.sqrt(len(te)))))
    index = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
    rng = np.random.default_rng(0)
    index.train(te[rng.choice(len(te), min(len(te), 50 * nlist), replace=False)].astype(np.float32))
    for i in range(0, len(te), 1_000_000):
        index.add(te[i:i + 1_000_000].astype(np.float32))
    index.nprobe = a.nprobe
    out = []
    for i in range(0, len(qe), 200_000):
        sc, ix = index.search(qe[i:i + 200_000].astype(np.float32), a.topk)
        valid = ix >= 0
        rows = np.broadcast_to(np.arange(i, i + len(ix))[:, None], ix.shape)
        out.append(pd.DataFrame({"s1": qc[rows[valid]], "cand": tc[ix[valid]], "score": sc[valid].astype(np.float32),
                                 "rank": np.broadcast_to(np.arange(a.topk, dtype=np.int16), ix.shape)[valid]}))
    from .channels import channel_path
    df = pd.concat(out, ignore_index=True)
    df.to_parquet(channel_path(a.tag, a.country, a.source, "embed"), index=False)
    print(f"{a.split} {a.country} S{a.source} embed: {len(df):,} pairs ({dev}, {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
