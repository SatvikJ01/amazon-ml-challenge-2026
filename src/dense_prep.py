"""E047: inputs for the dense-retrieval rescue channel (Kaggle kernel ``kaggle/e047_dense``).

Our lexical retrieval misses 1.8 % of the true matches (11,227 holdout pairs, F0.5 loss 0.0059).  E047
fine-tunes a multilingual bi-encoder on the ground-truth pairs of the non-holdout training entities, retrieves
kNN candidates in both directions within each country, keeps the pairs NOT already in our candidate set and
scores them with the E046 cross-encoder; a small rescue model (``src/dense_rescue.py``) then decides.

Writes experiments/E047/up/:
    rec_{train,test}_s{1,2,3}.parquet   code, cty (0 US / 1 India / 2 France), text "name | address"
                                        (built separately, see the E047 log)
    translit.parquet                    code, tr: dictionary transliteration of native-script names
    gt_train.parquet                    s1, m: ground-truth pairs of the training S1 that are NOT in the E039 holdout
    holdout_s1.npy                      the E039 holdout S1 codes (queries of the holdout evaluation)
    bloom.npy, bloom.json               Bloom filter of every existing (s1, cand) pair: holdout candidate table
                                        (all 4.23M pairs) + every test candidate pair (scores_frdown, 43M)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .ce_data import _native
from .l2_train import holdout_entities

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / "experiments" / "E047" / "up"
BLOOM_BITS = 1 << 29          # 64 MB; ~11 bits per key for 47M keys
BLOOM_K = 7
_T0 = time.time()


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), "%6.0fs" % (time.time() - _T0), *a, flush=True)


def pair_hash(s1: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """64-bit mix of a (s1, cand) pair (splitmix64 finaliser); identical code runs in the kernel."""
    with np.errstate(over="ignore"):
        x = (s1.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) ^ cand.astype(np.uint64)
        x ^= x >> np.uint64(33)
        x *= np.uint64(0xFF51AFD7ED558CCD)
        x ^= x >> np.uint64(33)
        x *= np.uint64(0xC4CEB9FE1A85EC53)
        x ^= x >> np.uint64(33)
    return x


def bloom_positions(h: np.ndarray, i: int) -> np.ndarray:
    with np.errstate(over="ignore"):
        h1 = h & np.uint64(BLOOM_BITS - 1)
        h2 = (h >> np.uint64(32)) | np.uint64(1)
        return ((h1 + np.uint64(i) * h2) & np.uint64(BLOOM_BITS - 1)).astype(np.int64)


def main() -> None:
    UP.mkdir(parents=True, exist_ok=True)
    # holdout entities and training ground truth without them
    ho = holdout_entities()
    np.save(UP / "holdout_s1.npy", ho)
    s1 = np.load(ROOT / "data/interim/gt_pair_s1.npy")
    m = np.load(ROOT / "data/interim/gt_pair_m.npy")
    keep = ~np.isin(s1, ho)
    pd.DataFrame({"s1": s1[keep], "m": m[keep]}).to_parquet(UP / "gt_train.parquet", index=False)
    log(f"gt_train: {int(keep.sum()):,} pairs (holdout pairs removed: {int((~keep).sum()):,}); holdout S1 {len(ho):,}")
    # transliterations of native-script names (same rule as the E046 cross-encoder texts)
    parts = []
    for split in ("train", "test"):
        for s in (1, 2, 3):
            for c in ("India", "US", "France"):
                f = ROOT / "data/processed" / f"{split}T_s{s}_{c}.parquet"
                if not f.exists():
                    continue
                t = pq.read_table(f, columns=["code", "name_norm", "nonascii"])
                t = t.filter(pc.equal(t.column("nonascii"), 1)).select(["code", "name_norm"]).to_pandas()
                if len(t):
                    raw = pq.read_table(ROOT / "data/interim" / f"{split}_s{s}.parquet", columns=["code", "business_name"]).to_pandas()
                    raw = raw[raw.code.isin(t.code)].set_index("code").business_name.fillna("")
                    nat = raw.map(_native)
                    t = t[t.code.map(nat).fillna(False).to_numpy()]
                    parts.append(t.rename(columns={"name_norm": "tr"}))
    T = pd.concat(parts, ignore_index=True).drop_duplicates("code")
    T.to_parquet(UP / "translit.parquet", index=False)
    log(f"translit: {len(T):,} native-script names")
    # Bloom filter of the existing candidate pairs
    flags = np.zeros(BLOOM_BITS, bool)
    n = 0

    def add(a: np.ndarray, b: np.ndarray) -> None:
        h = pair_hash(a, b)
        for i in range(BLOOM_K):
            flags[bloom_positions(h, i)] = True

    D = pd.read_parquet(ROOT / "experiments/E046/holdout_pairs.parquet", columns=["s1", "cand"])
    add(D.s1.to_numpy(), D.cand.to_numpy())
    n += len(D)
    for f in sorted((ROOT / "experiments/E039_test/scores_frdown").glob("*_p*.parquet")):
        t = pq.read_table(f, columns=["s1", "cand"])
        add(t.column("s1").to_numpy(), t.column("cand").to_numpy())
        n += t.num_rows
    np.save(UP / "bloom.npy", np.packbits(flags, bitorder="little"))
    fill = float(flags.mean())
    json.dump({"bits": BLOOM_BITS, "k": BLOOM_K, "keys": n, "fill": fill, "fpr": fill ** BLOOM_K},
              open(UP / "bloom.json", "w"), indent=1)
    log(f"bloom: {n:,} keys, fill {fill:.3f}, false-positive rate {fill ** BLOOM_K:.4f}")


if __name__ == "__main__":
    main()
