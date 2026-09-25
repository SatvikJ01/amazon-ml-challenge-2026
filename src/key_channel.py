"""Exact-key blocking channel: no top-k limit, so no saturation by lookalikes.

Loss analysis of the E023B holdout: 0.018 of the 0.033 F0.5 loss is pairs never
retrieved, and most of them are *easy* (median name similarity 85, address 82).
They are lost because generic names and cities fill each S1 entity's top-30 list
with lookalikes.  A key join has no depth limit: every (S1, target) pair that
shares a key becomes a candidate, as long as the key is not too common.

Keys per record (normalised text, country-agnostic):
    T|<skel>|<num>     consonant skeleton of a name token  x  one of the first two
                       canonical (leading-zero-stripped) address numbers
    N|<skel1>|<skel2>  adjacent name-token skeleton bigram
Keys are hashed to int64.  A key is used only if it occurs in at most
``max_s1`` S1 records and ``max_t`` target records of the block (frequency caps
bound the candidate volume).  Output per (country, source):
    s1, cand, key_hits (number of shared usable keys), key_minfreq

Usage:
    python -m src.key_channel --split train --tag trnall --country India --source 2
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .normalize import NULL_TOKENS, skeleton_token

import zlib
from array import array


def _h64(k: str) -> int:
    """Deterministic 64-bit key hash (Python's hash() is salted per process)."""
    b = k.encode()
    return ((zlib.crc32(b) << 32) | zlib.crc32(b[::-1])) - (1 << 63)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def key_path(tag: str, country: str, source: int) -> Path:
    return PROCESSED / f"cands_{tag}_{country}_s{source}_key.parquet"


def record_keys(name_norm: np.ndarray, addr_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(row index, int64 key hash) for every key of every record."""
    rows, keys = array("q"), array("q")
    for i, (name, addr) in enumerate(zip(name_norm, addr_norm)):
        nums = [t.lstrip("0") or "0" for t in addr.split() if t.isdigit()][:2]
        toks = [s for s in (skeleton_token(t) for t in name.split() if not t.isdigit() and t not in NULL_TOKENS)
                if len(s) >= 2]
        ks = {f"T|{t}|{n}" for t in toks for n in nums}
        ks.update(f"N|{a}|{b}" for a, b in zip(toks, toks[1:]))
        for k in ks:
            rows.append(i)
            keys.append(_h64(k))
    return np.frombuffer(rows, np.int64).copy(), np.frombuffer(keys, np.int64).copy()


def join(q_rows, q_keys, t_rows, t_keys, max_s1: int, max_t: int):
    """All (q_row, t_row) sharing a key whose frequencies are within the caps."""
    qu, qc = np.unique(q_keys, return_counts=True)
    tu, tc = np.unique(t_keys, return_counts=True)
    common, qi, ti = np.intersect1d(qu, tu, assume_unique=True, return_indices=True)
    ok = (qc[qi] <= max_s1) & (tc[ti] <= max_t)
    usable = common[ok]
    freq = pd.Series(qc[qi][ok] * tc[ti][ok], index=usable)
    qm = np.isin(q_keys, usable); tm = np.isin(t_keys, usable)
    qk, qr = q_keys[qm], q_rows[qm]
    tk, tr = t_keys[tm], t_rows[tm]
    qo, to = np.argsort(qk, kind="stable"), np.argsort(tk, kind="stable")
    qk, qr, tk, tr = qk[qo], qr[qo], tk[to], tr[to]
    # group boundaries per key on both sides (same sorted key order)
    qs = np.searchsorted(qk, usable); qe = np.searchsorted(qk, usable, side="right")
    ts = np.searchsorted(tk, usable); te = np.searchsorted(tk, usable, side="right")
    na, nb = qe - qs, te - ts
    tot = int((na * nb).sum())
    kid = np.repeat(np.arange(usable.size), na * nb)
    within = np.arange(tot) - np.repeat(np.cumsum(na * nb) - na * nb, na * nb)
    a_idx = qs[kid] + within // nb[kid]
    b_idx = ts[kid] + within % nb[kid]
    return qr[a_idx], tr[b_idx], freq.to_numpy()[kid]


def run(split: str, tag: str, country: str, source: int, max_s1: int, max_t: int) -> pd.DataFrame:
    t0 = time.time()
    q = pd.read_parquet(PROCESSED / f"{split}_s1_{country}.parquet", columns=["code", "name_norm", "addr_norm"])
    qr, qk = record_keys(q["name_norm"].to_numpy(), q["addr_norm"].to_numpy())
    qcodes = q["code"].to_numpy(); del q
    t = pd.read_parquet(PROCESSED / f"{split}_s{source}_{country}.parquet", columns=["code", "name_norm", "addr_norm"])
    tr, tk = record_keys(t["name_norm"].to_numpy(), t["addr_norm"].to_numpy())
    tcodes = t["code"].to_numpy(); del t
    gc.collect()
    a, b, f = join(qr, qk, tr, tk, max_s1, max_t)
    df = pd.DataFrame({"s1": qcodes[a], "cand": tcodes[b], "f": f})
    out = df.groupby(["s1", "cand"], as_index=False).agg(key_hits=("f", "size"), key_minfreq=("f", "min"))
    out["src"] = np.int8(source)
    print(f"  keys {split}-{country}-S{source}: {len(out):,} pairs for {out['s1'].nunique():,} S1 "
          f"({len(out) / qcodes.size:.1f}/S1) in {time.time() - t0:.0f}s", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--tag", default="trnall")
    ap.add_argument("--country", required=True)
    ap.add_argument("--source", type=int, required=True)
    ap.add_argument("--max-s1", type=int, default=5)
    ap.add_argument("--max-t", type=int, default=30)
    args = ap.parse_args()
    out = run(args.split, args.tag, args.country, args.source, args.max_s1, args.max_t)
    out.to_parquet(key_path(args.tag, args.country, args.source), index=False, compression="zstd")


if __name__ == "__main__":
    main()
