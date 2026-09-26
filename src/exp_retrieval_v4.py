"""Retrieval v4 pilot: do single-field channels recover the remaining retrieval misses?

After E035, 46 % of the holdout loss is retrieval (4,447 true pairs never become
candidates, even with anchor retrieval): empty-address records, alias / domain
names at the right address, native-script names.  The forward channel scores the
name+address blob jointly, so a record that matches on only one field is outranked
by records that half-match on both.  Two single-field channels:

* ``name``  S1 name terms (words, bigrams, skeleton keys) vs target names only;
* ``addr``  S1 address terms (numbers, words, bigrams, skeletons) vs target addresses only.

Measured on the 60k-entity holdout of E035: recovered misses per channel and depth,
by miss type, and the number of *new* candidates each channel adds per entity.

Usage:
    python -m src.exp_retrieval_v4 search --country India --source 2     (one block per process)
    python -m src.exp_retrieval_v4 report
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .blocking import BlockingConfig, RareTermIndex, build_blobs, search_topk
from .train import entity_fold, truth_for

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
INTERIM = ROOT / "data" / "interim"
OUT = ROOT / "experiments" / "E037_retrieval_v4"
K = 20
INDIC = re.compile(r"[ऀ-෿]")


def holdout_entities() -> np.ndarray:
    e = np.load(PROCESSED / "entities_v3c.npy")
    return e[entity_fold(e, 5) == 0]


def search(country: str, source: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ents = holdout_entities()
    q = pd.read_parquet(PROCESSED / f"train_s1_{country}.parquet", columns=["code", "name_norm", "addr_norm"])
    q = q[np.isin(q["code"].to_numpy(), ents)]
    t = pd.read_parquet(PROCESSED / f"train_s{source}_{country}.parquet", columns=["code", "name_norm", "addr_norm"])
    tcodes = t["code"].to_numpy()
    cfg = BlockingConfig(topk=K)
    empty = np.full(len(t), "", dtype=object)
    qempty = np.full(len(q), "", dtype=object)
    parts = []
    for ch, (tn, ta, qn, qa) in {"name": (t["name_norm"].to_numpy(), empty, q["name_norm"].to_numpy(), qempty),
                                 "addr": (empty, t["addr_norm"].to_numpy(), qempty, q["addr_norm"].to_numpy())}.items():
        index = RareTermIndex(tn, ta, cfg)
        idx, sc = search_topk(index, build_blobs(qn, qa, cfg), label=f"{country}-S{source}-{ch}")
        del index
        valid = idx >= 0
        rows = np.broadcast_to(np.arange(len(q))[:, None], idx.shape)
        parts.append(pd.DataFrame({"s1": q["code"].to_numpy()[rows[valid]], "cand": tcodes[idx[valid]],
                                   "channel": ch, "rank": np.broadcast_to(np.arange(K), idx.shape)[valid].astype(np.int16),
                                   "score": sc[valid]}))
    pd.concat(parts, ignore_index=True).to_parquet(OUT / f"hits_{country}_s{source}.parquet", index=False)
    print(f"{country} S{source}: {sum(len(p) for p in parts):,} hits for {len(q):,} entities ({time.time() - t0:.0f}s)", flush=True)


def report() -> None:
    ents = holdout_entities()
    truth = truth_for(ents)
    ev = pd.read_parquet(ROOT / "experiments" / "E035_stage3" / "holdout_pred.parquet", columns=["s1", "cand"])
    have = set(zip(ev.s1.tolist(), ev.cand.tolist()))
    miss = pd.DataFrame([(e, x) for e in ents for x in truth[e] if (e, x) not in have], columns=["s1", "cand"])
    hits = pd.concat([pd.read_parquet(p) for p in sorted(OUT.glob("hits_*.parquet"))], ignore_index=True)
    raw = pd.concat([pd.read_parquet(INTERIM / f"train_s{s}.parquet", columns=["code", "business_name", "business_address"],
                                     filters=[("code", "in", miss.cand.unique().tolist())]) for s in (2, 3)]).set_index("code")
    r = raw.reindex(miss.cand)
    miss["type"] = np.where(r.business_name.fillna("").str.contains(INDIC).to_numpy(), "native",
                            np.where((r.business_address.fillna("").str.strip() == "").to_numpy(), "empty_addr", "other"))
    print(f"retrieval misses: {len(miss):,}  ({miss.type.value_counts().to_dict()})")
    n_ent = len(ents)
    for depth in (5, 10, 20):
        h = hits[hits["rank"] < depth]
        for ch in ("name", "addr", "both"):
            hh = h if ch == "both" else h[h.channel == ch]
            keys = set(zip(hh.s1.tolist(), hh.cand.tolist()))
            rec = np.array([(a, b) in keys for a, b in zip(miss.s1.tolist(), miss.cand.tolist())])
            new = len(keys - have)
            by = miss.assign(r=rec).groupby("type")["r"].mean().round(3).to_dict()
            print(f"  depth {depth:2d} {ch:4s}: recovers {rec.sum():5d} / {len(miss)} ({rec.mean():.1%})  by type {by}  "
                  f"new candidates/entity {new / n_ent:.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["search", "report"])
    ap.add_argument("--country")
    ap.add_argument("--source", type=int)
    a = ap.parse_args()
    search(a.country, a.source) if a.step == "search" else report()


if __name__ == "__main__":
    main()
