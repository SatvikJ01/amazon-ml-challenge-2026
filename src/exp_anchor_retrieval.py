"""E022 pilot: can confident matches (anchors) retrieve the records the S1 text misses?

For residual blocking misses (after forward ∪ reverse top-3, India S2), take the
entity's *retrieved* true siblings as queries against the full target pool and
check whether the missed record appears among each query's top-k neighbours.

Using true siblings as anchors is an upper bound; production anchors are
stage-2 matches with p >= 0.9, which are ~99 % precise, so the bound is tight.
The pilot also reports how many *extra* candidates per entity the channel adds,
the cost side of the decision.

Usage: python -m src.exp_anchor_retrieval --country India --source 2 --k 5
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .blocking import BlockingConfig, RareTermIndex, build_blobs, search_topk
from .candidates import read_cands
from .ids import OFFSET

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
INTERIM = ROOT / "data" / "interim"
REPORTS = ROOT / "reports"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="India")
    ap.add_argument("--source", type=int, default=2)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    t0 = time.time()
    cfg = BlockingConfig(topk=args.k + 1)          # +1: a query's nearest neighbour is often itself

    miss = pd.read_parquet(REPORTS / f"residual_misses_{args.country}_S{args.source}.parquet")
    ents = miss["s1"].unique()
    ps1, pm = np.load(INTERIM / "gt_pair_s1.npy"), np.load(INTERIM / "gt_pair_m.npy")
    gt = pd.DataFrame({"s1": ps1, "m": pm})
    gt = gt[gt["s1"].isin(ents)]
    fwd = read_cands("trnall", args.country, ["s1", "cand"])
    fwd = fwd[fwd["s1"].isin(ents)]
    anchors = gt.merge(fwd, left_on=["s1", "m"], right_on=["s1", "cand"])[["s1", "m"]]
    anchors = anchors[~anchors["m"].isin(miss["cand"])]

    # Index the pool the missed records live in; query with anchors from both sources.
    pool = pd.read_parquet(PROCESSED / f"train_s{args.source}_{args.country}.parquet",
                           columns=["code", "name_norm", "addr_norm"])
    pcodes = pool["code"].to_numpy()
    index = RareTermIndex(pool["name_norm"].to_numpy(), pool["addr_norm"].to_numpy(), cfg)
    del pool
    texts = pd.concat([pd.read_parquet(PROCESSED / f"train_s{s}_{args.country}.parquet",
                                       columns=["code", "name_norm", "addr_norm"]) for s in (2, 3)]).set_index("code")
    aq = texts.loc[anchors["m"].to_numpy()]
    idx, _ = search_topk(index, build_blobs(aq["name_norm"].to_numpy(), aq["addr_norm"].to_numpy(), cfg),
                         label="anchor")
    del texts

    key = lambda a, b: (a % OFFSET) * OFFSET + (b % OFFSET)
    miss_keys = key(miss["s1"].to_numpy(), miss["cand"].to_numpy())
    res = {"residual_misses": int(len(miss)), "entities": int(ents.size), "anchor_queries": int(len(anchors)),
           "by_k": {}}
    a_s1 = anchors["s1"].to_numpy()
    a_self = anchors["m"].to_numpy()
    for k in range(1, args.k + 1):
        sub = idx[:, : k + 1]
        rows = np.broadcast_to(np.arange(sub.shape[0])[:, None], sub.shape)
        ok = sub >= 0
        got_c = pcodes[sub[ok]]
        got_s1 = a_s1[rows[ok]]
        not_self = got_c != a_self[rows[ok]]
        got_c, got_s1 = got_c[not_self], got_s1[not_self]
        new = np.unique(key(got_s1, got_c))
        rec = float(np.isin(miss_keys, new).mean())
        res["by_k"][k] = {"recovered_frac_of_residual": rec, "pairs_per_entity": float(new.size / ents.size)}
        print(f"  anchor top{k}: recovers {rec:.1%} of residual misses; "
              f"{new.size / ents.size:.1f} candidate pairs per affected entity", flush=True)
    res["elapsed_sec"] = round(time.time() - t0, 1)
    (REPORTS / f"exp_anchor_{args.country}_S{args.source}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
