"""Exact blocking recall and oracle F0.5 ceiling by candidate depth, over every
S1 entity of a country block (not a sample).

The oracle keeps exactly the true matches present among the first ``d``
candidates per (entity, source); its macro F0.5 is the best score any matcher
could reach at that depth.

Usage: python -m src.blocking_ceiling --tag trnall --country India
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
INTERIM = ROOT / "data" / "interim"
REPORTS = ROOT / "reports"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="trnall")
    ap.add_argument("--country", required=True)
    args = ap.parse_args()

    from .candidates import read_cands
    t = read_cands(args.tag, args.country, ["s1", "cand", "blk_rank"])
    s1, cand, rank = t["s1"].to_numpy(), t["cand"].to_numpy(), t["blk_rank"].to_numpy()
    del t

    ps1 = np.load(INTERIM / "gt_pair_s1.npy")
    pm = np.load(INTERIM / "gt_pair_m.npy")
    order = np.argsort(pm)
    pm_s, owner_s = pm[order], ps1[order]
    pos = np.searchsorted(pm_s, cand)
    pos = np.minimum(pos, pm_s.size - 1)
    is_true = (pm_s[pos] == cand) & (owner_s[pos] == s1)
    del pos, cand

    ents = pd.read_parquet(PROCESSED / f"train_s1_{args.country}.parquet", columns=["code"])["code"].to_numpy()
    ents.sort()
    ntrue = np.zeros(ents.size, np.int32)
    in_block = np.isin(ps1, ents)
    idx = np.searchsorted(ents, ps1[in_block])
    np.add.at(ntrue, idx, 1)

    ent_idx = np.searchsorted(ents, s1)
    res = {"country": args.country, "n_entities": int(ents.size), "n_true_pairs": int(ntrue.sum()), "by_depth": {}}
    for d in (3, 5, 10, 15, 20, 25, 30):
        sel = is_true & (rank < d)
        tp = np.bincount(ent_idx[sel], minlength=ents.size)
        # oracle predicts exactly the retrieved true matches: P=1, R=tp/ntrue
        f = np.where(ntrue == 0, 1.0,
                     np.where(tp == 0, 0.0, 1.25 * tp / (0.25 * np.maximum(ntrue, 1) + np.maximum(tp, 1))))
        res["by_depth"][d] = {
            "pair_recall": float(tp.sum() / ntrue.sum()),
            "oracle_f05": float(f.mean()),
            "entities_fully_covered": float(np.mean(tp[ntrue > 0] == ntrue[ntrue > 0])),
            "cands_per_entity": float((rank < d).sum() / ents.size),
        }
        r = res["by_depth"][d]
        print(f"depth {d:>2}/src: recall {r['pair_recall']:.4f}  oracle F0.5 {r['oracle_f05']:.4f}  "
              f"full-cover {r['entities_fully_covered']:.4f}  cands/entity {r['cands_per_entity']:.1f}", flush=True)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"blocking_ceiling_{args.tag}_{args.country}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
