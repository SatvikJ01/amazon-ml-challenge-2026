"""Counterfactual loss budget of an experiment's holdout predictions.

For each error category, the F0.5 gained if exactly those errors were fixed
(missed pairs added, or false positives removed), everything else unchanged.
Categories: blocking misses (not in the final candidate set) vs matcher misses,
split by native-script name / empty address / ambiguous name; false positives.

Usage: python -m src.loss_budget --exp E030_stage2 --entities data/processed/entities_v3c.npy
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .decision import select_sets
from .inference import enforce_exclusivity
from .metrics import f05_single
from .train import entity_fold, truth_for

ROOT = Path(__file__).resolve().parents[1]
I = ROOT / "data" / "interim"
INDIC = re.compile(r"[ऀ-෿]")


def filt(path, col, vals, cols):
    t = pq.read_table(path, columns=cols)
    return t.filter(pc.is_in(t.column(col), value_set=pa.array(vals))).to_pandas()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--entities", required=True)
    args = ap.parse_args()
    ev = pd.read_parquet(ROOT / "experiments" / args.exp / "holdout_pred.parquet")
    ents = np.load(ROOT / args.entities); ents = ents[entity_fold(ents, 5) == 0]
    truth = truth_for(ents)
    s1, c, p = ev["s1"].to_numpy(), ev["cand"].to_numpy(), ev["prob"].to_numpy()
    k = enforce_exclusivity(s1, c, p)
    sets, _, _ = select_sets(s1[k], c[k], p[k])
    pred = {int(e): set(np.asarray(sets.get(int(e), [])).tolist()) for e in ents}
    base = np.mean([f05_single(pred[e], truth[e]) for e in ents])
    ps1, pm = np.load(I / "gt_pair_s1.npy"), np.load(I / "gt_pair_m.npy")
    m = np.isin(ps1, ents); tp = pd.DataFrame({"s1": ps1[m], "cand": pm[m]})
    tp = tp[[cc not in pred[s] for s, cc in zip(tp.s1.tolist(), tp.cand.tolist())]]
    raw = pd.concat([filt(I / f"train_s{s}.parquet", "code", tp.cand.unique(), ["code", "business_name", "business_address"])
                     for s in (2, 3)]).set_index("code").loc[tp.cand]
    s1raw = filt(I / "train_s1.parquet", "code", tp.s1.unique(), ["code", "business_name", "country"]).set_index("code")
    name_cnt = pd.read_parquet(I / "train_s1.parquet", columns=["business_name"])["business_name"].str.lower().value_counts()
    # Explicit bool arrays: pandas' nullable-boolean -> object arrays make `~True == -2`.
    tp["indic"] = np.asarray(raw.business_name.str.contains(INDIC).fillna(False), dtype=bool)
    tp["empty_addr"] = np.asarray((raw.business_address.fillna("").str.strip() == "").fillna(False), dtype=bool)
    tp["amb"] = name_cnt.reindex(s1raw.loc[tp.s1].business_name.str.lower()).fillna(1).to_numpy() >= 2
    tp["country"] = s1raw.loc[tp.s1].country.to_numpy()
    cs = set(zip(ev.s1.tolist(), ev.cand.tolist()))
    tp["blk"] = np.array([x not in cs for x in zip(tp.s1.tolist(), tp.cand.tolist())], dtype=bool)
    tp["amb"] = tp["amb"].astype(bool)

    def gain(mask):
        p2 = {e: set(v) for e, v in pred.items()}
        for s, cc in zip(tp.s1[mask].tolist(), tp.cand[mask].tolist()):
            p2[s].add(cc)
        return np.mean([f05_single(p2[e], truth[e]) for e in ents]) - base

    print(f"{args.exp}: holdout F0.5 {base:.4f} (loss {1 - base:.4f}), {len(ents):,} entities, {len(tp):,} missed pairs")
    cats = {"blocking miss, native script": tp.blk & tp.indic,
            "blocking miss, empty address": tp.blk & ~tp.indic & tp.empty_addr,
            "blocking miss, other": tp.blk & ~tp.indic & ~tp.empty_addr,
            "matcher miss, empty addr + shared name": ~tp.blk & tp.empty_addr & tp.amb,
            "matcher miss, empty addr + unique name": ~tp.blk & tp.empty_addr & ~tp.amb,
            "matcher miss, native script": ~tp.blk & ~tp.empty_addr & tp.indic,
            "matcher miss, other": ~tp.blk & ~tp.empty_addr & ~tp.indic}
    for name, msk in cats.items():
        print(f"  {name:42s} pairs {int(msk.sum()):>6}  gain if fixed {gain(msk.to_numpy()):+.4f}   "
              f"(US {int((msk & (tp.country == 'US')).sum())}, India {int((msk & (tp.country == 'India')).sum())})")
    fp_fix = {e: pred[e] & truth[e] for e in ents}
    print(f"  {'false positives removed':42s}               gain if fixed "
          f"{np.mean([f05_single(fp_fix[e], truth[e]) for e in ents]) - base:+.4f}")


if __name__ == "__main__":
    main()
