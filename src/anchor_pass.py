"""E023: predicted-anchor retrieval (pass 2), evaluated on the standard holdout.

Anchors are *predicted* matches only: stage-2 probabilities p2 >= ``ANCHOR_P``.
On train they come from the cross-fitted OOF p2 of E021p (every entity scored by
a model that never saw it); on test from the E013 stage-2 scores.  No ground
truth is used to choose anchors.

Steps (one country per process, bounded memory):

``retrieve``  each anchor queries the full S2 and S3 pools of its country and
              keeps its top-K non-self neighbours -> (s1, cand, anc_score,
              anc_rank, anc_hits) per entity.
``build``     stage-3 training table = existing survivors (feats_trn2c_sib) ∪
              anchor-retrieved pairs not already present.  New pairs get the
              full pairwise features, retrieval/competition features looked up
              in the forward table (NaN when the forward pass never saw the
              pair), context and sibling features recomputed on the union, and
              p2 = NaN (they never went through stage 2).  ``--no-new`` keeps
              the old pairs only (anchor features attached), isolating the
              effect of *adding* candidates.
``stats``     label-free anchor statistics per country (also for France test).

The rollback baseline (E013 / E021p files) is never modified; outputs use the
tags ``trn2c_anc`` / ``trn2c_ancA`` and ``experiments/E023_*``.
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .blocking import BlockingConfig, RareTermIndex, build_blobs, search_topk
from .build_features import label_pairs
from .candidates import cand_path
from .collective import ANCHOR_P, _texts, sibling_features
from .features import TokenStats, add_context_features, pair_features
from .ids import OFFSET
from .make_stage2 import CONTEXT_PREFIXES

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
EXPERIMENTS = ROOT / "experiments"
OUT = EXPERIMENTS / "E023_anchor"
K = 3
SIB_COLS = ["p2", "p2_rank", "n_anchors", "anchor_p_mean", "sib_name", "sib_addr", "sib_all",
            "sib_num_eq", "is_anchor"]


def _texts_full(split: str, country: str, codes: np.ndarray) -> pd.DataFrame:
    """Like collective._texts but also returns the ``nonascii`` flag used by pair_features."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    want = pa.array(np.unique(codes))
    parts = []
    for s in (2, 3):
        t = pq.read_table(PROCESSED / f"{split}_s{s}_{country}.parquet", columns=["code", "name_norm", "addr_norm", "nonascii"])
        parts.append(t.filter(pc.is_in(t.column("code"), value_set=want)).to_pandas())
    return pd.concat(parts, ignore_index=True).set_index("code")


def load_anchors(split: str, country: str) -> pd.DataFrame:
    """(s1, cand, p2) of predicted anchors."""
    if split == "train":
        o = pd.read_parquet(EXPERIMENTS / "E021p_collective_on_E013" / "oof_p2.parquet")
        o = o[o["country"] == country][["s1", "cand", "p2"]]
    else:
        d = EXPERIMENTS / "E013_cascade_stage2" / "test_scores_d30_casc"
        o = pd.concat([pd.read_parquet(p) for p in sorted(d.glob(f"{country}_p*.parquet"))], ignore_index=True)
        o = o.rename(columns={"prob": "p2"})[["s1", "cand", "p2"]]
    return o[o["p2"] >= ANCHOR_P].reset_index(drop=True)


def retrieve(split: str, country: str) -> pd.DataFrame:
    anc = load_anchors(split, country)
    tx = _texts(split, country, anc["cand"].to_numpy())
    an, aa = tx.loc[anc["cand"].to_numpy(), "name_norm"].to_numpy(), tx.loc[anc["cand"].to_numpy(), "addr_norm"].to_numpy()
    del tx
    cfg = BlockingConfig(topk=K + 1)          # +1 because an anchor's nearest neighbour is usually itself
    hits = []
    for src in (2, 3):
        pool = pd.read_parquet(PROCESSED / f"{split}_s{src}_{country}.parquet", columns=["code", "name_norm", "addr_norm"])
        pcodes = pool["code"].to_numpy()
        index = RareTermIndex(pool["name_norm"].to_numpy(), pool["addr_norm"].to_numpy(), cfg)
        del pool
        gc.collect()
        for b0 in range(0, len(anc), 250_000):
            b1 = min(b0 + 250_000, len(anc))
            idx, sc = search_topk(index, build_blobs(an[b0:b1], aa[b0:b1], cfg), label=f"anchor {split}-{country}-S{src}")
            valid = idx >= 0
            code = np.where(valid, pcodes[np.maximum(idx, 0)], -1)
            valid &= code != anc["cand"].to_numpy()[b0:b1, None]          # drop self
            keep = valid & (np.cumsum(valid, axis=1) <= K)                  # first K non-self
            rk = np.cumsum(valid, axis=1) - 1
            rows = np.broadcast_to(np.arange(b0, b1)[:, None], idx.shape)
            hits.append(pd.DataFrame({"s1": anc["s1"].to_numpy()[rows[keep]], "cand": code[keep],
                                      "anc_score": sc[keep], "anc_rank": rk[keep].astype(np.int16)}))
        del index
        gc.collect()
    h = pd.concat(hits, ignore_index=True)
    return h.groupby(["s1", "cand"], as_index=False).agg(anc_score=("anc_score", "max"),
                                                         anc_rank=("anc_rank", "min"),
                                                         anc_hits=("anc_score", "size"))


def competition_lookup(country: str, q: pd.DataFrame, tag: str = "trnall") -> pd.DataFrame:
    """blk_score / blk_rank / comp_* for arbitrary (s1, cand) pairs.  Identical to
    ``competition_features`` for pairs present in the forward table; for pairs the
    forward pass never retrieved, own score is NaN and comp_* describe how the
    candidate is claimed by other S1 entities."""
    out = []
    for src in (2, 3):
        sub = q[(q["cand"] // OFFSET) == src]
        f = pd.read_parquet(cand_path(tag, country, src), columns=["s1", "cand", "blk_score", "blk_rank"])
        s1u = np.unique(np.concatenate([f["s1"].to_numpy(), sub["s1"].to_numpy()]))
        key = lambda a, b: (np.searchsorted(s1u, a).astype(np.int64) << 34) | (b % OFFSET)
        fk = key(f["s1"].to_numpy(), f["cand"].to_numpy())
        qk = key(sub["s1"].to_numpy(), sub["cand"].to_numpy())
        o = np.argsort(fk, kind="stable")
        fks = fk[o]
        del fk
        p = np.minimum(np.searchsorted(fks, qk), fks.size - 1)
        hit = fks[p] == qk
        del fks
        own = np.where(hit, f["blk_score"].to_numpy()[o[p]], np.nan).astype(np.float32)
        rank = np.where(hit, f["blk_rank"].to_numpy()[o[p]], 99).astype(np.int16)
        # per-candidate aggregates over the full forward table
        so = np.lexsort((-f["blk_score"].to_numpy(), f["cand"].to_numpy()))
        cs = f["cand"].to_numpy()[so]
        st = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
        uc, cnt = cs[st], np.diff(np.r_[st, cs.size])
        top1 = f["blk_score"].to_numpy()[so[st]]
        top1_s1 = f["s1"].to_numpy()[so[st]]
        top2 = np.where(cnt > 1, f["blk_score"].to_numpy()[so[np.minimum(st + 1, cs.size - 1)]], 0.0)
        pos = np.minimum(np.searchsorted(uc, sub["cand"].to_numpy()), uc.size - 1)
        known = uc[pos] == sub["cand"].to_numpy()
        is_top = known & (top1_s1[pos] == sub["s1"].to_numpy())
        best_other = np.where(known, np.where(is_top, top2[pos], top1[pos]), 0.0).astype(np.float32)
        out.append(pd.DataFrame({"s1": sub["s1"].to_numpy(), "cand": sub["cand"].to_numpy(),
                                 "blk_score": own, "blk_rank": rank,
                                 "comp_best_other": best_other, "comp_margin": (own - best_other).astype(np.float32),
                                 "comp_is_top": is_top.astype(np.int8),
                                 "comp_n": np.where(known, cnt[pos], 0).astype(np.int16)}))
        del f
        gc.collect()
    return pd.concat(out, ignore_index=True)


def _new_pair_features(country: str, new: pd.DataFrame) -> pd.DataFrame:
    """Full stage-2 feature set for anchor-only pairs, computed in memory-bounded
    steps (competition lookup, then text features), each freed before the next."""
    comp = competition_lookup(country, new[["s1", "cand"]])
    new = new.merge(comp, on=["s1", "cand"], how="left")
    del comp
    gc.collect()
    s1t = pd.read_parquet(PROCESSED / f"train_s1_{country}.parquet")
    stats = TokenStats(s1t["name_norm"].to_numpy(), s1t["addr_norm"].to_numpy())
    s1t = s1t[np.isin(s1t["code"].to_numpy(), new["s1"].unique())].set_index("code")
    ct = _texts_full("train", country, new["cand"].to_numpy())
    qx, cx = s1t.loc[new["s1"].to_numpy()], ct.loc[new["cand"].to_numpy()]
    pf = pair_features(qx["name_norm"].to_numpy(), qx["addr_norm"].to_numpy(),
                       cx["name_norm"].to_numpy(), cx["addr_norm"].to_numpy(), cx["nonascii"].to_numpy(), stats)
    del s1t, ct, qx, cx, stats
    gc.collect()
    new = pd.concat([new.reset_index(drop=True), pf], axis=1)
    new["src"] = (new["cand"] // OFFSET).astype(np.int8)
    new["label"] = label_pairs(new["s1"].to_numpy(), new["cand"].to_numpy())
    new["from_anchor_only"] = np.float32(1)
    new["p2"] = np.float32(np.nan)
    return new


def build(country: str, no_new: bool) -> None:
    import pyarrow.parquet as pq
    t0 = time.time()
    tag_out = "trn2c_ancA" if no_new else "trn2c_anc"
    old_path = PROCESSED / f"feats_trn2c_sib_{country}.parquet"
    okeys = pq.read_table(old_path, columns=["s1", "cand"]).to_pandas()
    hits = pd.read_parquet(OUT / f"hits_train_{country}.parquet")
    hits = hits[np.isin(hits["s1"].to_numpy(), okeys["s1"].unique())].reset_index(drop=True)
    new = None
    if not no_new:
        m = hits.merge(okeys.assign(_old=1), on=["s1", "cand"], how="left")
        new = hits[m["_old"].isna().to_numpy()].reset_index(drop=True)
        del m
        new = _new_pair_features(country, new)
        print(f"  {country}: +{len(new):,} anchor-only pairs, pos_rate {new['label'].mean():.4f}", flush=True)
    del okeys
    gc.collect()
    old = pd.read_parquet(old_path).merge(hits, on=["s1", "cand"], how="left")
    old["anc_hits"] = old["anc_hits"].fillna(0).astype(np.float32)
    old["from_anchor_only"] = np.float32(0)
    df = old if new is None else pd.concat([old, new], ignore_index=True)
    del old, new, hits
    gc.collect()
    p2 = df["p2"].to_numpy().astype(np.float32)
    # Recompute context and sibling features on the (possibly enlarged) candidate lists.
    df = df.drop(columns=[c for c in df.columns if c.startswith(CONTEXT_PREFIXES)] + SIB_COLS)
    df = add_context_features(df)
    cc = df["cand"].to_numpy()
    tx = _texts("train", country, cc)
    sf = sibling_features(df["s1"].to_numpy(), cc, p2,
                          tx.loc[cc, "name_norm"].to_numpy(), tx.loc[cc, "addr_norm"].to_numpy())
    del tx
    df = pd.concat([df.reset_index(drop=True), sf], axis=1)
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    df.to_parquet(PROCESSED / f"feats_{tag_out}_{country}.parquet", index=False, compression="zstd")
    shutil.copy(PROCESSED / "entities_trn2c.npy", PROCESSED / f"entities_{tag_out}.npy")
    print(f"{country}: {len(df):,} pairs -> feats_{tag_out}_{country} ({time.time() - t0:.0f}s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["retrieve", "build"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--country", required=True)
    ap.add_argument("--no-new", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.step == "retrieve":
        t0 = time.time()
        h = retrieve(args.split, args.country)
        h.to_parquet(OUT / f"hits_{args.split}_{args.country}.parquet", index=False)
        print(f"{args.split} {args.country}: {len(h):,} anchor-retrieved pairs ({time.time() - t0:.0f}s)", flush=True)
    else:
        build(args.country, args.no_new)


if __name__ == "__main__":
    main()
