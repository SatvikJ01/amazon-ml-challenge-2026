"""Retrieval v4 pilot: do single-field channels recover the remaining retrieval misses?

After E035, 46 % of the holdout loss is retrieval (4,447 true pairs never become
candidates, even with anchor retrieval): empty-address records, alias / domain
names at the right address, native-script names.  The forward channel scores the
name+address blob jointly, so a record that matches on only one field is outranked
by records that half-match on both.  Two single-field channels:

* ``name``  S1 name terms (words, bigrams, skeleton keys) vs target names only;
* ``addr``  S1 address terms (numbers, words, bigrams, skeletons) vs target addresses only
            (structured: number and number+street bigrams are terms);
* ``cname`` / ``caddr``  character 3-gram TF-IDF (within-token, boundary-marked) on name / address;
* ``bm25``  BM25 (k1=1.2, b=0.75) over the full name+address term set of the forward channel.

``report`` also scores reciprocal-rank fusion (RRF, k=60) across channels at a fixed
number of added candidates per entity.

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

from .blocking import BlockingConfig, RareTermIndex, _hasher, build_blobs, search_topk
from .normalize import NULL_TOKENS
from .train import entity_fold, truth_for

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
INTERIM = ROOT / "data" / "interim"
OUT = ROOT / "experiments" / "E037_retrieval_v4"
K = 20
INDIC = re.compile(r"[ऀ-෿]")


def _char_blobs(texts: np.ndarray) -> list[str]:
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
    """RareTermIndex over arbitrary pre-built term blobs, cosine or BM25 weighting."""

    def __init__(self, blob_fn, texts: np.ndarray, cfg: BlockingConfig, mode: str = "cosine", chunk: int = 500_000):
        import gc
        import scipy.sparse as sp
        self.cfg, self.hasher, self.mode, self.blob_fn = cfg, _hasher(cfg), mode, blob_fn
        parts = [self.hasher.transform(blob_fn(texts[i:i + chunk])).tocsr() for i in range(0, len(texts), chunk)]
        X = sp.vstack(parts, format="csr") if len(parts) > 1 else parts[0]
        del parts
        self.n_targets = X.shape[0]
        df = np.bincount(X.indices, minlength=cfg.n_features).astype(np.int64)
        self.df = df
        keep = (df > 0) & (df <= cfg.max_df_abs)
        self.idf = np.zeros(cfg.n_features, np.float32)
        self.idf[keep] = np.log((self.n_targets - df[keep] + 0.5) / (df[keep] + 0.5) + 1.0) if mode == "bm25" \
            else np.log((self.n_targets + 1.0) / (df[keep] + 1.0)) + 1.0
        if mode == "bm25":
            X.data = self.idf[X.indices]
            X.eliminate_zeros()
            ln = np.diff(X.indptr).astype(np.float32)
            k1, b = 1.2, 0.75
            X.data *= np.repeat((k1 + 1) / (1 + k1 * (1 - b + b * ln / max(ln.mean(), 1.0))), np.diff(X.indptr)).astype(np.float32)
        else:
            self._weight_inplace(X)
        self.BT = X.T.tocsr()
        del X
        gc.collect()

    def query_matrix(self, query_blobs):
        A = self.hasher.transform(query_blobs).tocsr()
        if self.mode == "bm25":
            A.data = (self.idf[A.indices] > 0).astype(np.float32)
            A.eliminate_zeros()
            return A
        self._weight_inplace(A)
        return A


def holdout_entities() -> np.ndarray:
    e = np.load(PROCESSED / "entities_v3c.npy")
    return e[entity_fold(e, 5) == 0]


def search(country: str, source: int, channels: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ents = holdout_entities()
    q = pd.read_parquet(PROCESSED / f"train_s1_{country}.parquet", columns=["code", "name_norm", "addr_norm"])
    q = q[np.isin(q["code"].to_numpy(), ents)]
    t = pd.read_parquet(PROCESSED / f"train_s{source}_{country}.parquet", columns=["code", "name_norm", "addr_norm"])
    tcodes = t["code"].to_numpy()
    tn, ta = t["name_norm"].to_numpy(), t["addr_norm"].to_numpy()
    qn, qa = q["name_norm"].to_numpy(), q["addr_norm"].to_numpy()
    del t
    e_t, e_q = np.full(len(tn), "", dtype=object), np.full(len(qn), "", dtype=object)
    for ch in channels:
        t0 = time.time()
        cfg = BlockingConfig(topk=K)
        if ch == "name":
            index, qb = RareTermIndex(tn, e_t, cfg), build_blobs(qn, e_q, cfg)
        elif ch == "addr":
            index, qb = RareTermIndex(e_t, ta, cfg), build_blobs(e_q, qa, cfg)
        elif ch in ("cname", "caddr"):
            cfg = BlockingConfig(topk=K, max_df_abs=3000)
            tt, qq = (tn, qn) if ch == "cname" else (ta, qa)
            index, qb = ChannelIndex(_char_blobs, tt, cfg), _char_blobs(qq)
        elif ch == "bm25":
            fb = lambda x: build_blobs(x[:, 0], x[:, 1], cfg)
            index = ChannelIndex(fb, np.stack([tn, ta], axis=1), cfg, mode="bm25")
            qb = build_blobs(qn, qa, cfg)
        else:
            raise ValueError(ch)
        idx, sc = search_topk(index, qb, label=f"{country}-S{source}-{ch}")
        del index, qb
        valid = idx >= 0
        rows = np.broadcast_to(np.arange(len(qn))[:, None], idx.shape)
        pd.DataFrame({"s1": q["code"].to_numpy()[rows[valid]], "cand": tcodes[idx[valid]], "channel": ch,
                      "rank": np.broadcast_to(np.arange(K), idx.shape)[valid].astype(np.int16), "score": sc[valid]}
                     ).to_parquet(OUT / f"hits_{country}_s{source}_{ch}.parquet", index=False)
        print(f"{country} S{source} {ch}: {int(valid.sum()):,} hits ({time.time() - t0:.0f}s)", flush=True)


def report() -> None:
    ents = holdout_entities()
    truth = truth_for(ents)
    ev = pd.read_parquet(ROOT / "experiments" / "E035_stage3" / "holdout_pred.parquet", columns=["s1", "cand", "country"])
    country = ev.drop_duplicates("s1").set_index("s1")["country"]
    have = set(zip(ev.s1.tolist(), ev.cand.tolist()))
    miss = pd.DataFrame([(e, x) for e in ents for x in truth[e] if (e, x) not in have], columns=["s1", "cand"])
    hits = pd.concat([pd.read_parquet(p) for p in sorted(OUT.glob("hits_*.parquet"))], ignore_index=True)
    hits = hits.drop_duplicates(["s1", "cand", "channel"])
    raw = pd.concat([pd.read_parquet(INTERIM / f"train_s{s}.parquet", columns=["code", "business_name", "business_address"],
                                     filters=[("code", "in", miss.cand.unique().tolist())]) for s in (2, 3)]).set_index("code")
    r = raw.reindex(miss.cand)
    miss["type"] = np.where(r.business_name.fillna("").str.contains(INDIC).to_numpy(), "native",
                            np.where((r.business_address.fillna("").str.strip() == "").to_numpy(), "empty_addr", "other"))
    miss["country"] = country.reindex(miss.s1).to_numpy()
    mkey = set(zip(miss.s1.tolist(), miss.cand.tolist()))
    n_true = sum(len(truth[e]) for e in ents)
    print(f"true pairs {n_true:,}; retrieval misses of the current union (E035 candidates): {len(miss):,} "
          f"{miss.type.value_counts().to_dict()} {miss.country.value_counts().to_dict()}")
    # only new candidates matter: drop hits already in the union
    hk = list(zip(hits.s1.tolist(), hits.cand.tolist()))
    hits = hits[[k not in have for k in hk]].copy()
    hits["hit"] = [k in mkey for k in zip(hits.s1.tolist(), hits.cand.tolist())]
    n_ent = len(ents)

    def line(label, h):
        keys = set(zip(h.s1.tolist(), h.cand.tolist()))
        rec = np.array([k in keys for k in zip(miss.s1.tolist(), miss.cand.tolist())])
        by_t = miss.assign(r=rec).groupby("type")["r"].mean().round(3).to_dict()
        by_c = miss.assign(r=rec).groupby("country")["r"].mean().round(3).to_dict()
        print(f"  {label:26s} recovers {rec.sum():5d}/{len(miss)} ({rec.mean():5.1%})  +recall {rec.sum() / n_true:.4f}  "
              f"new cands/entity {len(keys) / n_ent:5.2f}  precision of new {rec.sum() / max(len(keys), 1):.3f}  {by_t} {by_c}")

    chans = sorted(hits.channel.unique())
    for depth in (5, 10, 20):
        print(f"depth {depth}:")
        h = hits[hits["rank"] < depth]
        for ch in chans:
            line(ch, h[h.channel == ch])
        line("union of all", h)
    # reciprocal-rank fusion across channels, fixed budget of new candidates per entity (per source)
    hits["rrf"] = 1.0 / (60 + hits["rank"])
    hits["src"] = hits.cand // 10**10
    f = hits.groupby(["s1", "src", "cand"], as_index=False)["rrf"].sum()
    f["r"] = f.groupby(["s1", "src"])["rrf"].rank(ascending=False, method="first")
    for m in (2, 5, 10):
        line(f"RRF top-{m}/source", f[f["r"] <= m])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["search", "report"])
    ap.add_argument("--country")
    ap.add_argument("--source", type=int)
    ap.add_argument("--channels", default="name,addr")
    a = ap.parse_args()
    search(a.country, a.source, a.channels.split(",")) if a.step == "search" else report()


if __name__ == "__main__":
    main()
