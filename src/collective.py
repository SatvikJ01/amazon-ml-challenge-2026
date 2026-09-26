"""Collective (sibling-aware) matching: stage 3 of the cascade.

Motivation (E019): each S1 entity has several S2/S3 records, each noised
independently.  93 % of residual misses have a retrieved true sibling, and for
33 % that sibling is >= 15 similarity points closer to the missed record than
the S1 text is.  So a candidate's resemblance to the entity's *confident*
matches is evidence the pairwise model cannot see.

Procedure
---------
1. Stage-2 probabilities are **cross-fitted** over the standard 5 entity folds
   (``oof_stage2``), so every training pair's probability comes from a model that
   never saw its entity.  In-sample probabilities would leak labels into the
   sibling features.
2. Anchors = candidates of the same entity with p2 >= ``ANCHOR_P``.
3. For every candidate, similarity to its best *other* anchor (name, address,
   name|address, canonical first house number) plus anchor count / confidence.
4. Stage 3 = LightGBM on stage-2 features + sibling features + p2.

At test time p2 comes from the stage-2 model; anchors and sibling features are
computed identically.

Usage:
    python -m src.collective oof   --tag trn3c --exp E020_reverse_stage2 --out E021_collective
    python -m src.collective train --tag trn3c --out E021_collective
"""
from __future__ import annotations

import argparse
import os
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from .train import LGB_PARAMS, NON_FEATURES, as_sets, entity_fold, truth_for

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
EXPERIMENTS = ROOT / "experiments"

ANCHOR_P = 0.9
SIB_FEATURES = ["p2", "p2_rank", "n_anchors", "anchor_p_mean", "sib_name", "sib_addr", "sib_all",
                "sib_num_eq", "is_anchor"]


def _first_num(addr: str) -> str:
    for t in addr.split():
        if t.isdigit():
            return t.lstrip("0") or "0"
    return ""


def sibling_features(s1: np.ndarray, cand: np.ndarray, p2: np.ndarray,
                     c_name: np.ndarray, c_addr: np.ndarray) -> pd.DataFrame:
    """Sibling features for aligned pair arrays (any row order)."""
    n = s1.size
    order = np.lexsort((-p2, s1))
    s1o, p2o = s1[order], p2[order]
    starts = np.flatnonzero(np.r_[True, s1o[1:] != s1o[:-1]])
    bounds = np.r_[starts, n]
    grp = np.repeat(np.arange(starts.size), np.diff(bounds))
    rank = np.arange(n) - starts[grp]
    anchor = p2o >= ANCHOR_P
    n_anch = np.add.reduceat(anchor.astype(np.int32), starts)
    p_sum = np.add.reduceat(np.where(anchor, p2o, 0.0), starts)

    # (candidate, anchor) index pairs within each entity, excluding self.
    ci, ai = [], []
    for g, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        anc = np.flatnonzero(anchor[lo:hi]) + lo
        if anc.size == 0:
            continue
        rows = np.arange(lo, hi)
        cc = np.repeat(rows, anc.size)
        aa = np.tile(anc, rows.size)
        keep = cc != aa
        ci.append(cc[keep]); ai.append(aa[keep])
    sib_name = np.zeros(n, np.float32); sib_addr = np.zeros(n, np.float32)
    sib_all = np.zeros(n, np.float32); sib_num = np.zeros(n, np.float32)
    if ci:
        ci = np.concatenate(ci); ai = np.concatenate(ai)
        nm, ad = c_name[order], c_addr[order]
        al = np.array([f"{a} {b}" for a, b in zip(nm, ad)], dtype=object)
        num = np.array([_first_num(a) for a in ad], dtype=object)
        s_n = process.cpdist(nm[ci], nm[ai], scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        s_a = process.cpdist(ad[ci], ad[ai], scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        s_l = process.cpdist(al[ci], al[ai], scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        s_d = ((num[ci] == num[ai]) & (num[ci] != "")).astype(np.float32)
        np.maximum.at(sib_name, ci, s_n); np.maximum.at(sib_addr, ci, s_a)
        np.maximum.at(sib_all, ci, s_l); np.maximum.at(sib_num, ci, s_d)

    out = pd.DataFrame({
        "p2": p2o.astype(np.float32), "p2_rank": rank.astype(np.float32),
        "n_anchors": n_anch[grp].astype(np.float32),
        "anchor_p_mean": np.where(n_anch[grp] > 0, p_sum[grp] / np.maximum(n_anch[grp], 1), 0).astype(np.float32),
        "sib_name": sib_name, "sib_addr": sib_addr, "sib_all": sib_all, "sib_num_eq": sib_num,
        "is_anchor": anchor.astype(np.float32),
    })
    inv = np.empty(n, np.int64); inv[order] = np.arange(n)
    return out.iloc[inv].reset_index(drop=True)


def _texts(split: str, country: str, codes: np.ndarray) -> pd.DataFrame:
    """Name/address of the given target codes, filtered at the Arrow level so a
    3 M-record source file is never materialised as pandas strings."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    want = pa.array(np.unique(codes))
    parts = []
    for s in (2, 3):
        t = pq.read_table(PROCESSED / f"{split}_s{s}_{country}.parquet", columns=["code", "name_norm", "addr_norm"])
        parts.append(t.filter(pc.is_in(t.column("code"), value_set=want)).to_pandas())
        del t
    return pd.concat(parts, ignore_index=True).set_index("code")


def oof_stage2(tag: str, exp: str, out: Path, countries=("India", "US"), model: str = "lgb",
               folds: list[int] | None = None) -> None:
    """5-fold cross-fitted stage-2 probabilities for every sampled training pair.

    Column-wise loading into one float32 matrix (no frame copies).  ``model="xgb"``
    trains XGBoost on the GPU (~9x faster than LightGBM on this machine; same F0.5
    on this data, prediction correlation 0.999 -- E024).
    """
    import gc
    import pyarrow.parquet as pq
    feats = json.loads((EXPERIMENTS / exp / "features.json").read_text())
    paths = [PROCESSED / f"feats_{tag}_{c}.parquet" for c in countries]
    sizes = [pq.ParquetFile(p).metadata.num_rows for p in paths]
    n = sum(sizes)
    X = np.empty((n, len(feats)), np.float32)
    y = np.empty(n, np.float32); s1 = np.empty(n, np.int64); cand = np.empty(n, np.int64)
    country = np.empty(n, dtype=object)
    off = 0
    for p, sz, c in zip(paths, sizes, countries):
        for j, col in enumerate(feats):
            X[off:off + sz, j] = pq.read_table(p, columns=[col]).column(col).to_numpy()
        y[off:off + sz] = pq.read_table(p, columns=["label"]).column("label").to_numpy()
        s1[off:off + sz] = pq.read_table(p, columns=["s1"]).column("s1").to_numpy()
        cand[off:off + sz] = pq.read_table(p, columns=["cand"]).column("cand").to_numpy()
        country[off:off + sz] = c
        off += sz
    gc.collect()
    fold = entity_fold(s1, 5)
    es = entity_fold(s1 * 7 + 3, 10) == 0
    p2 = np.zeros(n, np.float32)
    dall = None
    for k in (folds if folds is not None else range(5)):
        tr = np.flatnonzero((fold != k) & ~es); va = np.flatnonzero((fold != k) & es); te = np.flatnonzero(fold == k)
        if model == "xgb":
            import xgboost as xgb
            params = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device=os.environ.get("XGB_DEVICE", "cuda"),
                          learning_rate=0.05, max_depth=9, min_child_weight=5, subsample=0.8,
                          colsample_bytree=0.8, reg_lambda=1.0, max_bin=128, seed=0)
            # One quantised matrix over all rows; each fold's training set is selected by
            # zero sample weights (no zero-weight row contributes to any gradient or
            # hessian), which avoids copying ~2/3 of X per fold.
            if dall is None:          # built in 1M-row batches: no full-size temporary copy of X
                class _Batches(xgb.DataIter):
                    def __init__(self):
                        self.i = 0
                        super().__init__()
                    def next(self, input_data):
                        if self.i >= n:
                            return 0
                        j = min(self.i + 1_000_000, n)
                        input_data(data=X[self.i:j], label=y[self.i:j])
                        self.i = j
                        return 1
                    def reset(self):
                        self.i = 0
                dall = xgb.QuantileDMatrix(_Batches(), max_bin=128)
            w = np.zeros(n, np.float32); w[tr] = 1.0
            dall.set_weight(w)
            dtr = dall
            dva = xgb.QuantileDMatrix(X[va], y[va], ref=dall, max_bin=128)
            b = xgb.train(params, dtr, 3000, evals=[(dva, "es")], early_stopping_rounds=100, verbose_eval=False)
            p2[te] = b.predict(xgb.DMatrix(X[te]), iteration_range=(0, b.best_iteration + 1))
            best = b.best_iteration
            del dva, b
        else:
            m = lgb.train(LGB_PARAMS, lgb.Dataset(X[tr], y[tr]), 3000, valid_sets=[lgb.Dataset(X[va], y[va])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            p2[te] = m.predict(X[te], num_iteration=m.best_iteration)
            best = m.best_iteration
        print(f"  fold {k}: best_iter {best}", flush=True)
        gc.collect()
    out.mkdir(parents=True, exist_ok=True)
    if folds is not None:      # one process per fold (parallel on a large machine); merged by `oof-merge`
        m = np.isin(fold, folds)
        pd.DataFrame({"s1": s1[m], "cand": cand[m], "country": country[m], "p2": p2[m]}).to_parquet(
            out / f"oof_p2_fold{'_'.join(map(str, folds))}.parquet", index=False)
        return
    pd.DataFrame({"s1": s1, "cand": cand, "country": country, "p2": p2}).to_parquet(out / "oof_p2.parquet", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["oof", "oof-merge", "build", "train"])
    ap.add_argument("--fold", type=int, default=None, help="oof: run only this fold")
    ap.add_argument("--country", default=None, help="build: one country per process")
    ap.add_argument("--split", default="train", help="build: data version for candidate texts")
    ap.add_argument("--model", choices=["lgb", "xgb"], default="lgb", help="oof: model family")
    ap.add_argument("--tag", default="trn3c")
    ap.add_argument("--exp", default="E020_reverse_stage2")
    ap.add_argument("--out", default="E021_collective")
    args = ap.parse_args()
    out = EXPERIMENTS / args.out
    t0 = time.time()
    if args.step == "oof":
        oof_stage2(args.tag, args.exp, out, model=args.model, folds=None if args.fold is None else [args.fold])
        return
    if args.step == "oof-merge":
        parts = sorted(out.glob("oof_p2_fold*.parquet"))
        assert len(parts) == 5, parts
        pd.concat([pd.read_parquet(x) for x in parts], ignore_index=True).to_parquet(out / "oof_p2.parquet", index=False)
        print(f"merged {len(parts)} folds -> {out / 'oof_p2.parquet'}")
        return

    if args.step == "build":
        # Stage-3 training features for one country -> feats_{tag}_sib_{country}.parquet,
        # then train with the standard, memory-lean `src.train --tag {tag}_sib`.
        import shutil
        oof = pd.read_parquet(out / "oof_p2.parquet")
        c = args.country
        oof = oof[oof.country == c][["s1", "cand", "p2"]]
        f = pd.read_parquet(PROCESSED / f"feats_{args.tag}_{c}.parquet").merge(oof, on=["s1", "cand"])
        del oof
        cc = f["cand"].to_numpy()
        tx = _texts(args.split, c, cc)
        sf = sibling_features(f["s1"].to_numpy(), cc, f["p2"].to_numpy(),
                              tx.loc[cc, "name_norm"].to_numpy(), tx.loc[cc, "addr_norm"].to_numpy())
        del tx
        f = pd.concat([f.drop(columns="p2"), sf], axis=1)
        f.to_parquet(PROCESSED / f"feats_{args.tag}_sib_{c}.parquet", index=False, compression="zstd")
        shutil.copy(PROCESSED / f"entities_{args.tag}.npy", PROCESSED / f"entities_{args.tag}_sib.npy")
        print(f"{c}: {len(f):,} pairs with sibling features ({time.time() - t0:.0f}s)", flush=True)
        return

    from .decision import select_sets, threshold_sets
    from .metrics import evaluate, evaluate_by_group
    oof = pd.read_parquet(out / "oof_p2.parquet")
    parts = []
    for c in ("India", "US"):
        f = pd.read_parquet(PROCESSED / f"feats_{args.tag}_{c}.parquet")
        f = f.merge(oof[oof.country == c][["s1", "cand", "p2"]], on=["s1", "cand"])
        tx = _texts("train", c, np.unique(f["cand"].to_numpy()))
        sf = sibling_features(f["s1"].to_numpy(), f["cand"].to_numpy(), f["p2"].to_numpy(),
                              tx.loc[f["cand"].to_numpy(), "name_norm"].to_numpy(),
                              tx.loc[f["cand"].to_numpy(), "addr_norm"].to_numpy())
        parts.append(pd.concat([f.drop(columns="p2"), sf], axis=1).assign(country=c))
        print(f"  {c}: sibling features for {len(f):,} pairs ({time.time() - t0:.0f}s)", flush=True)
        del f, tx, sf
    df = pd.concat(parts, ignore_index=True)
    feats = [c for c in df.columns if c not in NON_FEATURES]
    s1 = df["s1"].to_numpy(); fold = entity_fold(s1, 5); es = entity_fold(s1 * 7 + 3, 10) == 0
    tr, va, ho = (fold != 0) & ~es, (fold != 0) & es, fold == 0
    X = df[feats].to_numpy(np.float32); y = df["label"].to_numpy()
    m = lgb.train(LGB_PARAMS, lgb.Dataset(X[tr], y[tr], feature_name=feats), 3000,
                  valid_sets=[lgb.Dataset(X[va], y[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
    m.save_model(str(out / "model.txt"))
    (out / "features.json").write_text(json.dumps(feats))
    prob = m.predict(X[ho], num_iteration=m.best_iteration)

    sampled = np.load(PROCESSED / f"entities_{args.tag}.npy")
    ents = sampled[entity_fold(sampled, 5) == 0]
    truth = truth_for(ents)
    hs1, hc = s1[ho], df["cand"].to_numpy()[ho]
    res = {}
    for thr in (0.5, 0.6, 0.7, 0.8):
        res[f"thr_{thr}"] = evaluate(as_sets(threshold_sets(hs1, hc, prob, thr), ents), truth)
    res["expF"] = evaluate(as_sets(select_sets(hs1, hc, prob)[0], ents), truth)
    # stage-2 alone on the same rows, for a paired comparison
    p2h = df["p2"].to_numpy()[ho]
    res["stage2_thr_0.7"] = evaluate(as_sets(threshold_sets(hs1, hc, p2h, 0.7), ents), truth)
    res["stage2_expF"] = evaluate(as_sets(select_sets(hs1, hc, p2h)[0], ents), truth)
    best = max((k for k in res if not k.startswith("stage2")), key=res.get)
    rule_sets = (as_sets(select_sets(hs1, hc, prob)[0], ents) if best == "expF"
                 else as_sets(threshold_sets(hs1, hc, prob, float(best[4:])), ents))
    cmap = dict(zip(df["s1"].to_numpy()[ho], df["country"].to_numpy()[ho]))
    groups = {e: cmap.get(e, "none") for e in ents}
    rep = {"rules": res, "best": best, "best_f05": res[best], "best_iter": m.best_iteration,
           "by_country": evaluate_by_group(rule_sets, truth, groups),
           "top_features": pd.Series(m.feature_importance("gain"), index=feats).sort_values(ascending=False).head(12).round(0).to_dict(),
           "elapsed_sec": round(time.time() - t0, 1)}
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=float))
    print(json.dumps({k: rep[k] for k in ("rules", "best", "best_f05", "by_country")}, indent=2, default=float))


if __name__ == "__main__":
    main()
