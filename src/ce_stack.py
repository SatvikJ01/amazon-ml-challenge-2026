"""E046: combine the cross-encoder scores with the shipped probability (a residual stacker).

The cross-encoder (``kaggle/e046_ce/ce_kernel.py``) scores the band pairs (1e-3 <= pf < 0.999, where
the oracle says all the recoverable F0.5 is).  A small LightGBM residual learns, on the E039 holdout,
how far to move ``logit(pf)`` given the cross-encoder logit(s) and entity / claimant context:

    p_new = sigmoid(logit(pf) + booster(X))        (init_score = logit(pf), like the E044 residual)

Features (all label-free, computed the same way on holdout and test):
    lpf, lp39                       shipped prob and stage-3 prob (logits)
    ce{k}                           cross-encoder logit of backbone k (and their mean ``ce``)
    ce_rank, ce_gap, ce_max_o       rank of the pair's ce among the entity's band pairs, ce - best other
    n_band, n_sure, pf_max_o        entity: number of band pairs, of pairs with pf >= 0.999, best other pf
    c_ce_max_o, c_pf_max_o, c_n     candidate: best ce / pf of the OTHER S1 claiming the same record, count
    src                             2 or 3

``holdout``: 2-fold entity CV on the holdout (the stacker never sees the entity it scores), F0.5 with
the submission rule at several round checkpoints, per country and for down-only variants.
``test``:    trains on the whole holdout band with the chosen rounds and writes a scores directory
(same layout as E039_test/scores_*) with the band probabilities replaced, per-country rule.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "experiments" / "E046"
LO, HI = 1e-3, 0.999
_T0 = time.time()
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, num_threads=12, verbose=-1, seed=46)


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), "%6.0fs" % (time.time() - _T0), *a, flush=True)


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def max_other(g: np.ndarray, v: np.ndarray):
    """Per row: max of v over the OTHER rows of its group g (-inf if alone), rank of v in its group
    (0 = best) and group size."""
    order = np.lexsort((-v, g))
    gs, vs = g[order], v[order]
    first = np.r_[True, gs[1:] != gs[:-1]]
    gid = np.cumsum(first) - 1
    starts = np.flatnonzero(first)
    sizes = np.diff(np.r_[starts, len(gs)])
    top1 = vs[starts]
    top2 = np.where(sizes > 1, vs[np.minimum(starts + 1, len(vs) - 1)], -np.inf)
    mo = np.empty(len(v), np.float64)
    rk = np.empty(len(v), np.int32)
    sz = np.empty(len(v), np.int32)
    mo[order] = np.where(first, top2[gid], top1[gid])
    rk[order] = np.arange(len(gs)) - starts[gid]
    sz[order] = sizes[gid]
    return mo, rk, sz


def features(B: pd.DataFrame, A: pd.DataFrame, ce_cols: list[str]) -> pd.DataFrame:
    """B: band pairs (s1, cand, src, pf, p39, ce...); A: all pairs with pf >= 1e-3 (s1, cand, pf)."""
    X = pd.DataFrame(index=B.index)
    X["lpf"] = _logit(B.pf)
    X["lp39"] = _logit(B.p39)
    for c in ce_cols:
        X[c] = B[c].to_numpy(np.float64)
    ce = B[ce_cols].to_numpy(np.float64).mean(1)
    if len(ce_cols) > 1:
        X["ce"] = ce
    s1, cand = B.s1.to_numpy(), B.cand.to_numpy()
    mo, rk, sz = max_other(s1, ce)
    X["ce_rank"], X["ce_max_o"], X["ce_gap"], X["n_band"] = rk, np.where(np.isfinite(mo), mo, -20), \
        np.where(np.isfinite(mo), ce - mo, 20), sz
    cmo, _, csz = max_other(cand, ce)
    X["c_ce_max_o"] = np.where(np.isfinite(cmo), cmo, -20)
    # context over all pairs with pf >= 1e-3
    a_s1, a_c, a_pf = A.s1.to_numpy(), A.cand.to_numpy(), A.pf.to_numpy(np.float64)
    pmo, _, _ = max_other(a_s1, a_pf)
    cpmo, _, cn = max_other(a_c, a_pf)
    sure = pd.Series((a_pf >= HI).astype(np.int32)).groupby(a_s1).sum()
    key = pd.MultiIndex.from_arrays([a_s1, a_c])
    idx = key.get_indexer(pd.MultiIndex.from_arrays([s1, cand]))
    assert (idx >= 0).all(), "band pairs missing from the all-pairs table"
    X["pf_max_o"] = np.where(np.isfinite(pmo[idx]), pmo[idx], 0.0)
    X["c_pf_max_o"] = np.where(np.isfinite(cpmo[idx]), cpmo[idx], 0.0)
    X["c_n"] = cn[idx]
    X["n_sure"] = sure.reindex(s1).fillna(0).to_numpy()
    X["src"] = B.src.to_numpy()
    return X.astype(np.float32)


# Test-vs-train composition of the candidate pools (record counts only, no labels):
# unowned decoy records per S1 are x1.89 (US) / x1.94 (India) in test, and the number of other
# businesses (S1 owners) per country is x0.50 (US) / x0.92 (India).
SHIFT_W = {"US": {"decoy": 1.89, "sib": 0.50}, "India": {"decoy": 1.94, "sib": 0.92}}


def negative_weights(B: pd.DataFrame) -> np.ndarray:
    """Per-pair weight simulating the test pool composition: 1 for matches, SHIFT_W[country]['decoy'] for
    negatives whose candidate record has no owner (a decoy), SHIFT_W[country]['sib'] for negatives owned
    by another S1.  Uses the training ground truth (holdout rows only)."""
    owned = pd.Index(np.unique(np.load(ROOT / "data" / "interim" / "gt_pair_m.npy")))
    own = owned.get_indexer(B.cand.to_numpy()) >= 0
    w = np.ones(len(B))
    neg = B.label.to_numpy() == 0
    for c, d in SHIFT_W.items():
        m = neg & (B.country.to_numpy() == c)
        w[m & ~own] = d["decoy"]
        w[m & own] = d["sib"]
    return w


def weighted_scorer(D: pd.DataFrame):
    """Per-entity F0.5 where each false positive counts with its test-composition weight (decoy /
    other-owner), an approximation of the F0.5 the decision would get on a test-like pool."""
    from .decision import select_sets
    from .inference import enforce_exclusivity
    from .l2_train import holdout_entities
    from .train import truth_for

    ents = holdout_entities()
    T = truth_for(ents)
    owned = pd.Index(np.unique(np.load(ROOT / "data" / "interim" / "gt_pair_m.npy")))
    cty = dict(zip(D.s1.tolist(), D.country.tolist()))
    s1a, ca = D.s1.to_numpy(), D.cand.to_numpy()
    decoys = set(np.unique(ca[owned.get_indexer(ca) < 0]).tolist())     # candidate records without owner

    def f(prob: np.ndarray) -> np.ndarray:
        pp = np.asarray(prob, np.float64)
        kk = enforce_exclusivity(s1a, ca, pp)
        sets_, _, _ = select_sets(s1a[kk], ca[kk], pp[kk])
        out = np.empty(len(ents))
        for i, e in enumerate(ents):
            S = set(np.asarray(sets_.get(int(e), [])).tolist())
            t = T[e]
            if not t:
                out[i] = 1.0 if not S else 0.0
                continue
            tp = len(S & t)
            fp_ids = [x for x in S if x not in t]
            wd = SHIFT_W.get(cty.get(int(e), ""), {"decoy": 1.0, "sib": 1.0})
            fp = sum(wd["decoy"] if x in decoys else wd["sib"] for x in fp_ids)
            den = 1.25 * tp + 0.25 * (len(t) - tp) + fp
            out[i] = 1.25 * tp / den if den > 0 else 0.0
        return out

    return f


def load_ce(split: str, ks: list[int], d: Path) -> pd.DataFrame:
    out = None
    for k in ks:
        t = pd.read_parquet(d / f"ce{k}_{split}.parquet").rename(columns={"logit": f"ce{k}"})
        out = t if out is None else out.merge(t, on=["s1", "cand"], how="inner")
    return out


# ============================================================================ holdout CV
def cmd_holdout(a) -> None:
    import lightgbm as lgb

    from .ce_data import entity_scorer

    ks = [int(x) for x in a.ce.split(",")]
    D = pd.read_parquet(E / "holdout_pairs.parquet")
    A = D[D.pf >= LO][["s1", "cand", "pf"]]
    B = D[(D.pf >= LO) & (D.pf < HI)].reset_index(drop=True)
    C = load_ce("hold", ks, Path(a.ce_dir))
    n0 = len(B)
    B = B.merge(C, on=["s1", "cand"], how="inner")
    assert len(B) == n0, f"cross-encoder scores cover {len(B)} of {n0} band pairs"
    ce_cols = [f"ce{k}" for k in ks]
    X = features(B, A, ce_cols)
    if a.extra:
        ex = a.extra.split(",")
        Xe = B[["s1", "cand"]].merge(pd.read_parquet(E / "hold_extra.parquet", columns=["s1", "cand"] + ex),
                                     on=["s1", "cand"], how="left")
        for c in ex:
            X[c] = Xe[c].to_numpy(np.float32)
    feats = list(X.columns) if a.feats == "all" else a.feats.split(",") + (a.extra.split(",") if a.extra else [])
    y = B.label.to_numpy()
    init = X.lpf.to_numpy(np.float64)
    ent_f, ents, ec = entity_scorer(D)
    pf_all = D.pf.to_numpy().astype(np.float64)
    key = pd.MultiIndex.from_arrays([D.s1.to_numpy(), D.cand.to_numpy()])
    pos = key.get_indexer(pd.MultiIndex.from_arrays([B.s1.to_numpy(), B.cand.to_numpy()]))
    f0 = ent_f(pf_all)
    wf = weighted_scorer(D)
    f0w = wf(pf_all)
    wtr = negative_weights(B) if a.shift else np.ones(len(B))
    log(f"band pairs {len(B):,}; shipped F={f0.mean():.6f} (test-composition weighted {f0w.mean():.6f}); "
        f"shift-weighted training {a.shift}; features {feats}")
    for c in ce_cols + (["ce"] if "ce" in X else []):
        from .ce_data import _sigmoid  # noqa: F401
        yy = y.astype(bool)
        r = pd.Series(X[c].to_numpy()).rank().to_numpy()
        auc = (r[yy].sum() - yy.sum() * (yy.sum() + 1) / 2) / (yy.sum() * (~yy).sum())
        log(f"band AUC {c}: {auc:.5f}")
    r = pd.Series(init).rank().to_numpy()
    yy = y.astype(bool)
    log("band AUC pf: %.5f" % ((r[yy].sum() - yy.sum() * (yy.sum() + 1) / 2) / (yy.sum() * (~yy).sum())))
    # 2-fold entity CV
    u = np.unique(B.s1.to_numpy())
    fold_of = pd.Series(np.random.default_rng(46).integers(0, 2, len(u)), index=u)
    fold = fold_of.reindex(B.s1.to_numpy()).to_numpy()
    cps = [int(x) for x in a.checkpoints.split(",")]
    raw = {it: np.zeros(len(B)) for it in cps}
    for f in (0, 1):
        tr, te = fold != f, fold == f
        ds = lgb.Dataset(X.loc[tr, feats], y[tr], init_score=init[tr], weight=wtr[tr], free_raw_data=False)
        m = lgb.train(PARAMS, ds, num_boost_round=max(cps))
        for it in cps:
            raw[it][te] = m.predict(X.loc[te, feats], raw_score=True, num_iteration=it)
    if a.dump:
        np.save(E / f"stack_raw_{a.tag}.npy", np.stack([raw[it] for it in cps]))
    res = []
    us = (B.country.to_numpy() == "US")
    for it in cps:
        pn = 1 / (1 + np.exp(-(init + raw[it])))
        for rule in ("full", "US_down", "down"):
            q = pn.copy()
            if rule == "US_down":
                q[us] = np.minimum(q[us], B.pf.to_numpy()[us])
            elif rule == "down":
                q = np.minimum(q, B.pf.to_numpy())
            p = pf_all.copy()
            p[pos] = q
            fn = ent_f(p)
            d = fn - f0
            dw = wf(p) - f0w
            r_ = dict(tag=a.tag, it=it, rule=rule, F=round(float(fn.mean()), 6), gain=round(float(d.mean()), 6),
                      wgain=round(float(dw.mean()), 6), wgain_US=round(float(dw[ec == "US"].mean()), 6),
                      wgain_India=round(float(dw[ec == "India"].mean()), 6),
                      se=round(float(d.std() / np.sqrt(len(d))), 6),
                      gain_India=round(float(d[ec == "India"].mean()), 6), gain_US=round(float(d[ec == "US"].mean()), 6),
                      changed=int((d != 0).sum()))
            res.append(r_)
            log("STACK", json.dumps(r_))
    out = E / f"stack_{a.tag}.json"
    out.write_text(json.dumps(res, indent=1))
    log("->", out)


# ============================================================================ test
def cmd_test(a) -> None:
    import lightgbm as lgb
    import pyarrow.compute as pc

    ks = [int(x) for x in a.ce.split(",")]
    ce_cols = [f"ce{k}" for k in ks]
    # train on the whole holdout band
    D = pd.read_parquet(E / "holdout_pairs.parquet")
    A = D[D.pf >= LO][["s1", "cand", "pf"]]
    B = D[(D.pf >= LO) & (D.pf < HI)].reset_index(drop=True).merge(load_ce("hold", ks, Path(a.ce_dir)), on=["s1", "cand"])
    X = features(B, A, ce_cols)
    ex = a.extra.split(",") if a.extra else []
    if ex:
        Xe = B[["s1", "cand"]].merge(pd.read_parquet(E / "hold_extra.parquet", columns=["s1", "cand"] + ex),
                                     on=["s1", "cand"], how="left")
        for c in ex:
            X[c] = Xe[c].to_numpy(np.float32)
    feats = list(X.columns) if a.feats == "all" else a.feats.split(",") + ex
    wtr = negative_weights(B) if a.shift else None
    m = lgb.train(PARAMS, lgb.Dataset(X[feats], B.label.to_numpy(), init_score=X.lpf.to_numpy(np.float64), weight=wtr),
                  num_boost_round=a.rounds)
    del D, A, B, X
    # test band + context
    T = pd.read_parquet(E / "test_pairs.parquet")               # s1, cand, src, pf, country (pf >= 1e-3)
    A = T[["s1", "cand", "pf"]]
    B = T[(T.pf >= LO) & (T.pf < HI)].reset_index(drop=True)
    p39, keys = [], B[["s1", "cand"]]
    for f in sorted((ROOT / "experiments" / a.run / "scores_feats").glob("*.parquet")):
        t = pq.read_table(f, columns=["s1", "cand", "prob3"] + ex).to_pandas()
        p39.append(t.merge(keys, on=["s1", "cand"], how="inner"))     # keep only the band pairs (memory)
        del t
    p39 = pd.concat(p39, ignore_index=True).rename(columns={"prob3": "p39"})
    n0 = len(B)
    B = B.merge(p39, on=["s1", "cand"], how="left")
    del p39
    assert len(B) == n0 and not B.p39.isna().any(), "stage-3 prob missing for some test band pairs"
    B = B.merge(load_ce("test", ks, Path(a.ce_dir)), on=["s1", "cand"], how="inner")
    assert len(B) == n0, f"cross-encoder test scores cover {len(B)} of {n0} band pairs"
    X = features(B, A, ce_cols)
    for c in ex:
        X[c] = B[c].to_numpy(np.float32)
    raw = m.predict(X[feats], raw_score=True)
    pn = 1 / (1 + np.exp(-(X.lpf.to_numpy(np.float64) + raw)))
    rules = dict(r.split("=") for r in a.rule)
    cty = B.country.to_numpy()
    pf = B.pf.to_numpy(np.float64)
    q = pf.copy()
    for c in ("India", "US", "France"):
        mm = cty == c
        rule = rules.get(c, "raw")
        if rule == "full":
            q[mm] = pn[mm]
        elif rule == "down":
            q[mm] = np.minimum(pn[mm], pf[mm])
        log(f"{c}: rule {rule}; band pairs {int(mm.sum()):,}; mean shift {float((_logit(q[mm]) - _logit(pf[mm])).mean()):+.3f}; "
            f"lowered {int((q[mm] < pf[mm]).sum()):,} raised {int((q[mm] > pf[mm]).sum()):,}")
    new = pd.DataFrame({"s1": B.s1.to_numpy(), "cand": B.cand.to_numpy(), "pn": q.astype(np.float32)})
    src_dir = ROOT / "experiments" / a.run / a.base
    out = ROOT / "experiments" / a.run / a.out
    out.mkdir(parents=True, exist_ok=True)
    tot = 0
    for f in sorted(src_dir.glob("*_p*.parquet")):
        t = pd.read_parquet(f)
        t = t.merge(new, on=["s1", "cand"], how="left")
        hit = t.pn.notna().to_numpy()
        t["prob"] = np.where(hit, t.pn.to_numpy(), t.prob.to_numpy()).astype(np.float32)
        tot += int(hit.sum())
        t.drop(columns=["pn"]).to_parquet(out / f.name, index=False)
    assert tot == len(new), f"replaced {tot} of {len(new)} band pairs"
    json.dump(dict(ce=ks, feats=feats, rounds=a.rounds, rule=rules, base=a.base, band_pairs=len(new)),
              open(out / "meta_ce.json", "w"), indent=1)
    log(f"-> {out} ({tot:,} probabilities replaced)")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("holdout")
    s.add_argument("--ce", default="0,1")
    s.add_argument("--ce-dir", default=str(E / "kaggle_out"))
    s.add_argument("--feats", default="all")
    s.add_argument("--checkpoints", default="100,200,400,800")
    s.add_argument("--tag", default="all")
    s.add_argument("--dump", action="store_true")
    s.add_argument("--shift", action="store_true", help="train with test-composition negative weights")
    s.add_argument("--extra", default="", help="comma list of stage-3 features added to the stacker")
    s = sub.add_parser("test")
    s.add_argument("--ce", default="0,1")
    s.add_argument("--ce-dir", default=str(E / "kaggle_out"))
    s.add_argument("--feats", default="all")
    s.add_argument("--rounds", type=int, required=True)
    s.add_argument("--run", default="E039_test")
    s.add_argument("--base", default="scores_frdown")
    s.add_argument("--out", required=True)
    s.add_argument("--rule", nargs="+", required=True, help="Country=full|down|raw")
    s.add_argument("--shift", action="store_true", help="train with test-composition negative weights")
    s.add_argument("--extra", default="", help="comma list of stage-3 features added to the stacker")
    a = ap.parse_args()
    {"holdout": cmd_holdout, "test": cmd_test}[a.cmd](a)


if __name__ == "__main__":
    main()
