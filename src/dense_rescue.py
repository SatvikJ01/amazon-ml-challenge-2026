"""E047: rescue model for the dense-retrieval pairs that are NOT in our candidate set.

The Kaggle kernel (``kaggle/e047_dense/dense_kernel.py``) returns, per country, the new pairs found by a fine-tuned
e5-small bi-encoder (forward top-5 / reverse top-2, Bloom-filtered against the existing 47M candidate pairs) with
their cosine, ranks and the E046 LaBSE cross-encoder logit.  A LightGBM model gives each new pair a match
probability; the new pairs are then added to the existing table and the usual decision (one-owner exclusivity +
expected-F0.5 prefix) runs on the union.

Features (label-free, same on holdout and test): cos, rank_f, rank_r, ce, src, the pair's ce/cos rank among the
entity's new pairs, number of new pairs, and the entity's existing confidence (best current probability, number
of current pairs >= 0.5, their sum).  No same-candidate claimant features (the holdout holds 8 % of the S1).

``holdout``: 2-fold entity CV on the holdout new pairs; F0.5 of existing-only vs existing + rescued.
``test``:    trains on all holdout new pairs, scores the test new pairs and writes <base> + <country>_px.parquet.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
E46 = ROOT / "experiments" / "E046"
E47 = ROOT / "experiments" / "E047"
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1, seed=47)
FEATS = ["cos", "rank_f", "rank_r", "ce", "src", "ce_rk", "cos_rk", "n_new", "e_pmax", "e_nhi", "e_psum"]
_T0 = time.time()


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), "%6.0fs" % (time.time() - _T0), *a, flush=True)


def _sig(x):
    return 1 / (1 + np.exp(-x))


def _logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def add_features(N: pd.DataFrame, P: pd.DataFrame) -> pd.DataFrame:
    """N: new pairs (s1, cand, cos, rank_f, rank_r, ce); P: existing pairs (s1, prob) of the same S1."""
    N = N.copy()
    N["src"] = (N.cand // 10**10).astype(np.int8)
    g = N.groupby("s1")
    N["ce_rk"] = g.ce.rank(ascending=False, method="first").astype(np.float32)
    N["cos_rk"] = g.cos.rank(ascending=False, method="first").astype(np.float32)
    N["n_new"] = g.cand.transform("size").astype(np.float32)
    e = P.groupby("s1").prob.agg(e_pmax="max", e_psum="sum", e_nhi=lambda x: float((x >= 0.5).sum()))
    N = N.merge(e, left_on="s1", right_index=True, how="left")
    for c in ("e_pmax", "e_psum", "e_nhi"):
        N[c] = N[c].fillna(0.0).astype(np.float32)
    return N


def load_new(split: str, countries, d: Path) -> pd.DataFrame:
    parts = []
    for c in countries:
        f = d / f"dense_{split}_{c}.parquet"
        if f.exists():
            t = pd.read_parquet(f)
            t["country"] = c
            parts.append(t)
    return pd.concat(parts, ignore_index=True)


def holdout_final_probs(tag: str) -> pd.DataFrame:
    """Holdout pairs with the CV probability of the chosen E046 stack (full rule) on the band, pf elsewhere."""
    from .ce_stack import HI, LO
    D = pd.read_parquet(E46 / "holdout_pairs.parquet")
    raw = np.load(E46 / f"stack_raw_{tag}.npy")[-1]
    band = np.flatnonzero((D.pf.to_numpy() >= LO) & (D.pf.to_numpy() < HI))
    assert len(band) == len(raw), f"stack dump has {len(raw)} rows, band {len(band)}"
    p = D.pf.to_numpy().astype(np.float64)
    p[band] = _sig(_logit(p[band]) + raw)
    D["prob"] = p
    return D


def cmd_holdout(a) -> None:
    import lightgbm as lgb

    from .ce_data import entity_scorer
    from .l2_train import holdout_entities
    from .train import truth_for

    D = holdout_final_probs(a.stack_tag)
    N = load_new("hold", ("US", "India"), Path(a.dense_dir))
    T = truth_for(holdout_entities())
    N["label"] = np.fromiter((c in T.get(int(s), ()) for s, c in zip(N.s1.to_numpy(), N.cand.to_numpy())), bool, len(N)).astype(np.int8)
    # how much of the retrieval loss do the new pairs cover?
    pos = D[D.label == 1].groupby("s1").cand.apply(set).to_dict()
    missed = sum(len(T[e] - pos.get(int(e), set())) for e in T)
    log(f"holdout new pairs {len(N):,} (positives {int(N.label.sum()):,} = {N.label.sum() / max(missed, 1):.1%} of the {missed:,} "
        f"missed true matches); by country {N.country.value_counts().to_dict()}")
    N = add_features(N, D[["s1", "prob"]])
    # 2-fold entity CV
    u = np.unique(N.s1.to_numpy())
    fold = pd.Series(np.random.default_rng(47).integers(0, 2, len(u)), index=u).reindex(N.s1.to_numpy()).to_numpy()
    pr = np.zeros(len(N))
    y = N.label.to_numpy()
    for f in (0, 1):
        m = lgb.train(PARAMS, lgb.Dataset(N.loc[fold != f, FEATS], y[fold != f]), num_boost_round=a.rounds)
        pr[fold == f] = m.predict(N.loc[fold == f, FEATS])
    r = pd.Series(pr).rank().to_numpy()
    yy = y.astype(bool)
    log("rescue AUC %.4f; pairs with p>=0.5: %d (precision %.3f)" % (
        (r[yy].sum() - yy.sum() * (yy.sum() + 1) / 2) / (yy.sum() * (~yy).sum()), int((pr >= 0.5).sum()),
        float(y[pr >= 0.5].mean()) if (pr >= 0.5).any() else float("nan")))
    base = D[["s1", "cand", "country", "prob"]]
    ent_f0, ents, ec = entity_scorer(base)
    f0 = ent_f0(base.prob.to_numpy())
    res = []
    for scale in a.scales:
        U = pd.concat([base, pd.DataFrame({"s1": N.s1, "cand": N.cand, "country": N.country,
                                           "prob": _sig(_logit(pr) + np.log(scale))})], ignore_index=True)
        ent_f, _, _ = entity_scorer(U)
        f1 = ent_f(U.prob.to_numpy())
        d = f1 - f0
        r_ = dict(scale=scale, F_base=round(float(f0.mean()), 6), F=round(float(f1.mean()), 6), gain=round(float(d.mean()), 6),
                  se=round(float(d.std() / np.sqrt(len(d))), 6), gain_India=round(float(d[ec == "India"].mean()), 6),
                  gain_US=round(float(d[ec == "US"].mean()), 6), changed=int((d != 0).sum()))
        res.append(r_)
        log("RESCUE", json.dumps(r_))
    (E47 / f"rescue_{a.tag}.json").write_text(json.dumps(res, indent=1))


def cmd_test(a) -> None:
    import lightgbm as lgb

    from .l2_train import holdout_entities
    from .train import truth_for

    if a.model_in:                                   # packaged route: apply the shipped rescue model
        m = lgb.Booster(model_file=a.model_in)
        assert m.feature_name() == FEATS, f"shipped rescue features {m.feature_name()} != {FEATS}"
    else:
        D = holdout_final_probs(a.stack_tag)
        N = load_new("hold", ("US", "India"), Path(a.dense_dir))
        T = truth_for(holdout_entities())
        N["label"] = np.fromiter((c in T.get(int(s), ()) for s, c in zip(N.s1.to_numpy(), N.cand.to_numpy())), bool, len(N)).astype(np.int8)
        N = add_features(N, D[["s1", "prob"]])
        m = lgb.train(PARAMS, lgb.Dataset(N[FEATS], N.label.to_numpy()), num_boost_round=a.rounds)
        if a.model_out:
            Path(a.model_out).parent.mkdir(parents=True, exist_ok=True)
            m.save_model(a.model_out)
            log("rescue model saved ->", a.model_out)
        del D, N
    base = ROOT / "experiments" / a.run / a.base
    out = ROOT / "experiments" / a.run / a.out
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(base, out)
    P = pd.concat([pd.read_parquet(f, columns=["s1", "prob"]) for f in sorted(base.glob("*_p*.parquet"))], ignore_index=True)
    P = P[P.prob >= 1e-3]
    for c in a.countries:
        Nt = load_new("test", (c,), Path(a.dense_dir))
        Nt = add_features(Nt, P[P.s1.isin(Nt.s1.unique())])
        p = _sig(_logit(m.predict(Nt[FEATS])) + np.log(a.scale))
        res = pd.DataFrame({"s1": Nt.s1.to_numpy(), "cand": Nt.cand.to_numpy(), "src": Nt.src.to_numpy(), "prob": p.astype(np.float32)})
        res.to_parquet(out / f"{c}_px.parquet", index=False)
        log(f"{c}: {len(res):,} new pairs; p>=0.5: {int((p >= 0.5).sum()):,}; p>=0.9: {int((p >= 0.9).sum()):,}")
    json.dump(dict(base=a.base, rounds=a.rounds, scale=a.scale, countries=a.countries), open(out / "meta_rescue.json", "w"))
    log("->", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("holdout", "test"):
        s = sub.add_parser(name)
        s.add_argument("--dense-dir", default=str(E47 / "k4"))
        s.add_argument("--stack-tag", default="mdl_dump")
        s.add_argument("--rounds", type=int, default=300)
    sub.choices["holdout"].add_argument("--scales", type=float, nargs="+", default=[0.5, 1.0])
    sub.choices["holdout"].add_argument("--tag", default="v1")
    t = sub.choices["test"]
    t.add_argument("--run", default="E039_test")
    t.add_argument("--base", default="scores_mdl_C")
    t.add_argument("--out", required=True)
    t.add_argument("--countries", nargs="+", default=["US", "India"])
    t.add_argument("--scale", type=float, default=1.0)
    t.add_argument("--model-out", default="", help="save the rescue model trained on the holdout here")
    t.add_argument("--model-in", default="", help="apply this saved rescue model instead of training")
    a = ap.parse_args()
    {"holdout": cmd_holdout, "test": cmd_test}[a.cmd](a)


if __name__ == "__main__":
    main()
