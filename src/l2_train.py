"""Training of the E044 second-level RESIDUAL model (the final correction applied on top of E039 stage 3).

This module is the repo port of the EC2 research scripts that produced ``experiments/L2_E044``
(originally ``/home/ubuntu/final_push/level2_prod/{crossfit,feat_train,build,train_e044,finalize_e044}.py``
plus the helpers they imported from ``final_push/common.py``).  Folds, seeds, LightGBM parameters,
thresholds, row filters and column order are unchanged; only the paths are repo-relative.

Idea
----
E039 stage 3 gives a probability ``prob`` for every (S1, candidate) pair.  The residual model learns a
correction in logit space from features the pairwise model cannot see (entity context: how the pair
ranks among the entity's candidates and among other S1 claiming the same record; name edit type;
empty-address ownership among S1 that share a name), see ``src/level2.py``:

    prob_new = sigmoid(logit(prob) + booster.predict(X, raw_score=True))     for rows with prob >= 1e-3

The residual is trained with ``init_score = logit(prob)``, so it starts from stage 3 and only fits
what stage 3 got wrong.  Its training ``prob`` must be out-of-sample, therefore stage 3 is first
CROSS-FITTED inside the E039 training entities (the E039 holdout is never read into any residual
training matrix).  Note: the residual's number of trees (600) was chosen from the 200/400/600/800
checkpoints evaluated on this same holdout (0.985327-0.985504; 600 preferred over 800 because the
mean logit shift kept growing with rounds), so 0.985458 carries a small selection effect.

Pipeline (sub-commands, outputs under ``experiments/E044/`` unless ``--work`` says otherwise)
----------------------------------------------------------------------------------------------
``crossfit``  (research: crossfit.py)
    Training rows = rows of ``data/processed/feats_v5c_ancz_{India,US}.parquet`` (the E039 stage-3
    table) with ``entity_fold(s1, 5) != 0``.  4 folds ``entity_fold(s1*17+3, 4)``; inner early-stopping
    split ``entity_fold(s1*7+3, 10) == 0`` (as in src/train.py).  Fold k: LightGBM with the E039
    parameters (``src.train.LGB_PARAMS``, 8 threads) on the 114 E039 stage-3 features
    (``experiments/E039_stage3/features.json`` order), up to 3000 rounds, early stopping 100 on the inner
    split, predicts all training rows of fold k.
    -> stage3_fold{k}.txt, oof_fold{k}.parquet (country, row, s1, cand, label, prob), crossfit_folds.json
``features``  (research: feat_train.py)
    Production level-2 features (``src.level2.features``, tables of split trainT) for every training row,
    with prob = the OOF stage-3 prob.  Each fold file holds whole entities, so entity aggregates see the
    full candidate list of every S1.  Keeps rows with OOF prob >= 1e-3.
    -> l2_fold{k}_{country}.parquet (row, s1, cand, label, prob + the 95 L2_COLS)
``holdout``   (research: common._build_holdout + build.py)
    The E039 holdout table (entities_v5c_ancz with entity_fold == 0; ids, label, country, the 114
    features, prob = experiments/E039_stage3/holdout_pred.parquet) and its production level-2 features.
    -> holdout.parquet, holdout_l2.parquet (row, s1, cand + the 95 L2_COLS)
``train``     (research: train_e044.py)
    Residual LightGBM, variant ALL_nc (204 columns = ``level2.model_columns(E039 features, "ALL_nc")``),
    on the training rows of ``--folds`` (prob >= 1e-3), label = match, init_score = logit(OOF prob),
    params LGB2 (below) with ``num_leaves`` / ``learning_rate`` from the command line.  Then evaluates on
    the untouched holdout (prob = E039 holdout prob, production features) with the submission rule
    (one-owner exclusivity + expected-F0.5 prefix selection, macro F0.5 against the full ground truth)
    at each checkpoint (number of trees).
    -> model_{tag}/{model.txt, features.json, ref_X5000.npy, entf_it{it}.npy, eval.json}
``evaluate``  (the holdout-evaluation half of train_e044.py, for an existing model)
    -> <out>/{entf_it{it}.npy, eval.json}; also checks the rebuilt holdout matrix against the model
    dir's ref_X5000.npy when present.
``finalize``  (research: finalize_e044.py)
    Truncates model_{tag} to the chosen iteration and writes a deployable model dir (model.txt,
    features.json, ref_X5000.npy, ref_raw5000.npy, meta.json) as read by ``scripts/apply_l2.py``.
    -> final_E044/  (the shipped copy is experiments/L2_E044/)

Exact commands of the final model (EC2 r7i.2xlarge, 8 vCPU / 61 GB, run from the repo root after
``scripts/aws/run_e039.sh``; timings of the original run).  Every sub-command takes ``--work DIR``
(default experiments/E044).
------------------------------------------------------------------------------------------------
    python -m src.l2_train crossfit                    # 4 folds x 16-21 min (best it 1431/1428/999/1647), 7.7 GB matrix
    python -m src.l2_train features --wait             # ran concurrently with crossfit (~1 min / fold)
    python -m src.l2_train holdout                     # ~3 min (level-2 part), 7.7 GB peak
    L2_THREADS=8 python -m src.l2_train train --tag f0123 --num-leaves 63 --rounds 800 \
        --checkpoints 200,300,400,600,800 --lr 0.05 --folds 0,1,2,3     # 3 min train, 8.9 GB peak
    python -m src.l2_train finalize --tag f0123 --iteration 600
    # training: 3,509,054 rows, 702,007 entities, pos rate 0.6970
    # holdout F0.5 (180,091 entities): raw E039 0.984231; @200 0.985327, @400 0.985385,
    # @600 0.985458 (+0.00123, the shipped model), @800 0.985504.  experiments/L2_E044 == final_E044.
    # (The research run also passed the per-entity F0.5 of the E042p holdout-CV residual,
    #  entf_prod_ALL_nc_s0.npy, which only fills the vs_cv / se_vs_cv fields: add --cv-ref FILE.)

Notes: ``features`` and ``holdout`` must use the same ``--work`` as ``crossfit`` (they read its
oof_fold{k}.parquet / write next to it).  ``--wait`` only matters when ``features`` runs concurrently
with ``crossfit`` (it polls for each fold file); run sequentially it is a no-op.

Re-evaluate a model dir without training (same code as the train step):
    python -m src.l2_train evaluate --model-dir experiments/L2_E044 --checkpoints 600

Port verification (EC2, against the research artefacts, 2 threads): evaluate(final_E044 @600) and
evaluate(model_f0123 @200..800) reproduce eval.json and the per-entity F0.5 arrays exactly (F 0.985458);
finalize(model_f0123, 600) is byte-identical to final_E044 (all 5 files); holdout level-2 features
(98 columns x 4,230,376 rows) and features for cross-fit fold 0 are bit-identical; crossfit_design
reproduces the OOF ids/labels and fold sizes, and the saved fold models reproduce the OOF prob
exactly; 5-round runs of train / crossfit give the same trees as the research models (leaf values
within 1e-12, thread-count rounding).

Needs: data/processed/feats_v5c_ancz_{India,US}.parquet, entities_v5c_ancz.npy, trainT_s{1,2,3}_*.parquet,
data/interim/{train_s1,train_s2,train_s3}.parquet + gt_pair_{s1,m}.npy, experiments/E039_stage3/
{features.json, holdout_pred.parquet} (all produced by scripts/aws/run_e039.sh and src/audit_gt.py).
Test-time application: ``scripts/apply_l2.py --model-dir experiments/L2_E044`` (see scripts/run_day3_l2b.sh).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import level2 as L2
from .decision import select_sets
from .inference import enforce_exclusivity
from .metrics import f05_single
from .train import LGB_PARAMS, entity_fold, truth_for

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
EXPERIMENTS = ROOT / "experiments"
WORK = EXPERIMENTS / "E044"
COUNTRIES = ("India", "US")           # France has no labels: it keeps the stage-3 prob at test time
SUB = 1e-3                            # the residual is trained on / applied to rows with prob >= SUB
N_CROSSFIT = 4                        # stage-3 cross-fit folds inside the training entities

# Residual LightGBM parameters (final_push/common.py LGB2); train overrides num_threads, seed=0,
# num_leaves and learning_rate from the command line.
LGB2 = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=127, num_threads=2, verbose=-1, seed=0)

_T0 = time.time()


def log(*a) -> None:
    """Timestamped progress line (wall clock + seconds since start)."""
    print(time.strftime("%H:%M:%S"), "%6.0fs" % (time.time() - _T0), *a, flush=True)


def feats114() -> list[str]:
    """The 114 E039 stage-3 feature names in model order."""
    return json.loads((EXPERIMENTS / "E039_stage3" / "features.json").read_text())


def holdout_entities(tag: str = "v5c_ancz") -> np.ndarray:
    """E039 holdout S1 entities: entity_fold(e, 5) == 0 of data/processed/entities_<tag>.npy."""
    e = np.load(PROCESSED / f"entities_{tag}.npy")
    return e[entity_fold(e, 5) == 0]


def _logit(p: np.ndarray) -> np.ndarray:
    """logit(p) in float64 with p clipped to [1e-7, 1 - 1e-7] (the residual's init_score)."""
    p = np.clip(np.asarray(p, np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Atomic parquet write (partial file, then rename) so an interrupted run never leaves a truncated file."""
    tmp = path.with_suffix(".partial")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


# ============================================================================ crossfit
def crossfit_design(tag: str = "v5c_ancz"):
    """Design of the stage-3 cross-fit: X (114 E039 features, float32), y, ids and the fold split of every
    TRAINING row (entity_fold(s1, 5) != 0) of feats_<tag>_{India,US}, India rows first, file order.
    Returns dict(X, y, s1, cand, row, cty (0 India / 1 US), inner (early-stopping rows), cf (cross-fit fold))."""
    FEATS = feats114()
    meta = []
    for ci, c in enumerate(COUNTRIES):
        t = pq.read_table(PROCESSED / f"feats_{tag}_{c}.parquet", columns=["s1", "cand"])
        s1 = t.column("s1").to_numpy()
        cand = t.column("cand").to_numpy()
        rows = np.flatnonzero(entity_fold(s1, 5) != 0)        # holdout entities (fold 0) never enter
        meta.append((c, ci, rows, s1[rows], cand[rows]))
        log(c, "rows", len(s1), "training rows", len(rows), "training entities", len(np.unique(s1[rows])))
    n = sum(len(m[2]) for m in meta)
    X = np.empty((n, len(FEATS)), np.float32)
    y = np.empty(n, np.float32)
    off = 0
    for c, ci, rows, _, _ in meta:                             # column by column to bound memory
        path = PROCESSED / f"feats_{tag}_{c}.parquet"
        for j, col in enumerate(FEATS):
            X[off:off + len(rows), j] = pq.read_table(path, columns=[col]).column(col).to_numpy().astype(
                np.float32, copy=False)[rows]
        y[off:off + len(rows)] = pq.read_table(path, columns=["label"]).column("label").to_numpy()[rows]
        off += len(rows)
    s1_all = np.concatenate([m[3] for m in meta])
    assert (entity_fold(s1_all, 5) != 0).all()
    return dict(X=X, y=y, s1=s1_all, cand=np.concatenate([m[4] for m in meta]),
                row=np.concatenate([m[2] for m in meta]),
                cty=np.concatenate([np.full(len(m[2]), m[1], np.int8) for m in meta]),
                inner=entity_fold(s1_all * 7 + 3, 10) == 0,     # early-stopping entities (as src/train.py)
                cf=entity_fold(s1_all * 17 + 3, N_CROSSFIT))     # cross-fit fold of each entity


def cmd_crossfit(a) -> None:
    """E044 step 1: 4-fold cross-fit of E039 stage 3 within the training entities -> OOF prob."""
    import lightgbm as lgb
    D = Path(a.work)
    D.mkdir(parents=True, exist_ok=True)
    FEATS = feats114()
    Z = crossfit_design(a.tag)
    X, y, inner, cf = Z["X"], Z["y"], Z["inner"], Z["cf"]
    s1_all, cand_all, row_all, cty_all = Z["s1"], Z["cand"], Z["row"], Z["cty"]
    del Z
    log("X", X.shape, "GB %.1f" % (X.nbytes / 1e9), "pos rate %.4f" % y.mean(), "inner frac %.3f" % inner.mean(),
        "fold sizes", np.bincount(cf).tolist())
    params = dict(LGB_PARAMS, num_threads=a.threads)
    dall = lgb.Dataset(X, y, feature_name=FEATS, params=params, free_raw_data=True).construct()
    log("dataset constructed")
    res = []
    for k in range(N_CROSSFIT):
        tk = time.time()
        tr_idx = np.flatnonzero((cf != k) & ~inner)
        es_idx = np.flatnonzero((cf != k) & inner)
        te_idx = np.flatnonzero(cf == k)
        dtr = dall.subset(tr_idx)
        dva = dall.subset(es_idx)
        mdl = lgb.train(params, dtr, num_boost_round=a.rounds, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])
        mdl.save_model(str(D / f"stage3_fold{k}.txt"))
        prob = mdl.predict(X[te_idx], num_iteration=mdl.best_iteration).astype(np.float32)
        out = pd.DataFrame({"country": np.where(cty_all[te_idx] == 0, "India", "US"), "row": row_all[te_idx],
                            "s1": s1_all[te_idx], "cand": cand_all[te_idx], "label": y[te_idx].astype(np.int8),
                            "prob": prob})
        _write_parquet(out, D / f"oof_fold{k}.parquet")
        yy = y[te_idx]
        pp = np.clip(prob.astype(np.float64), 1e-7, 1 - 1e-7)
        ll = float(-np.mean(yy * np.log(pp) + (1 - yy) * np.log(1 - pp)))
        res.append(dict(fold=k, best_iteration=mdl.best_iteration, train_rows=len(tr_idx), es_rows=len(es_idx),
                        pred_rows=len(te_idx), oof_logloss=ll, minutes=round((time.time() - tk) / 60, 1)))
        log("FOLD", res[-1])
        del dtr, dva, mdl, out
    json.dump(res, open(D / "crossfit_folds.json", "w"), indent=1)
    log("done")


# ============================================================================ features
def cmd_features(a) -> None:
    """E044 step 2: production level-2 features of the training rows, prob = OOF stage-3 prob."""
    D = Path(a.work)
    T = {c: L2.Tables(a.split, c, root=ROOT, log=log) for c in COUNTRIES}
    for k in range(N_CROSSFIT):
        f = D / f"oof_fold{k}.parquet"
        while a.wait and not f.exists():                       # original ran concurrently with crossfit
            time.sleep(20)
        if not f.exists():
            raise SystemExit(f"missing {f} (run crossfit first, or pass --wait)")
        O = pd.read_parquet(f)
        for c in COUNTRIES:
            dst = D / f"l2_fold{k}_{c}.parquet"
            if dst.exists():
                continue
            Oc = O[O.country == c].reset_index(drop=True)
            rows = Oc.row.to_numpy()
            assert (np.diff(rows) > 0).all()
            t = pq.read_table(PROCESSED / f"feats_{a.tag}_{c}.parquet",
                              columns=["s1", "cand", "src", "name_tset", "addr_tset", "addr_empty_c"]).take(rows).to_pandas()
            assert (t.s1.to_numpy() == Oc.s1.to_numpy()).all() and (t.cand.to_numpy() == Oc.cand.to_numpy()).all()
            t["prob"] = Oc.prob.to_numpy().astype(np.float32)
            tk = time.time()
            F = L2.features(t, T[c], as_dict=True)
            m = t.prob.to_numpy() >= SUB
            out = pd.DataFrame({"row": rows[m], "s1": Oc.s1.to_numpy()[m], "cand": Oc.cand.to_numpy()[m],
                                "label": Oc.label.to_numpy()[m], "prob": Oc.prob.to_numpy()[m]})
            for col in L2.L2_COLS:
                out[col] = F[col][m]
            _write_parquet(out, dst)
            log(f"fold {k} {c}: {len(t):,} rows, {len(np.unique(rows)):,}, entities {Oc.s1.nunique():,}, "
                f"saved {int(m.sum()):,} rows prob>=1e-3 ({time.time() - tk:.0f}s)")
            del t, F, out
    log("done")


# ============================================================================ holdout
def build_holdout_table(tag: str = "v5c_ancz", cols: list[str] | None = None) -> pd.DataFrame:
    """Holdout rows of the E039 stage-3 table (India then US, file order) + the E039 holdout prob.
    cols=None -> all 114 features.  float64 columns are stored as float32 (as the research cache)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    ents = pa.array(holdout_entities(tag))
    FEATS = feats114()
    want = ["s1", "cand", "src", "label"] + (FEATS if cols is None else
                                             [c for c in cols if c not in ("s1", "cand", "src", "label")])
    parts = []
    for c in COUNTRIES:
        t = pq.read_table(PROCESSED / f"feats_{tag}_{c}.parquet", columns=[x for x in dict.fromkeys(want)])
        t = t.filter(pc.is_in(t.column("s1"), value_set=ents)).to_pandas()
        t["country"] = c
        parts.append(t)
    H = pd.concat(parts, ignore_index=True)
    hp = pd.read_parquet(EXPERIMENTS / "E039_stage3" / "holdout_pred.parquet", columns=["s1", "cand", "prob"])
    H = H.merge(hp, on=["s1", "cand"], how="inner")
    for c in H.columns:
        if H[c].dtype == np.float64:
            H[c] = H[c].astype(np.float32)
    return H


def holdout_l2_features(H: pd.DataFrame, split: str = "trainT", n_entities: int = 0) -> pd.DataFrame:
    """Production level-2 features of the holdout table H (per country, whole entities).
    n_entities > 0: only the first n entities (sorted codes) per country (quick check).
    Returns row (index into H), s1, cand + the 95 L2_COLS, sorted by row."""
    outs = []
    for c in COUNTRIES:
        ix = np.flatnonzero(H.country.to_numpy() == c)
        if n_entities:
            ents = np.unique(H.s1.to_numpy()[ix])[:n_entities]
            ix = ix[np.isin(H.s1.to_numpy()[ix], ents)]
        T = L2.Tables(split, c, root=ROOT, log=log)
        t1 = time.time()
        F = L2.features(H.iloc[ix].reset_index(drop=True), T)
        log(c, "features", len(ix), "rows %.0fs" % (time.time() - t1))
        F.index = ix
        outs.append(F)
        del T
    F = pd.concat(outs).sort_index()
    rows = F.index.to_numpy()
    F = F.reset_index(drop=True)
    F.insert(0, "row", rows)
    F.insert(1, "s1", H.s1.to_numpy()[rows])
    F.insert(2, "cand", H.cand.to_numpy()[rows])
    return F


def cmd_holdout(a) -> None:
    """E039 holdout table + its production level-2 features (inputs of the train-step evaluation)."""
    D = Path(a.work)
    D.mkdir(parents=True, exist_ok=True)
    if a.from_table:                   # reuse an existing holdout table (e.g. to rebuild only the L2 part)
        H = pd.read_parquet(a.from_table, columns=["s1", "cand", "src", "label", "country", "prob",
                                                   "name_tset", "addr_tset", "addr_empty_c"])
        log("holdout table read from", a.from_table, H.shape)
    else:
        H = build_holdout_table(a.tag)
        _write_parquet(H, D / "holdout.parquet")
        log("holdout table", H.shape, "->", D / "holdout.parquet")
        H = H[["s1", "cand", "src", "label", "country", "prob", "name_tset", "addr_tset", "addr_empty_c"]]
    F = holdout_l2_features(H, a.split, a.n_entities)
    dst = D / ("holdout_l2_q.parquet" if a.n_entities else "holdout_l2.parquet")
    _write_parquet(F, dst)
    log("saved", F.shape, "->", dst)


# ============================================================================ train / evaluate
def _holdout_matrix(cols: list[str], holdout: Path, holdout_l2: Path):
    """Holdout design matrix of the residual (rows prob >= SUB), in model column order.
    Returns (H ids frame, idx of the scored rows, Xh, E039 prob of all rows)."""
    H = pd.read_parquet(holdout, columns=["s1", "cand", "src", "label", "country", "prob"])
    ph = H.prob.to_numpy().astype(np.float64)
    idx = np.flatnonzero(ph >= SUB)
    t = pd.read_parquet(holdout_l2, columns=["s1", "cand"])
    assert (t.s1.to_numpy() == H.s1.to_numpy()).all() and (t.cand.to_numpy() == H.cand.to_numpy()).all(), \
        "holdout level-2 features are not row-aligned with the holdout table"
    del t
    PF = pq.ParquetFile(holdout_l2).schema_arrow.names
    Xh = np.empty((len(idx), len(cols)), np.float32)
    for j, col in enumerate(cols):
        if col == "prob":
            Xh[:, j] = ph[idx].astype(np.float32)
        elif col in PF:
            Xh[:, j] = pd.read_parquet(holdout_l2, columns=[col])[col].to_numpy(np.float32)[idx]
        else:
            Xh[:, j] = pd.read_parquet(holdout, columns=[col])[col].to_numpy(np.float32)[idx]
    return H, idx, Xh, ph


def evaluate_holdout(mdl, cols: list[str], checkpoints: list[int], out_dir: Path, tag: str, holdout: Path,
                     holdout_l2: Path, cv_ref: Path | None, threads: int, ent_tag: str = "v5c_ancz",
                     save_ref: bool = False, ref_check: Path | None = None) -> list[dict]:
    """Holdout F0.5 of the residual at each checkpoint with the submission rule (exclusivity + expF),
    against the full ground truth of every holdout entity (retrieval misses count).  Writes
    entf_it{it}.npy (per-entity F0.5) and eval.json to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    H, idx, Xh, ph = _holdout_matrix(cols, holdout, holdout_l2)
    s1a, ca = H.s1.to_numpy(), H.cand.to_numpy()
    lgh = _logit(ph)[idx]
    if save_ref:
        np.save(out_dir / "ref_X5000.npy", Xh[:5000])
    if ref_check is not None and ref_check.exists():
        R = np.load(ref_check)
        same = (R == Xh[:len(R)]) | (np.isnan(R) & np.isnan(Xh[:len(R)]))
        log(f"holdout matrix vs {ref_check}: {int((~same).sum())} of {same.size} values differ")
    ents = holdout_entities(ent_tag)
    T = truth_for(ents)
    n3 = len(ents) // 3
    nt = np.array([len(T[e]) for e in ents])
    cty_of = dict(zip(s1a.tolist(), H.country.tolist()))
    ec = np.array([cty_of.get(int(e), "?") for e in ents])

    def ent_f(prob):
        """Per-entity F0.5 of the holdout for pair probabilities ``prob``: submission rule (one-owner
        exclusivity + expected-F0.5 prefix) against the full ground truth, in ``ents`` order."""
        pp = prob.astype(np.float64)
        kk = enforce_exclusivity(s1a, ca, pp)
        sets_, _, _ = select_sets(s1a[kk], ca[kk], pp[kk])
        S = [set(np.asarray(sets_.get(int(e), [])).tolist()) for e in ents]
        return np.array([f05_single(S[i], T[e]) for i, e in enumerate(ents)]), S

    fr, Sr = ent_f(ph)
    fcv = np.load(cv_ref) if cv_ref is not None and Path(cv_ref).exists() else None
    log("raw F=%.6f (India %.6f US %.6f)" % (fr.mean(), fr[ec == "India"].mean(), fr[ec == "US"].mean())
        + ("; holdout-CV %s F=%.6f" % (Path(cv_ref).name, fcv.mean()) if fcv is not None else ""))
    res = []
    for it in checkpoints:
        raw = mdl.predict(Xh, raw_score=True, num_iteration=it, num_threads=threads)
        pn = ph.copy()
        pn[idx] = 1 / (1 + np.exp(-(lgh + raw)))
        fn, Sn = ent_f(pn)
        np.save(out_dir / f"entf_it{it}.npy", fn)
        chg = np.array([x != z for x, z in zip(Sr, Sn)])
        dl = _logit(pn[idx]) - lgh
        gate = dict(India_per1000=round(1000 * float(chg[ec == "India"].mean()), 2),
                    US_per1000=round(1000 * float(chg[ec == "US"].mean()), 2),
                    dl_mean=round(float(dl.mean()), 4), dl_sd=round(float(dl.std()), 4),
                    dl_p1=round(float(np.percentile(dl, 1)), 3), dl_p99=round(float(np.percentile(dl, 99)), 3),
                    dl_abs_gt1=round(float((np.abs(dl) > 1).mean()), 4))
        d = fn - fr
        grp = {g: round(float(fn[mm].mean()), 5) for g, mm in
               (("0", nt == 0), ("1", nt == 1), ("2-3", (nt >= 2) & (nt <= 3)), ("4+", nt >= 4))}
        r = dict(tag=tag, it=it, F=round(float(fn.mean()), 6), gain_vs_raw=round(float(d.mean()), 5),
                 se=round(float(d.std() / np.sqrt(len(d))), 5))
        if fcv is not None:
            dc = fn - fcv
            r.update(vs_cv=round(float(dc.mean()), 5), se_vs_cv=round(float(dc.std() / np.sqrt(len(dc))), 5))
        else:
            r.update(vs_cv=None, se_vs_cv=None)
        r.update(first3=round(float(d[:n3].mean()), 5), rest=round(float(d[n3:].mean()), 5),
                 India=round(float(fn[ec == "India"].mean()), 6), US=round(float(fn[ec == "US"].mean()), 6),
                 gain_India=round(float(d[ec == "India"].mean()), 5), gain_US=round(float(d[ec == "US"].mean()), 5),
                 groups=grp, changed=int((d != 0).sum()), gates=gate)
        res.append(r)
        log("EVAL", json.dumps(r))
    json.dump(res, open(out_dir / "eval.json", "w"), indent=1)
    return res


def train_design(work: Path, tag: str = "v5c_ancz", folds=(0, 1, 2, 3)):
    """Residual training matrix: rows of <work>/l2_fold{k}_{country}.parquet (OOF prob >= 1e-3), India then
    US, folds in the given order; columns = model_columns(E039 features, "ALL_nc") (prob = OOF prob,
    level-2 columns from the fold files, the E039 stage-3 features from feats_<tag>_<country> by row).
    Returns X (float32), y, OOF prob (float64), column list, number of entities."""
    cols = L2.model_columns(feats114(), "ALL_nc")
    assert len(cols) == 204
    l2set = set(L2.L2_COLS)
    Xs, ys, ps, n_ent = [], [], [], 0
    for c in COUNTRIES:
        L = pd.concat([pd.read_parquet(Path(work) / f"l2_fold{k}_{c}.parquet") for k in folds], ignore_index=True)
        assert (entity_fold(L.s1.to_numpy(), 5) != 0).all()          # no holdout entity
        rows = L.row.to_numpy()
        n_ent += L.s1.nunique()
        Xc = np.empty((len(L), len(cols)), np.float32)
        for j, col in enumerate(cols):
            if col == "prob":
                Xc[:, j] = L.prob.to_numpy(np.float32)
            elif col in l2set:
                Xc[:, j] = L[col].to_numpy(np.float32)
            else:                                                     # E039 stage-3 feature of the same row
                Xc[:, j] = pq.read_table(PROCESSED / f"feats_{tag}_{c}.parquet", columns=[col]).column(
                    col).to_numpy().astype(np.float32)[rows]
        Xs.append(Xc)
        ys.append(L.label.to_numpy().astype(np.float32))
        ps.append(L.prob.to_numpy().astype(np.float64))
        del L
    X = np.vstack(Xs)
    del Xs
    return X, np.concatenate(ys), np.concatenate(ps), cols, n_ent


def cmd_train(a) -> None:
    """E044 step 3: residual ALL_nc on the training entities, init = logit(OOF prob); holdout evaluation."""
    import lightgbm as lgb
    E = Path(a.work)
    folds = [int(x) for x in a.folds.split(",")]
    cps = [int(x) for x in a.checkpoints.split(",")]
    X, y, p, cols, n_ent = train_design(E, a.tag, folds)
    init = _logit(p)
    log("train rows", len(y), "entities (with >=1 row prob>=1e-3)", n_ent, "pos rate %.4f" % y.mean(),
        "X GB %.2f" % (X.nbytes / 1e9))
    prm = dict(LGB2, num_threads=a.threads, seed=0, num_leaves=a.num_leaves, learning_rate=a.lr)
    mdl = lgb.train(prm, lgb.Dataset(X, y, init_score=init, feature_name=cols, free_raw_data=True), a.rounds)
    del X
    md = E / f"model_{a.tag_model}"
    md.mkdir(parents=True, exist_ok=True)
    mdl.save_model(str(md / "model.txt"))
    json.dump(cols, open(md / "features.json", "w"))
    log("trained", a.rounds, "rounds, params", prm, "folds", folds)
    # ---------------- holdout (untouched): E039 prob + production features
    evaluate_holdout(mdl, cols, cps, md, a.tag_model, Path(a.holdout or E / "holdout.parquet"),
                     Path(a.holdout_l2 or E / "holdout_l2.parquet"), Path(a.cv_ref) if a.cv_ref else None,
                     a.threads, save_ref=True)
    imp = mdl.feature_importance("gain")
    top = sorted(zip(imp, cols), reverse=True)[:20]
    log("top gain", [(f, round(g / imp.sum(), 3)) for g, f in top])
    log("done")


def cmd_evaluate(a) -> None:
    """Holdout evaluation (the train step's code) of an existing residual model directory."""
    import lightgbm as lgb
    E = Path(a.work)
    md = Path(a.model_dir)
    cols = json.loads((md / "features.json").read_text())
    assert cols == L2.model_columns(feats114(), "ALL_nc"), "features.json is not the ALL_nc column order"
    mdl = lgb.Booster(model_file=str(md / "model.txt"))
    assert mdl.num_feature() == len(cols) == 204
    cps = [int(x) for x in a.checkpoints.split(",")]
    assert max(cps) <= mdl.num_trees(), (cps, mdl.num_trees())
    log("model", md, "trees", mdl.num_trees())
    evaluate_holdout(mdl, cols, cps, Path(a.out or E / f"eval_{md.name}"), a.label or md.name,
                     Path(a.holdout or E / "holdout.parquet"), Path(a.holdout_l2 or E / "holdout_l2.parquet"),
                     Path(a.cv_ref) if a.cv_ref else None, a.threads, ref_check=md / "ref_X5000.npy")
    log("done")


# ============================================================================ finalize
def cmd_finalize(a) -> None:
    """Save an E044 variant as a deployable model dir (model truncated to the chosen iteration)."""
    import lightgbm as lgb
    E = Path(a.work)
    tag, it = a.tag_model, a.iteration
    src = Path(a.src or E / f"model_{tag}")
    out = Path(a.out or E / "final_E044")
    out.mkdir(parents=True, exist_ok=True)
    full = lgb.Booster(model_file=str(src / "model.txt"))
    full.save_model(str(out / "model.txt"), num_iteration=it)
    cols = json.load(open(src / "features.json"))
    json.dump(cols, open(out / "features.json", "w"))
    b = lgb.Booster(model_file=str(out / "model.txt"))
    X = np.load(src / "ref_X5000.npy")
    r_full = full.predict(X, raw_score=True, num_iteration=it)
    r = b.predict(X, raw_score=True)
    assert b.num_trees() == it and b.num_feature() == len(cols) == 204, (b.num_trees(), b.num_feature())
    print("trees", b.num_trees(), "truncated vs full@it max diff", float(np.abs(r - r_full).max()))
    np.save(out / "ref_X5000.npy", X)
    np.save(out / "ref_raw5000.npy", r)
    ev = [e for e in json.load(open(src / "eval.json")) if e["it"] == it][0]
    prm = [ln for ln in open(src / "model.txt") if ln.startswith("[")]
    json.dump({"variant": "ALL_nc", "experiment": "E044", "tag": tag, "rounds": it,
               "train": "TRAINING entities of E039 (entity_fold(s1,5)!=0, 720k entities), cross-fitted stage-3 OOF "
                        "prob (4 folds entity_fold(s1*17+3,4), E039 params), rows OOF prob>=1e-3, init=logit(OOF prob), "
                        "production features src/level2.py (trainT)",
               "holdout_eval": ev, "lgb_params_lines": prm[:40]}, open(out / "meta.json", "w"), indent=1)
    print("saved", out, "holdout", json.dumps(ev))


# ============================================================================ CLI
def main(argv=None) -> None:
    """CLI: parse the sub-command (crossfit / features / holdout / train / evaluate / finalize) and run it."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--work", default=str(WORK), help="E044 work directory (default experiments/E044)")

    s = sub.add_parser("crossfit", parents=[common], help="4-fold cross-fit of E039 stage 3 within the training entities -> OOF prob")
    s.add_argument("--tag", default="v5c_ancz", help="stage-3 feature table data/processed/feats_<tag>_<country>")
    s.add_argument("--threads", type=int, default=8)
    s.add_argument("--rounds", type=int, default=3000, help="max boosting rounds per fold (early stopping 100)")
    s.set_defaults(fn=cmd_crossfit)

    s = sub.add_parser("features", parents=[common], help="production level-2 features of the training rows (per fold / country)")
    s.add_argument("--tag", default="v5c_ancz")
    s.add_argument("--split", default="trainT", help="data version of the level-2 tables")
    s.add_argument("--wait", action="store_true", help="poll for oof_fold{k}.parquet (run next to crossfit)")
    s.set_defaults(fn=cmd_features)

    s = sub.add_parser("holdout", parents=[common], help="E039 holdout table + its production level-2 features")
    s.add_argument("--tag", default="v5c_ancz")
    s.add_argument("--split", default="trainT")
    s.add_argument("--from-table", default=None, help="existing holdout table: rebuild only the level-2 features")
    s.add_argument("--n-entities", type=int, default=0, help="first n entities per country only (quick check)")
    s.set_defaults(fn=cmd_holdout)

    def eval_args(s):
        """Add the holdout-evaluation options shared by ``train`` and ``evaluate`` to sub-parser ``s``."""
        s.add_argument("--holdout", default=None, help="holdout table (default <work>/holdout.parquet)")
        s.add_argument("--holdout-l2", default=None, help="holdout level-2 features (default <work>/holdout_l2.parquet)")
        s.add_argument("--cv-ref", default=None,
                       help="optional per-entity F0.5 of the holdout-CV residual (research entf_prod_ALL_nc_s0.npy) "
                            "for the vs_cv columns")
        s.add_argument("--threads", type=int, default=int(os.environ.get("L2_THREADS", "8")))

    s = sub.add_parser("train", parents=[common], help="residual ALL_nc on the training entities + holdout evaluation")
    s.add_argument("--tag", dest="tag_model", required=True, help="model name -> <work>/model_<tag>/")
    s.add_argument("--num-leaves", type=int, default=63)
    s.add_argument("--rounds", type=int, default=800)
    s.add_argument("--checkpoints", default="200,300,400,600,800", help="comma list of tree counts to evaluate")
    s.add_argument("--lr", type=float, default=0.05)
    s.add_argument("--folds", default="0,1,2,3", help="cross-fit folds whose training rows are used")
    s.add_argument("--table-tag", dest="tag", default="v5c_ancz", help="stage-3 feature table tag")
    eval_args(s)
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("evaluate", parents=[common], help="holdout evaluation of an existing residual model dir")
    s.add_argument("--model-dir", required=True)
    s.add_argument("--checkpoints", default="600")
    s.add_argument("--out", default=None, help="output dir (default <work>/eval_<model dir name>)")
    s.add_argument("--label", default=None, help="tag written into eval.json (default: model dir name)")
    eval_args(s)
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("finalize", parents=[common], help="truncate a trained residual and write a deployable model dir")
    s.add_argument("--tag", dest="tag_model", required=True)
    s.add_argument("--iteration", type=int, required=True)
    s.add_argument("--src", default=None, help="model dir to finalize (default <work>/model_<tag>)")
    s.add_argument("--out", default=None, help="output dir (default <work>/final_E044)")
    s.set_defaults(fn=cmd_finalize)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
