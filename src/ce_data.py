"""E046: pair tables for the pretrained cross-encoder (a multilingual transformer fine-tuned as a
"same business?" classifier on (S1 record, candidate record) text pairs).

The cross-encoder only re-scores the pairs whose decision is still open after E039 stage 3 + the E044
residual, so every table carries the current probability next to the texts:

``holdout``  (EC2)  E039 holdout pairs with ``p39`` (stage 3) and ``pf`` = the shipped probability
                    (E044 correction: India full, US down-only).  Also prints the oracle value of each
                    probability band (F0.5 if the pairs of the band were scored perfectly).
                    -> experiments/E046/holdout_pairs.parquet
``train``    (EC2)  training-entity pairs (E039 training entities, 4-fold cross-fitted OOF stage-3 prob
                    ``p3`` from E044), band-sampled.  -> experiments/E046/train_pairs.parquet
``test``   (laptop) test pairs from a scores directory (the shipped day3_frdown probabilities).
                    -> experiments/E046/test_pairs.parquet
``texts``           attaches the record texts of both sides (raw name / address as in the dataset, plus the
                    native-script dictionary transliteration of names that are not ASCII).

Codes: S1/S2/S3 ids are stored as ``source * 10**10 + number`` (data/interim/*_s{1,2,3}.parquet).
"""
from __future__ import annotations

import argparse
import json
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
INTERIM = ROOT / "data" / "interim"
EXPERIMENTS = ROOT / "experiments"
OUT = EXPERIMENTS / "E046"
_T0 = time.time()


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), "%6.0fs" % (time.time() - _T0), *a, flush=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


# ============================================================================ holdout
def holdout_table(holdout: Path, holdout_l2: Path, model_dir: Path) -> pd.DataFrame:
    """E039 holdout pairs with the stage-3 prob and the shipped (E044-corrected) prob."""
    import lightgbm as lgb

    from .l2_train import SUB, _holdout_matrix, _logit

    cols = json.loads((model_dir / "features.json").read_text())
    mdl = lgb.Booster(model_file=str(model_dir / "model.txt"))
    H, idx, Xh, ph = _holdout_matrix(cols, holdout, holdout_l2)
    ref = model_dir / "ref_X5000.npy"
    if ref.exists():
        R = np.load(ref)
        same = (R == Xh[:len(R)]) | (np.isnan(R) & np.isnan(Xh[:len(R)]))
        log(f"holdout matrix vs ref_X5000: {int((~same).sum())} of {same.size} values differ")
    raw = mdl.predict(Xh, raw_score=True, num_threads=8)
    del Xh
    pc_ = ph.copy()
    pc_[idx] = _sigmoid(_logit(ph[idx]) + raw)
    us = H.country.to_numpy() == "US"
    pf = np.where(us, np.minimum(ph, pc_), pc_)
    out = H[["s1", "cand", "src", "country", "label"]].copy()
    out["p39"] = ph.astype(np.float32)
    out["pf"] = pf.astype(np.float32)
    log(f"holdout pairs {len(out):,}; scored by the residual {len(idx):,} (prob >= {SUB})")
    return out


def entity_scorer(D: pd.DataFrame):
    """Per-entity F0.5 with the submission rule (one-owner exclusivity + expected-F0.5 prefix) against the
    full ground truth of every holdout entity (retrieval misses count)."""
    from .decision import select_sets
    from .inference import enforce_exclusivity
    from .l2_train import holdout_entities
    from .metrics import f05_single
    from .train import truth_for

    ents = holdout_entities()
    T = truth_for(ents)
    s1a, ca = D.s1.to_numpy(), D.cand.to_numpy()
    cty = dict(zip(s1a.tolist(), D.country.tolist()))
    ec = np.array([cty.get(int(e), "?") for e in ents])

    def ent_f(prob: np.ndarray) -> np.ndarray:
        pp = np.asarray(prob, np.float64)
        kk = enforce_exclusivity(s1a, ca, pp)
        sets_, _, _ = select_sets(s1a[kk], ca[kk], pp[kk])
        return np.array([f05_single(set(np.asarray(sets_.get(int(e), [])).tolist()), T[e]) for e in ents])

    return ent_f, ents, ec


def cmd_holdout(a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    D = holdout_table(Path(a.holdout), Path(a.holdout_l2), Path(a.model_dir))
    ent_f, ents, ec = entity_scorer(D)
    pf, y = D.pf.to_numpy().astype(np.float64), D.label.to_numpy()
    f0 = ent_f(pf)
    log("shipped holdout F=%.6f  India %.6f  US %.6f  (%d entities)" % (
        f0.mean(), f0[ec == "India"].mean(), f0[ec == "US"].mean(), len(ents)))
    f39 = ent_f(D.p39.to_numpy())
    log("stage-3 only F=%.6f" % f39.mean())
    res = {"F_shipped": float(f0.mean()), "F_stage3": float(f39.mean()), "bands": []}
    for lo, hi in [(1e-3, 0.999), (3e-3, 0.997), (0.01, 0.99), (0.02, 0.98), (0.05, 0.95), (0.1, 0.9),
                   (0.999, 1.01), (0.99, 1.01), (1e-4, 1e-3), (0.0, 1e-4)]:
        m = (pf >= lo) & (pf < hi)
        po = pf.copy()
        po[m] = y[m]
        fo = ent_f(po)
        r = dict(lo=lo, hi=hi, pairs=int(m.sum()), pos=int(y[m].sum()), F_oracle=round(float(fo.mean()), 6),
                 gain=round(float(fo.mean() - f0.mean()), 6),
                 gain_India=round(float(fo[ec == "India"].mean() - f0[ec == "India"].mean()), 6),
                 gain_US=round(float(fo[ec == "US"].mean() - f0[ec == "US"].mean()), 6))
        res["bands"].append(r)
        log("oracle band", json.dumps(r))
    (OUT / "holdout_oracle.json").write_text(json.dumps(res, indent=1))
    D.to_parquet(OUT / "holdout_pairs.parquet", index=False)
    np.save(OUT / "holdout_entf_shipped.npy", f0)
    log("->", OUT / "holdout_pairs.parquet")


# ============================================================================ train
def cmd_train(a) -> None:
    """Training-entity pairs with the cross-fitted OOF stage-3 prob, band-sampled:
    all pairs with lo <= p3 < hi, plus a ``--easy-frac`` sample of the pairs outside it (p3 >= 1e-3)."""
    OUT.mkdir(parents=True, exist_ok=True)
    parts = []
    for k in range(4):
        t = pd.read_parquet(Path(a.oof_dir) / f"oof_fold{k}.parquet", columns=["country", "s1", "cand", "label", "prob"])
        t = t[t.prob >= 1e-3]
        parts.append(t)
    T = pd.concat(parts, ignore_index=True)
    del parts
    p = T.prob.to_numpy()
    band = (p >= a.lo) & (p < a.hi)
    rng = np.random.default_rng(46)
    easy = ~band & (rng.random(len(T)) < a.easy_frac)
    T = T[band | easy].reset_index(drop=True)
    T["src"] = (T.cand.to_numpy() // 10**10).astype(np.int8)
    T = T.rename(columns={"prob": "p3"})
    T["p3"] = T.p3.astype(np.float32)
    T["band"] = ((T.p3 >= a.lo) & (T.p3 < a.hi)).astype(np.int8)
    log(f"training pairs {len(T):,} (band {int(band.sum()):,}, easy sample {int(easy.sum()):,}); "
        f"pos rate {T.label.mean():.3f}; by country {T.country.value_counts().to_dict()}")
    T.to_parquet(OUT / "train_pairs.parquet", index=False)
    log("->", OUT / "train_pairs.parquet")


# ============================================================================ test
def cmd_test(a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    parts = []
    for f in sorted(Path(a.scores).glob("*_p*.parquet")):
        country = f.name.split("_p")[0]
        t = pq.read_table(f)
        t = t.filter(pc.greater_equal(t.column("prob"), a.lo)).to_pandas()
        t["country"] = country
        parts.append(t)
    T = pd.concat(parts, ignore_index=True).rename(columns={"prob": "pf"})
    log(f"test pairs with pf >= {a.lo}: {len(T):,}; by country {T.country.value_counts().to_dict()}; "
        f"band [{a.lo}, {a.hi}): {int(((T.pf >= a.lo) & (T.pf < a.hi)).sum()):,}")
    T.to_parquet(OUT / "test_pairs.parquet", index=False)
    log("->", OUT / "test_pairs.parquet")


# ============================================================================ texts
def _clean(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def _native(s: str) -> bool:
    """True if the name contains a letter outside the Latin script (Devanagari, Bengali, ...).  Accented
    Latin names (French) are not native-script: the model reads them as they are."""
    return (not s.isascii()) and any(ch.isalpha() and "LATIN" not in unicodedata.name(ch, "") for ch in s)


def _texts(split: str, codes: np.ndarray, source: int) -> pd.DataFrame:
    """Raw name / address of the given codes of one source, plus the transliterated normalized name for
    native-script names, indexed by code."""
    want = pa.array(np.unique(codes))
    t = pq.read_table(INTERIM / f"{split}_s{source}.parquet", columns=["code", "business_name", "business_address"])
    t = t.filter(pc.is_in(t.column("code"), value_set=want)).to_pandas().set_index("code")
    t["business_name"] = _clean(t.business_name)
    t["business_address"] = _clean(t.business_address)
    nonascii = t.business_name.map(_native)
    tr = []
    for c in ("India", "US", "France"):
        f = PROCESSED / f"{split}T_s{source}_{c}.parquet"
        if f.exists():
            x = pq.read_table(f, columns=["code", "name_norm", "nonascii"])
            x = x.filter(pc.and_(pc.equal(x.column("nonascii"), 1), pc.is_in(x.column("code"), value_set=want)))
            tr.append(x.select(["code", "name_norm"]).to_pandas())
    tr = pd.concat(tr).drop_duplicates("code").set_index("code").name_norm if tr else pd.Series(dtype=str)
    t["translit"] = ""
    idx = t.index[nonascii.to_numpy()]
    t.loc[idx, "translit"] = tr.reindex(idx).fillna("").to_numpy()
    return t


def cmd_texts(a) -> None:
    D = pd.read_parquet(a.pairs)
    A = _texts(a.split, D.s1.to_numpy(), 1)
    D["a_name"] = A.business_name.reindex(D.s1).to_numpy()
    D["a_addr"] = A.business_address.reindex(D.s1).to_numpy()
    D["a_tr"] = A.translit.reindex(D.s1).to_numpy()
    del A
    for c in ("b_name", "b_addr", "b_tr"):
        D[c] = ""
    for s in (2, 3):
        m = (D.cand.to_numpy() // 10**10) == s
        B = _texts(a.split, D.cand.to_numpy()[m], s)
        ids = D.cand.to_numpy()[m]
        D.loc[m, "b_name"] = B.business_name.reindex(ids).to_numpy()
        D.loc[m, "b_addr"] = B.business_address.reindex(ids).to_numpy()
        D.loc[m, "b_tr"] = B.translit.reindex(ids).to_numpy()
        del B
    miss = D.a_name.isna() | D.b_name.isna()
    log(f"{len(D):,} pairs; missing texts {int(miss.sum())}; non-ASCII S1 names {int((D.a_tr != '').sum()):,}, "
        f"candidate names {int((D.b_tr != '').sum()):,}")
    for c in ("a_name", "a_addr", "a_tr", "b_name", "b_addr", "b_tr"):
        D[c] = D[c].fillna("")
    D = D.sort_values(["s1", "cand"]).reset_index(drop=True)
    pq.write_table(pa.Table.from_pandas(D, preserve_index=False), a.out, compression="zstd", compression_level=9)
    log("->", a.out, f"{Path(a.out).stat().st_size / 1e6:.1f} MB")
    print(D.sample(6, random_state=1)[["a_name", "a_addr", "b_name", "b_addr", "b_tr"]].to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("holdout")
    s.add_argument("--holdout", required=True)
    s.add_argument("--holdout-l2", required=True)
    s.add_argument("--model-dir", default=str(EXPERIMENTS / "L2_E044"))
    s = sub.add_parser("train")
    s.add_argument("--oof-dir", required=True)
    s.add_argument("--lo", type=float, default=0.01)
    s.add_argument("--hi", type=float, default=0.99)
    s.add_argument("--easy-frac", type=float, default=0.25)
    s = sub.add_parser("test")
    s.add_argument("--scores", default=str(EXPERIMENTS / "E039_test" / "scores_frdown"))
    s.add_argument("--lo", type=float, default=1e-3)
    s.add_argument("--hi", type=float, default=0.999)
    s = sub.add_parser("texts")
    s.add_argument("--split", required=True, choices=["train", "test"])
    s.add_argument("--pairs", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    {"holdout": cmd_holdout, "train": cmd_train, "test": cmd_test, "texts": cmd_texts}[a.cmd](a)


if __name__ == "__main__":
    main()
