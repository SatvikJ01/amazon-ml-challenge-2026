"""Level-2 RESIDUAL features for the E039 stage-3 pairs (production port of the final-push research).

Research sources (EC2, holdout only, row-aligned with final_push/common.load_holdout()):
  CTX  level2-context/build_ctx.py         61 entity-context columns (``logit`` is not a model input)
  ED   error-mining/feats.py               the 8 ``ed_*`` edit-type columns only
  EA   empty-address/03_feats.py           21 columns
       empty-address/09_tiefeat.py         5 columns (sup_*)
Column names and semantics are identical, with two deliberate differences:
  * the two random competitor samples (03_feats: 30 S1 of the same exact-name group; 09_tiefeat: 40
    superset S1) are replaced by a deterministic, evenly spaced subset of the same size (identical to
    the research value whenever the group is not larger than the cap);
  * row-level text features are computed only for the rows the residual model scores
    (``rows``; default prob >= SUB) and are NaN elsewhere.  Entity-level aggregates always use every
    candidate of the entity.

``features(df, tables)`` takes WHOLE entities of ONE country (every S1's full candidate list; the
pass-2 chunk files are entity-disjoint) with columns
    s1, cand, src, prob (E039 stage-3 probability = prob3), name_tset, addr_tset, addr_empty_c
and returns the level-2 columns (float32, same row order).  ``Tables(split, country)`` holds the
per-country tables (full S1 table, exact-name groups, word inverted index, empty-address target pool,
target text hashes) built from the FULL split data of that country: split ``trainT`` for the holdout,
``testT`` for the test (raw names from data/interim/{train,test}_s{1,2,3}.parquet).

Residual model: prob_new = sigmoid(logit(prob) + booster.predict(X, raw_score=True)) for rows with
prob >= SUB (others keep prob); X columns = ``model_columns(E039 features, variant)`` (comb.py order).
"""
from __future__ import annotations

import os
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUB = 1e-3                 # residual applies to rows with prob >= SUB
CAP_COMP, CAP_SUP = 30, 40  # competitor subset sizes (research: random samples of this size)

_P = ("n", "max_other", "sum_other", "rank", "gap_top", "share")
CTX_COLS = ([f"e_{c}" for c in _P] + ["e_n10_other", "e_n50_other", "e_n90_other", "e_gap_next", "e_gap_prev",
                                       "e_expcnt_minus_rank"]
            + [f"{g}_{c}" for g in ("es", "dn", "da", "dd", "dnd", "dsim") for c in _P + ("n50_other",)]
            + ["xs_name_max_other_src", "xs_namedig_max_other_src", "ent_empty_psum", "ent_nonempty_psum",
               "is_empty", "src", "is_us"])
ED_COLS = ["ed_ocr", "ed_anagram", "ed_append", "ed_drop", "ed_sub1", "ed_other", "ed_minlen", "ed_short_sub"]
EA1_COLS = ["raw_eq", "rawl_eq", "rawl_ratio", "cand_nn_s1_n", "pool_empty_nn_n", "pool_empty_per_s1",
            "comp_rawl_best", "rawl_margin", "comp_rawl_eq_n", "sib_rawl_eq", "sib_nn_var_eq", "sib_rawl_best",
            "name_digit_supported", "n_conf_src_other", "n_conf_othersrc", "sum_p_src_other", "src_cap_room",
            "ent_empty_sum_p_other", "ent_empty_conf_other", "ent_conf_n", "empty_prob_rank"]
EA2_COLS = ["sup_n", "sup_self", "sup_n_other", "sup_best_other", "sup_margin"]
NC = {"cand_nn_s1_n", "pool_empty_nn_n", "comp_rawl_eq_n", "sup_n", "sup_n_other"}   # table-size counts
L2_COLS = CTX_COLS + ED_COLS + EA1_COLS + EA2_COLS
INPUT_COLS = ["s1", "cand", "src", "prob", "name_tset", "addr_tset", "addr_empty_c"]
assert len(CTX_COLS) == 61 and len(L2_COLS) == 95


def model_columns(feats114: list[str], variant: str = "ALL_nc") -> list[str]:
    """Exact column order of planner/comb.py for set ALL / ALL_nc (prob + CTX + ED + EA + remaining FEATS)."""
    ea1, ea2 = EA1_COLS, EA2_COLS
    if variant.endswith("_nc"):
        ea1, ea2 = [c for c in ea1 if c not in NC], [c for c in ea2 if c not in NC]
    cols = ["prob"] + CTX_COLS + ED_COLS + ea1 + ea2
    return cols + [f for f in feats114 if f not in cols]


def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def apply_residual(booster, X: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """prob (float) of the rows of X -> residual-corrected probability (float64)."""
    return 1 / (1 + np.exp(-(logit(prob) + booster.predict(X, raw_score=True))))


# ----------------------------------------------------------------------------- helpers
def _low(x):
    return " ".join(str(x).lower().split()) if isinstance(x, str) else ""


def _obj(arr) -> np.ndarray:
    """pyarrow (Chunked)Array -> numpy object array (None for nulls)."""
    return np.asarray(arr.to_pandas(), dtype=object)


def _fill(a: np.ndarray) -> np.ndarray:
    """pandas fillna("") on an object array (None / NaN -> "")."""
    a = a.copy()
    a[pd.isna(a)] = ""
    return a


def _align(src_codes: np.ndarray, dst_codes: np.ndarray) -> np.ndarray:
    """For every dst code, its index in src_codes (-1 if absent)."""
    if len(src_codes) == 0:
        return np.full(len(dst_codes), -1, np.int64)
    o = np.argsort(src_codes, kind="stable")
    sc = src_codes[o]
    p = np.minimum(np.searchsorted(sc, dst_codes), len(sc) - 1)
    return np.where(sc[p] == dst_codes, o[p], -1)


def _lookup(codes_sorted: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.minimum(np.searchsorted(codes_sorted, q), len(codes_sorted) - 1)
    return np.where(codes_sorted[p] == q, p, -1)


def _take(arr, pos: np.ndarray) -> np.ndarray:
    """pyarrow Array take with -1 -> None; returns an object array."""
    import pyarrow as pa
    return _obj(arr.take(pa.array(np.maximum(pos, 0), mask=pos < 0)))


def _subset(a: np.ndarray, cap: int) -> np.ndarray:
    """Deterministic evenly spaced subset of size cap (replaces rng.choice(a, cap, replace=False))."""
    return a if len(a) <= cap else a[(np.arange(cap) * len(a)) // cap]


_HASH_EMPTY = None


def _hash(a: np.ndarray) -> np.ndarray:
    return pd.util.hash_array(np.asarray(a, dtype=object))


DRE = re.compile(r"\d+")
DIG = re.compile(r"\d{3,}")


# ----------------------------------------------------------------------------- per-country tables
class Tables:
    """Per-country lookup tables from the FULL split data (split 'trainT' for the holdout, 'testT' for test)."""

    def __init__(self, split: str, country: str, root: str | Path | None = None, log=print):
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
        root = Path(root or os.environ.get("L2_ROOT", ROOT))
        proc, inter = root / "data" / "processed", root / "data" / "interim"
        rs = "train" if split.startswith("train") else "test"
        self.split, self.country, self.log = split, country, log
        t0 = time.time()
        # ---------------- S1 table (sorted by code)
        s1 = pq.read_table(proc / f"{split}_s1_{country}.parquet", columns=["code", "name_norm"])
        codes = s1.column("code").to_numpy()
        o = np.argsort(codes, kind="stable")
        self.s1_codes = codes[o]
        self.s1_nn = _obj(s1.column("name_norm"))[o]            # raw name_norm (None for null)
        del s1
        r = pq.read_table(inter / f"{rs}_s1.parquet", columns=["code", "business_name"],
                          filters=[("country", "==", country)])
        rp = _align(r.column("code").to_numpy(), self.s1_codes)
        self.s1_raw = _take(r.column("business_name").combine_chunks(), rp)
        del r
        self.s1_l = np.array([_low(x) for x in self.s1_raw] + [None], dtype=object)[:-1]
        # exact-name groups (value_counts / groupby of name_norm over the full S1 table)
        fac, uniq = pd.factorize(self.s1_nn)
        self.s1_fac = fac.astype(np.int64)
        self.grp_cnt = np.bincount(fac[fac >= 0], minlength=len(uniq)).astype(np.int64)
        go = np.argsort(fac, kind="stable")
        self.grp_order = go.astype(np.int64)
        self.grp_start = np.searchsorted(fac[go], np.arange(len(uniq)))
        self.s1_name_index = pd.Index(uniq)
        # word inverted index over S1 name_norm tokens (positions in s1_codes order, ascending)
        inv = defaultdict(list)
        for i, x in enumerate(self.s1_nn):
            if isinstance(x, str) and x:
                for w in set(x.split()):
                    inv[w].append(i)
        self.inv = {w: np.array(v, np.int32) for w, v in inv.items()}
        del inv
        log(f"  [L2 tables {country}/{split}] S1 {len(self.s1_codes):,} ({time.time() - t0:.0f}s)")
        # ---------------- targets (S2 + S3), sorted by code, deduplicated; addr_norm streamed (hashes only)
        files = [pq.ParquetFile(proc / f"{split}_s{s}_{country}.parquet") for s in (2, 3)]
        codes = np.concatenate([f.read(columns=["code"]).column("code").to_numpy() for f in files])
        o = np.argsort(codes, kind="stable")
        cs = codes[o]
        keep = np.ones(len(cs), bool)
        keep[1:] = cs[1:] != cs[:-1]
        o = o[keep]
        self.t_codes = cs[keep]
        del cs, keep
        name = pa.concat_arrays([f.read(columns=["name_norm"]).column("name_norm").combine_chunks() for f in files])
        name = name.take(pa.array(o))
        nf = len(codes)
        addr_h, dig_h, emp = np.empty(nf, np.uint64), np.empty(nf, np.uint64), np.empty(nf, bool)
        a0 = 0
        for f in files:                           # file order, then permuted like the codes
            for b in f.iter_batches(batch_size=500_000, columns=["addr_norm"]):
                ad = _fill(_obj(b.column(0)))
                addr_h[a0:a0 + len(ad)] = _hash(ad)
                dig_h[a0:a0 + len(ad)] = _hash(np.array([" ".join(sorted(DRE.findall(x))) for x in ad], dtype=object))
                emp[a0:a0 + len(ad)] = np.array([len(x) == 0 for x in ad], bool)
                a0 += len(ad)
                del ad
        assert a0 == nf
        self.t_addr_h, self.t_dig_h, self.t_empty = addr_h[o], dig_h[o], emp[o]
        del addr_h, dig_h, emp, codes, o, files
        nt = len(self.t_codes)
        self.t_name = name
        self.t_name_h = np.empty(nt, np.uint64)
        B = 500_000
        for a0 in range(0, nt, B):
            nm = _fill(_obj(name.slice(a0, B)))
            self.t_name_h[a0:a0 + len(nm)] = _hash(nm)
            del nm
        vc = pd.Series(_obj(name.filter(pa.array(self.t_empty)))).value_counts()   # empty-address target pool
        self.pool_index, self.pool_cnt = pd.Index(vc.index), vc.to_numpy().astype(np.int64)
        del vc
        rr = pa.concat_tables([pq.read_table(inter / f"{rs}_s{s}.parquet", columns=["code", "business_name"],
                                             filters=[("country", "==", country)]) for s in (2, 3)])
        rp = _align(rr.column("code").to_numpy(), self.t_codes)
        bn = rr.column("business_name").combine_chunks()
        del rr
        self.t_raw = bn.take(pa.array(np.maximum(rp, 0), mask=rp < 0))
        del bn, rp
        global _HASH_EMPTY
        _HASH_EMPTY = _hash(np.array([""], dtype=object))[0]
        self.hash_empty = _HASH_EMPTY
        self.dig_empty = _HASH_EMPTY
        self.sup_cache: dict = {}
        self.sup_cache_size = 0
        import gc
        gc.collect()
        pa.default_memory_pool().release_unused()
        log(f"  [L2 tables {country}/{split}] targets {nt:,}, empty-address pool {int(self.t_empty.sum()):,} "
            f"({time.time() - t0:.0f}s)")

    def superset(self, key: str) -> np.ndarray:
        """Positions (ascending) of the S1 whose name token set contains every token of ``key``."""
        v = self.sup_cache.get(key)
        if v is None:
            ws = key.split()
            if not ws:
                v = np.zeros(0, np.int32)
            else:
                empty = np.zeros(0, np.int32)
                lists = sorted((self.inv.get(w, empty) for w in set(ws)), key=len)
                v = lists[0]
                for l in lists[1:]:
                    if len(v) == 0:
                        break
                    v = np.intersect1d(v, l, assume_unique=True)
            if self.sup_cache_size > 15_000_000:      # bound the memo (~60 MB of positions)
                self.sup_cache, self.sup_cache_size = {}, 0
            self.sup_cache[key] = v
            self.sup_cache_size += len(v) + 16
        return v


# ----------------------------------------------------------------------------- CTX (build_ctx.py)
def _grp_stats(keys, p, prefix, thr=(0.1, 0.5, 0.9), want_rank=True, want_nb=False):
    n = len(p)
    order = np.lexsort([-p] + keys[::-1])  # last key is primary in lexsort; ties -> input row order
    ks = [k[order] for k in keys]
    brk = np.zeros(n, bool)
    brk[0] = True
    for k in ks:
        brk[1:] |= k[1:] != k[:-1]
    start_idx = np.flatnonzero(brk)
    gid = np.cumsum(brk) - 1
    ps = p[order]
    gsize = np.diff(np.r_[start_idx, n])
    gsum = np.add.reduceat(ps, start_idx)
    pos = np.arange(n) - start_idx[gid]
    gmax = ps[start_idx]
    second = np.where(gsize > 1, ps[np.minimum(start_idx + 1, n - 1)], 0.0)
    max_other = np.where(pos == 0, second[gid], gmax[gid])
    out = {prefix + "n": gsize[gid].astype(np.float32), prefix + "max_other": max_other,
           prefix + "sum_other": gsum[gid] - ps}
    if want_rank:
        out[prefix + "rank"] = (pos + 1).astype(np.float32)
        out[prefix + "gap_top"] = gmax[gid] - ps
        out[prefix + "share"] = ps / np.maximum(gsum[gid], 1e-9)
    for th in thr:
        c = np.add.reduceat((ps >= th).astype(np.float64), start_idx)
        out[prefix + f"n{int(th * 100):02d}_other"] = c[gid] - (ps >= th)
    if want_nb:
        nxt = np.where((pos + 1 < gsize[gid]), ps[np.minimum(np.arange(n) + 1, n - 1)], 0.0)
        prv = np.where(pos > 0, ps[np.maximum(np.arange(n) - 1, 0)], 1.0)
        out[prefix + "gap_next"] = ps - nxt
        out[prefix + "gap_prev"] = prv - ps
        out[prefix + "expcnt_minus_rank"] = gsum[gid] - (pos + 1)
    inv = np.empty(n, np.int64)
    inv[order] = np.arange(n)
    return {k: v[inv].astype(np.float32) for k, v in out.items()}


def _max_other_src(s1, key, src, p):
    D = pd.DataFrame({"s1": s1, "k": key, "src": src, "p": p})
    mx = D.groupby(["s1", "k", "src"])["p"].max().rename("mx").reset_index()
    mx["src"] = np.where(mx["src"] == 2, 3, 2)   # swap to the other source (research: min<->max of {2,3})
    return D.merge(mx, on=["s1", "k", "src"], how="left")["mx"].fillna(-1).to_numpy().astype(np.float32)


# ----------------------------------------------------------------------------- ED (feats.py, ed_* only)
_OCR = str.maketrans("0158634729", "olsbgeatzg")


def _ocr(s):
    return s.translate(_OCR).replace("i", "l")


def _edit_class(q, c):
    if _ocr(q) == _ocr(c):
        return 0  # ocr
    if sorted(q) == sorted(c):
        return 1  # anagram
    if c.startswith(q) and 1 <= len(c) - len(q) <= 3:
        return 2  # suffix append
    if q.startswith(c) and 1 <= len(q) - len(c) <= 3:
        return 3  # suffix drop
    if len(q) == len(c) and sum(a != b for a, b in zip(_ocr(q), _ocr(c))) == 1:
        return 4  # single non-ocr substitution
    return 5


_EC: dict = {}


def _name_feats(qn, cn):
    from rapidfuzz.distance import Levenshtein
    key = (qn, cn)
    v = _EC.get(key)
    if v is not None:
        return v
    Q = [t for t in dict.fromkeys(qn.split())]
    C = cn.split()
    cnt = [0] * 6
    minlen = 99
    short_sub = 0
    Cset = set(C)
    for q in Q:
        if q in Cset or len(q) < 2:
            continue
        if not C:
            break
        best = max(C, key=lambda c: Levenshtein.normalized_similarity(q, c))
        sim = Levenshtein.normalized_similarity(q, best)
        if sim < 0.5:
            continue
        k = _edit_class(q, best)
        cnt[k] += 1
        if k >= 2:
            minlen = min(minlen, len(q))
        if k in (2, 4) and len(q) <= 4:
            short_sub += 1
    v = cnt + [minlen if minlen < 99 else 0, short_sub]
    if len(_EC) > 300_000:          # bounded memo (pure function; results do not depend on it)
        _EC.clear()
    _EC[key] = v
    return v


# ----------------------------------------------------------------------------- main entry
def features(df: pd.DataFrame, T: Tables, rows: np.ndarray | None = None, as_dict: bool = False):
    """Level-2 columns (L2_COLS, float32) for whole entities of one country (row order of df).
    rows: rows that need the row-level text features (default prob >= SUB); as_dict: return
    {column: array} instead of a DataFrame (avoids one copy)."""
    from rapidfuzz import fuzz
    import pyarrow as pa  # noqa: F401
    n = len(df)
    s1 = df["s1"].to_numpy().astype(np.int64)
    cand = df["cand"].to_numpy().astype(np.int64)
    src = np.rint(df["src"].to_numpy().astype(np.float64)).astype(np.int64)
    p32 = df["prob"].to_numpy().astype(np.float32)
    p = p32.astype(np.float64)
    aec = df["addr_empty_c"].to_numpy().astype(np.float32)
    m = (p32 >= SUB) if rows is None else np.asarray(rows, bool)
    tpos = _lookup(T.t_codes, cand)
    spos = _lookup(T.s1_codes, s1)
    nmiss_t, nmiss_s = int((tpos < 0).sum()), int((spos < 0).sum())
    if nmiss_t or nmiss_s:
        T.log(f"  [L2] WARNING {T.country}: {nmiss_t} cands / {nmiss_s} s1 rows without text (treated as empty)")
    F = {}
    # ======================= CTX
    with np.errstate(invalid="ignore", over="ignore"):
        F.update(_grp_stats([s1], p, "e_", want_nb=True))
        F.update(_grp_stats([s1, src], p, "es_", thr=(0.5,), want_nb=False))
        ok = tpos >= 0
        tp0 = np.maximum(tpos, 0)
        name_h = np.where(ok, T.t_name_h[tp0], T.hash_empty).astype(np.uint64)
        addr_h = np.where(ok, T.t_addr_h[tp0], T.hash_empty).astype(np.uint64)
        dig_h = np.where(ok, T.t_dig_h[tp0], T.dig_empty).astype(np.uint64)
        nmdig_h = (name_h * np.uint64(1000003)) ^ dig_h
        empty = aec > 0.5
        for key, pre in ((name_h, "dn_"), (addr_h, "da_"), (dig_h, "dd_"), (nmdig_h, "dnd_")):
            F.update(_grp_stats([s1, key.view(np.int64)], p, pre, thr=(0.5,), want_rank=True))
        nts = df["name_tset"].to_numpy().astype(np.float32)
        ats = df["addr_tset"].to_numpy().astype(np.float32)
        sk = (np.round(nts * 100).astype(np.int64) * 1000 + np.round(ats * 100).astype(np.int64))
        F.update(_grp_stats([s1, sk], p, "dsim_", thr=(0.5,), want_rank=True))
    for k in list(F):
        if k.startswith("da_") or k.startswith("dd_"):
            F[k] = np.where(empty, np.nan, F[k]).astype(np.float32)
    F["xs_name_max_other_src"] = _max_other_src(s1, name_h.view(np.int64), src, p)
    F["xs_namedig_max_other_src"] = _max_other_src(s1, nmdig_h.view(np.int64), src, p)
    De = pd.DataFrame({"s1": s1, "pe": np.where(empty, p, 0.0), "pn": np.where(~empty, p, 0.0)})
    g = De.groupby("s1")
    F["ent_empty_psum"] = (g["pe"].transform("sum") - De["pe"]).to_numpy().astype(np.float32)
    F["ent_nonempty_psum"] = (g["pn"].transform("sum") - De["pn"]).to_numpy().astype(np.float32)
    del De, g
    F["is_empty"] = empty.astype(np.float32)
    F["src"] = src.astype(np.float32)
    F["is_us"] = np.full(n, float(T.country == "US"), np.float32)

    # ======================= texts of the rows that need them (model rows + confident siblings)
    conf = p32 >= 0.8
    need_txt = m | conf
    ti = np.flatnonzero(need_txt)
    c_nn = np.full(n, None, dtype=object)          # raw cand name_norm (None for null)
    c_raw = np.full(n, None, dtype=object)
    c_nn[ti] = _take(T.t_name, tpos[ti])
    c_raw[ti] = _take(T.t_raw, tpos[ti])
    c_l = np.full(n, "", dtype=object)
    c_l[ti] = np.array([_low(x) for x in c_raw[ti]] + [None], dtype=object)[:-1]
    sok = spos >= 0
    sp0 = np.maximum(spos, 0)
    s1_nn = np.where(sok, T.s1_nn[sp0], None)
    s1_raw = np.where(sok, T.s1_raw[sp0], None)
    s1_l = np.where(sok, T.s1_l[sp0], "")
    idx = np.flatnonzero(m)

    # ======================= ED
    ed = np.full((n, 8), np.nan, np.float32)
    qn, cn = _fill(s1_nn[idx]), _fill(c_nn[idx])
    for j, i in enumerate(idx):
        ed[i] = _name_feats(qn[j], cn[j])
    for k, c in enumerate(ED_COLS):
        F[c] = ed[:, k]
    del ed, qn, cn

    # ======================= EA (03_feats.py)
    nanv = lambda: np.full(n, np.nan, np.float32)  # noqa: E731
    raw_eq, rawl_eq, rawl_ratio = nanv(), nanv(), nanv()
    raw_eq[idx] = (s1_raw[idx] == c_raw[idx]).astype(np.float32)
    rawl_eq[idx] = (s1_l[idx] == c_l[idx]).astype(np.float32)
    rawl_ratio[idx] = np.array([fuzz.ratio(x, y) for x, y in zip(s1_l[idx], c_l[idx])], np.float32)
    F["raw_eq"], F["rawl_eq"], F["rawl_ratio"] = raw_eq, rawl_eq, rawl_ratio
    cnn_n, pool_n = nanv(), nanv()
    j1 = T.s1_name_index.get_indexer(c_nn[idx])
    cnn_n[idx] = np.where(j1 >= 0, T.grp_cnt[np.maximum(j1, 0)], 0).astype(np.float32)
    j2 = T.pool_index.get_indexer(c_nn[idx])
    pool_n[idx] = np.where(j2 >= 0, T.pool_cnt[np.maximum(j2, 0)], 0).astype(np.float32)
    F["cand_nn_s1_n"], F["pool_empty_nn_n"] = cnn_n, pool_n
    F["pool_empty_per_s1"] = pool_n / np.maximum(cnn_n, 1)
    # competitor raw similarity within the S1 exact-name group (empty-address cands of shared-name S1s)
    fac = np.where(sok, T.s1_fac[sp0], -1)
    grp_size = np.where(fac >= 0, T.grp_cnt[np.maximum(fac, 0)], 0)
    need = m & (aec > 0) & (grp_size > 1)
    cb, ceq = nanv(), nanv()
    for i in np.flatnonzero(need):
        f = fac[i]
        g_ = T.grp_order[T.grp_start[f]:T.grp_start[f] + T.grp_cnt[f]]
        g_ = _subset(g_[g_ != spos[i]], CAP_COMP)
        ls = T.s1_l[g_]
        ci = c_l[i]
        cb[i] = max(fuzz.ratio(ci, y) for y in ls)
        ceq[i] = float(sum(y == ci for y in ls))
    F["comp_rawl_best"] = cb
    F["rawl_margin"] = rawl_ratio - cb
    F["comp_rawl_eq_n"] = ceq
    # sibling features: confident siblings (prob >= 0.8, not self) of the same entity
    sib_eq, sib_nnvar, sib_best, sib_dig = np.zeros(n, np.float32), np.zeros(n, np.float32), nanv(), nanv()
    order = np.argsort(s1, kind="stable")
    s1s = s1[order]
    starts = np.r_[0, np.flatnonzero(s1s[1:] != s1s[:-1]) + 1, n]
    for b in range(len(starts) - 1):
        ids = order[starts[b]:starts[b + 1]]
        mi = ids[m[ids]]
        if len(mi) == 0:
            continue
        ci_ = ids[conf[ids]]
        sib_l = [(j, c_l[j], c_nn[j]) for j in ci_]
        s1dig = set(DIG.findall(s1_l[ids[0]]))
        sibdig = set().union(*[set(DIG.findall(x[1])) for x in sib_l]) if sib_l else set()
        for i in mi:
            others = [x for x in sib_l if x[0] != i]
            if others:
                sib_eq[i] = float(any(x[1] == c_l[i] for x in others))
                sib_nnvar[i] = float(any((x[2] == c_nn[i]) and (c_nn[i] != s1_nn[i]) for x in others))
                sib_best[i] = max(fuzz.ratio(c_l[i], x[1]) for x in others)
            dg = set(DIG.findall(c_l[i]))
            if dg:
                sib_dig[i] = float(bool(dg & (s1dig | sibdig)))
    sib_eq[~m] = np.nan
    sib_nnvar[~m] = np.nan
    F["sib_rawl_eq"], F["sib_nn_var_eq"], F["sib_rawl_best"], F["name_digit_supported"] = sib_eq, sib_nnvar, sib_best, sib_dig
    # prob-based entity / per-source aggregates (all rows)
    H = pd.DataFrame({"s1": s1, "src": src, "prob": p32})
    g = H.groupby("s1", sort=False)
    conf5 = (H["prob"] >= 0.5).astype(np.float32)
    e = pd.Series((aec > 0).astype(np.float32))
    H["_c5"] = conf5
    H["_e"] = e
    H["_ep"] = H["prob"] * e
    H["_ec5"] = conf5 * e
    src_c5 = H.groupby(["s1", "src"])["_c5"].transform("sum")
    F["n_conf_src_other"] = (src_c5 - conf5).to_numpy(np.float32)
    F["n_conf_othersrc"] = (g["_c5"].transform("sum") - src_c5).to_numpy(np.float32)
    F["sum_p_src_other"] = (H.groupby(["s1", "src"])["prob"].transform("sum") - H["prob"]).to_numpy(np.float32)
    F["src_cap_room"] = (np.where(H["src"] == 2, 5, 6) - F["n_conf_src_other"]).astype(np.float32)
    F["ent_empty_sum_p_other"] = (g["_ep"].transform("sum") - H["_ep"]).to_numpy(np.float32)
    F["ent_empty_conf_other"] = (g["_ec5"].transform("sum") - H["_ec5"]).to_numpy(np.float32)
    F["ent_conf_n"] = g["_c5"].transform("sum").to_numpy(np.float32)
    pe = H["prob"].where(H["_e"] > 0)
    F["empty_prob_rank"] = pe.groupby(H["s1"]).rank(ascending=False, method="min").to_numpy(np.float32)
    del H, g, pe

    # ======================= EA tie features (09_tiefeat.py)
    sub_n, self_sub, best_o, sup_m = nanv(), nanv(), nanv(), nanv()
    E = np.flatnonzero(m & (aec > 0))
    cnnE = _fill(c_nn[E])
    for j, i in enumerate(E):
        sup = T.superset(cnnE[j])
        me = spos[i]
        sub_n[i] = len(sup)
        k = np.searchsorted(sup, me) if len(sup) else 0
        mine = bool(len(sup) and k < len(sup) and sup[k] == me)
        self_sub[i] = float(mine)
        own = fuzz.ratio(c_l[i], s1_l[i])
        oth = np.delete(sup, k) if mine else sup
        if len(oth):
            oth = _subset(oth, CAP_SUP)
            best_o[i] = max(fuzz.ratio(c_l[i], T.s1_l[jj]) for jj in oth)
            sup_m[i] = own - best_o[i]
    F["sup_n"] = sub_n
    F["sup_self"] = self_sub
    F["sup_n_other"] = sub_n - self_sub
    F["sup_best_other"] = best_o
    F["sup_margin"] = sup_m
    out = {c: np.asarray(F[c], dtype=np.float32) for c in L2_COLS}
    return out if as_dict else pd.DataFrame(out)
