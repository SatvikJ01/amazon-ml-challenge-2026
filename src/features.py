"""Pairwise features for (Source-1 record, candidate) pairs.

Every feature is a *comparison* between the two records, never a property that
identifies a record.  That is what makes it safe to train on pairs whose target
records also appear as negatives for validation entities: the model can only
learn "how similar is similar enough", not "which record is which".

Country is deliberately **not** a feature.  France is absent from training, so a
country indicator would be an unseen category at test time.  Country-specific
behaviour, where it exists, has to show up through the comparisons themselves.

Feature groups
--------------
skeleton   the same ratios on consonant skeletons, so a phonetically
           transliterated name ("praaprttiis") still matches ("properties")
name       fuzzy ratios (rapidfuzz, C++ and multi-threaded), a "core" variant
           with high-frequency tokens removed (legal suffixes, generic words --
           learned per country block from document frequency), a no-space
           variant that catches ``bhgfoundation.com``, IDF-weighted token overlap
address    fuzzy ratios, numeric-token agreement (house / plot numbers are the
           most reliable address evidence and survive reordering), IDF overlap
blocking   retrieval score and rank within the entity's candidate list, gap to
           the best candidate: whether a candidate is a match is *relative*
context    per-entity ranks of the key similarities, candidate count
meta       target source (S2 vs S3), empty-address flags, script flags
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from .normalize import skeleton_tokens

__all__ = ["TokenStats", "pair_features", "add_context_features", "FEATURE_COLS"]


class TokenStats:
    """Per-country token document frequencies, used for IDF weights and to
    define the 'core name' (a name with its frequent tokens stripped)."""

    def __init__(self, name_norm: np.ndarray, addr_norm: np.ndarray, core_max_df: float = 0.003):
        from collections import Counter
        n = len(name_norm)
        self.n = n
        # Counters hold only distinct tokens; materialising every token as a
        # pandas Series cost >1 GB on the 1.3 M-record US block.
        name_df: Counter = Counter()
        for x in name_norm:
            name_df.update(set(x.split()))
        addr_df: Counter = Counter()
        for x in addr_norm:
            addr_df.update(set(x.split()))
        ln = np.log(n + 1)
        self.name_idf = {t: float(ln - np.log(c + 1)) for t, c in name_df.items()}
        self.addr_idf = {t: float(ln - np.log(c + 1)) for t, c in addr_df.items()}
        self.name_default = float(ln)
        self.addr_default = float(ln)
        self.frequent_name = {t for t, c in name_df.items() if c > core_max_df * n}
        # How many S1 records share an exact / core name.  A unique S1 name is
        # strong evidence for a name-only (empty-address) candidate; a name
        # shared by dozens of records is not.
        self.name_count = Counter(name_norm)
        self.core_count = Counter(self.core(x) for x in name_norm)

    def core(self, name: str) -> str:
        toks = [t for t in name.split() if t not in self.frequent_name]
        return " ".join(toks) if toks else name


def _idf_overlap(a: list[str], b: list[str], idf: dict, default: float) -> tuple[float, float, float]:
    """IDF-weighted overlap: (shared/a, shared/b, shared/union)."""
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0, 0.0, 0.0
    wa = sum(idf.get(t, default) for t in sa)
    wb = sum(idf.get(t, default) for t in sb)
    ws = sum(idf.get(t, default) for t in sa & sb)
    wu = wa + wb - ws
    # Guard: tokens present in every document have IDF 0.
    return (ws / wa if wa > 0 else 0.0), (ws / wb if wb > 0 else 0.0), (ws / wu if wu > 0 else 0.0)


def _map_unique(arr: np.ndarray, fn) -> np.ndarray:
    """Apply ``fn`` once per distinct string and broadcast back.  Each S1 record
    appears in ~60 pairs and each candidate in ~15, so this removes most of the
    per-pair Python work."""
    codes, uniques = pd.factorize(arr)
    vals = np.empty(len(uniques), dtype=object)
    for i, u in enumerate(uniques):
        vals[i] = fn(u)
    return vals[codes]


def _digits(s: str) -> list[str]:
    return [t for t in s.split() if t.isdigit()]


def _num_sim(a: str, b: str) -> float:
    """Similarity of two canonical (leading-zero-stripped) numbers.

    1.0 equal; 0.8 when one is the other with a digit dropped at either end
    (``324``/``32``, ``18704``/``8704``); 0.6 for a single substitution or
    insertion elsewhere; 0 otherwise.  These are the corruptions observed in the
    missed true pairs (reports/E010 error analysis).
    """
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if abs(la - lb) == 1:
        lo, hi = (a, b) if la < lb else (b, a)
        if hi.startswith(lo) or hi.endswith(lo):
            return 0.8
    if abs(la - lb) <= 1 and Levenshtein.distance(a, b) <= 1:
        return 0.6
    return 0.0


def pair_features(
    q_name: np.ndarray,
    q_addr: np.ndarray,
    c_name: np.ndarray,
    c_addr: np.ndarray,
    c_raw_nonascii: np.ndarray,
    stats: TokenStats,
    workers: int = -1,
) -> pd.DataFrame:
    """Compute comparison features for aligned arrays of pairs."""
    n = len(q_name)
    f: dict[str, np.ndarray] = {}

    def cp(scorer, a, b, **kw):
        return process.cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32, **kw)

    # ---- name ----
    f["name_ratio"] = cp(fuzz.ratio, q_name, c_name)
    f["name_partial"] = cp(fuzz.partial_ratio, q_name, c_name)
    f["name_tsort"] = cp(fuzz.token_sort_ratio, q_name, c_name)
    f["name_tset"] = cp(fuzz.token_set_ratio, q_name, c_name)
    f["name_jw"] = cp(JaroWinkler.normalized_similarity, q_name, c_name)

    q_core = _map_unique(q_name, stats.core)
    c_core = _map_unique(c_name, stats.core)
    f["core_ratio"] = cp(fuzz.ratio, q_core, c_core)
    f["core_tset"] = cp(fuzz.token_set_ratio, q_core, c_core)
    f["core_partial"] = cp(fuzz.partial_ratio, q_core, c_core)

    q_ns = _map_unique(q_name, lambda x: x.replace(" ", ""))
    c_ns = _map_unique(c_name, lambda x: x.replace(" ", ""))
    f["nospace_partial"] = cp(fuzz.partial_ratio, q_ns, c_ns)
    f["nospace_lev"] = cp(Levenshtein.normalized_similarity, q_ns, c_ns)
    qc_ns = _map_unique(q_core, lambda x: x.replace(" ", ""))
    cc_ns = _map_unique(c_core, lambda x: x.replace(" ", ""))
    f["core_nospace_partial"] = cp(fuzz.partial_ratio, qc_ns, cc_ns)

    # ---- consonant skeleton (transliteration-robust, see normalize.py) ----
    q_sk = _map_unique(q_name, lambda x: " ".join(skeleton_tokens(x)))
    c_sk = _map_unique(c_name, lambda x: " ".join(skeleton_tokens(x)))
    f["skel_ratio"] = cp(fuzz.ratio, q_sk, c_sk)
    f["skel_tset"] = cp(fuzz.token_set_ratio, q_sk, c_sk)
    f["skel_nospace_partial"] = cp(
        fuzz.partial_ratio,
        _map_unique(q_sk, lambda x: x.replace(" ", "")),
        _map_unique(c_sk, lambda x: x.replace(" ", "")),
    )
    qa_sk = _map_unique(q_addr, lambda x: " ".join(skeleton_tokens(x)))
    ca_sk = _map_unique(c_addr, lambda x: " ".join(skeleton_tokens(x)))
    f["addr_skel_tset"] = cp(fuzz.token_set_ratio, qa_sk, ca_sk)

    # ---- address ----
    f["addr_tset"] = cp(fuzz.token_set_ratio, q_addr, c_addr)
    f["addr_tsort"] = cp(fuzz.token_sort_ratio, q_addr, c_addr)
    f["addr_partial"] = cp(fuzz.partial_token_set_ratio, q_addr, c_addr)
    f["addr_ratio"] = cp(fuzz.ratio, q_addr, c_addr)

    # ---- token-level loops (IDF overlap, digits) ----
    n_ov_a = np.zeros(n, np.float32); n_ov_b = np.zeros(n, np.float32); n_ov_u = np.zeros(n, np.float32)
    a_ov_a = np.zeros(n, np.float32); a_ov_b = np.zeros(n, np.float32); a_ov_u = np.zeros(n, np.float32)
    d_jac = np.zeros(n, np.float32); d_first = np.zeros(n, np.float32)
    d_q = np.zeros(n, np.float32); d_c = np.zeros(n, np.float32); d_any = np.zeros(n, np.float32)
    nt_q = np.zeros(n, np.float32); nt_c = np.zeros(n, np.float32)
    at_c = np.zeros(n, np.float32)
    dz_jac = np.zeros(n, np.float32); dz_first = np.zeros(n, np.float32)
    dz_best = np.zeros(n, np.float32); dz_first_sim = np.zeros(n, np.float32)
    nm_cnt = np.zeros(n, np.float32); core_cnt = np.zeros(n, np.float32)
    nc_cnt = np.zeros(n, np.float32)

    ni, nd, ai, ad = stats.name_idf, stats.name_default, stats.addr_idf, stats.addr_default
    for i in range(n):
        qn, cn = q_name[i].split(), c_name[i].split()
        qa, ca = q_addr[i].split(), c_addr[i].split()
        n_ov_a[i], n_ov_b[i], n_ov_u[i] = _idf_overlap(qn, cn, ni, nd)
        a_ov_a[i], a_ov_b[i], a_ov_u[i] = _idf_overlap(qa, ca, ai, ad)
        qd, cd = [t for t in qa if t.isdigit()], [t for t in ca if t.isdigit()]
        sq, sc = set(qd), set(cd)
        d_q[i], d_c[i] = len(sq), len(sc)
        if sq and sc:
            inter = len(sq & sc)
            d_jac[i] = inter / len(sq | sc)
            d_any[i] = 1.0 if inter else 0.0
            d_first[i] = 1.0 if qd[0] == cd[0] else 0.0
        nt_q[i], nt_c[i], at_c[i] = len(qn), len(cn), len(ca)
        # canonical digits: strip leading zeros ("02818" -> "2818")
        qz = [t.lstrip("0") or "0" for t in qd]
        cz = [t.lstrip("0") or "0" for t in cd]
        if qz and cz:
            szq, szc = set(qz), set(cz)
            dz_jac[i] = len(szq & szc) / len(szq | szc)
            dz_first[i] = 1.0 if qz[0] == cz[0] else 0.0
            dz_first_sim[i] = max(_num_sim(qz[0], y) for y in cz)
            dz_best[i] = max(_num_sim(x, y) for x in szq for y in szc)
        nm_cnt[i] = stats.name_count.get(q_name[i], 0)
        core_cnt[i] = stats.core_count.get(q_core[i], 0)
        nc_cnt[i] = stats.core_count.get(c_core[i], 0)

    f.update(
        name_idf_q=n_ov_a, name_idf_c=n_ov_b, name_idf_u=n_ov_u,
        addr_idf_q=a_ov_a, addr_idf_c=a_ov_b, addr_idf_u=a_ov_u,
        dig_jaccard=d_jac, dig_first_eq=d_first, dig_any=d_any,
        dig_n_q=d_q, dig_n_c=d_c,
        name_ntok_q=nt_q, name_ntok_c=nt_c, name_ntok_diff=nt_q - nt_c,
        addr_ntok_c=at_c, addr_empty_c=(at_c == 0).astype(np.float32),
        name_nonascii_c=c_raw_nonascii.astype(np.float32),
        digz_jaccard=dz_jac, digz_first_eq=dz_first, digz_best_sim=dz_best,
        digz_first_sim=dz_first_sim,
        s1_name_count=nm_cnt, s1_core_count=core_cnt, cand_core_count_in_s1=nc_cnt,
        name_len_ratio=np.array(
            [min(len(a), len(b)) / max(len(a), len(b), 1) for a, b in zip(q_name, c_name)],
            np.float32,
        ),
    )
    return pd.DataFrame(f)


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-entity relative features.  ``df`` must contain ``s1``, ``src`` and the
    base features.  A candidate's absolute similarity means little on its own; how
    it compares with the entity's other candidates is what separates the true
    match from its nearest distractor."""
    g = df.groupby("s1", sort=False)
    for col in ("blk_score", "name_tset", "addr_tset", "name_idf_u", "addr_idf_u"):
        mx = g[col].transform("max")
        df[f"{col}_gap"] = mx - df[col]
        df[f"{col}_rank"] = g[col].rank(ascending=False, method="min").astype(np.float32)
    gs = df.groupby(["s1", "src"], sort=False)
    df["blk_score_src_gap"] = gs["blk_score"].transform("max") - df["blk_score"]
    df["n_cands"] = g["blk_score"].transform("size").astype(np.float32)
    return df


FEATURE_COLS: list[str] | None = None  # resolved at training time from the frame
