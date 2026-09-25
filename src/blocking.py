"""Candidate generation (blocking) with bounded memory.

Blocking sets the hard ceiling on achievable score: a true match never proposed
as a candidate can never be recovered.  F_0.5 is precision-heavy, so the goal is
**high recall at a modest candidate count**; precision is left to the matcher and
the decision layer.

Partitioning
------------
The search space is cut by ``country`` first.  The ground-truth audit showed this
is lossless: 0 of 7,638,365 true training pairs cross a country boundary.

Retrieval: sparse IDF cosine over rare terms
--------------------------------------------
Each record becomes a bag of namespaced terms:

    n:<tok>    name unigram              nb:<a>_<b>  adjacent name bigram
    a:<tok>    address alphabetic token  ab:<a>_<b>  adjacent address bigram
    d:<num>    address numeric token     nc:<pfx>    first 10 chars of the name
                                                     with spaces removed
    k:<skel>   name consonant skeleton   kb:/kc:     skeleton bigram / prefix
    ak:<skel>  address consonant skeleton

Skeleton terms were added after the miss analysis showed 76 % of India misses
were phonetically transliterated names (``praaprttiis`` for ``properties``).

``nc:`` targets the token-boundary damage the generator applies, e.g.
``BHG Foundation Private Limited -> bhgfoundation.com``: both start with
``bhgfoundat``.  Bigrams keep evidence like ``sai farms`` or ``415 suttle``
usable after the frequent unigrams ``sai``, ``farms``, ``415`` are pruned.

Why it no longer blows up memory
--------------------------------
The first version pruned terms by *relative* document frequency (2 %).  In a
2 M-document block that still allows terms shared by 40 000 documents, and one
query chunk produced billions of non-zeros (OOM, exit 137).  This version

1. hashes terms (``HashingVectorizer``), so there is no Python vocabulary dict,
2. drops every term whose document frequency exceeds an **absolute** cap
   (``max_df_abs``), which bounds each query row's product size by
   ``n_terms * max_df_abs``,
3. sizes each query chunk from the *exact* predicted output size (sum of the
   document frequencies of the query's surviving terms), so the sparse product
   never exceeds ``nnz_budget`` non-zeros.

IDF weights are country-agnostic by construction: legal suffixes, street words
and region names are frequent in whichever country they belong to, so they get
pruned or down-weighted in the France block without any French word list.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer

from .normalize import NULL_TOKENS, skeleton_token

__all__ = ["BlockingConfig", "build_blobs", "RareTermIndex", "search_topk"]


@dataclass
class BlockingConfig:
    """Tunable knobs for candidate generation (see ``src/bench_blocking.py``)."""

    topk: int = 30                  # neighbours kept per query per target source
    max_df_abs: int = 2_000         # drop terms present in more target docs than this
    n_features: int = 2**23         # hash space; collisions are negligible at this size
    nnz_budget: int = 30_000_000    # max non-zeros in one query-chunk product
    nc_prefix: int = 10             # length of the concatenated-name prefix key
    canon_digits: bool = False      # strip leading zeros from numeric terms ("02818" -> "2818")


# --------------------------------------------------------------------------- #
# Term construction
# --------------------------------------------------------------------------- #
def _clean(tokens: list[str]) -> list[str]:
    return [t for t in tokens if len(t) > 1 and t not in NULL_TOKENS]


def build_blobs(name_norm: np.ndarray, addr_norm: np.ndarray, cfg: BlockingConfig) -> list[str]:
    """One space-joined string of namespaced terms per record.

    Built as plain strings so ``HashingVectorizer`` can tokenise with its C
    regex path (``token_pattern=r"\\S+"``) instead of a Python callable.
    """
    out: list[str] = []
    p = cfg.nc_prefix
    for name, addr in zip(name_norm, addr_norm):
        nt = _clean(name.split())
        at = _clean(addr.split())
        if cfg.canon_digits:
            at = [(t.lstrip("0") or "0") if t.isdigit() else t for t in at]
        terms = ["n:" + t for t in nt]
        terms += ["nb:" + a + "_" + b for a, b in zip(nt, nt[1:])]
        joined = "".join(nt)
        if len(joined) >= 4:
            terms.append("nc:" + joined[:p])
        for t in at:
            terms.append(("d:" if t.isdigit() else "a:") + t)
        terms += ["ab:" + a + "_" + b for a, b in zip(at, at[1:])]
        # Consonant skeletons: transliteration-robust keys (see normalize.py).
        nk = [k for k in (skeleton_token(t) for t in nt if not t.isdigit()) if len(k) >= 2]
        terms += ["k:" + k for k in nk]
        terms += ["kb:" + a + "_" + b for a, b in zip(nk, nk[1:])]
        if nk:
            terms.append("kc:" + "".join(nk)[:8])
        terms += ["ak:" + k for k in (skeleton_token(t) for t in at if not t.isdigit()) if len(k) >= 2]
        out.append(" ".join(terms))
    return out


def _hasher(cfg: BlockingConfig) -> HashingVectorizer:
    return HashingVectorizer(
        n_features=cfg.n_features,
        token_pattern=r"\S+",
        lowercase=False,
        alternate_sign=False,
        norm=None,
        binary=True,
        dtype=np.float32,
    )


# --------------------------------------------------------------------------- #
# Index over one target pool
# --------------------------------------------------------------------------- #
class RareTermIndex:
    """Inverted index over the rare terms of one target pool (one country, one
    source).  Holds the transposed, IDF-weighted, L2-normalised term matrix.

    Memory: terms are hashed in chunks straight from the normalised text (the
    blob strings of a 3 M-record pool alone take ~1.6 GB), and weighting and
    normalisation happen in place, so peak usage is about two copies of the
    sparse matrix (during the final transpose).  The first version built all
    blobs up front and copied the matrix four times, and was OOM-killed on the
    3.17 M-record US S3 pool under a 5.5 GB cap.
    """

    def __init__(self, name_norm: np.ndarray, addr_norm: np.ndarray, cfg: BlockingConfig,
                 chunk: int = 500_000):
        t0 = time.time()
        self.cfg = cfg
        self.hasher = _hasher(cfg)
        parts = []
        for i in range(0, len(name_norm), chunk):
            blobs = build_blobs(name_norm[i:i + chunk], addr_norm[i:i + chunk], cfg)
            parts.append(self.hasher.transform(blobs).tocsr())
            del blobs
        X = sp.vstack(parts, format="csr") if len(parts) > 1 else parts[0]
        del parts
        gc.collect()
        self.n_targets = X.shape[0]

        df = np.bincount(X.indices, minlength=cfg.n_features).astype(np.int64)
        self.df = df
        keep = (df > 0) & (df <= cfg.max_df_abs)
        idf = np.zeros(cfg.n_features, dtype=np.float32)
        idf[keep] = np.log((self.n_targets + 1.0) / (df[keep] + 1.0)) + 1.0
        self.idf = idf

        self._weight_inplace(X)
        self.BT = X.T.tocsr()          # (n_features, n_targets)
        del X
        gc.collect()
        self.build_sec = time.time() - t0

    def _weight_inplace(self, X: sp.csr_matrix) -> None:
        """IDF weights (zero for pruned terms), drop zeros, L2-normalise rows."""
        X.data = self.idf[X.indices]
        X.eliminate_zeros()
        sq = X.data.astype(np.float64) ** 2
        cs = np.concatenate([[0.0], np.cumsum(sq)])
        norms = np.sqrt(cs[X.indptr[1:]] - cs[X.indptr[:-1]])
        norms[norms == 0] = 1.0
        X.data /= np.repeat(norms, np.diff(X.indptr)).astype(np.float32)

    def query_matrix(self, query_blobs: list[str]) -> sp.csr_matrix:
        A = self.hasher.transform(query_blobs).tocsr()
        self._weight_inplace(A)
        return A

    def row_costs(self, A: sp.csr_matrix) -> np.ndarray:
        """Upper bound on the product's non-zeros per query row: the sum of the
        document frequencies of the row's surviving terms."""
        cs = np.concatenate([[0], np.cumsum(self.df[A.indices])])
        return cs[A.indptr[1:]] - cs[A.indptr[:-1]]


def _chunks_by_budget(costs: np.ndarray, budget: int, max_rows: int = 20_000):
    """Yield (start, stop) query ranges whose summed cost stays under budget."""
    n = costs.size
    start = 0
    csum = np.concatenate([[0], np.cumsum(costs)])
    while start < n:
        limit = csum[start] + budget
        stop = int(np.searchsorted(csum, limit, side="right")) - 1
        stop = max(stop, start + 1)
        stop = min(stop, start + max_rows, n)
        yield start, stop
        start = stop


def search_topk(
    index: RareTermIndex,
    query_blobs: list[str],
    *,
    topk: int | None = None,
    verbose: bool = True,
    label: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k cosine neighbours for every query.

    Returns ``(idx, score)`` of shape ``(n_queries, topk)``: positional indices
    into the target pool, padded with ``-1`` / ``0.0`` where fewer than ``topk``
    targets share any surviving term with the query.
    """
    t0 = time.time()
    k = topk or index.cfg.topk
    A = index.query_matrix(query_blobs)
    costs = index.row_costs(A)

    n_q = A.shape[0]
    idx_out = np.full((n_q, k), -1, dtype=np.int32)
    sc_out = np.zeros((n_q, k), dtype=np.float32)
    peak = 0

    for start, stop in _chunks_by_budget(costs, index.cfg.nnz_budget):
        C = (A[start:stop] @ index.BT).tocsr()
        peak = max(peak, C.nnz)
        ip, ind, dat = C.indptr, C.indices, C.data
        for r in range(stop - start):
            lo, hi = ip[r], ip[r + 1]
            n = hi - lo
            if n == 0:
                continue
            d = dat[lo:hi]
            part = np.argpartition(d, -k)[-k:] if n > k else np.arange(n)
            order = part[np.argsort(-d[part], kind="stable")]
            idx_out[start + r, : order.size] = ind[lo:hi][order]
            sc_out[start + r, : order.size] = d[order]
        del C

    if verbose:
        print(
            f"    [{label}] {n_q:,} q x {index.n_targets:,} t  top{k}  "
            f"mean_cost={costs.mean():.0f}  peak_nnz={peak:,}  "
            f"build={index.build_sec:.0f}s search={time.time() - t0:.0f}s"
        )
    return idx_out, sc_out
