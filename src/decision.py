"""Decision layer: turn per-candidate match probabilities into a predicted set.

Why not a single threshold
--------------------------
The score is F_0.5 computed **per Source-1 entity** and macro-averaged.  So the
right question for each entity is: *which subset of my candidates maximises the
expected F_0.5 for this entity?*  A global probability threshold answers a
different question (pair-level accuracy) and is systematically wrong at the
edges.  Two examples:

* One candidate at p = 0.4.  Under independence, abstaining scores
  P(no match) = 0.6 in expectation versus 0.4 for predicting, so abstain --
  *unless* the entity likely has true matches the blocker missed (``missed``
  > 0), in which case abstaining is worth ~0 and predicting wins.  The
  singleton prior (~6 %) therefore matters only through ``missed`` or a
  separate singleton model; this is logged as an experiment.
* Five candidates at p = 0.45 each (a common case: 3.5 true matches on average).
  Taking all five beats taking none by a wide margin, but a 0.5 threshold takes
  none.

Expected F_0.5 of a prefix
--------------------------
With closed form ``F = 1.25 TP / (0.25 |T| + |P|)`` and candidates sorted by
probability, let the prediction be the top-``k`` prefix.  With ``A`` = number of
true matches inside the prefix and ``B`` = number among the remaining candidates,
treated as independent Poisson-binomial variables,

    E[F_k] = sum_{a,b} P(A=a) P(B=b) * 1.25 a / (0.25 (a + b + m) + k)   (k >= 1)
    E[F_0] = P(A=0, B=0) * [m == 0]      = prod_i (1 - p_i)              (k = 0)

where ``m`` is an optional expected count of true matches the blocker never
surfaced (defaults to 0).  For F-measures under independence the optimal subset
is always a prefix of the probability-sorted list (Lewis 1995; Ye et al. 2012),
so scanning ``k = 0..n`` is exact.

Supports are truncated at ``MAX_TRUE`` (training max is 11 matches per entity),
which makes the scan O(n * MAX_TRUE^2) per entity and lets numba process ~1.7 M
entities in seconds.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

__all__ = ["expected_f05_best_k", "select_sets", "threshold_sets"]

MAX_TRUE = 16


@njit(cache=True)
def _poisson_binomial(p: np.ndarray, cap: int) -> np.ndarray:
    """P(sum of Bernoulli(p_i) = j) for j = 0..cap (tail folded into cap)."""
    dist = np.zeros(cap + 1)
    dist[0] = 1.0
    for pi in p:
        for j in range(cap, 0, -1):
            dist[j] = dist[j] * (1.0 - pi) + dist[j - 1] * pi
        dist[0] *= 1.0 - pi
    return dist


@njit(cache=True)
def expected_f05_best_k(p_sorted: np.ndarray, missed: float = 0.0) -> tuple:
    """Best prefix length and its expected F_0.5 for one entity.

    ``p_sorted`` must be sorted in descending order.  Returns ``(k, E[F_k])``.
    """
    n = p_sorted.size
    cap = MAX_TRUE

    # Suffix distributions: suf[k] = distribution of successes among items k..n-1.
    suf = np.zeros((n + 1, cap + 1))
    suf[n, 0] = 1.0
    for k in range(n - 1, -1, -1):
        pi = p_sorted[k]
        suf[k, 0] = suf[k + 1, 0] * (1.0 - pi)
        for j in range(1, cap + 1):
            suf[k, j] = suf[k + 1, j] * (1.0 - pi) + suf[k + 1, j - 1] * pi

    best_k = 0
    best = suf[0, 0] if missed == 0.0 else 0.0     # E[F_0]

    pre = np.zeros(cap + 1)
    pre[0] = 1.0
    for k in range(1, n + 1):
        pi = p_sorted[k - 1]
        for j in range(cap, 0, -1):
            pre[j] = pre[j] * (1.0 - pi) + pre[j - 1] * pi
        pre[0] *= 1.0 - pi

        B = suf[k]
        e = 0.0
        for a in range(1, cap + 1):
            if pre[a] < 1e-12:
                continue
            s = 0.0
            for b in range(cap + 1):
                if B[b] < 1e-12:
                    continue
                s += B[b] / (0.25 * (a + b + missed) + k)
            e += pre[a] * 1.25 * a * s
        if e > best:
            best = e
            best_k = k
    return best_k, best


@njit(parallel=True, cache=True)
def _select_all(offsets: np.ndarray, probs: np.ndarray, missed: np.ndarray,
                max_cands: int) -> tuple:
    """Vectorised over entities. ``probs`` is grouped by entity and sorted
    descending within each group; ``offsets`` are CSR-style group boundaries."""
    n_ent = offsets.size - 1
    ks = np.zeros(n_ent, dtype=np.int32)
    ef = np.zeros(n_ent)
    for e in prange(n_ent):
        lo = offsets[e]
        hi = min(offsets[e + 1], lo + max_cands)
        if hi == lo:
            ks[e] = 0
            ef[e] = 1.0 if missed[e] == 0.0 else 0.0
            continue
        k, v = expected_f05_best_k(probs[lo:hi], missed[e])
        ks[e] = k
        ef[e] = v
    return ks, ef


def select_sets(
    entity: np.ndarray,
    cand: np.ndarray,
    prob: np.ndarray,
    *,
    missed: np.ndarray | float = 0.0,
    max_cands: int = 40,
    calib_power: float = 1.0,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Expected-F_0.5 set selection for every entity.

    Parameters
    ----------
    entity, cand, prob:
        Flat pair arrays (one row per candidate pair).
    missed:
        Expected number of true matches absent from each entity's candidate list
        (scalar or per-entity array aligned to ``np.unique(entity)``).
    max_cands:
        Only the top ``max_cands`` candidates per entity are considered; the tail
        beyond that has negligible probability mass after a trained matcher.
    calib_power:
        Optional ``p -> p**calib_power`` sharpening, tuned on validation.  1.0
        means trust the model's probabilities as-is.

    Returns
    -------
    (sets, entities, expected_f) where ``sets`` maps entity code -> array of
    chosen candidate codes.
    """
    p = np.clip(prob.astype(np.float64), 0.0, 1.0) ** calib_power
    order = np.lexsort((-p, entity))
    ent_s, cand_s, p_s = entity[order], cand[order], p[order]

    uniq, start = np.unique(ent_s, return_index=True)
    offsets = np.append(start, ent_s.size).astype(np.int64)
    miss = np.broadcast_to(np.asarray(missed, dtype=np.float64), uniq.shape).copy()

    ks, ef = _select_all(offsets, p_s, miss, max_cands)
    sets = {int(u): cand_s[offsets[i]: offsets[i] + ks[i]] for i, u in enumerate(uniq)}
    return sets, uniq, ef


def threshold_sets(entity: np.ndarray, cand: np.ndarray, prob: np.ndarray,
                   thr: float) -> dict:
    """Baseline decision rule: keep every candidate with p >= thr."""
    keep = prob >= thr
    df_e, df_c = entity[keep], cand[keep]
    order = np.argsort(df_e, kind="stable")
    df_e, df_c = df_e[order], df_c[order]
    uniq, start = np.unique(df_e, return_index=True)
    bounds = np.append(start, df_e.size)
    return {int(u): df_c[bounds[i]: bounds[i + 1]] for i, u in enumerate(uniq)}
