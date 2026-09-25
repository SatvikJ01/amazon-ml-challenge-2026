"""Exact reproduction of the Amazon ML Challenge 2026 evaluation metric.

The official metric (problem statement, "Evaluation Criteria") is

    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

computed **per Source-1 entity** and then **macro-averaged** across all Source-1
entities in the evaluation set, singletons included.

Closed form
-----------
Let ``TP = |P n T|``, ``|P|`` = size of the predicted set, ``|T|`` = size of the
true set.  Then ``Precision = TP/|P|`` and ``Recall = TP/|T|``, so

    F_0.5 = 1.25 * (TP/|P|)(TP/|T|) / (0.25*TP/|P| + TP/|T|)
          = 1.25 * TP^2/(|P||T|) * (|P||T|) / (TP * (0.25*|T| + |P|))
          = 1.25 * TP / (0.25*|T| + |P|)

This closed form is exact whenever ``TP > 0`` and is what we evaluate, because it
avoids two separate divisions and is numerically cleaner.  It is unit-tested
against the naive definition in ``tests/test_metrics.py``.

Boundary conventions (taken verbatim from the problem statement)
---------------------------------------------------------------
* ``|P| = 0`` and ``|T| = 0``  -> **1.0**
  "A Source 1 entity with no true matches scores 1.0 when you correctly predict
  an empty list"
* ``|P| > 0`` and ``|T| = 0``  -> **0.0**
  "and 0.0 when you predict any match for it"
* ``|P| = 0`` and ``|T| > 0``  -> **0.0** (recall is 0, so F is 0)
* ``TP = 0`` with both non-empty -> **0.0**

All functions take *sets of entity-id strings*.  Nothing here knows about models.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Hashable, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "BETA",
    "fbeta_single",
    "f05_single",
    "evaluate",
    "evaluate_cv",
    "evaluate_by_group",
    "precision_recall_single",
    "micro_prf",
    "candidate_recall_ceiling",
]

BETA: float = 0.5
_BETA_SQ: float = BETA * BETA          # 0.25
_ONE_PLUS_BETA_SQ: float = 1.0 + _BETA_SQ  # 1.25


# --------------------------------------------------------------------------- #
# Per-entity metric
# --------------------------------------------------------------------------- #
def fbeta_single(pred: Iterable[Hashable], true: Iterable[Hashable], beta: float = BETA) -> float:
    """F-beta for one Source-1 entity, with the challenge's empty-set conventions.

    Parameters
    ----------
    pred, true:
        Iterables of entity ids.  Converted to sets internally, so duplicates in
        the input are collapsed (the submission format forbids duplicates anyway,
        and the validator rejects them separately).
    beta:
        Defaults to 0.5, the competition value.  Exposed only so the unit tests
        can cross-check against F1.
    """
    p = pred if isinstance(pred, (set, frozenset)) else set(pred)
    t = true if isinstance(true, (set, frozenset)) else set(true)

    n_p, n_t = len(p), len(t)

    # Both empty: a correctly identified singleton earns full credit.
    if n_p == 0 and n_t == 0:
        return 1.0
    # Exactly one empty: precision or recall is 0, hence F is 0.
    if n_p == 0 or n_t == 0:
        return 0.0

    tp = len(p & t)
    if tp == 0:
        return 0.0

    b2 = beta * beta
    return (1.0 + b2) * tp / (b2 * n_t + n_p)


def f05_single(pred: Iterable[Hashable], true: Iterable[Hashable]) -> float:
    """F_0.5 for one Source-1 entity. Thin alias for the competition beta."""
    return fbeta_single(pred, true, BETA)


def precision_recall_single(
    pred: Iterable[Hashable], true: Iterable[Hashable]
) -> tuple[float, float]:
    """Per-entity (precision, recall) using the same empty-set conventions.

    Both are defined as 1.0 when the corresponding set is empty *and* the other
    is too; this keeps diagnostics consistent with ``fbeta_single``.
    """
    p = set(pred)
    t = set(true)
    if not p and not t:
        return 1.0, 1.0
    tp = len(p & t)
    prec = tp / len(p) if p else 0.0
    rec = tp / len(t) if t else 0.0
    return prec, rec


# --------------------------------------------------------------------------- #
# Dataset-level metric
# --------------------------------------------------------------------------- #
def evaluate(
    pred: Mapping[Hashable, Iterable[Hashable]],
    true: Mapping[Hashable, Iterable[Hashable]],
    *,
    strict: bool = True,
) -> float:
    """Macro-averaged F_0.5 over every Source-1 entity in ``true``.

    The average is taken over **the keys of ``true``** -- i.e. over the evaluation
    set -- because the challenge requires every Source-1 entity to be scored.  An
    entity absent from ``pred`` is treated as an empty prediction, which is what
    a missing row would mean semantically (the real submission validator would
    reject the file outright; see ``utils/validate_submission.py``).

    Parameters
    ----------
    strict:
        If True (default), raise when ``pred`` contains entities not present in
        ``true``.  Silent extra keys are almost always an alignment bug, and an
        alignment bug that scores fine locally but fails on the leaderboard is
        the most expensive kind.
    """
    if strict:
        extra = set(pred) - set(true)
        if extra:
            raise ValueError(
                f"pred contains {len(extra)} entity ids absent from the truth "
                f"mapping, e.g. {sorted(map(str, extra))[:5]}. This is an "
                f"alignment bug; pass strict=False only if intentional."
            )

    if not true:
        raise ValueError("empty truth mapping -- nothing to evaluate")

    empty: frozenset = frozenset()
    total = 0.0
    for key, t in true.items():
        total += fbeta_single(pred.get(key, empty), t)
    return total / len(true)


def evaluate_cv(
    oof_pred: Mapping[Hashable, Iterable[Hashable]],
    true: Mapping[Hashable, Iterable[Hashable]],
    folds: Mapping[Hashable, int] | None = None,
    *,
    strict: bool = True,
) -> Dict[str, object]:
    """Score out-of-fold predictions, overall and per fold.

    Returns a dict with ``overall``, and when ``folds`` is supplied also
    ``fold_scores`` (list), ``fold_mean``, ``fold_std``, ``fold_sizes``.

    The *overall* number is the honest one to report: it is the macro-average
    over every entity exactly once.  ``fold_mean`` differs from it slightly when
    folds are unequal in size, and ``fold_std`` is the quantity to watch when
    deciding whether a CV improvement is real or noise.
    """
    out: Dict[str, object] = {"overall": evaluate(oof_pred, true, strict=strict)}

    if folds is None:
        return out

    by_fold: Dict[int, Dict[Hashable, Iterable[Hashable]]] = defaultdict(dict)
    for key, t in true.items():
        by_fold[folds[key]][key] = t

    fold_ids = sorted(by_fold)
    scores, sizes = [], []
    for f in fold_ids:
        sub_true = by_fold[f]
        sub_pred = {k: oof_pred[k] for k in sub_true if k in oof_pred}
        scores.append(evaluate(sub_pred, sub_true, strict=False))
        sizes.append(len(sub_true))

    out.update(
        fold_ids=fold_ids,
        fold_scores=scores,
        fold_sizes=sizes,
        fold_mean=float(np.mean(scores)),
        fold_std=float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
    )
    return out


def evaluate_by_group(
    pred: Mapping[Hashable, Iterable[Hashable]],
    true: Mapping[Hashable, Iterable[Hashable]],
    groups: Mapping[Hashable, Hashable],
) -> Dict[Hashable, Dict[str, float]]:
    """Macro F_0.5 broken down by an arbitrary per-entity group label.

    Use for the slices that actually matter here: ``country`` (does India lag
    US? what does that imply for France?), ``n_true`` bucket (singleton / 1 /
    2 / 3+), candidate-count bucket, and name-length bucket.

    Returns ``{group: {"f05", "precision", "recall", "n"}}`` where precision and
    recall are themselves macro-averaged, for interpretability.
    """
    acc: Dict[Hashable, list] = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    empty: frozenset = frozenset()

    for key, t in true.items():
        p = pred.get(key, empty)
        g = groups[key]
        prec, rec = precision_recall_single(p, t)
        a = acc[g]
        a[0] += fbeta_single(p, t)
        a[1] += prec
        a[2] += rec
        a[3] += 1

    out: Dict[Hashable, Dict[str, float]] = {}
    for g, (s, pr, rc, n) in acc.items():
        out[g] = {
            "f05": s / n,
            "precision": pr / n,
            "recall": rc / n,
            "n": float(n),
        }
    return out


# --------------------------------------------------------------------------- #
# Secondary diagnostics (NOT the competition metric -- for error analysis only)
# --------------------------------------------------------------------------- #
def micro_prf(
    pred: Mapping[Hashable, Iterable[Hashable]],
    true: Mapping[Hashable, Iterable[Hashable]],
) -> Dict[str, float]:
    """Pair-level micro precision / recall / F_0.5 / F_1.

    Deliberately *not* the scored metric.  It is useful because it is insensitive
    to the per-entity macro weighting, so comparing it against ``evaluate`` tells
    you whether a change helped the many-match entities or the singletons.
    """
    tp = fp = fn = 0
    empty: frozenset = frozenset()
    for key, t in true.items():
        p, t = set(pred.get(key, empty)), set(t)
        tp += len(p & t)
        fp += len(p - t)
        fn += len(t - p)

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f05 = (
        _ONE_PLUS_BETA_SQ * prec * rec / (_BETA_SQ * prec + rec)
        if (prec + rec) > 0
        else 0.0
    )
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "micro_precision": prec,
        "micro_recall": rec,
        "micro_f05": f05,
        "micro_f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def candidate_recall_ceiling(
    candidates: Mapping[Hashable, Iterable[Hashable]],
    true: Mapping[Hashable, Iterable[Hashable]],
) -> Dict[str, float]:
    """Upper bound on achievable score given a blocking/candidate set.

    ``max_f05`` is the score of the *oracle* matcher that keeps exactly the true
    matches present in the candidate set and nothing else.  This is the single
    most important number for judging a blocking stage: no amount of downstream
    modelling can exceed it.

    Also reports pair recall and the reduction ratio proxy (mean candidates per
    Source-1 entity), which is what the organisers audit ``candidate_pairs.tsv``
    for.
    """
    empty: frozenset = frozenset()
    oracle: Dict[Hashable, set] = {}
    covered = total = 0
    n_cands = 0

    for key, t in true.items():
        c = set(candidates.get(key, empty))
        t = set(t)
        hit = c & t
        oracle[key] = hit
        covered += len(hit)
        total += len(t)
        n_cands += len(c)

    n_entities = len(true)
    return {
        "max_f05": evaluate(oracle, true, strict=False),
        "pair_recall": covered / total if total else 1.0,
        "n_true_pairs": total,
        "n_recovered_pairs": covered,
        "n_missed_pairs": total - covered,
        "mean_candidates_per_entity": n_cands / n_entities if n_entities else 0.0,
        "total_candidate_pairs": n_cands,
    }
