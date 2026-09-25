"""Unit tests: the local metric must match the problem statement exactly.

Run:  .venv/bin/python -m pytest -q tests/   (or: python -m tests.test_metrics)
"""

from __future__ import annotations

import math
import random

from src.metrics import (
    candidate_recall_ceiling,
    evaluate,
    evaluate_by_group,
    evaluate_cv,
    f05_single,
    fbeta_single,
    micro_prf,
)


def naive_f(pred, true, beta=0.5):
    """Textbook definition, used as the reference implementation."""
    p, t = set(pred), set(true)
    if not p and not t:
        return 1.0
    if not p or not t:
        return 0.0
    tp = len(p & t)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(p), tp / len(t)
    b2 = beta * beta
    return (1 + b2) * prec * rec / (b2 * prec + rec)


def test_problem_statement_example():
    # Predicted [S2-00047, S2-00193, S3-00812], truth [S2-00047, S3-00812] -> 0.714
    s = f05_single(["S2-00047", "S2-00193", "S3-00812"], ["S2-00047", "S3-00812"])
    assert math.isclose(s, 2.5 / 3.5)
    assert round(s, 3) == 0.714


def test_singleton_conventions():
    assert f05_single([], []) == 1.0            # correct empty prediction
    assert f05_single(["S2-1"], []) == 0.0      # any match on a singleton
    assert f05_single([], ["S2-1"]) == 0.0      # missed everything
    assert f05_single(["S2-9"], ["S2-1"]) == 0.0


def test_precision_weighting():
    # |T|=1: exact hit 1.0; one extra wrong id -> 1.25/(0.25+2)
    assert f05_single(["a"], ["a"]) == 1.0
    assert math.isclose(f05_single(["a", "b"], ["a"]), 1.25 / 2.25)
    # Half recall, full precision beats full recall, half precision.
    assert f05_single(["a"], ["a", "b"]) > f05_single(["a", "b"], ["a"])


def test_closed_form_matches_naive_random():
    rng = random.Random(0)
    universe = [f"S2-{i}" for i in range(12)]
    for _ in range(5000):
        p = rng.sample(universe, rng.randint(0, 8))
        t = rng.sample(universe, rng.randint(0, 8))
        assert math.isclose(fbeta_single(p, t), naive_f(p, t), abs_tol=1e-12)
        assert math.isclose(fbeta_single(p, t, beta=1.0), naive_f(p, t, 1.0), abs_tol=1e-12)


def test_duplicates_collapsed():
    assert f05_single(["a", "a"], ["a"]) == 1.0


def test_macro_average_and_missing_rows():
    true = {"S1-1": {"a"}, "S1-2": set(), "S1-3": {"b", "c"}}
    pred = {"S1-1": {"a"}, "S1-3": {"b"}}          # S1-2 missing -> empty -> 1.0
    exp = (1.0 + 1.0 + f05_single({"b"}, {"b", "c"})) / 3
    assert math.isclose(evaluate(pred, true), exp)


def test_extra_keys_rejected():
    try:
        evaluate({"S1-X": {"a"}}, {"S1-1": set()})
    except ValueError:
        return
    raise AssertionError("extra prediction keys must raise in strict mode")


def test_cv_and_groups():
    true = {"1": {"a"}, "2": set(), "3": {"b"}, "4": {"c"}}
    pred = {"1": {"a"}, "2": {"x"}, "3": {"b"}, "4": set()}
    cv = evaluate_cv(pred, true, folds={"1": 0, "2": 0, "3": 1, "4": 1})
    assert math.isclose(cv["overall"], 0.5)
    assert cv["fold_scores"] == [0.5, 0.5]
    g = evaluate_by_group(pred, true, {"1": "US", "2": "US", "3": "IN", "4": "IN"})
    assert math.isclose(g["US"]["f05"], 0.5) and g["US"]["n"] == 2


def test_diagnostics():
    true = {"1": {"a", "b"}, "2": set()}
    cands = {"1": {"a", "z"}, "2": {"q"}}
    c = candidate_recall_ceiling(cands, true)
    assert c["pair_recall"] == 0.5
    assert math.isclose(c["max_f05"], (f05_single({"a"}, {"a", "b"}) + 1.0) / 2)
    m = micro_prf({"1": {"a", "z"}, "2": set()}, true)
    assert m["tp"] == 1 and m["fp"] == 1 and m["fn"] == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
