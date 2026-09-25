"""Exact check of the expected-F_0.5 solver against brute-force enumeration."""
import itertools
import math

import numpy as np

from src.decision import expected_f05_best_k, select_sets
from src.metrics import f05_single


def brute(p):
    """E[F] of every prefix by enumerating all 2^n truth outcomes."""
    n = len(p)
    best_k, best = 0, -1.0
    for k in range(n + 1):
        e = 0.0
        for bits in itertools.product([0, 1], repeat=n):
            pr = math.prod(pi if b else 1 - pi for pi, b in zip(p, bits))
            truth = {i for i, b in enumerate(bits) if b}
            e += pr * f05_single(set(range(k)), truth)
        if e > best + 1e-12:
            best_k, best = k, e
    return best_k, best


def test_against_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = rng.integers(0, 9)
        p = np.sort(rng.beta(0.6, 0.6, size=n))[::-1].copy()
        k, v = expected_f05_best_k(p, 0.0)
        bk, bv = brute(p)
        assert math.isclose(v, bv, abs_tol=1e-9), (p, v, bv)
        assert k == bk or math.isclose(v, bv, abs_tol=1e-9)


def test_intuitions():
    # A lone 0.4 candidate: abstain under independence (0.6 > 0.4) ...
    assert expected_f05_best_k(np.array([0.4]), 0.0)[0] == 0
    # ... but predict when true matches are expected outside the candidate list.
    assert expected_f05_best_k(np.array([0.4]), 1.0)[0] == 1
    # Five candidates at 0.45: take them all rather than none.
    assert expected_f05_best_k(np.full(5, 0.45), 0.0)[0] >= 3
    # Everything tiny: abstain.
    assert expected_f05_best_k(np.array([0.05, 0.03]), 0.0)[0] == 0


def test_select_sets_shapes():
    ent = np.array([1, 1, 1, 2, 2])
    cand = np.array([10, 11, 12, 20, 21])
    prob = np.array([0.9, 0.1, 0.8, 0.02, 0.01])
    sets, uniq, ef = select_sets(ent, cand, prob)
    assert set(sets[1].tolist()) == {10, 12}
    assert sets[2].size == 0


if __name__ == "__main__":
    test_against_brute_force(); test_intuitions(); test_select_sets_shapes(); print("decision tests ok")
