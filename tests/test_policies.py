import itertools

import numpy as np

from policies import exact_burst_selection, lagrangian_burst_selection


def _brute_force(weights, allowed, budget):
    best_value = -1.0
    best = None
    n = len(weights)
    for bits in itertools.product((0, 1), repeat=n):
        chosen = np.array(bits, dtype=bool)
        if chosen.sum() > budget or np.any(chosen & ~allowed) or np.any(chosen[:-1] & chosen[1:]):
            continue
        value = float(weights[chosen].sum())
        if value > best_value:
            best_value = value
            best = chosen
    return best_value, best


def test_p2prime_exact_dp_hand_checked_t20_case():
    # Three separated peaks. With B=3 the unique hand-checked optimum is
    # t={1,5,10}: 9 + 12 + 20 = 41.
    weights = np.zeros(20)
    weights[[0, 1, 2]] = [8, 9, 8]
    weights[[4, 5, 6]] = [7, 12, 7]
    weights[10] = 20
    selected = exact_burst_selection(weights, np.ones(20, dtype=bool), budget=3)
    assert set(np.flatnonzero(selected)) == {1, 5, 10}
    assert not np.any(selected[:-1] & selected[1:])


def test_exact_dp_matches_bruteforce_with_disallowed_slots():
    weights = np.array([4.0, 9.0, 6.0, 7.0, 12.0, 3.0, 5.0, 11.0])
    allowed = np.array([1, 1, 0, 1, 1, 1, 1, 1], dtype=bool)
    expected_value, _ = _brute_force(weights, allowed, budget=3)
    selected = exact_burst_selection(weights, allowed, budget=3)
    assert weights[selected].sum() == expected_value


def test_lagrangian_large_path_respects_budget_and_adjacency():
    rng = np.random.default_rng(7)
    weights = rng.random(3_000)
    allowed = np.ones(3_000, dtype=bool)
    selected = lagrangian_burst_selection(weights, allowed, budget=200)
    assert selected.sum() <= 200
    assert not np.any(selected[:-1] & selected[1:])


def test_lagrangian_t2000_matches_exact_selected_value_to_four_decimals():
    rng = np.random.default_rng(7)
    weights = rng.random(2_000)
    allowed = rng.random(2_000) > 0.1
    exact = exact_burst_selection(weights, allowed, budget=200)
    lagrangian = lagrangian_burst_selection(weights, allowed, budget=200)

    assert round(float(weights[lagrangian].sum()), 4) == round(
        float(weights[exact].sum()), 4
    )
