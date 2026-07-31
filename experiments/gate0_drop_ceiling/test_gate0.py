from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from config import default_experiment
from dnn_profile import minimum_good_deadline_ms
from experiments.gate0_drop_ceiling import run


@pytest.fixture(scope="module")
def small_results() -> pd.DataFrame:
    return run.calculate_experiment(
        sample_count=8_000,
        seed=12345,
        deadline_ratios=(1.2, 1.5, 2.0),
        epsilons=run.EPSILONS,
        chunk_size=2_000,
    )


def test_p2_energy_never_exceeds_p1(small_results: pd.DataFrame):
    feasible = small_results[small_results["status"] == "feasible"]
    assert len(feasible) > 0
    assert np.all(
        feasible["p2_energy_mean"].to_numpy()
        <= feasible["p1_energy_mean"].to_numpy() + 1e-14
    )


def test_p2_saving_is_monotone_in_epsilon(small_results: pd.DataFrame):
    for _, group in small_results.groupby("deadline_ratio"):
        saving = (
            group[group["status"] == "feasible"]
            .sort_values("epsilon")["p2_saving_percent"]
            .to_numpy()
        )
        assert np.all(np.diff(saving) >= -1e-12)


def test_epsilon_below_forced_floor_is_infeasible():
    config = default_experiment()
    rates = np.full(200, config.channel.r_bad_mbps * 0.5)
    profile = config.devices[0].profile
    d_min = minimum_good_deadline_ms(
        profile, config.channel.r_good_mbps, config.channel.tx_power_w
    )
    evaluation = run.evaluate_deadline(
        rates,
        profile,
        deadline_ms=1.2 * d_min,
        tx_power_w=config.channel.tx_power_w,
        chunk_size=50,
    )
    assert evaluation.epsilon_min > 0
    row = run.evaluate_epsilon(evaluation, evaluation.epsilon_min / 2.0)
    assert row["status"] == "infeasible"
    assert np.isnan(row["p2_saving_percent"])


def test_p2_saving_is_nonnegative(small_results: pd.DataFrame):
    feasible = small_results[small_results["status"] == "feasible"]
    assert np.all(feasible["p2_saving_percent"].to_numpy() >= -1e-12)


def test_rho_is_not_an_input_or_result_axis(small_results: pd.DataFrame):
    assert "rho" not in small_results.columns
    for function in (
        run.sample_stationary_rates,
        run.evaluate_deadline,
        run.calculate_experiment,
    ):
        assert "rho" not in inspect.signature(function).parameters


def test_same_seed_reproduces_identical_rates_and_results():
    first_rates = run.sample_stationary_rates(2_000, 9182)
    second_rates = run.sample_stationary_rates(2_000, 9182)
    np.testing.assert_array_equal(first_rates, second_rates)

    arguments = {
        "sample_count": 2_000,
        "seed": 9182,
        "deadline_ratios": (1.5,),
        "epsilons": (0.005, 0.01),
        "chunk_size": 500,
    }
    pd.testing.assert_frame_equal(
        run.calculate_experiment(**arguments),
        run.calculate_experiment(**arguments),
        check_exact=True,
    )


def test_fractional_budget_uses_fractional_boundary_item():
    savings = np.array([10.0, 4.0, 1.0])
    oracle = run.compute_fractional_oracle(
        savings_descending=savings,
        n_discretionary_drops=0.25,
        sample_count=100,
        p1_energy_mean=1.0,
    )
    assert oracle["p2_energy_mean"] == pytest.approx(0.975)
    assert oracle["p2_saving_percent"] == pytest.approx(2.5)
    assert oracle["boundary_saving_lambda"] == pytest.approx(10.0)


def test_epsilon_equal_forced_floor_is_feasible_with_zero_p2_saving():
    config = default_experiment()
    rates = run.sample_stationary_rates(4_000, 771)
    profile = config.devices[0].profile
    d_min = minimum_good_deadline_ms(
        profile, config.channel.r_good_mbps, config.channel.tx_power_w
    )
    evaluation = run.evaluate_deadline(
        rates,
        profile,
        deadline_ms=1.2 * d_min,
        tx_power_w=config.channel.tx_power_w,
        chunk_size=1_000,
    )
    row = run.evaluate_epsilon(evaluation, evaluation.epsilon_min)
    assert row["status"] == "feasible"
    assert row["discretionary_budget"] == 0
    assert row["n_discretionary_drops"] == 0
    assert row["p2_saving_percent"] == pytest.approx(0.0, abs=1e-14)
    if evaluation.feasible_count:
        assert row["boundary_saving_lambda"] > 0
