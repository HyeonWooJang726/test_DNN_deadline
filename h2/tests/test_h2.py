import sys
from dataclasses import replace
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
H2_DIR = ROOT_DIR / "h2"
for path in (ROOT_DIR, H2_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pytest

from config import DeviceConfig
from h2_channel import generate_h2_trace, generate_h2_traces, trace_hash
from h2_config import default_h2_experiment
from h2_simulator import (
    h2_decomposition_metrics,
    h2_sanity_rows,
    independent_policy_sets,
    minimum_h2_deadline_ms,
    shared_action_options,
    simulate_h2_trace,
    solve_j1,
)
from simulator import simulate_trace


def test_h2_capacity_grid_contains_every_equal_share():
    config = default_h2_experiment("quick")
    assert all(config.sweep.capacity_bins % n == 0 for n in (2, 4, 8))
    with pytest.raises(ValueError, match="does not contain 1/N"):
        replace(config.sweep, capacity_bins=201)


def test_h2_common_traces_match_and_independent_traces_differ():
    config = default_h2_experiment("smoke")
    common = generate_h2_traces(
        config.channel, 0.975, 1_000, 1701, 4, "common"
    )
    independent = generate_h2_traces(
        config.channel, 0.975, 1_000, 1701, 4, "independent"
    )
    assert len({trace_hash(trace) for trace in common}) == 1
    assert len({trace_hash(trace) for trace in independent}) == 4
    repeated = generate_h2_trace(config.channel, 0.975, 1_000, 1701, 3)
    np.testing.assert_array_equal(repeated.state, independent[3].state)
    np.testing.assert_array_equal(repeated.rate_mbps, independent[3].rate_mbps)


def test_h2_n1_independent_path_bitwise_reduces_to_h1_p1_p2():
    config = default_h2_experiment("smoke")
    trace = generate_h2_trace(config.channel, 0.75, 2_000, 88, 0)
    d_min = minimum_h2_deadline_ms(
        config.profile, config.channel, config.sweep.server_total_speedup, 1
    )
    deadline = 2.0 * d_min
    h1 = simulate_trace(
        trace,
        DeviceConfig(profile=config.profile),
        config.channel,
        deadline,
        2.0,
        0.15,
        "drop",
        config.sweep.lambda_iterations * (10.0,),
        88,
        0.75,
        0,
        0.005,
    )
    i1, i2, _ = independent_policy_sets(
        (trace,),
        config.profile,
        config.channel,
        deadline,
        0.15,
        "drop",
        config.sweep.server_total_speedup,
    )
    for h1_name, h2_result in (
        ("P1", i1.per_device[0]),
        ("P2", i2.per_device[0]),
    ):
        expected = h1.policies[h1_name]
        np.testing.assert_array_equal(h2_result.violate, expected.violate)
        np.testing.assert_array_equal(h2_result.split_p, expected.split_p)
        np.testing.assert_array_equal(h2_result.local_mode, expected.local_mode)
        np.testing.assert_array_equal(h2_result.energy_j, expected.energy_j)


def _small_case(n_devices=2, t_slots=200):
    config = default_h2_experiment("smoke")
    traces = generate_h2_traces(
        config.channel, 0.0, t_slots, 1701, n_devices, "independent"
    )
    d_min = minimum_h2_deadline_ms(
        config.profile,
        config.channel,
        config.sweep.server_total_speedup,
        n_devices,
    )
    return config, traces, 1.35 * d_min


def test_i1_actions_are_representable_after_capacity_rounding():
    for n_devices in (2, 4, 8):
        config, traces, deadline = _small_case(n_devices, 100)
        i1, _, _ = independent_policy_sets(
            traces,
            config.profile,
            config.channel,
            deadline,
            0.15,
            "drop",
            config.sweep.server_total_speedup,
        )
        total_units = np.zeros(100, dtype=np.int64)
        mode_count = len(config.profile.local_modes)
        for device, trace in enumerate(traces):
            options = shared_action_options(
                config.profile,
                trace.rate_mbps,
                deadline,
                config.channel.tx_power_w,
                "drop",
                config.sweep.server_total_speedup,
                config.sweep.capacity_bins,
            )
            result = i1.per_device[device]
            meet = ~result.violate
            action = result.split_p * mode_count + result.local_mode
            units = options.capacity_units[np.arange(100), np.maximum(action, 0)]
            assert np.all(units[meet] <= config.sweep.capacity_bins // n_devices)
            total_units[meet] += units[meet]
        assert np.all(total_units <= config.sweep.capacity_bins)


def test_joint_grid_200_and_1000_energy_differ_below_point_one_percent():
    config = default_h2_experiment("smoke")
    traces = generate_h2_traces(
        config.channel, 0.0, 200, 1701, 2, "common"
    )
    deadline = 1.2 * minimum_h2_deadline_ms(
        config.profile,
        config.channel,
        config.sweep.server_total_speedup,
        2,
    )
    i1, _, _ = independent_policy_sets(
        traces,
        config.profile,
        config.channel,
        deadline,
        0.15,
        "drop",
        config.sweep.server_total_speedup,
    )

    def solve_at(bins):
        options = tuple(
            shared_action_options(
                config.profile,
                trace.rate_mbps,
                deadline,
                config.channel.tx_power_w,
                "drop",
                config.sweep.server_total_speedup,
                bins,
            )
            for trace in traces
        )
        return solve_j1(
            i1,
            options,
            traces,
            config.profile,
            config.channel,
            "drop",
            bins,
        ).system_mean_energy_j

    coarse = solve_at(200)
    fine = solve_at(1_000)
    assert abs(coarse - fine) / max(abs(fine), 1e-15) < 0.001


def test_h2_small_simulation_capacity_budget_and_energy_sanity():
    config, traces, deadline = _small_case(2, 200)
    sweep = replace(config.sweep, t_slots=200, lambda_iterations=3)
    simulation = simulate_h2_trace(
        traces,
        config.profile,
        config.channel,
        deadline,
        0.15,
        "late",
        sweep,
    )
    failures = [
        row
        for row in h2_sanity_rows(simulation, 0.15, 0.005, 1e-12)
        if not row["passed"]
    ]
    assert failures == []
    metrics = h2_decomposition_metrics(simulation)
    assert (
        metrics["I1_violation_set_sha256_by_device_json"]
        == metrics["J1_violation_set_sha256_by_device_json"]
    )


def test_pareto_pruning_preserves_joint_dp_results_exactly():
    config, traces, deadline = _small_case(2, 120)
    base = replace(
        config.sweep,
        t_slots=120,
        lambda_iterations=3,
        pareto_pruning=False,
    )
    pruned = replace(base, pareto_pruning=True)
    without_pruning = simulate_h2_trace(
        traces,
        config.profile,
        config.channel,
        deadline,
        0.15,
        "late",
        base,
    )
    with_pruning = simulate_h2_trace(
        traces,
        config.profile,
        config.channel,
        deadline,
        0.15,
        "late",
        pruned,
    )

    assert without_pruning.policies.keys() == with_pruning.policies.keys()
    for name in without_pruning.policies:
        expected = without_pruning.policies[name]
        actual = with_pruning.policies[name]
        np.testing.assert_array_equal(
            actual.allocation_fraction, expected.allocation_fraction
        )
        for expected_device, actual_device in zip(
            expected.per_device, actual.per_device
        ):
            np.testing.assert_array_equal(
                actual_device.violate, expected_device.violate
            )
            np.testing.assert_array_equal(
                actual_device.split_p, expected_device.split_p
            )
            np.testing.assert_array_equal(
                actual_device.local_mode, expected_device.local_mode
            )
            np.testing.assert_array_equal(
                actual_device.energy_j, expected_device.energy_j
            )
