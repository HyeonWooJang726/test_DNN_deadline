"""One-device simulation kernel and analytical pre-flight checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from channel import ChannelTrace
from config import ChannelConfig, DeviceConfig
from dnn_profile import (
    SlotCosts,
    compute_slot_costs,
    infeasible_probability_with_jitter,
    state_feasibility,
)
from policies import (
    P0RandomDrop,
    P1AlwaysMeet,
    P2BurstOracle,
    P2OfflineOracle,
    P3OnlineThreshold,
    PolicyResult,
)


@dataclass(frozen=True)
class PreflightResult:
    good_feasible: bool
    bad_feasible: bool
    good_points: tuple[int, ...]
    bad_points: tuple[int, ...]
    forced_violation_expected: float
    valid: bool
    reason: str


@dataclass(frozen=True)
class SimulationResult:
    costs: SlotCosts
    policies: dict[str, PolicyResult]
    forced_count: int
    discretionary_budget: int


def preflight_check(
    device: DeviceConfig,
    channel: ChannelConfig,
    deadline_ms: float,
    epsilon: float,
) -> PreflightResult:
    good_ok, good_points = state_feasibility(
        device.profile, channel.r_good_mbps, deadline_ms, channel.tx_power_w
    )
    bad_ok, bad_points = state_feasibility(
        device.profile, channel.r_bad_mbps, deadline_ms, channel.tx_power_w
    )
    good_forced_probability = infeasible_probability_with_jitter(
        device.profile,
        channel.r_good_mbps,
        deadline_ms,
        channel.rate_jitter_sigma_log,
    )
    bad_forced_probability = infeasible_probability_with_jitter(
        device.profile,
        channel.r_bad_mbps,
        deadline_ms,
        channel.rate_jitter_sigma_log,
    )
    forced = (
        (1.0 - channel.pi_bad) * good_forced_probability
        + channel.pi_bad * bad_forced_probability
    )
    valid = forced < epsilon / 2.0
    if valid:
        reason = "valid: expected forced-violation rate is below epsilon/2"
    else:
        missing = []
        if not good_ok:
            missing.append("Good")
        if not bad_ok:
            missing.append("Bad")
        if missing:
            cause = f"no feasible split/mode at nominal rate in {','.join(missing)} state(s)"
        else:
            cause = "jitter-induced infeasibility"
        reason = (
            f"invalid: {cause}; expected forced rate {forced:.6f} "
            f">= epsilon/2 {epsilon/2:.6f}"
        )
    return PreflightResult(
        good_feasible=good_ok,
        bad_feasible=bad_ok,
        good_points=good_points,
        bad_points=bad_points,
        forced_violation_expected=float(forced),
        valid=bool(valid),
        reason=reason,
    )


def _combination_rng(
    seed: int,
    deadline_ratio: float,
    epsilon: float,
    skip_mode: str,
    rho: float,
    device_index: int,
) -> np.random.Generator:
    # SeedSequence avoids Python's process-randomized hash(), giving bitwise
    # reproducibility across independent interpreter runs. Including rho and
    # device_index intentionally changes P0's random stream from the v0.1-h1
    # results and prevents collisions ahead of the H2 multi-device extension.
    entropy = [
        int(seed),
        int(round(deadline_ratio * 10_000)),
        int(round(epsilon * 1_000_000)),
        0 if skip_mode == "drop" else 1,
        int(round((rho + 1.0) * 1_000_000)),
        int(device_index),
    ]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def simulate_trace(
    trace: ChannelTrace,
    device: DeviceConfig,
    channel: ChannelConfig,
    deadline_ms: float,
    deadline_ratio: float,
    epsilon: float,
    skip_mode: str,
    p3_v_values: tuple[float, ...],
    seed: int,
    rho: float,
    device_index: int,
    p3_violation_tolerance: float = 0.005,
) -> SimulationResult:
    costs = compute_slot_costs(
        device.profile,
        trace.rate_mbps,
        deadline_ms,
        channel.tx_power_w,
        skip_mode,
    )
    forced_count = int((~costs.feasible).sum())
    total_budget = int(np.floor(epsilon * len(trace.rate_mbps)))
    discretionary_budget = total_budget - forced_count
    if discretionary_budget < 0:
        raise ValueError(
            f"actual forced count {forced_count} exceeds floor(epsilon*T)={total_budget}; "
            "the combination must be marked invalid"
        )

    policies = {
        "P1": P1AlwaysMeet().run(costs),
        "P0": P0RandomDrop(
            discretionary_budget,
            _combination_rng(
                seed,
                deadline_ratio,
                epsilon,
                skip_mode,
                rho,
                device_index,
            ),
        ).run(costs),
        "P2": P2OfflineOracle(discretionary_budget).run(costs),
        "P2prime": P2BurstOracle(discretionary_budget).run(costs),
        "P3": P3OnlineThreshold(
            epsilon,
            p3_v_values,
            discretionary_budget,
            p3_violation_tolerance,
        ).run(costs),
    }
    return SimulationResult(costs, policies, forced_count, discretionary_budget)
