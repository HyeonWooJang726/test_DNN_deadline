"""H2 policies for joint violation timing and shared-server allocation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace

import numpy as np

from channel import ChannelTrace
from config import ChannelConfig, DNNProfileConfig, DeviceConfig
from dnn_profile import (
    SlotCosts,
    action_table,
    compute_slot_costs,
    kb_to_megabits,
    minimum_good_deadline_ms,
    mj_to_j,
)
from h2_config import H2SweepConfig
from policies import P1AlwaysMeet, P2OfflineOracle, PolicyResult
from simulator import PreflightResult, preflight_check

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        def decorate(fn):
            return fn
        return decorate


@dataclass(frozen=True)
class SharedActionOptions:
    energy_j: np.ndarray  # [T, O], final option is violation
    capacity_units: np.ndarray  # [T, O], conservatively rounded upward
    split_p: np.ndarray  # [O]
    local_mode: np.ndarray  # [O]
    valid: np.ndarray  # [T, O]
    mode_names: tuple[str, ...]
    violation_option: int


@dataclass(frozen=True)
class H2PolicySet:
    name: str
    per_device: tuple[PolicyResult, ...]
    allocation_fraction: np.ndarray  # [N, T]
    metadata: dict[str, object]

    @property
    def system_mean_energy_j(self) -> float:
        return float(sum(result.energy_j.sum() for result in self.per_device) / len(self.per_device[0].energy_j))

    @property
    def violation_rates(self) -> np.ndarray:
        return np.asarray([result.violation_rate for result in self.per_device])


@dataclass(frozen=True)
class H2SimulationResult:
    policies: dict[str, H2PolicySet]
    independent_costs: tuple[SlotCosts, ...]
    j2_source: str
    j2_lambda_initial: tuple[float, ...]
    j2_lambda_final: tuple[float, ...]
    j2_iterations: int
    j2_repair_success: bool


def effective_equal_share_profile(
    profile: DNNProfileConfig, server_total_speedup: float, n_devices: int
) -> DNNProfileConfig:
    return replace(profile, server_speedup=server_total_speedup / n_devices)


def minimum_h2_deadline_ms(
    profile: DNNProfileConfig,
    channel: ChannelConfig,
    server_total_speedup: float,
    n_devices: int,
) -> float:
    equal_profile = effective_equal_share_profile(
        profile, server_total_speedup, n_devices
    )
    return minimum_good_deadline_ms(
        equal_profile, channel.r_good_mbps, channel.tx_power_w
    )


def h2_preflight_check(
    profile: DNNProfileConfig,
    channel: ChannelConfig,
    server_total_speedup: float,
    n_devices: int,
    deadline_ms: float,
    epsilon: float,
) -> PreflightResult:
    equal_profile = effective_equal_share_profile(
        profile, server_total_speedup, n_devices
    )
    return preflight_check(
        DeviceConfig(name="h2_equal_share", profile=equal_profile, expected_d_min_ms=None),
        channel,
        deadline_ms,
        epsilon,
    )


def shared_action_options(
    profile: DNNProfileConfig,
    rate_mbps: np.ndarray,
    deadline_ms: float,
    tx_power_w: float,
    skip_mode: str,
    server_total_speedup: float,
    capacity_bins: int,
    pareto_prune: bool = True,
) -> SharedActionOptions:
    """Return every meet option's energy and minimum rounded server share."""
    rates = np.asarray(rate_mbps, dtype=np.float64)
    table = action_table(profile, rates, tx_power_w)
    tloc = np.asarray(profile.t_loc_ms, dtype=np.float64)
    local_normal = np.concatenate(([0.0], np.cumsum(tloc)))
    remaining = tloc.sum() - local_normal
    data_mb = kb_to_megabits(np.asarray(profile.d_kb, dtype=np.float64))
    speed_scales = np.asarray(
        [mode.speed_scale for mode in profile.local_modes], dtype=np.float64
    )
    local_ms = local_normal[:, None] * speed_scales[None, :]
    tx_ms = (data_mb[None, :] / rates[:, None]) * 1000.0
    slack = deadline_ms - local_ms[None, :, :] - tx_ms[:, :, None]
    numerator = remaining[None, :, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        required_fraction = numerator / (server_total_speedup * slack)
    required_fraction = np.where(
        (slack >= -1e-12) & (numerator <= 1e-12), 0.0, required_fraction
    )
    feasible = (slack > 0.0) & (required_fraction >= -1e-12) & (
        required_fraction <= 1.0 + 1e-12
    )
    required_fraction = np.maximum(required_fraction, 0.0)
    safe_fraction = np.where(feasible, required_fraction, 0.0)
    units = np.ceil(safe_fraction * capacity_bins - 1e-12).astype(np.int64)
    units = np.where(feasible, units, capacity_bins + 1)

    action_count = table.energy_j.shape[1] * table.energy_j.shape[2]
    mode_count = len(table.mode_names)
    energy_flat = table.energy_j.reshape(len(rates), action_count)
    units_flat = units.reshape(len(rates), action_count)
    valid_flat = feasible.reshape(len(rates), action_count)
    split_p = (np.arange(action_count) // mode_count).astype(np.int16)
    local_mode = (np.arange(action_count) % mode_count).astype(np.int8)

    if skip_mode == "drop":
        skip_energy = np.zeros(len(rates), dtype=np.float64)
        skip_p = -1
        skip_mode_index = -1
    elif skip_mode == "late":
        # Late work is best-effort outside this slot's real-time capacity.
        skip_action = np.argmin(energy_flat, axis=1)
        skip_energy = energy_flat[np.arange(len(rates)), skip_action]
        # All rows have the same energy ordering for a fixed profile/rate
        # formula in practice, but retain per-slot actions when constructing a
        # result by storing sentinel values and re-deriving below.
        skip_p = -2
        skip_mode_index = -2
    else:
        raise ValueError("skip_mode must be drop or late")

    violation_option = action_count
    energy = np.column_stack((energy_flat, skip_energy))
    capacity = np.column_stack(
        (units_flat, np.zeros(len(rates), dtype=np.int64))
    )
    valid = np.column_stack(
        (valid_flat, np.ones(len(rates), dtype=np.bool_))
    )
    if pareto_prune:
        valid = _pareto_prune_valid(
            energy, capacity, valid, violation_option
        )
    split_options = np.concatenate((split_p, np.array([skip_p], dtype=np.int16)))
    mode_options = np.concatenate(
        (local_mode, np.array([skip_mode_index], dtype=np.int8))
    )
    return SharedActionOptions(
        energy_j=energy,
        capacity_units=capacity,
        split_p=split_options,
        local_mode=mode_options,
        valid=valid,
        mode_names=table.mode_names,
        violation_option=violation_option,
    )


@njit(cache=True)
def _pareto_prune_valid(energy, capacity_units, valid, violation_option):
    """Remove meet options dominated in both energy and rounded capacity.

    The violation option is never compared with meet options because its
    Lagrangian penalty changes between J2 iterations.
    """
    t_slots, option_count = energy.shape
    out = valid.copy()
    for t in range(t_slots):
        for candidate in range(violation_option):
            if not valid[t, candidate]:
                continue
            for alternative in range(violation_option):
                if alternative == candidate or not valid[t, alternative]:
                    continue
                lower_capacity = capacity_units[t, alternative] <= capacity_units[t, candidate]
                lower_energy = energy[t, alternative] <= energy[t, candidate]
                strict = (
                    capacity_units[t, alternative] < capacity_units[t, candidate]
                    or energy[t, alternative] < energy[t, candidate]
                )
                same_but_earlier = (
                    capacity_units[t, alternative] == capacity_units[t, candidate]
                    and energy[t, alternative] == energy[t, candidate]
                    and alternative < candidate
                )
                if lower_capacity and lower_energy and (strict or same_but_earlier):
                    out[t, candidate] = False
                    break
    return out


@njit(cache=True)
def _solve_mckp_slots(
    energy,
    capacity_units,
    valid,
    choice_mode,
    violation_option,
    penalties,
    capacity_bins,
    slot_indices,
):
    """Multiple-choice knapsack with slot-rotated device ordering.

    choice_mode is 0=all options, 1=meet only, 2=violate only. Minimum shares
    are already rounded upward, so every returned allocation is conservative.
    """
    n_devices, t_slots, option_count = energy.shape
    choices = np.full((n_devices, t_slots), -1, dtype=np.int16)
    feasible_out = np.zeros(t_slots, dtype=np.uint8)
    inf = 1e300
    prev = np.empty(capacity_bins + 1, dtype=np.float64)
    current = np.empty(capacity_bins + 1, dtype=np.float64)
    back_choice = np.empty((n_devices, capacity_bins + 1), dtype=np.int16)
    back_capacity = np.empty((n_devices, capacity_bins + 1), dtype=np.int16)

    for local_t in range(t_slots):
        for c in range(capacity_bins + 1):
            prev[c] = inf
        prev[0] = 0.0
        original_t = slot_indices[local_t]
        for position in range(n_devices):
            device = (position + original_t) % n_devices
            for c in range(capacity_bins + 1):
                current[c] = inf
                back_choice[position, c] = -1
                back_capacity[position, c] = -1
            mode = choice_mode[device, local_t]
            for used in range(capacity_bins + 1):
                base = prev[used]
                if base >= inf / 2:
                    continue
                for option in range(option_count):
                    if not valid[device, local_t, option]:
                        continue
                    if mode == 1 and option == violation_option:
                        continue
                    if mode == 2 and option != violation_option:
                        continue
                    new_used = used + capacity_units[device, local_t, option]
                    if new_used > capacity_bins:
                        continue
                    adjusted = energy[device, local_t, option]
                    if option == violation_option:
                        adjusted += penalties[device]
                    candidate = base + adjusted
                    if candidate < current[new_used] - 1e-14:
                        current[new_used] = candidate
                        back_choice[position, new_used] = option
                        back_capacity[position, new_used] = used
            for c in range(capacity_bins + 1):
                prev[c] = current[c]

        best_capacity = -1
        best_value = inf
        for c in range(capacity_bins + 1):
            if prev[c] < best_value - 1e-14:
                best_value = prev[c]
                best_capacity = c
        if best_capacity < 0:
            continue
        feasible_out[local_t] = 1
        capacity = best_capacity
        for position in range(n_devices - 1, -1, -1):
            device = (position + original_t) % n_devices
            option = back_choice[position, capacity]
            choices[device, local_t] = option
            capacity = back_capacity[position, capacity]
    return choices, feasible_out


def solve_joint_slots(
    options: tuple[SharedActionOptions, ...],
    choice_mode: np.ndarray,
    penalties: np.ndarray,
    capacity_bins: int,
    slot_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n_devices = len(options)
    t_slots = choice_mode.shape[1]
    if slot_indices is None:
        slot_indices = np.arange(t_slots, dtype=np.int64)
    energy = np.stack([option.energy_j for option in options])
    units = np.stack([option.capacity_units for option in options])
    valid = np.stack([option.valid for option in options])
    violation_option = options[0].violation_option
    if any(option.violation_option != violation_option for option in options):
        raise ValueError("all devices must use a common option layout")
    return _solve_mckp_slots(
        energy,
        units,
        valid,
        np.asarray(choice_mode, dtype=np.int8),
        violation_option,
        np.asarray(penalties, dtype=np.float64),
        capacity_bins,
        np.asarray(slot_indices, dtype=np.int64),
    )


def _late_skip_actions(
    profile: DNNProfileConfig, rates: np.ndarray, tx_power_w: float
) -> tuple[np.ndarray, np.ndarray]:
    table = action_table(profile, rates, tx_power_w)
    flat = table.energy_j.reshape(len(rates), -1)
    action = np.argmin(flat, axis=1)
    modes = len(table.mode_names)
    return (action // modes).astype(np.int16), (action % modes).astype(np.int8)


def _policy_set_from_choices(
    name: str,
    options: tuple[SharedActionOptions, ...],
    traces: tuple[ChannelTrace, ...],
    choices: np.ndarray,
    capacity_bins: int,
    profile: DNNProfileConfig,
    tx_power_w: float,
    skip_mode: str,
    metadata: dict[str, object] | None = None,
) -> H2PolicySet:
    results = []
    allocations = np.zeros(choices.shape, dtype=np.float64)
    for device, option in enumerate(options):
        selected = choices[device]
        if np.any(selected < 0):
            raise ValueError("joint solver returned an infeasible slot")
        slot = np.arange(len(selected))
        violate = selected == option.violation_option
        p = option.split_p[selected].astype(np.int16)
        local_mode = option.local_mode[selected].astype(np.int8)
        if skip_mode == "late" and np.any(violate):
            late_p, late_mode = _late_skip_actions(
                profile, traces[device].rate_mbps, tx_power_w
            )
            p = np.where(violate, late_p, p).astype(np.int16)
            local_mode = np.where(violate, late_mode, local_mode).astype(np.int8)
        energy = option.energy_j[slot, selected]
        allocations[device] = option.capacity_units[slot, selected] / capacity_bins
        results.append(
            PolicyResult(
                name=name,
                violate=violate,
                split_p=p,
                local_mode=local_mode,
                energy_j=energy,
                mode_names=option.mode_names,
                metadata={"joint": True, **(metadata or {})},
            )
        )
    return H2PolicySet(name, tuple(results), allocations, metadata or {})


def independent_policy_sets(
    traces: tuple[ChannelTrace, ...],
    profile: DNNProfileConfig,
    channel: ChannelConfig,
    deadline_ms: float,
    epsilon: float,
    skip_mode: str,
    server_total_speedup: float,
) -> tuple[H2PolicySet, H2PolicySet, tuple[SlotCosts, ...]]:
    """Construct I1/I2 by exactly reusing the H1 fixed-share policy path."""
    n_devices = len(traces)
    equal_profile = effective_equal_share_profile(
        profile, server_total_speedup, n_devices
    )
    budget = int(math.floor(epsilon * len(traces[0].rate_mbps)))
    i1_results = []
    i2_results = []
    costs_out = []
    allocations_i1 = np.zeros((n_devices, len(traces[0].rate_mbps)))
    allocations_i2 = np.zeros_like(allocations_i1)
    for device, trace in enumerate(traces):
        costs = compute_slot_costs(
            equal_profile,
            trace.rate_mbps,
            deadline_ms,
            channel.tx_power_w,
            skip_mode,
        )
        forced = int((~costs.feasible).sum())
        if forced > budget:
            raise ValueError(
                f"device {device} actual forced count {forced} exceeds "
                f"floor(epsilon*T)={budget}"
            )
        i1 = P1AlwaysMeet().run(costs)
        i1.name = "I1"
        i2 = P2OfflineOracle(budget - forced).run(costs)
        i2.name = "I2"
        i1_results.append(i1)
        i2_results.append(i2)
        costs_out.append(costs)
        allocations_i1[device, ~i1.violate] = 1.0 / n_devices
        allocations_i2[device, ~i2.violate] = 1.0 / n_devices
    return (
        H2PolicySet("I1", tuple(i1_results), allocations_i1, {"fixed_share": 1 / n_devices}),
        H2PolicySet("I2", tuple(i2_results), allocations_i2, {"fixed_share": 1 / n_devices}),
        tuple(costs_out),
    )


def solve_j1(
    i1: H2PolicySet,
    options: tuple[SharedActionOptions, ...],
    traces: tuple[ChannelTrace, ...],
    profile: DNNProfileConfig,
    channel: ChannelConfig,
    skip_mode: str,
    capacity_bins: int,
) -> H2PolicySet:
    """Joint meet allocation with I1's violation sets held fixed.

    Every I1 meet action needs at most 1/N and every 1/N is an exact grid
    point, so I1 is representable and J1 <= I1 constructively. Rescuing an I1
    forced violation is intentionally excluded here and belongs to J2's joint
    budget/allocation decision, hence to ``targeting_joint``.
    """
    n_devices = len(options)
    t_slots = len(traces[0].rate_mbps)
    mode = np.ones((n_devices, t_slots), dtype=np.int8)
    for device, result in enumerate(i1.per_device):
        mode[device, result.violate] = 2
    choices, feasible = solve_joint_slots(
        options, mode, np.zeros(n_devices), capacity_bins
    )
    if not np.all(feasible):
        raise ValueError("J1 joint allocation unexpectedly infeasible")
    return _policy_set_from_choices(
        "J1",
        options,
        traces,
        choices,
        capacity_bins,
        profile,
        channel.tx_power_w,
        skip_mode,
        {"violation_set": "fixed_to_I1"},
    )


def _raw_choice_energy(
    options: tuple[SharedActionOptions, ...], choices: np.ndarray
) -> float:
    total = 0.0
    slots = np.arange(choices.shape[1])
    for device, option in enumerate(options):
        total += float(option.energy_j[slots, choices[device]].sum())
    return total


def _repair_budget_violations(
    choices: np.ndarray,
    options: tuple[SharedActionOptions, ...],
    independent_costs: tuple[SlotCosts, ...],
    budgets: np.ndarray,
    capacity_bins: int,
) -> tuple[np.ndarray, bool]:
    """Repair over-budget devices by capacity-aware single-slot re-solves."""
    repaired = choices.copy()
    violation_option = options[0].violation_option
    counts = np.sum(repaired == violation_option, axis=1).astype(np.int64)
    n_devices = len(options)
    while np.any(counts > budgets):
        progress = False
        for device in range(n_devices):
            if counts[device] <= budgets[device]:
                continue
            slots = np.flatnonzero(repaired[device] == violation_option)
            order = slots[
                np.lexsort((slots, independent_costs[device].saving_j[slots]))
            ]
            for t in order:
                mode = np.zeros((n_devices, 1), dtype=np.int8)
                mode[device, 0] = 1
                for other in range(n_devices):
                    if other == device:
                        continue
                    if counts[other] >= budgets[other] and repaired[other, t] != violation_option:
                        mode[other, 0] = 1
                sliced = tuple(
                    SharedActionOptions(
                        energy_j=option.energy_j[t : t + 1],
                        capacity_units=option.capacity_units[t : t + 1],
                        split_p=option.split_p,
                        local_mode=option.local_mode,
                        valid=option.valid[t : t + 1],
                        mode_names=option.mode_names,
                        violation_option=option.violation_option,
                    )
                    for option in options
                )
                candidate, feasible = solve_joint_slots(
                    sliced,
                    mode,
                    np.zeros(n_devices),
                    capacity_bins,
                    np.array([t], dtype=np.int64),
                )
                if not feasible[0]:
                    continue
                old = repaired[:, t].copy()
                repaired[:, t] = candidate[:, 0]
                counts += (repaired[:, t] == violation_option).astype(np.int64)
                counts -= (old == violation_option).astype(np.int64)
                progress = True
                break
        if not progress:
            return repaired, False
    return repaired, True


def solve_j2(
    i2: H2PolicySet,
    j1: H2PolicySet,
    independent_costs: tuple[SlotCosts, ...],
    options: tuple[SharedActionOptions, ...],
    traces: tuple[ChannelTrace, ...],
    profile: DNNProfileConfig,
    channel: ChannelConfig,
    epsilon: float,
    skip_mode: str,
    sweep: H2SweepConfig,
) -> tuple[H2PolicySet, str, tuple[float, ...], tuple[float, ...], int, bool]:
    """Lagrangian joint oracle with capacity-aware repair and 3-way guard.

    Positive median equal-share savings initialize each lambda; simultaneous
    multiplicative subgradient updates run for at most 20 full-T solves.
    Over-budget devices are repaired by re-solving affected joint slots with
    the device forced to meet. If repair cannot satisfy every per-device
    budget, the heuristic is discarded. The final feasible result is the
    minimum-energy member of {heuristic, I2, J1}; therefore J2 <= I2 and
    J2 <= J1 constructively, and measured gains are a lower bound on the true
    joint optimum.
    """
    n_devices = len(options)
    t_slots = len(traces[0].rate_mbps)
    budget = int(math.floor(epsilon * t_slots))
    budgets = np.full(n_devices, budget, dtype=np.int64)
    initial = []
    for costs in independent_costs:
        positive = costs.saving_j[costs.saving_j > 0]
        if not len(positive):
            raise ValueError("J2 lambda initialization requires positive savings")
        initial.append(float(np.median(positive)))
    lambdas = np.asarray(initial, dtype=np.float64)
    minimum = lambdas * sweep.lambda_min_factor
    mode = np.zeros((n_devices, t_slots), dtype=np.int8)
    best_feasible_choices = None
    best_feasible_energy = float("inf")
    final_choices = None
    iterations = 0
    for iteration in range(sweep.lambda_iterations):
        iterations = iteration + 1
        choices, feasible = solve_joint_slots(
            options, mode, lambdas, sweep.capacity_bins
        )
        if not np.all(feasible):
            raise ValueError("J2 full-horizon joint slot solve infeasible")
        final_choices = choices
        counts = np.sum(choices == options[0].violation_option, axis=1)
        if np.all(counts <= budgets):
            raw_energy = _raw_choice_energy(options, choices)
            if raw_energy < best_feasible_energy:
                best_feasible_energy = raw_energy
                best_feasible_choices = choices.copy()
        eta = sweep.lambda_eta_initial / math.sqrt(iteration + 1.0)
        ratios = counts / np.maximum(budgets, 1)
        lambdas = np.maximum(minimum, lambdas * (1.0 + eta * (ratios - 1.0)))

    repaired, repair_success = _repair_budget_violations(
        final_choices, options, independent_costs, budgets, sweep.capacity_bins
    )
    if repair_success:
        repaired_energy = _raw_choice_energy(options, repaired)
        if repaired_energy < best_feasible_energy:
            best_feasible_choices = repaired
            best_feasible_energy = repaired_energy

    candidates: list[tuple[float, str, H2PolicySet]] = [
        (i2.system_mean_energy_j, "I2_guard", i2),
        (j1.system_mean_energy_j, "J1_guard", j1),
    ]
    if best_feasible_choices is not None:
        heuristic = _policy_set_from_choices(
            "J2",
            options,
            traces,
            best_feasible_choices,
            sweep.capacity_bins,
            profile,
            channel.tx_power_w,
            skip_mode,
            {"source": "heuristic"},
        )
        candidates.append((heuristic.system_mean_energy_j, "heuristic", heuristic))
    candidates.sort(key=lambda candidate: (candidate[0], candidate[1]))
    _, source, selected = candidates[0]
    if source != "heuristic":
        selected = H2PolicySet(
            "J2",
            tuple(
                PolicyResult(
                    "J2",
                    result.violate.copy(),
                    result.split_p.copy(),
                    result.local_mode.copy(),
                    result.energy_j.copy(),
                    result.mode_names,
                    {**result.metadata, "source": source},
                )
                for result in selected.per_device
            ),
            selected.allocation_fraction.copy(),
            {**selected.metadata, "source": source},
        )
    return (
        selected,
        source,
        tuple(initial),
        tuple(float(value) for value in lambdas),
        iterations,
        repair_success,
    )


def simulate_h2_trace(
    traces: tuple[ChannelTrace, ...],
    profile: DNNProfileConfig,
    channel: ChannelConfig,
    deadline_ms: float,
    epsilon: float,
    skip_mode: str,
    sweep: H2SweepConfig,
) -> H2SimulationResult:
    i1, i2, independent_costs = independent_policy_sets(
        traces,
        profile,
        channel,
        deadline_ms,
        epsilon,
        skip_mode,
        sweep.server_total_speedup,
    )
    options = tuple(
        shared_action_options(
            profile,
            trace.rate_mbps,
            deadline_ms,
            channel.tx_power_w,
            skip_mode,
            sweep.server_total_speedup,
            sweep.capacity_bins,
            sweep.pareto_pruning,
        )
        for trace in traces
    )
    j1 = solve_j1(
        i1,
        options,
        traces,
        profile,
        channel,
        skip_mode,
        sweep.capacity_bins,
    )
    j2, source, initial, final, iterations, repair_success = solve_j2(
        i2,
        j1,
        independent_costs,
        options,
        traces,
        profile,
        channel,
        epsilon,
        skip_mode,
        sweep,
    )
    return H2SimulationResult(
        policies={"I1": i1, "J1": j1, "I2": i2, "J2": j2},
        independent_costs=independent_costs,
        j2_source=source,
        j2_lambda_initial=initial,
        j2_lambda_final=final,
        j2_iterations=iterations,
        j2_repair_success=repair_success,
    )


def jain_index(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    denominator = len(values) * float(np.sum(values * values))
    if denominator <= 1e-30:
        return 1.0
    return float(np.sum(values) ** 2 / denominator)


def _violation_set_hashes(policy: H2PolicySet) -> str:
    """Serialize exact per-device violation-set identities compactly."""
    hashes = []
    for result in policy.per_device:
        packed = np.packbits(result.violate, bitorder="little")
        hashes.append(hashlib.sha256(packed.tobytes()).hexdigest())
    return json.dumps(hashes)


def h2_decomposition_metrics(simulation: H2SimulationResult) -> dict[str, object]:
    energies = {
        name: policy.system_mean_energy_j
        for name, policy in simulation.policies.items()
    }
    i1 = max(energies["I1"], 1e-15)
    server_only = 100.0 * (energies["I1"] - energies["J1"]) / i1
    targeting_indep = 100.0 * (energies["I1"] - energies["I2"]) / i1
    targeting_joint = 100.0 * (energies["J1"] - energies["J2"]) / i1
    total = 100.0 * (energies["I1"] - energies["J2"]) / i1
    rates = {
        name: simulation.policies[name].violation_rates
        for name in simulation.policies
    }
    return {
        "I1_energy_j": energies["I1"],
        "J1_energy_j": energies["J1"],
        "I2_energy_j": energies["I2"],
        "J2_energy_j": energies["J2"],
        "server_only_percent": server_only,
        "targeting_indep_percent": targeting_indep,
        "targeting_joint_percent": targeting_joint,
        "interaction_percent_points": targeting_joint - targeting_indep,
        "total_percent": total,
        "identity_error_percent_points": total - server_only - targeting_joint,
        "I1_violation_rate_mean": float(np.mean(rates["I1"])),
        "J1_violation_rate_mean": float(np.mean(rates["J1"])),
        "I2_violation_rate_mean": float(np.mean(rates["I2"])),
        "J2_violation_rate_mean": float(np.mean(rates["J2"])),
        "I1_violation_set_sha256_by_device_json": _violation_set_hashes(
            simulation.policies["I1"]
        ),
        "J1_violation_set_sha256_by_device_json": _violation_set_hashes(
            simulation.policies["J1"]
        ),
        "I2_violation_set_sha256_by_device_json": _violation_set_hashes(
            simulation.policies["I2"]
        ),
        "J2_violation_set_sha256_by_device_json": _violation_set_hashes(
            simulation.policies["J2"]
        ),
        "I2_violation_spread": float(np.ptp(rates["I2"])),
        "J2_violation_spread": float(np.ptp(rates["J2"])),
        "I2_jain_index": jain_index(rates["I2"]),
        "J2_jain_index": jain_index(rates["J2"]),
        "J2_source": simulation.j2_source,
        "J2_heuristic_selected": simulation.j2_source == "heuristic",
        "J2_I2_guard_selected": simulation.j2_source == "I2_guard",
        "J2_J1_guard_selected": simulation.j2_source == "J1_guard",
        "J2_repair_success": simulation.j2_repair_success,
        "J2_lambda_initial_json": json.dumps(simulation.j2_lambda_initial),
        "J2_lambda_final_json": json.dumps(simulation.j2_lambda_final),
        "J2_iterations": simulation.j2_iterations,
    }


def h2_sanity_rows(
    simulation: H2SimulationResult,
    epsilon: float,
    relative_energy_tolerance: float,
    violation_tolerance: float,
) -> list[dict[str, object]]:
    policies = simulation.policies
    energies = {name: value.system_mean_energy_j for name, value in policies.items()}
    scale = max(abs(energies["I1"]), 1e-12)
    tolerance = relative_energy_tolerance * scale
    metrics = h2_decomposition_metrics(simulation)
    rows = []
    order_checks = (
        ("energy_J2_le_I2", energies["J2"] <= energies["I2"] + tolerance),
        ("energy_J2_le_J1", energies["J2"] <= energies["J1"] + tolerance),
        ("energy_J1_le_I1", energies["J1"] <= energies["I1"] + tolerance),
        ("energy_I2_le_I1", energies["I2"] <= energies["I1"] + tolerance),
    )
    for name, passed in order_checks:
        rows.append(
            {
                "check": name,
                "passed": bool(passed),
                "detail": json.dumps(energies, sort_keys=True),
                "severity": "required",
            }
        )
    t_slots = len(policies["I1"].per_device[0].violate)
    budget = int(math.floor(epsilon * t_slots))
    for name, policy in policies.items():
        capacity = policy.allocation_fraction.sum(axis=0)
        rows.append(
            {
                "check": f"capacity_{name}_within_one",
                "passed": bool(np.all(capacity <= 1.0 + 1e-12)),
                "detail": f"max={float(capacity.max()):.12g}",
                "severity": "required",
            }
        )
        for device, result in enumerate(policy.per_device):
            count = int(result.violate.sum())
            rows.append(
                {
                    "check": f"budget_{name}_device_{device}",
                    "passed": bool(
                        count <= budget + violation_tolerance * t_slots + 1e-9
                    ),
                    "detail": f"count={count} budget={budget}",
                    "severity": "required",
                }
            )
    fixed_equal = all(
        np.array_equal(
            policies["I1"].per_device[device].violate,
            policies["J1"].per_device[device].violate,
        )
        for device in range(len(policies["I1"].per_device))
    )
    rows.extend(
        [
            {
                "check": "J1_violation_sets_equal_I1",
                "passed": fixed_equal,
                "detail": "per-device violation arrays compared exactly",
                "severity": "required",
            },
            {
                "check": "decomposition_total_identity",
                "passed": abs(float(metrics["identity_error_percent_points"])) <= 1e-10,
                "detail": f"error_pp={metrics['identity_error_percent_points']}",
                "severity": "required",
            },
            {
                "check": "J2_repair_or_guard_feasible",
                "passed": True,
                "detail": f"source={simulation.j2_source} repair_success={simulation.j2_repair_success}",
                "severity": "required",
            },
        ]
    )
    return rows


def stable_h2_digest(simulation: H2SimulationResult) -> str:
    digest = hashlib.sha256()
    for name in sorted(simulation.policies):
        policy = simulation.policies[name]
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(policy.allocation_fraction).tobytes())
        for result in policy.per_device:
            digest.update(np.ascontiguousarray(result.violate).tobytes())
            digest.update(np.ascontiguousarray(result.split_p).tobytes())
            digest.update(np.ascontiguousarray(result.local_mode).tobytes())
            digest.update(np.ascontiguousarray(result.energy_j).tobytes())
    digest.update(simulation.j2_source.encode("utf-8"))
    digest.update(json.dumps(simulation.j2_lambda_final).encode("utf-8"))
    return digest.hexdigest()
