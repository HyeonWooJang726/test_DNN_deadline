"""DNN action costs with explicit, testable unit conversions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from config import DNNProfileConfig


def kb_to_megabits(value_kb):
    """Convert decimal kilobytes to megabits (KB * 8 / 1000)."""
    return np.asarray(value_kb) * 8.0 / 1000.0


def ms_to_s(value_ms):
    return np.asarray(value_ms) / 1000.0


def s_to_ms(value_s):
    return np.asarray(value_s) * 1000.0


def mj_to_j(value_mj):
    return np.asarray(value_mj) / 1000.0


def j_to_mj(value_j):
    return np.asarray(value_j) * 1000.0


@dataclass(frozen=True)
class ActionTable:
    latency_ms: np.ndarray  # [T, K+1, M]
    energy_j: np.ndarray  # [T, K+1, M]
    mode_names: tuple[str, ...]


@dataclass(frozen=True)
class SlotCosts:
    feasible: np.ndarray
    meet_p: np.ndarray
    meet_local_mode: np.ndarray
    meet_energy_j: np.ndarray
    skip_p: np.ndarray
    skip_local_mode: np.ndarray
    skip_energy_j: np.ndarray
    saving_j: np.ndarray
    mode_names: tuple[str, ...]


def action_table(
    profile: DNNProfileConfig,
    rate_mbps: np.ndarray,
    tx_power_w: float,
) -> ActionTable:
    """Return latency and energy for every ``(slot, split, local mode)``."""
    rates = np.asarray(rate_mbps, dtype=np.float64)
    if rates.ndim != 1 or np.any(rates <= 0):
        raise ValueError("rate_mbps must be a positive one-dimensional array")

    tloc = np.asarray(profile.t_loc_ms, dtype=np.float64)
    eloc = np.asarray(profile.e_loc_mj, dtype=np.float64)
    data_mb = kb_to_megabits(np.asarray(profile.d_kb, dtype=np.float64))

    local_ms_normal = np.concatenate(([0.0], np.cumsum(tloc)))
    local_j_normal = np.concatenate(([0.0], np.cumsum(mj_to_j(eloc))))
    remaining_ms = tloc.sum() - local_ms_normal
    server_ms = remaining_ms / profile.server_speedup
    speed_scales = np.asarray([mode.speed_scale for mode in profile.local_modes])
    energy_scales = np.asarray([mode.energy_scale for mode in profile.local_modes])
    local_ms = local_ms_normal[:, None] * speed_scales[None, :]
    local_j = local_j_normal[:, None] * energy_scales[None, :]

    tx_s = data_mb[None, :] / rates[:, None]
    latency_ms = local_ms[None, :, :] + s_to_ms(tx_s)[:, :, None] + server_ms[None, :, None]
    energy_j = local_j[None, :, :] + (tx_s * tx_power_w)[:, :, None]
    return ActionTable(
        latency_ms=latency_ms,
        energy_j=energy_j,
        mode_names=tuple(mode.name for mode in profile.local_modes),
    )


def minimum_good_deadline_ms(
    profile: DNNProfileConfig, r_good_mbps: float, tx_power_w: float
) -> float:
    """Good-state minimum latency using *normal mode only*.

    Keeping this baseline independent of boost prevents the absolute deadline
    from shrinking when a faster local mode is added.
    """
    table = action_table(profile, np.array([r_good_mbps]), tx_power_w)
    normal_index = table.mode_names.index("normal")
    return float(table.latency_ms[0, :, normal_index].min())


def compute_slot_costs(
    profile: DNNProfileConfig,
    rate_mbps: np.ndarray,
    deadline_ms: float,
    tx_power_w: float,
    skip_mode: str,
) -> SlotCosts:
    table = action_table(profile, rate_mbps, tx_power_w)
    meets = table.latency_ms <= deadline_ms + 1e-12
    feasible = meets.any(axis=(1, 2))
    action_count = table.latency_ms.shape[1] * table.latency_ms.shape[2]
    mode_count = len(table.mode_names)
    energy_flat = table.energy_j.reshape(len(rate_mbps), action_count)
    meets_flat = meets.reshape(len(rate_mbps), action_count)

    masked = np.where(meets_flat, energy_flat, np.inf)
    meet_action = np.argmin(masked, axis=1)
    meet_p = (meet_action // mode_count).astype(np.int16)
    meet_local_mode = (meet_action % mode_count).astype(np.int8)
    meet_energy = masked[np.arange(len(rate_mbps)), meet_action]

    if skip_mode == "drop":
        skip_p = np.full(len(rate_mbps), -1, dtype=np.int16)
        skip_local_mode = np.full(len(rate_mbps), -1, dtype=np.int8)
        skip_energy = np.zeros(len(rate_mbps), dtype=np.float64)
    elif skip_mode == "late":
        skip_action = np.argmin(energy_flat, axis=1)
        skip_p = (skip_action // mode_count).astype(np.int16)
        skip_local_mode = (skip_action % mode_count).astype(np.int8)
        skip_energy = energy_flat[np.arange(len(rate_mbps)), skip_action]
    else:
        raise ValueError("skip_mode must be 'drop' or 'late'")

    # Infeasible slots are forced violations, so E_meet/saving are undefined.
    saving = np.where(feasible, meet_energy - skip_energy, 0.0)
    meet_energy = np.where(feasible, meet_energy, skip_energy)
    meet_p = np.where(feasible, meet_p, skip_p).astype(np.int16)
    meet_local_mode = np.where(feasible, meet_local_mode, skip_local_mode).astype(np.int8)
    if np.any(saving < -1e-10):
        raise AssertionError("E_skip cannot exceed the minimum feasible energy")
    saving = np.maximum(saving, 0.0)
    return SlotCosts(
        feasible=feasible,
        meet_p=meet_p,
        meet_local_mode=meet_local_mode,
        meet_energy_j=meet_energy,
        skip_p=skip_p,
        skip_local_mode=skip_local_mode,
        skip_energy_j=skip_energy,
        saving_j=saving,
        mode_names=table.mode_names,
    )


def state_feasibility(
    profile: DNNProfileConfig,
    rate_mbps: float,
    deadline_ms: float,
    tx_power_w: float,
) -> tuple[bool, tuple[int, ...]]:
    table = action_table(profile, np.array([rate_mbps]), tx_power_w)
    feasible_actions = table.latency_ms[0] <= deadline_ms + 1e-12
    points = tuple(np.flatnonzero(feasible_actions.any(axis=1)).tolist())
    return bool(points), points


def minimum_required_rate_mbps(
    profile: DNNProfileConfig,
    deadline_ms: float,
) -> float:
    """Smallest rate at which at least one split/mode meets ``deadline_ms``."""
    tloc = np.asarray(profile.t_loc_ms, dtype=np.float64)
    local_normal = np.concatenate(([0.0], np.cumsum(tloc)))
    server_ms = (tloc.sum() - local_normal) / profile.server_speedup
    data_mb = kb_to_megabits(np.asarray(profile.d_kb, dtype=np.float64))
    required: list[float] = []
    for mode in profile.local_modes:
        fixed_ms = local_normal * mode.speed_scale + server_ms
        slack_ms = deadline_ms - fixed_ms
        action_required = np.where(
            slack_ms > 0.0,
            data_mb * 1000.0 / slack_ms,
            np.inf,
        )
        required.append(float(np.min(action_required)))
    return min(required)


def infeasible_probability_with_jitter(
    profile: DNNProfileConfig,
    base_rate_mbps: float,
    deadline_ms: float,
    sigma_log: float,
    clip_low: float = 0.5,
    clip_high: float = 2.0,
) -> float:
    """Probability no action meets the deadline under clipped lognormal jitter."""
    if base_rate_mbps <= 0:
        raise ValueError("base_rate_mbps must be positive")
    if sigma_log < 0:
        raise ValueError("sigma_log must be nonnegative")
    required_ratio = minimum_required_rate_mbps(profile, deadline_ms) / base_rate_mbps
    if sigma_log == 0.0:
        return float(required_ratio > 1.0 + 1e-15)
    if required_ratio <= clip_low:
        return 0.0
    if required_ratio > clip_high:
        return 1.0
    z = math.log(required_ratio) / sigma_log
    return float(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
