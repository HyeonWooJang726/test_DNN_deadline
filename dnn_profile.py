"""DNN action costs with explicit, testable unit conversions."""

from __future__ import annotations

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
    latency_ms: np.ndarray  # [T, K+1]
    energy_j: np.ndarray  # [T, K+1]


@dataclass(frozen=True)
class SlotCosts:
    feasible: np.ndarray
    meet_p: np.ndarray
    meet_energy_j: np.ndarray
    skip_p: np.ndarray
    skip_energy_j: np.ndarray
    saving_j: np.ndarray


def action_table(
    profile: DNNProfileConfig,
    rate_mbps: np.ndarray,
    tx_power_w: float,
) -> ActionTable:
    """Return latency and energy for every slot and split point."""
    rates = np.asarray(rate_mbps, dtype=np.float64)
    if rates.ndim != 1 or np.any(rates <= 0):
        raise ValueError("rate_mbps must be a positive one-dimensional array")

    tloc = np.asarray(profile.t_loc_ms, dtype=np.float64)
    eloc = np.asarray(profile.e_loc_mj, dtype=np.float64)
    data_mb = kb_to_megabits(np.asarray(profile.d_kb, dtype=np.float64))

    local_ms = np.concatenate(([0.0], np.cumsum(tloc)))
    local_j = np.concatenate(([0.0], np.cumsum(mj_to_j(eloc))))
    remaining_ms = tloc.sum() - local_ms
    server_ms = remaining_ms / profile.server_speedup

    tx_s = data_mb[None, :] / rates[:, None]
    latency_ms = local_ms[None, :] + s_to_ms(tx_s) + server_ms[None, :]
    energy_j = local_j[None, :] + tx_s * tx_power_w
    return ActionTable(latency_ms=latency_ms, energy_j=energy_j)


def minimum_good_deadline_ms(
    profile: DNNProfileConfig, r_good_mbps: float, tx_power_w: float
) -> float:
    table = action_table(profile, np.array([r_good_mbps]), tx_power_w)
    return float(table.latency_ms.min())


def compute_slot_costs(
    profile: DNNProfileConfig,
    rate_mbps: np.ndarray,
    deadline_ms: float,
    tx_power_w: float,
    skip_mode: str,
) -> SlotCosts:
    table = action_table(profile, rate_mbps, tx_power_w)
    meets = table.latency_ms <= deadline_ms + 1e-12
    feasible = meets.any(axis=1)

    masked = np.where(meets, table.energy_j, np.inf)
    meet_p = np.argmin(masked, axis=1).astype(np.int16)
    meet_energy = masked[np.arange(len(rate_mbps)), meet_p]

    if skip_mode == "drop":
        skip_p = np.full(len(rate_mbps), -1, dtype=np.int16)
        skip_energy = np.zeros(len(rate_mbps), dtype=np.float64)
    elif skip_mode == "late":
        skip_p = np.argmin(table.energy_j, axis=1).astype(np.int16)
        skip_energy = table.energy_j[np.arange(len(rate_mbps)), skip_p]
    else:
        raise ValueError("skip_mode must be 'drop' or 'late'")

    # Infeasible slots are forced violations, so E_meet/saving are undefined.
    saving = np.where(feasible, meet_energy - skip_energy, 0.0)
    meet_energy = np.where(feasible, meet_energy, skip_energy)
    meet_p = np.where(feasible, meet_p, skip_p).astype(np.int16)
    if np.any(saving < -1e-10):
        raise AssertionError("E_skip cannot exceed the minimum feasible energy")
    saving = np.maximum(saving, 0.0)
    return SlotCosts(feasible, meet_p, meet_energy, skip_p, skip_energy, saving)


def state_feasibility(
    profile: DNNProfileConfig,
    rate_mbps: float,
    deadline_ms: float,
    tx_power_w: float,
) -> tuple[bool, tuple[int, ...]]:
    table = action_table(profile, np.array([rate_mbps]), tx_power_w)
    points = tuple(np.flatnonzero(table.latency_ms[0] <= deadline_ms + 1e-12).tolist())
    return bool(points), points

