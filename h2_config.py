"""Configuration for H2 multi-device server-capacity multiplexing sweeps."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from config import ChannelConfig, DNNProfileConfig


@dataclass(frozen=True)
class H2SweepConfig:
    mode: str
    t_slots: int
    seeds: tuple[int, ...]
    n_devices: tuple[int, ...]
    rho_values: tuple[float, ...]
    deadline_ratios: tuple[float, ...]
    epsilons: tuple[float, ...]
    skip_modes: tuple[str, ...] = ("drop", "late")
    channel_sync_values: tuple[str, ...] = ("independent", "common")
    server_total_speedup: float = 20.0
    capacity_bins: int = 200
    lambda_iterations: int = 20
    lambda_eta_initial: float = 0.5
    lambda_min_factor: float = 1e-4
    relative_energy_tolerance: float = 0.005
    violation_tolerance: float = 1e-12
    grid_resolution_tolerance: float = 0.001
    heuristic_adoption_warning_threshold: float = 0.5
    # Enabled only after the first unpruned smoke projected 6.08 full hours,
    # exceeding the preregistered four-hour optimization threshold.
    pareto_pruning: bool = True

    def __post_init__(self) -> None:
        if self.t_slots <= 0 or not self.seeds:
            raise ValueError("H2 t_slots and seeds must be non-empty and positive")
        if self.server_total_speedup <= 0 or self.capacity_bins <= 0:
            raise ValueError("server speedup and capacity bins must be positive")
        if self.lambda_iterations <= 0 or self.lambda_eta_initial <= 0:
            raise ValueError("J2 lambda controls must be positive")
        if not 0 < self.lambda_min_factor < 1:
            raise ValueError("lambda_min_factor must lie in (0, 1)")
        if any(n <= 0 for n in self.n_devices):
            raise ValueError("all H2 device counts must be positive")
        # Exact representation of every equal share is a premise of J1 <= I1.
        incompatible = [n for n in self.n_devices if self.capacity_bins % n != 0]
        if incompatible:
            raise ValueError(
                f"capacity grid 1/{self.capacity_bins} does not contain 1/N "
                f"for N={incompatible}"
            )
        if any(mode not in ("drop", "late") for mode in self.skip_modes):
            raise ValueError("H2 skip modes must be drop or late")
        if any(sync not in ("independent", "common") for sync in self.channel_sync_values):
            raise ValueError("channel_sync must be independent or common")


@dataclass(frozen=True)
class H2ExperimentConfig:
    profile: DNNProfileConfig = field(default_factory=DNNProfileConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    sweep: H2SweepConfig | None = None
    # In late mode, missed work runs outside real-time capacity in best-effort
    # or idle time; it consumes energy but no capacity in the missed slot.
    late_uses_realtime_capacity: bool = False

    def __post_init__(self) -> None:
        if self.sweep is None:
            raise ValueError("H2 sweep configuration is required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


H2_PREREGISTRATION = {
    "scope": "late mode with independent channels",
    "high_load_region": "N in {4,8}, rho>=0.75, D/D_min(N)<=1.35",
    "H2a_adopt": "interaction >= 2 percentage points in the high-load region",
    "H2a_conditional": "interaction in [0.5, 2) percentage points",
    "H2a_reject": "interaction < 0.5 percentage points throughout",
    "H2b": "independent interaction exceeds common interaction broadly",
    "H2c": "J2 fairness is not worse than I2 by violation spread/Jain index",
}


def default_h2_experiment(mode: str = "smoke") -> H2ExperimentConfig:
    if mode == "smoke":
        sweep = H2SweepConfig(
            mode="smoke",
            t_slots=10_000,
            seeds=(1701, 1702),
            n_devices=(2, 4),
            rho_values=(0.0, 0.975),
            deadline_ratios=(1.35,),
            epsilons=(0.15,),
        )
    elif mode == "quick":
        sweep = H2SweepConfig(
            mode="quick",
            t_slots=10_000,
            seeds=(1701, 1702, 1703),
            n_devices=(2, 4, 8),
            rho_values=(0.0, 0.75, 0.975),
            deadline_ratios=(1.2, 1.35, 1.5),
            epsilons=(0.05, 0.15),
        )
    elif mode == "full":
        sweep = H2SweepConfig(
            mode="full",
            t_slots=50_000,
            seeds=tuple(range(1701, 1711)),
            n_devices=(2, 4, 8),
            rho_values=(0.0, 0.75, 0.975),
            deadline_ratios=(1.2, 1.35, 1.5),
            epsilons=(0.05, 0.15),
        )
    else:
        raise ValueError("H2 mode must be smoke, quick, or full")
    return H2ExperimentConfig(sweep=sweep)


def default_h2_output_dir(mode: str) -> Path:
    return Path("results") / f"h2_{mode}"
