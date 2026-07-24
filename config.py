"""Experiment configuration for the DNN split-offloading simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LocalMode:
    """Local execution mode applied to the locally executed DNN prefix."""

    name: str
    speed_scale: float
    energy_scale: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("local mode name must be non-empty")
        if self.speed_scale <= 0 or self.energy_scale <= 0:
            raise ValueError("local mode scales must be positive")


@dataclass(frozen=True)
class DNNProfileConfig:
    """One device's chain-DNN profile.

    The deliberately back-loaded default compute profile keeps at least one
    nontrivial split feasible in the 10-Mbps state for part of the requested
    D/D_min range.  Only total AlexNet-class magnitudes are assumed by the
    experiment; users should replace these arrays with measured profiles.
    """

    name: str = "synthetic_alexnet_scale"
    t_loc_ms: tuple[float, ...] = (2.0, 2.5, 3.0, 3.5, 4.0, 6.1, 75.0, 103.9)
    e_loc_mj: tuple[float, ...] = (10.0, 14.0, 18.0, 22.0, 30.0, 46.0, 260.0, 400.0)
    # d[p] is the tensor sent after p local blocks; d[0] is the input.
    d_kb: tuple[float, ...] = (150.0, 400.0, 600.0, 450.0, 200.0, 80.0, 30.0, 10.0, 4.0)
    server_speedup: float = 20.0
    local_modes: tuple[LocalMode, ...] = (
        LocalMode("normal", 1.0, 1.0),
        # CutEdge Jetson TX2 GPU profile (Lim et al., IEEE TSC 17(6):3300-3316, 2024).
        # normal = 850 MHz, boost = 1300 MHz (the TX2 DVFS maximum).
        #   speed  = 850/1300 = 0.654                            (T ~ 1/f)
        #   energy = [P(1300)/P(850)] * 0.654 = 2.510 * 0.654 = 1.64
        # from P(f) = alpha*f^gamma + beta with
        # (alpha, beta, gamma) = (3.8351e-8, 0.7312, 2.6343).
        # Frequency-based toy approximations, not measured ratios.
        LocalMode("boost", 0.654, 1.64),
    )

    def __post_init__(self) -> None:
        if len(self.t_loc_ms) != len(self.e_loc_mj):
            raise ValueError("t_loc_ms and e_loc_mj must have equal length")
        if len(self.d_kb) != len(self.t_loc_ms) + 1:
            raise ValueError("d_kb must contain K+1 entries, indexed by split p")
        if min(self.t_loc_ms) < 0 or min(self.e_loc_mj) < 0 or min(self.d_kb) < 0:
            raise ValueError("profile values must be nonnegative")
        if self.server_speedup <= 0:
            raise ValueError("server_speedup must be positive")
        if not self.local_modes:
            raise ValueError("at least one local mode is required")
        names = [mode.name for mode in self.local_modes]
        if len(names) != len(set(names)):
            raise ValueError("local mode names must be unique")
        if "normal" not in names:
            raise ValueError("local_modes must contain a normal mode")

    @property
    def k(self) -> int:
        return len(self.t_loc_ms)


@dataclass(frozen=True)
class DeviceConfig:
    name: str = "device_0"
    profile: DNNProfileConfig = field(default_factory=DNNProfileConfig)
    # Update this frozen-profile guard when profile measurements change, or
    # set it to None to record the measured D_min without enforcing a value.
    expected_d_min_ms: float | None = 35.0

    def __post_init__(self) -> None:
        if self.expected_d_min_ms is not None and self.expected_d_min_ms <= 0:
            raise ValueError("expected_d_min_ms must be positive or None")


@dataclass(frozen=True)
class ChannelConfig:
    # CutEdge's trace-driven simulation uses a measured uplink trace spanning
    # 10.864-53.196 Mbps. The Bad and Good states represent the low and high
    # ends of that range. These are representative two-state throughputs, not
    # the measured goodput of any specific MCS.
    r_good_mbps: float = 48.0
    r_bad_mbps: float = 16.0
    pi_bad: float = 0.2
    rate_jitter_sigma_log: float = 0.25
    tx_power_w: float = 1.2
    enforce_marginal_tolerance: float = 0.01
    max_trace_resamples: int = 1_000


@dataclass(frozen=True)
class SweepConfig:
    mode: str
    t_slots: int
    seeds: tuple[int, ...]
    rho_values: tuple[float, ...] = (0.0, 0.375, 0.75, 0.875, 0.9375, 0.975)
    deadline_ratios: tuple[float, ...] = (1.05, 1.1, 1.2, 1.35, 1.5, 2.0)
    epsilons: tuple[float, ...] = (0.01, 0.05, 0.1, 0.15)
    skip_modes: tuple[str, ...] = ("drop", "late")
    p3_v_values: tuple[float, ...] = (
        10.0,
        20.0,
        30.0,
        50.0,
        70.0,
        100.0,
        150.0,
        200.0,
        300.0,
        500.0,
        700.0,
        1_000.0,
        2_000.0,
        5_000.0,
        10_000.0,
    )
    relative_energy_tolerance: float = 0.005
    violation_tolerance: float = 0.005
    oracle_gap_mask_percent: float = 2.0
    deadline_axis_tolerance_percent: float = 0.01


@dataclass(frozen=True)
class ExperimentConfig:
    devices: tuple[DeviceConfig, ...]
    channel: ChannelConfig
    sweep: SweepConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_experiment(mode: str = "quick") -> ExperimentConfig:
    if mode == "smoke":
        sweep = SweepConfig(
            mode="smoke",
            t_slots=10_000,
            seeds=(1701, 1702),
            rho_values=(0.0, 0.75, 0.975),
        )
    elif mode == "quick":
        sweep = SweepConfig(mode="quick", t_slots=10_000, seeds=(1701, 1702, 1703))
    elif mode == "full":
        sweep = SweepConfig(
            mode="full",
            t_slots=100_000,
            seeds=tuple(range(1701, 1711)),
        )
    else:
        raise ValueError("mode must be 'smoke', 'quick', or 'full'")
    # A list/tuple of devices is used from day one so a future shared-server
    # extension does not need to change the configuration schema.
    return ExperimentConfig(
        devices=(DeviceConfig(),),
        channel=ChannelConfig(),
        sweep=sweep,
    )


def default_output_dir(mode: str) -> Path:
    return Path("results") / mode
