"""Common channel-trace interface and Gilbert--Elliott implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - tested environments install requirements.txt
    def njit(*args, **kwargs):
        def decorate(fn):
            return fn
        return decorate


GOOD = 0
BAD = 1


@dataclass(frozen=True)
class ChannelTrace:
    rate_mbps: np.ndarray
    state: np.ndarray  # 0=Good, 1=Bad; -1 when a measured CSV has no labels
    source: str
    metadata: dict[str, float | int | str]


class ChannelSource(ABC):
    @abstractmethod
    def generate(self, t_slots: int, seed: int) -> ChannelTrace:
        raise NotImplementedError


def transition_probabilities(pi_bad: float, mean_bad_dwell_slots: float) -> tuple[float, float]:
    """Return (P(B->G), P(G->B)) from stationary pi_B and bad dwell L."""
    if not 0.0 < pi_bad < 1.0:
        raise ValueError("pi_bad must lie strictly between zero and one")
    if mean_bad_dwell_slots < 1.0:
        raise ValueError("mean bad dwell must be at least one slot")
    p_bg = 1.0 / mean_bad_dwell_slots
    p_gb = pi_bad / (mean_bad_dwell_slots * (1.0 - pi_bad))
    if p_gb > 1.0:
        raise ValueError("parameters imply invalid P(G->B)>1")
    return p_bg, p_gb


def theoretical_lag1_autocorrelation(pi_bad: float, mean_bad_dwell_slots: float) -> float:
    p_bg, p_gb = transition_probabilities(pi_bad, mean_bad_dwell_slots)
    return 1.0 - p_bg - p_gb


@njit(cache=True)
def _markov_states(uniforms, initial_bad, p_bg, p_gb):
    n = len(uniforms)
    out = np.empty(n, dtype=np.int8)
    state = 1 if initial_bad else 0
    for t in range(n):
        out[t] = state
        u = uniforms[t]
        if state == 1:
            if u < p_bg:
                state = 0
        elif u < p_gb:
            state = 1
    return out


def lag1_autocorrelation(states: np.ndarray) -> float:
    x = np.asarray(states, dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0:
        return float("nan")
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


@dataclass(frozen=True)
class GilbertElliottChannel(ChannelSource):
    r_good_mbps: float
    r_bad_mbps: float
    pi_bad: float
    mean_bad_dwell_slots: float
    marginal_tolerance: float | None = 0.01
    max_resamples: int = 1_000

    def generate(self, t_slots: int, seed: int) -> ChannelTrace:
        if t_slots <= 0:
            raise ValueError("t_slots must be positive")
        p_bg, p_gb = transition_probabilities(self.pi_bad, self.mean_bad_dwell_slots)
        accepted = None
        observed = float("nan")
        attempt = 0
        # Starting in stationarity removes burn-in. Rejection only controls finite-T
        # marginal noise for the requested L-invariance sanity check; set tolerance
        # to None to obtain an entirely unconditioned Markov trace.
        for attempt in range(self.max_resamples):
            ss = np.random.SeedSequence([seed, int(self.mean_bad_dwell_slots * 1000), attempt])
            rng = np.random.default_rng(ss)
            initial_bad = rng.random() < self.pi_bad
            states = _markov_states(rng.random(t_slots), initial_bad, p_bg, p_gb)
            observed = float(states.mean())
            accepted = states
            if self.marginal_tolerance is None or abs(observed - self.pi_bad) <= self.marginal_tolerance:
                break
        else:
            raise RuntimeError(
                f"could not obtain pi_B within tolerance after {self.max_resamples} traces"
            )

        rates = np.where(accepted == BAD, self.r_bad_mbps, self.r_good_mbps).astype(np.float64)
        return ChannelTrace(
            rate_mbps=rates,
            state=accepted,
            source="gilbert_elliott",
            metadata={
                "p_bg": p_bg,
                "p_gb": p_gb,
                "pi_bad_observed": observed,
                "lag1_autocorr": lag1_autocorrelation(accepted),
                "lag1_autocorr_theory": theoretical_lag1_autocorrelation(
                    self.pi_bad, self.mean_bad_dwell_slots
                ),
                "resample_attempt": attempt,
            },
        )


@dataclass(frozen=True)
class CSVChannel(ChannelSource):
    """Measured rate trace using the same interface as the synthetic channel."""

    path: str | Path
    rate_column: str = "rate_mbps"
    state_column: str | None = "state"

    def generate(self, t_slots: int, seed: int) -> ChannelTrace:
        import pandas as pd

        frame = pd.read_csv(self.path)
        if self.rate_column not in frame:
            raise ValueError(f"missing CSV column: {self.rate_column}")
        if len(frame) < t_slots:
            raise ValueError(f"CSV contains {len(frame)} rows, need {t_slots}")
        rates = frame[self.rate_column].to_numpy(dtype=np.float64)[:t_slots]
        if np.any(rates <= 0):
            raise ValueError("measured rates must be positive")
        if self.state_column and self.state_column in frame:
            raw = frame[self.state_column].to_numpy()[:t_slots]
            if raw.dtype.kind in "OUS":
                states = np.array([BAD if str(v).lower().startswith("b") else GOOD for v in raw], dtype=np.int8)
            else:
                states = raw.astype(np.int8)
        else:
            states = np.full(t_slots, -1, dtype=np.int8)
        return ChannelTrace(
            rate_mbps=rates,
            state=states,
            source=f"csv:{Path(self.path).name}",
            metadata={"seed_ignored": seed},
        )
