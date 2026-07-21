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


def transition_probabilities(pi_bad: float, rho: float) -> tuple[float, float]:
    """Return transition probabilities from stationary ``pi_B`` and lag-1 ``rho``."""
    if not 0.0 < pi_bad < 1.0:
        raise ValueError("pi_bad must lie strictly between zero and one")
    if not -1.0 < rho < 1.0:
        raise ValueError("rho must lie strictly between -1 and one")
    p_bg = (1.0 - rho) * (1.0 - pi_bad)
    p_gb = (1.0 - rho) * pi_bad
    if p_bg > 1.0 or p_gb > 1.0:
        raise ValueError("pi_bad and rho imply an invalid transition probability")
    return p_bg, p_gb


def theoretical_lag1_autocorrelation(pi_bad: float, rho: float) -> float:
    transition_probabilities(pi_bad, rho)  # validate the implied matrix
    return float(rho)


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
    rho: float
    rate_jitter_sigma_log: float = 0.25
    marginal_tolerance: float | None = 0.01
    max_resamples: int = 1_000

    def generate(self, t_slots: int, seed: int) -> ChannelTrace:
        if t_slots <= 0:
            raise ValueError("t_slots must be positive")
        if self.rate_jitter_sigma_log < 0:
            raise ValueError("rate_jitter_sigma_log must be nonnegative")
        p_bg, p_gb = transition_probabilities(self.pi_bad, self.rho)
        accepted = None
        observed = float("nan")
        attempt = 0
        # Starting in stationarity removes burn-in. Rejection only controls finite-T
        # marginal noise for the requested rho-invariance sanity check; set tolerance
        # to None to obtain an entirely unconditioned Markov trace.
        for attempt in range(self.max_resamples):
            rho_key = int(round((self.rho + 1.0) * 1_000_000))
            ss = np.random.SeedSequence([seed, rho_key, attempt])
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

        base_rates = np.where(accepted == BAD, self.r_bad_mbps, self.r_good_mbps).astype(np.float64)
        if self.rate_jitter_sigma_log == 0.0:
            jitter = np.ones(t_slots, dtype=np.float64)
        else:
            jitter_rng = np.random.default_rng(
                np.random.SeedSequence([seed, rho_key, 0x4A495454])
            )
            jitter = np.clip(
                jitter_rng.lognormal(mean=0.0, sigma=self.rate_jitter_sigma_log, size=t_slots),
                0.5,
                2.0,
            )
        rates = base_rates * jitter
        jitter_p10, jitter_p50, jitter_p90 = np.quantile(jitter, [0.1, 0.5, 0.9])
        return ChannelTrace(
            rate_mbps=rates,
            state=accepted,
            source="gilbert_elliott",
            metadata={
                "p_bg": p_bg,
                "p_gb": p_gb,
                "rho": self.rho,
                "pi_bad_observed": observed,
                "lag1_autocorr": lag1_autocorrelation(accepted),
                "lag1_autocorr_theory": theoretical_lag1_autocorrelation(
                    self.pi_bad, self.rho
                ),
                "rate_jitter_sigma_log": self.rate_jitter_sigma_log,
                "rate_jitter_p10": float(jitter_p10),
                "rate_jitter_p50": float(jitter_p50),
                "rate_jitter_p90": float(jitter_p90),
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
