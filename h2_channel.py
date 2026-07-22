"""H2 channel generation with explicit device-index entropy and sync control."""

from __future__ import annotations

import hashlib

import numpy as np

from channel import (
    BAD,
    ChannelTrace,
    _markov_states,
    lag1_autocorrelation,
    theoretical_lag1_autocorrelation,
    transition_probabilities,
)
from config import ChannelConfig


def generate_h2_trace(
    channel: ChannelConfig,
    rho: float,
    t_slots: int,
    seed: int,
    device_index: int,
) -> ChannelTrace:
    """Generate one deterministic device trace.

    State resampling uses ``[seed, rho_key, device_index, attempt]`` and rate
    jitter uses ``[seed, rho_key, device_index, JITT]``. The caller implements
    common synchronization by requesting device_index zero for every device.
    There is deliberately no N==1 seed special case.
    """
    if t_slots <= 0 or device_index < 0:
        raise ValueError("t_slots must be positive and device_index nonnegative")
    p_bg, p_gb = transition_probabilities(channel.pi_bad, rho)
    rho_key = int(round((rho + 1.0) * 1_000_000))
    accepted = None
    observed = float("nan")
    attempt = 0
    for attempt in range(channel.max_trace_resamples):
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, rho_key, device_index, attempt])
        )
        initial_bad = rng.random() < channel.pi_bad
        states = _markov_states(rng.random(t_slots), initial_bad, p_bg, p_gb)
        observed = float(states.mean())
        accepted = states
        if abs(observed - channel.pi_bad) <= channel.enforce_marginal_tolerance:
            break
    else:
        raise RuntimeError(
            "could not obtain H2 pi_B within tolerance after "
            f"{channel.max_trace_resamples} traces"
        )

    base_rates = np.where(
        accepted == BAD, channel.r_bad_mbps, channel.r_good_mbps
    ).astype(np.float64)
    if channel.rate_jitter_sigma_log == 0:
        jitter = np.ones(t_slots, dtype=np.float64)
    else:
        jitter_rng = np.random.default_rng(
            np.random.SeedSequence([seed, rho_key, device_index, 0x4A495454])
        )
        jitter = np.clip(
            jitter_rng.lognormal(
                mean=0.0, sigma=channel.rate_jitter_sigma_log, size=t_slots
            ),
            0.5,
            2.0,
        )
    rates = base_rates * jitter
    p10, p50, p90 = np.quantile(jitter, [0.1, 0.5, 0.9])
    return ChannelTrace(
        rate_mbps=rates,
        state=accepted,
        source="h2_gilbert_elliott",
        metadata={
            "p_bg": p_bg,
            "p_gb": p_gb,
            "rho": rho,
            "device_index_entropy": device_index,
            "pi_bad_observed": observed,
            "lag1_autocorr": lag1_autocorrelation(accepted),
            "lag1_autocorr_theory": theoretical_lag1_autocorrelation(
                channel.pi_bad, rho
            ),
            "rate_jitter_sigma_log": channel.rate_jitter_sigma_log,
            "rate_jitter_p10": float(p10),
            "rate_jitter_p50": float(p50),
            "rate_jitter_p90": float(p90),
            "resample_attempt": attempt,
        },
    )


def generate_h2_traces(
    channel: ChannelConfig,
    rho: float,
    t_slots: int,
    seed: int,
    n_devices: int,
    channel_sync: str,
) -> tuple[ChannelTrace, ...]:
    if channel_sync == "common":
        shared = generate_h2_trace(channel, rho, t_slots, seed, 0)
        return tuple(shared for _ in range(n_devices))
    if channel_sync == "independent":
        return tuple(
            generate_h2_trace(channel, rho, t_slots, seed, index)
            for index in range(n_devices)
        )
    raise ValueError("channel_sync must be independent or common")


def trace_hash(trace: ChannelTrace) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(trace.state).tobytes())
    digest.update(np.ascontiguousarray(trace.rate_mbps).tobytes())
    return digest.hexdigest()
