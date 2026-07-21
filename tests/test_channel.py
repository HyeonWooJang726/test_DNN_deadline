from pathlib import Path

import numpy as np
import pandas as pd

from channel import (
    CSVChannel,
    GilbertElliottChannel,
    theoretical_lag1_autocorrelation,
    transition_probabilities,
)


def test_transition_probability_inversion():
    p_bg, p_gb = transition_probabilities(pi_bad=0.2, rho=0.75)
    assert p_bg == 0.2
    assert p_gb == 0.05
    stationary_bad = p_gb / (p_bg + p_gb)
    assert abs(stationary_bad - 0.2) < 1e-12
    assert theoretical_lag1_autocorrelation(0.2, 0.75) == 0.75


def test_rho_zero_is_exact_iid_transition_matrix():
    p_bg, p_gb = transition_probabilities(0.2, 0.0)
    assert p_bg == 0.8
    assert p_gb == 0.2
    assert theoretical_lag1_autocorrelation(0.2, 0.0) == 0.0


def test_ge_trace_is_reproducible_and_marginal_controlled():
    source = GilbertElliottChannel(40.0, 10.0, 0.2, 0.875, marginal_tolerance=0.01)
    a = source.generate(5_000, 123)
    b = source.generate(5_000, 123)
    np.testing.assert_array_equal(a.state, b.state)
    np.testing.assert_array_equal(a.rate_mbps, b.rate_mbps)
    assert abs(a.state.mean() - 0.2) <= 0.01
    assert abs(a.metadata["lag1_autocorr"] - 0.875) <= 0.03


def test_zero_jitter_recovers_exact_state_rates():
    source = GilbertElliottChannel(
        40.0, 10.0, 0.2, 0.0, rate_jitter_sigma_log=0.0
    )
    trace = source.generate(5_000, 321)
    np.testing.assert_array_equal(
        trace.rate_mbps,
        np.where(trace.state == 1, 10.0, 40.0),
    )
    assert trace.metadata["rate_jitter_p10"] == 1.0
    assert trace.metadata["rate_jitter_p50"] == 1.0
    assert trace.metadata["rate_jitter_p90"] == 1.0


def test_csv_channel_common_interface(tmp_path: Path):
    path = tmp_path / "trace.csv"
    pd.DataFrame({"rate_mbps": [40.0, 10.0, 40.0], "state": ["Good", "Bad", "Good"]}).to_csv(path, index=False)
    trace = CSVChannel(path).generate(3, seed=999)
    np.testing.assert_allclose(trace.rate_mbps, [40.0, 10.0, 40.0])
    np.testing.assert_array_equal(trace.state, [0, 1, 0])
