import numpy as np

from channel import GilbertElliottChannel
from config import default_experiment
from dnn_profile import minimum_good_deadline_ms
from metrics import assert_sanity, combination_sanity_rows, stable_simulation_digest
from simulator import _combination_rng, preflight_check, simulate_trace


def test_combination_rng_includes_rho_and_device_index():
    arguments = (88, 1.5, 0.05, "drop", 0.75, 0)
    reference = _combination_rng(*arguments).integers(0, 2**63, size=8)
    repeated = _combination_rng(*arguments).integers(0, 2**63, size=8)
    other_rho = _combination_rng(88, 1.5, 0.05, "drop", 0.875, 0).integers(
        0, 2**63, size=8
    )
    other_device = _combination_rng(88, 1.5, 0.05, "drop", 0.75, 1).integers(
        0, 2**63, size=8
    )

    assert (reference == repeated).all()
    assert not (reference == other_rho).all()
    assert not (reference == other_device).all()


def test_preflight_and_fixed_seed_pipeline_hash():
    config = default_experiment("quick")
    device = config.devices[0]
    channel = config.channel
    d_min = minimum_good_deadline_ms(device.profile, channel.r_good_mbps, channel.tx_power_w)
    assert d_min == 35.0
    assert not preflight_check(device, channel, 1.2 * d_min, 0.05).valid
    assert preflight_check(device, channel, 1.35 * d_min, 0.05).valid

    source = GilbertElliottChannel(40.0, 10.0, 0.2, 0.75, marginal_tolerance=0.01)
    trace = source.generate(1_000, 88)
    args = (
        trace,
        device,
        channel,
        1.5 * d_min,
        1.5,
        0.05,
        "drop",
        config.sweep.p3_v_values,
        88,
        0.75,
        0,
        config.sweep.violation_tolerance,
    )
    first = simulate_trace(*args)
    second = simulate_trace(*args)
    assert stable_simulation_digest(first) == stable_simulation_digest(second)
    assert_sanity(combination_sanity_rows(first, 0.05, 0.005, 0.005))
    assert first.policies["P3"].violate.sum() <= np.floor(
        0.05 * len(first.policies["P3"].violate)
    )
