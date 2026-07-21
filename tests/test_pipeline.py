from channel import GilbertElliottChannel
from config import default_experiment
from dnn_profile import minimum_good_deadline_ms
from metrics import assert_sanity, combination_sanity_rows, stable_simulation_digest
from simulator import preflight_check, simulate_trace


def test_preflight_and_fixed_seed_pipeline_hash():
    config = default_experiment("quick")
    device = config.devices[0]
    channel = config.channel
    d_min = minimum_good_deadline_ms(device.profile, channel.r_good_mbps, channel.tx_power_w)
    assert not preflight_check(device, channel, 1.3 * d_min, 0.05).valid
    assert preflight_check(device, channel, 1.5 * d_min, 0.05).valid

    source = GilbertElliottChannel(40.0, 10.0, 0.2, 5, marginal_tolerance=0.01)
    trace = source.generate(1_000, 88)
    args = (trace, device, channel, 1.5 * d_min, 1.5, 0.05, "drop", config.sweep.p3_v_values, 88)
    first = simulate_trace(*args)
    second = simulate_trace(*args)
    assert stable_simulation_digest(first) == stable_simulation_digest(second)
    assert_sanity(combination_sanity_rows(first, 0.05, 0.005, 0.005))

