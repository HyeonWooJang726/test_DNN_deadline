import numpy as np

from config import DNNProfileConfig
from dnn_profile import (
    compute_slot_costs,
    j_to_mj,
    kb_to_megabits,
    minimum_good_deadline_ms,
    mj_to_j,
    ms_to_s,
    s_to_ms,
)


def test_kb_to_megabits_decimal_units():
    assert kb_to_megabits(150.0) == 1.2
    np.testing.assert_allclose(kb_to_megabits([1.0, 1000.0]), [0.008, 8.0])


def test_time_conversions_round_trip():
    values = np.array([0.0, 1.0, 200.0, 1234.5])
    np.testing.assert_allclose(s_to_ms(ms_to_s(values)), values)


def test_energy_conversions_round_trip():
    values = np.array([0.0, 1.0, 800.0, 1200.5])
    np.testing.assert_allclose(j_to_mj(mj_to_j(values)), values)


def test_dmin_is_normal_only_and_boost_can_meet_tight_bad_slot():
    profile = DNNProfileConfig()
    d_min = minimum_good_deadline_ms(profile, 40.0, 1.2)
    assert d_min == 36.045
    costs = compute_slot_costs(
        profile,
        np.array([10.0]),
        deadline_ms=1.35 * d_min,
        tx_power_w=1.2,
        skip_mode="drop",
    )
    assert costs.feasible[0]
    assert costs.mode_names[costs.meet_local_mode[0]] == "boost"
