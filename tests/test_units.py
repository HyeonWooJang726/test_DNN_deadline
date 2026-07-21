import numpy as np

from dnn_profile import j_to_mj, kb_to_megabits, mj_to_j, ms_to_s, s_to_ms


def test_kb_to_megabits_decimal_units():
    assert kb_to_megabits(150.0) == 1.2
    np.testing.assert_allclose(kb_to_megabits([1.0, 1000.0]), [0.008, 8.0])


def test_time_conversions_round_trip():
    values = np.array([0.0, 1.0, 200.0, 1234.5])
    np.testing.assert_allclose(s_to_ms(ms_to_s(values)), values)


def test_energy_conversions_round_trip():
    values = np.array([0.0, 1.0, 800.0, 1200.5])
    np.testing.assert_allclose(j_to_mj(mj_to_j(values)), values)

