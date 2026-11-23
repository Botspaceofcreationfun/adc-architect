"""Tests for the binary-weighted capacitive DAC."""

import numpy as np
import pytest

from saradc.config import ADCConfig, NonidealityConfig
from saradc.dac import BinaryWeightedDAC


def test_ideal_dac_voltages():
    cfg = ADCConfig(n_bits=4, v_ref=1.0)
    dac = BinaryWeightedDAC(cfg, NonidealityConfig())
    assert dac.voltage(0) == 0.0
    assert dac.voltage(1) == pytest.approx(1.0 / 16)          # one LSB
    assert dac.voltage(2) == pytest.approx(2.0 / 16)
    assert dac.voltage(15) == pytest.approx(15.0 / 16)        # (2^N - 1) LSB
    assert dac.gain_error == pytest.approx(0.0)


def test_dac_table_consistency():
    cfg = ADCConfig(n_bits=10, v_ref=2.5)
    dac = BinaryWeightedDAC(cfg, NonidealityConfig())
    table = dac.voltage_table()
    for c in [0, 1, 5, 511, 1023]:
        assert table[c] == pytest.approx(dac.voltage(c))


def test_mismatch_zero_sigma_matches_ideal():
    cfg = ADCConfig(n_bits=8, v_ref=1.0)
    dac_ideal = BinaryWeightedDAC(cfg, NonidealityConfig())
    dac_mis = BinaryWeightedDAC(cfg, NonidealityConfig(cap_mismatch_sigma=0.0, seed=5))
    np.testing.assert_allclose(dac_ideal.weights, dac_mis.weights)


def test_mismatch_reproducible_and_positive():
    cfg = ADCConfig(n_bits=8, v_ref=1.0)
    d1 = BinaryWeightedDAC(cfg, NonidealityConfig(cap_mismatch_sigma=0.01, seed=123))
    d2 = BinaryWeightedDAC(cfg, NonidealityConfig(cap_mismatch_sigma=0.01, seed=123))
    np.testing.assert_allclose(d1.weights, d2.weights)
    assert np.all(d1.weights > 0)


def test_mismatch_breaks_ideal_linearity():
    cfg = ADCConfig(n_bits=8, v_ref=1.0)
    dac = BinaryWeightedDAC(cfg, NonidealityConfig(cap_mismatch_sigma=0.03, seed=7))
    lsb = cfg.lsb
    # At least one code deviates from the ideal code*LSB by more than a small
    # amount (i.e. mismatch actually does something).
    dev = np.abs(dac.voltage_table() - np.arange(256) * lsb)
    assert np.max(dev) > 0.5 * lsb


def test_mismatch_scales_with_sigma():
    cfg = ADCConfig(n_bits=10, v_ref=1.0)
    lsb = cfg.lsb
    dac_s = BinaryWeightedDAC(cfg, NonidealityConfig(cap_mismatch_sigma=0.0, seed=1))
    dac_m1 = BinaryWeightedDAC(cfg, NonidealityConfig(cap_mismatch_sigma=0.005, seed=1))
    dac_m2 = BinaryWeightedDAC(cfg, NonidealityConfig(cap_mismatch_sigma=0.02, seed=1))
    dev = lambda d: np.max(np.abs(d.voltage_table() - np.arange(d.n_codes) * lsb))
    assert dev(dac_m2) > dev(dac_m1) > dev(dac_s)


def test_ref_error_scales_voltages():
    cfg = ADCConfig(n_bits=8, v_ref=1.0)
    dac = BinaryWeightedDAC(cfg, NonidealityConfig(ref_error=0.05))
    assert dac.ref_actual == pytest.approx(1.05)
    assert dac.voltage(128) == pytest.approx(1.05 * 128.0 / 256)
