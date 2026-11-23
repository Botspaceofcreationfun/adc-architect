"""Tests for the configuration dataclasses."""

import pytest

from saradc.config import ADCConfig, NonidealityConfig


def test_default_config():
    cfg = ADCConfig()
    assert cfg.n_bits == 8
    assert cfg.v_ref == 1.0
    assert cfg.v_min == 0.0
    assert cfg.v_max == 1.0
    assert cfg.lsb == pytest.approx(1.0 / 256)
    assert cfg.n_codes == 256


@pytest.mark.parametrize("bits", [4, 8, 12, 16])
def test_bits_accepted(bits):
    ADCConfig(n_bits=bits)


@pytest.mark.parametrize("bits", [3, 17, 0, -4, 8.5])
def test_bits_rejected(bits):
    with pytest.raises(ValueError):
        ADCConfig(n_bits=bits)


def test_vref_rejected():
    with pytest.raises(ValueError):
        ADCConfig(v_ref=0)
    with pytest.raises(ValueError):
        ADCConfig(v_ref=-1.0)


def test_bad_coding_rejected():
    with pytest.raises(ValueError):
        ADCConfig(coding="bcd")
    with pytest.raises(ValueError):
        ADCConfig(tread="bogus")
    with pytest.raises(ValueError):
        ADCConfig(reconstruction="bogus")


def test_nonideal_validation():
    with pytest.raises(ValueError):
        NonidealityConfig(comparator_noise_rms=-1)
    with pytest.raises(ValueError):
        NonidealityConfig(sampling_jitter_rms=-0.1)
    with pytest.raises(ValueError):
        NonidealityConfig(switch_tau=-1e-9)


def test_nonideal_is_ideal_flag():
    assert NonidealityConfig().is_ideal
    assert not NonidealityConfig(comparator_offset=1e-6).is_ideal
    assert not NonidealityConfig(cap_mismatch_sigma=0.001).is_ideal
