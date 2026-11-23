"""Tests for the ideal quantizer module."""

import numpy as np
import pytest

from saradc.config import ADCConfig
from saradc.ideal import IdealADC


def test_offset_binary_codes():
    cfg = ADCConfig(n_bits=8, v_ref=1.0)
    adc = IdealADC(cfg)
    assert adc.code(0.0) == 0
    assert adc.code(1.0 / 256 - 1e-12) == 0
    assert adc.code(1.0 / 256) == 1
    assert adc.code(0.5) == 128
    assert adc.code(255.0 / 256) == 255
    assert adc.code(1.0) == 255          # clip at full scale
    assert adc.code(-1.0) == 0           # clip at zero


def test_reconstruction_midpoint_and_code():
    cfg = ADCConfig(n_bits=4, v_ref=1.0, reconstruction="midpoint")
    adc = IdealADC(cfg)
    assert adc.reconstruct(0) == pytest.approx(1.0 / 32)
    assert adc.reconstruct(1) == pytest.approx(3.0 / 32)

    cfg2 = ADCConfig(n_bits=4, v_ref=1.0, reconstruction="code")
    adc2 = IdealADC(cfg2)
    assert adc2.reconstruct(1) == pytest.approx(1.0 / 16)


def test_quant_error_bounded_midrise():
    cfg = ADCConfig(n_bits=8, v_ref=1.0)
    adc = IdealADC(cfg)
    v = np.linspace(0.0, 255.0 / 256, 100000)
    e = adc.quant_error(v)
    assert np.all(e >= -cfg.lsb / 2 - 1e-12)
    assert np.all(e <= cfg.lsb / 2 + 1e-12)
    # RMS of a uniform-in-LSB input equals LSB/sqrt(12)
    assert abs(np.sqrt(np.mean(e ** 2)) - cfg.lsb / np.sqrt(12)) < cfg.lsb * 1e-2


def test_quant_error_bounded_midtread():
    cfg = ADCConfig(n_bits=8, v_ref=1.0, tread="midtread")
    adc = IdealADC(cfg)
    v = np.linspace(0.0, 0.99, 100000)
    e = adc.quant_error(v)
    assert np.all(e >= -cfg.lsb / 2 - 1e-12)
    assert np.all(e <= cfg.lsb / 2 + 1e-12)


def test_twos_complement():
    cfg = ADCConfig(n_bits=8, v_ref=2.0, v_min=-1.0, coding="twos_complement")
    adc = IdealADC(cfg)
    assert adc.code(0.0) == 0
    assert adc.code(-1.0) == -128
    assert adc.code(1.0 - 1e-6) == 127
    assert adc.code(1.0) == 127
    # reconstruct round trip
    for v in (-0.75, -0.1, 0.0, 0.25, 0.99):
        c = adc.code(v)
        r = adc.reconstruct(c)
        assert abs(r - v) <= cfg.lsb / 2 + 1e-9


def test_transfer_monotonic():
    cfg = ADCConfig(n_bits=6, v_ref=1.0)
    adc = IdealADC(cfg)
    v, codes, _ = adc.transfer(10000)
    assert np.all(np.diff(codes) >= 0)


def test_midtread_zero_code():
    cfg = ADCConfig(n_bits=8, v_ref=1.0, tread="midtread", v_min=-0.5)
    adc = IdealADC(cfg)
    # midtread: input at exactly -0.5 (v_min) maps to code 0
    assert adc.code(-0.5) == 0
    assert adc.code(0.0) in (127, 128)  # half-way
    e = adc.quant_error(0.0)
    assert abs(e) <= cfg.lsb / 2 + 1e-9
