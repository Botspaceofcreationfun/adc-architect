"""Tests for the end-to-end simulator pipeline."""

import numpy as np
import pytest

from saradc.config import ADCConfig, NonidealityConfig
from saradc.pipeline import Simulator


def test_pipeline_ideal_sar_equals_ideal_quantizer_within_metrics():
    """With zero nonidealities the SAR metrics must match ideal theory."""
    cfg = ADCConfig(n_bits=10, v_ref=1.0)
    res = Simulator(cfg, NonidealityConfig(seed=1), fs=1e6, n=16384, n_cycles=997, amp=0.45).run()
    assert res.spectrum["sndr_db"] > 60.0
    assert res.metrics["inl_max_endpoint_lsb"] < 1e-9
    assert res.metrics["dnl_max_lsb"] < 1e-9
    assert res.metrics["gain_error_frac"] < 1e-9
    assert res.metrics["latency"]["comparison_cycles"] == 10


def test_pipeline_nonideal_degrades_metrics():
    cfg = ADCConfig(n_bits=10, v_ref=1.0)
    clean = Simulator(cfg, NonidealityConfig(seed=2), fs=1e6, n=16384, n_cycles=997, amp=0.45).run()
    dirty = Simulator(
        cfg,
        NonidealityConfig(
            cap_mismatch_sigma=0.02, comparator_noise_rms=1.5 * cfg.lsb,
            sampling_jitter_rms=8e-8, acquisition_time=2e-7, switch_tau=1e-7, seed=2,
        ),
        fs=1e6, n=16384, n_cycles=997, amp=0.45,
    ).run()
    assert dirty.spectrum["sndr_db"] < clean.spectrum["sndr_db"] - 8.0
    assert dirty.metrics["dnl_max_lsb"] > clean.metrics["dnl_max_lsb"]
    assert dirty.metrics["inl_max_endpoint_lsb"] > clean.metrics["inl_max_endpoint_lsb"]


def test_pipeline_16bit_runs():
    """16-bit is the top of the supported range; the record must convert fast."""
    import time

    cfg = ADCConfig(n_bits=16, v_ref=1.0)
    t0 = time.perf_counter()
    res = Simulator(cfg, NonidealityConfig(seed=3), fs=1e6, n=16384, n_cycles=997, amp=0.45).run()
    dt = time.perf_counter() - t0
    assert dt < 30.0  # generous bound; typical runtime is well under a second
    assert res.spectrum["enob"] > 10.0  # limited by record/window noise floor


def test_pipeline_reproducible_with_seed():
    cfg = ADCConfig(n_bits=12, v_ref=1.0)
    r1 = Simulator(cfg, NonidealityConfig(cap_mismatch_sigma=0.01, comparator_noise_rms=0.002, seed=99), fs=1e6, n=8192, n_cycles=499, amp=0.45).run()
    r2 = Simulator(cfg, NonidealityConfig(cap_mismatch_sigma=0.01, comparator_noise_rms=0.002, seed=99), fs=1e6, n=8192, n_cycles=499, amp=0.45).run()
    np.testing.assert_array_equal(r1.codes, r2.codes)
    assert r1.spectrum["sndr_db"] == pytest.approx(r2.spectrum["sndr_db"], abs=1e-12)


def test_pipeline_respects_input_range():
    cfg = ADCConfig(n_bits=8, v_ref=2.0, v_min=-1.0, coding="twos_complement")
    res = Simulator(cfg, NonidealityConfig(seed=4), fs=1e6, n=8192, n_cycles=499, amp=0.9, dc=0.0).run()
    assert res.codes.min() >= -128
    assert res.codes.max() <= 127
    assert np.all(np.abs(res.qerr_sar) <= cfg.lsb / 2 + 2 * cfg.lsb)  # near-ideal decode
