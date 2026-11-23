"""Tests for performance metrics: INL/DNL, SNDR, ENOB, latency."""

import numpy as np
import pytest

from saradc.config import ADCConfig, NonidealityConfig
from saradc.dac import BinaryWeightedDAC
from saradc.metrics import (
    conversion_latency,
    enob,
    inl_dnl_from_transitions,
    spectral_analysis,
    transition_levels_analytical,
)


def test_ideal_inl_dnl_zero():
    cfg = ADCConfig(n_bits=10, v_ref=1.0)
    dac = BinaryWeightedDAC(cfg, NonidealityConfig())
    T = transition_levels_analytical(cfg, dac, 0.0)
    m = inl_dnl_from_transitions(T, cfg.lsb, cfg.n_bits)
    assert m["dnl_max"] < 1e-9
    assert m["inl_max_endpoint"] < 1e-9
    assert m["gain_error_frac"] == pytest.approx(0.0)


def test_dnl_sums_to_gain_error():
    cfg = ADCConfig(n_bits=8, v_ref=1.0)
    dac = BinaryWeightedDAC(cfg, NonidealityConfig(cap_mismatch_sigma=0.02, seed=3), rng=np.random.default_rng(3))
    T = transition_levels_analytical(cfg, dac, 0.02 * cfg.lsb)
    m = inl_dnl_from_transitions(T, cfg.lsb, cfg.n_bits)
    # sum of DNL over the full span equals the endpoint gain error
    assert abs(np.sum(m["dnl"]) - m["gain_error_lsb"]) < 1e-6


def test_inl_consistency_endpoint_vs_bestfit():
    cfg = ADCConfig(n_bits=10, v_ref=1.0)
    dac = BinaryWeightedDAC(cfg, NonidealityConfig(cap_mismatch_sigma=0.01, seed=9), rng=np.random.default_rng(9))
    T = transition_levels_analytical(cfg, dac, 0.0)
    m = inl_dnl_from_transitions(T, cfg.lsb, cfg.n_bits)
    # best-fit INL can only be smaller (or equal) than endpoint INL
    assert m["inl_max_bestfit"] <= m["inl_max_endpoint"] + 1e-9


def test_sndr_ideal_quantizer_theory():
    """Ideal N-bit quantizer: SNDR = 6.02*N + 1.76 - backoff (dB)."""
    for bits, tol in ((6, 0.5), (8, 0.5), (10, 0.6), (12, 1.0)):
        cfg = ADCConfig(n_bits=bits, v_ref=1.0)
        from saradc.pipeline import Simulator

        amp = 0.45  # -0.915 dBFS
        res = Simulator(cfg, NonidealityConfig(seed=1), fs=1e6, n=32768, n_cycles=997, amp=amp).run()
        theory = 6.02 * bits + 1.76 - 20 * np.log10(0.5 / amp)
        assert abs(res.spectrum["sndr_db"] - theory) < tol, (
            f"{bits}-bit: measured {res.spectrum['sndr_db']:.2f} vs theory {theory:.2f}"
        )
        assert abs(res.spectrum["enob"] - (theory - 1.76) / 6.02) < tol / 6.02


def test_enob_derivation():
    assert enob(6.02 * 8 + 1.76) == pytest.approx(8.0)
    assert enob(74.0) == pytest.approx((74.0 - 1.76) / 6.02)


def test_harmonics_assigned_to_thd():
    """A 2nd-harmonic-heavy signal must show up in THD and SFDR."""
    n = 8192
    fs = 1e6
    f0 = 997 * fs / n
    t = np.arange(n) / fs
    # strong 2nd harmonic relative to a small fundamental
    x = 0.1 * np.sin(2 * np.pi * f0 * t) + 0.05 * np.sin(2 * np.pi * 2 * f0 * t)
    spec = spectral_analysis(x, fs, f0, n_harmonics=6)
    # THD is negative dB (harmonic below carrier, summed over harmonics);
    # SFDR is the *positive* distance of the largest spur below the carrier.
    assert spec["thd_db"] < -3            # ~ -6 dB harmonic
    assert 0 < spec["sfdr_db"] < 12
    assert spec["snr_db"] > 60


def test_sndr_no_window_vs_hann_consistent():
    n = 16384
    fs = 1e6
    f0 = 997 * fs / n
    t = np.arange(n) / fs
    x = 0.45 * np.sin(2 * np.pi * f0 * t)
    x = x + np.random.default_rng(0).normal(0, 1e-4, n)
    s_win = spectral_analysis(x, fs, f0, window="hann")
    s_now = spectral_analysis(x, fs, f0, window="none")
    assert abs(s_win["sndr_db"] - s_now["sndr_db"]) < 1.5


def test_latency_formula():
    lat = conversion_latency(n_bits=12, t_cycle=2e-9, acquisition_time=50e-9, fs=10e6)
    assert lat["comparison_cycles"] == 12
    assert lat["total_time_s"] == pytest.approx(12 * 2e-9 + 50e-9)
    assert lat["latency_sample_periods"] == pytest.approx((12 * 2e-9 + 50e-9) * 10e6)


def test_ramp_vs_analytical_transitions():
    cfg = ADCConfig(n_bits=8, v_ref=1.0)
    from saradc.pipeline import Simulator

    sim = Simulator(cfg, NonidealityConfig(cap_mismatch_sigma=0.01, seed=6), fs=1e6, n=4096, n_cycles=63, amp=0.45)
    T_an = transition_levels_analytical(cfg, sim.dac, 0.0)
    from saradc.metrics import transition_levels_ramp

    T_ramp = transition_levels_ramp(sim.sar, cfg, samples_per_code=32)
    np.testing.assert_allclose(T_ramp[1:-1], T_an[1:-1], atol=cfg.lsb)
