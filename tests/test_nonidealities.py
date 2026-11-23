"""Tests for nonideality models."""

import numpy as np
import pytest

from saradc.config import ADCConfig, NonidealityConfig
from saradc.frontend import SineWave, apply_sampling
from saradc.metrics import inl_dnl_from_transitions, transition_levels_analytical
from saradc.sar import SARConverter


def test_comparator_offset_measured_ramp_matches_closed_form():
    """Ramp-measured transitions must equal the analytical T[k] = DAC(k) - off."""
    cfg = ADCConfig(n_bits=8, v_ref=1.0)
    off = 0.5 * cfg.lsb
    sar = SARConverter(cfg, NonidealityConfig(comparator_offset=off, seed=1), rng=np.random.default_rng(1))
    T_an = transition_levels_analytical(cfg, sar.dac, off)

    from saradc.metrics import transition_levels_ramp

    T_ramp = transition_levels_ramp(sar, cfg, samples_per_code=32)
    np.testing.assert_allclose(T_ramp[1:-1], T_an[1:-1], atol=1.5 * cfg.lsb)


def test_ref_error_gain():
    cfg = ADCConfig(n_bits=8, v_ref=1.0)
    g = 0.05
    sar = SARConverter(cfg, NonidealityConfig(ref_error=g, seed=2), rng=np.random.default_rng(2))
    T = transition_levels_analytical(cfg, sar.dac, 0.0)
    m = inl_dnl_from_transitions(T, cfg.lsb, cfg.n_bits)
    # reference error is a pure gain error: best-fit INL stays ~0, gain ~g
    # (endpoint INL is large because the top code becomes unreachable -- that is
    #  exactly the physics of a reference/input-range mismatch).
    assert m["inl_max_bestfit"] < 0.05
    assert m["gain_error_frac"] == pytest.approx(g, rel=0.02)


def test_jitter_moves_sample_instants():
    cfg = ADCConfig(n_bits=8, v_ref=1.0)
    ni = NonidealityConfig(sampling_jitter_rms=5e-7, seed=4)
    rng = np.random.default_rng(4)
    wave = SineWave(0.45, 1e5, dc=0.5)
    t = np.linspace(0, 1e-5, 100)
    t_eff, _ = apply_sampling(wave, t, cfg, ni, rng)
    dt = t_eff - t
    assert np.abs(np.mean(dt)) < 1e-7
    assert abs(np.std(dt) - 5e-7) < 5e-8


def test_jitter_zero_is_identity():
    cfg = ADCConfig(n_bits=8, v_ref=1.0)
    ni = NonidealityConfig()
    rng = np.random.default_rng(4)
    wave = SineWave(0.45, 1e5, dc=0.5)
    t = np.linspace(0, 1e-5, 100)
    t_eff, v = apply_sampling(wave, t, cfg, ni, rng)
    np.testing.assert_allclose(t_eff, t)
    np.testing.assert_allclose(v, wave.eval(t))


def test_acquisition_error_grows_with_frequency():
    """Finite acquisition time produces an error that scales with slew rate."""
    cfg = ADCConfig(n_bits=12, v_ref=1.0)
    ni = NonidealityConfig(acquisition_time=1e-7, switch_tau=1e-7, seed=0)
    rng = np.random.default_rng(0)
    n = 2000

    def meas(freq):
        wave = SineWave(0.45, freq, dc=0.5)
        t = np.arange(n) / 2e6
        _, v_held = apply_sampling(wave, t, cfg, ni, rng)
        return np.mean(np.abs(wave.eval(t) - v_held))

    err_low = meas(10e3)
    err_high = meas(200e3)
    assert err_high > 3 * err_low


def test_acquisition_ideal_when_tau_zero():
    cfg = ADCConfig(n_bits=12, v_ref=1.0)
    ni = NonidealityConfig(acquisition_time=1e-7, switch_tau=0.0, seed=0)
    rng = np.random.default_rng(0)
    wave = SineWave(0.45, 100e3, dc=0.5)
    t = np.arange(500) / 2e6
    _, v_held = apply_sampling(wave, t, cfg, ni, rng)
    np.testing.assert_allclose(v_held, wave.eval(t), atol=1e-12)


def test_acquisition_converges_with_long_window():
    cfg = ADCConfig(n_bits=12, v_ref=1.0)
    rng = np.random.default_rng(0)
    wave = SineWave(0.45, 100e3, dc=0.5)
    t = np.arange(500) / 2e6
    long_acq = NonidealityConfig(acquisition_time=50e-6, switch_tau=1e-8, seed=0)
    _, v_held = apply_sampling(wave, t, cfg, long_acq, rng)
    np.testing.assert_allclose(v_held, wave.eval(t), atol=1e-8)


def test_slew_nl_creates_second_harmonic():
    """The optional S/H nonlinearity must put power at 2*fin."""
    cfg = ADCConfig(n_bits=14, v_ref=1.0)
    ni = NonidealityConfig(slew_nl=1e-9, seed=5)
    from saradc.pipeline import Simulator

    res = Simulator(cfg, ni, fs=1e6, n=16384, n_cycles=997, amp=0.45).run()
    spec = res.spectrum
    f0 = spec["freqs"][spec["k_sig"]]
    # find power at 2*f0 within one bin
    k2 = int(round(2 * f0 / (spec["fs"] / spec["n"])))
    p2 = spec["power"][k2]
    floor = np.median(spec["power"][: len(spec["power"]) // 2])
    assert p2 > 100 * floor


def test_comparator_noise_reduces_sndr():
    cfg = ADCConfig(n_bits=10, v_ref=1.0)
    from saradc.pipeline import Simulator

    res_clean = Simulator(cfg, NonidealityConfig(seed=1), fs=1e6, n=16384, n_cycles=997, amp=0.45).run()
    res_noisy = Simulator(
        cfg, NonidealityConfig(comparator_noise_rms=2 * cfg.lsb, seed=1), fs=1e6, n=16384, n_cycles=997, amp=0.45
    ).run()
    assert res_noisy.spectrum["sndr_db"] < res_clean.spectrum["sndr_db"] - 6.0


def test_mismatch_inl_dnl_grows():
    cfg = ADCConfig(n_bits=10, v_ref=1.0)
    from saradc.dac import BinaryWeightedDAC

    def inl(sigma):
        dac = BinaryWeightedDAC(cfg, NonidealityConfig(cap_mismatch_sigma=sigma, seed=7), rng=np.random.default_rng(7))
        T = transition_levels_analytical(cfg, dac, 0.0)
        return inl_dnl_from_transitions(T, cfg.lsb, cfg.n_bits)

    m0 = inl(0.0)
    m1 = inl(0.01)
    m2 = inl(0.04)
    assert m0["dnl_max"] < 1e-9
    assert m1["dnl_max"] < m2["dnl_max"]
    assert m1["inl_max_bestfit"] < m2["inl_max_bestfit"]
