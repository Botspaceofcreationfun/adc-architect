"""Tests for the cycle-by-cycle SAR converter -- the heart of the project."""

import numpy as np
import pytest

from saradc.config import ADCConfig, NonidealityConfig
from saradc.dac import BinaryWeightedDAC
from saradc.sar import SARConverter


def make(cfg: ADCConfig, ni: NonidealityConfig | None = None, seed: int = 0) -> SARConverter:
    return SARConverter(cfg, ni or NonidealityConfig(seed=seed), rng=np.random.default_rng(seed))


def test_sar_matches_ideal_floor():
    """With an ideal DAC the SAR must reproduce floor(v/LSB) exactly."""
    cfg = ADCConfig(n_bits=12, v_ref=1.0)
    sar = make(cfg)
    rng = np.random.default_rng(1)
    v = rng.uniform(0.0, 1.0 - 1e-9, size=20000)
    codes = sar.convert_bulk_int(v)
    ideal = np.floor(v * (1 << 12)).astype(np.int64)
    np.testing.assert_array_equal(codes, ideal)


def test_sar_trace_structure_and_correctness():
    """The traced conversion must expose every cycle and end on the right code."""
    cfg = ADCConfig(n_bits=8, v_ref=1.0)
    sar = make(cfg)
    tr = sar.convert_traced(0.63)

    assert len(tr.cycles) == 8
    assert [c.bit_index for c in tr.cycles] == [7, 6, 5, 4, 3, 2, 1, 0]
    assert tr.cycles[0].trial_code == 128                      # first trial = MSB
    # final register equals the bulk conversion result
    assert tr.final_code_int == int(sar.convert_bulk_int(np.array([0.63]))[0])
    # ideal decode within 1 LSB of the input (midpoint reconstruction)
    assert abs(tr.residual_ideal) <= cfg.lsb + 1e-12
    # register voltage trackable to the DAC model
    assert tr.recon_actual == pytest.approx(sar.dac.voltage(tr.final_code_int))


def test_sar_decisions_match_comparator_rule():
    """Each cycle's decision must equal (v_held + offset) >= v_trial."""
    cfg = ADCConfig(n_bits=6, v_ref=1.0)
    sar = make(cfg)
    tr = sar.convert_traced(0.42)
    for c in tr.cycles:
        assert c.decision == (0.42 >= c.trial_voltage - 1e-12)
        assert c.comparator_input == pytest.approx(0.42 - c.trial_voltage)


def test_sar_boundary_conditions():
    cfg = ADCConfig(n_bits=8, v_ref=1.0)
    sar = make(cfg)
    assert sar.convert_int(0.0) == 0
    assert sar.convert_int(-0.5) == 0                    # below range clamps
    assert sar.convert_int(1.0 - 1e-9) == 255
    assert sar.convert_int(1.0) == 255                   # at/above full scale
    assert sar.convert_int(5.0) == 255
    # saturation is flagged in the trace
    tr = sar.convert_traced(1.2)
    assert tr.saturated


def test_sar_monotonic():
    cfg = ADCConfig(n_bits=8, v_ref=1.0)
    sar = make(cfg)
    v = np.linspace(0.0, 1.0, 50000)
    codes = sar.convert_bulk_int(v)
    assert np.all(np.diff(codes) >= 0)


def test_sar_midtread():
    cfg = ADCConfig(n_bits=8, v_ref=1.0, tread="midtread")
    sar = make(cfg)
    # midtread is equivalent to midrise on (v + LSB/2)
    c = sar.convert_int(0.5)
    expected = int(np.floor((0.5 + 0.5 * cfg.lsb) * 256))
    assert c == expected
    # zero input near LSB/2 boundary
    assert sar.convert_int(0.0) == 0


def test_sar_twos_complement():
    cfg = ADCConfig(n_bits=8, v_ref=2.0, v_min=-1.0, coding="twos_complement")
    sar = make(cfg)
    assert sar.convert(-1.0) == -128
    assert sar.convert(0.0) == 0
    assert sar.convert(1.0 - 1e-9) == 127
    v = np.linspace(-1.0, 1.0, 1000)
    codes = sar.convert_bulk(v)
    assert codes.min() >= -128 and codes.max() <= 127


def test_sar_with_comparator_offset():
    """A comparator offset shifts decisions: code(v) behaves like code(v + off)."""
    cfg = ADCConfig(n_bits=10, v_ref=1.0)
    off = 0.001
    sar_off = make(cfg, NonidealityConfig(comparator_offset=off, seed=3), seed=3)
    sar_ideal = make(cfg, NonidealityConfig(seed=3), seed=3)
    rng = np.random.default_rng(9)
    v = rng.uniform(0.0, 0.99, size=5000)
    c_off = sar_off.convert_bulk_int(v)
    c_shift = sar_ideal.convert_bulk_int(v + off)
    np.testing.assert_array_equal(c_off, c_shift)


def test_sar_register_progression_binary_search():
    """The register must follow a genuine binary search (MSB first halving)."""
    cfg = ADCConfig(n_bits=8, v_ref=1.0)
    sar = make(cfg)
    tr = sar.convert_traced(0.31)
    reg = tr.cycles[0].register_code
    for c in tr.cycles[1:]:
        # after cycle i, the remaining uncertainty is halved: register differs
        # from the trial code by at most the bit just trialled
        assert abs(c.register_code - c.trial_code) in (0, 1 << c.bit_index)


def test_sar_bulk_scalar_consistency():
    cfg = ADCConfig(n_bits=8, v_ref=1.0)
    sar = make(cfg, NonidealityConfig(seed=11), seed=11)
    rng = np.random.default_rng(2)
    v = rng.uniform(0.0, 0.99, size=300)
    bulk = sar.convert_bulk_int(v)
    for vi, ci in zip(v, bulk):
        assert sar.convert_int(float(vi)) == ci


def test_latency_model():
    cfg = ADCConfig(n_bits=10)
    sar = make(cfg)
    lat = sar.latency(t_cycle=1e-9, fs=1e6)
    assert lat["comparison_cycles"] == 10
    assert lat["total_time_s"] == pytest.approx(10 * 1e-9 + lat["acquisition_time_s"])
    assert lat["latency_sample_periods"] == pytest.approx(lat["total_time_s"] * 1e6)


def test_sar_not_single_rounding():
    """The SAR must be a real bit-trialling algorithm, not a rounding op.

    With capacitor mismatch the final code for inputs near a mismatched
    transition differs from the ideal rounding of the same input, and the
    per-cycle trial voltages are the *actual* DAC outputs.
    """
    cfg = ADCConfig(n_bits=8, v_ref=1.0)
    sar = make(cfg, NonidealityConfig(cap_mismatch_sigma=0.03, seed=17), seed=17)
    # find a code whose actual transition differs from the ideal one
    T = sar.dac.voltage_table()
    ideal_T = np.arange(256) * cfg.lsb
    diffs = T - ideal_T
    k = int(np.argmax(np.abs(diffs[1:]))) + 1
    assert k > 0
    # at v slightly above the *actual* transition but below the ideal one the
    # SAR must produce a code the simple ideal rounding would not.
    v = T[k] + cfg.lsb / 8
    assert v < ideal_T[k]
    c = sar.convert_int(v)
    assert c >= k
    assert c != int(np.floor(v / cfg.lsb))
    # trial voltages come from the mismatched DAC, not code*LSB
    tr = sar.convert_traced(v)
    assert any(abs(c.trial_voltage - c.trial_code * cfg.lsb) > 0.2 * cfg.lsb for c in tr.cycles)
