"""Test-signal generation and the analog sample-and-hold front end.

Signals are generated *analytically* (a wave object that can be evaluated at
arbitrary instants), which is what makes sampling jitter and acquisition
settling physically meaningful: the converter samples ``wave.eval(t + jitter)``
rather than indexing a precomputed array.
"""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np

from .config import ADCConfig, NonidealityConfig


# ---------------------------------------------------------------------------
# Analytic waveforms
# ---------------------------------------------------------------------------

class SineWave:
    """Single-tone sine: ``v(t) = dc + amp * sin(2*pi*freq*t + phase)``."""

    def __init__(self, amp: float, freq: float, phase: float = 0.0, dc: float = 0.0) -> None:
        if amp < 0:
            raise ValueError("amp must be >= 0")
        if freq < 0:
            raise ValueError("freq must be >= 0")
        self.amp = float(amp)
        self.freq = float(freq)
        self.phase = float(phase)
        self.dc = float(dc)

    def eval(self, t: float | np.ndarray) -> float | np.ndarray:
        return self.dc + self.amp * np.sin(2.0 * np.pi * self.freq * np.asarray(t, dtype=float) + self.phase)

    def deriv(self, t: float | np.ndarray) -> float | np.ndarray:
        return self.amp * 2.0 * np.pi * self.freq * np.cos(2.0 * np.pi * self.freq * np.asarray(t, dtype=float) + self.phase)

    def __repr__(self) -> str:
        return f"SineWave(amp={self.amp:.4g}, freq={self.freq:.4g}, phase={self.phase:.4g}, dc={self.dc:.4g})"


def coherent_freq(fs: float, n: int, n_cycles: int, margin: float = 0.1) -> float:
    """Frequency that makes an ``n``-sample record coherent at ``fs``.

    Coherent sampling of ``n_cycles`` integer cycles guarantees zero spectral
    leakage of the test tone (fundamental lands on an exact DFT bin), which the
    SNDR methodology relies on. ``margin`` keeps the tone below Nyquist.
    """
    if n_cycles <= 0 or n_cycles >= (n // 2) * (1 - margin):
        raise ValueError(f"n_cycles={n_cycles} out of coherent range for n={n}")
    return n_cycles * fs / n


def coherent_sine(
    fs: float,
    n: int,
    n_cycles: int,
    amp: float = 0.45,
    dc: float = 0.0,
    phase: float = 0.0,
    margin: float = 0.1,
) -> SineWave:
    """Coherent single-tone test signal."""
    return SineWave(amp, coherent_freq(fs, n, n_cycles, margin), phase=phase, dc=dc)


# ---------------------------------------------------------------------------
# Sample-and-hold front end
# ---------------------------------------------------------------------------

def apply_sampling(
    wave: SineWave,
    t: np.ndarray,
    cfg: ADCConfig,
    nonideal: NonidealityConfig,
    rng: np.random.Generator,
    v_prev: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Model the sample-and-hold chain for sample times ``t``.

    Effects applied, in order:

    1. **Sampling jitter** -- each sample instant is perturbed by
       ``N(0, sampling_jitter_rms)``. The signal is then evaluated at the
       perturbed instant (the aperture-time view of jitter).
    2. **Finite acquisition time** -- the track switch closes ``T_acq`` before
       the sampling instant. The capacitor charges from the previously held
       value toward the instantaneous input with time constant ``tau``; the
       held value is the imperfectly-settled residue at the instant the switch
       opens::

           v_held = v(t) - (v(t) - v_prev) * exp(-T_acq / tau)

       This error grows with signal slew rate, i.e. with input frequency.
    3. **Slew-rate nonlinearity** (optional) -- a behavioral distortion term
       ``slew_nl * (dv/dt)**2`` (even-harmonic content). This is the model's
       "input-frequency-dependent distortion" mechanism; it is a simplified
       behavioral term, not a transistor-level effect.
    4. **Range clamp** -- held values outside ``[v_min, v_max]`` are clipped
       (the input buffer's saturation).

    Returns
    -------
    t_eff : np.ndarray
        The actually-used sampling instants (with jitter).
    v_held : np.ndarray
        The held analog values fed to the converter.
    """
    t_eff = np.asarray(t, dtype=float).copy()

    if nonideal.sampling_jitter_rms > 0.0:
        t_eff = t_eff + rng.normal(0.0, nonideal.sampling_jitter_rms, size=t_eff.shape)

    v_true = np.asarray(wave.eval(t_eff), dtype=float)

    if nonideal.switch_tau > 0.0 and nonideal.acquisition_time > 0.0:
        settle = np.exp(-nonideal.acquisition_time / nonideal.switch_tau)
        v_held = np.empty_like(v_true)
        prev = float(v_prev)
        for k in range(len(v_true)):
            vk = v_true[k]
            v_held[k] = vk - (vk - prev) * settle
            prev = v_held[k]
    else:
        v_held = v_true.copy()

    if nonideal.slew_nl != 0.0:
        d = np.asarray(wave.deriv(t_eff), dtype=float)
        v_held = v_held + nonideal.slew_nl * d * d

    v_lo, v_hi = cfg.v_min, cfg.v_max
    v_held = np.clip(v_held, v_lo, v_hi)

    return t_eff, v_held


def apply_sampling_scalar(
    wave: SineWave,
    t: float,
    cfg: ADCConfig,
    nonideal: NonidealityConfig,
    rng: np.random.Generator,
    v_prev: float = 0.0,
) -> float:
    """Scalar version of :func:`apply_sampling` (used by the cycle trace)."""
    (t_eff_arr, v_held_arr) = apply_sampling(
        wave, np.array([t]), cfg, nonideal, rng, v_prev=v_prev
    )
    return float(v_held_arr[0])
