"""Performance metrics: INL, DNL, SNDR, ENOB, THD, SFDR, conversion latency.

All measurement methodology is documented in :doc:`docs/TECHNICAL.md`; the
implementation below follows IEEE-1241-style definitions.

Transition levels
-----------------
The (ideal or nonideal) converter's code transitions ``T[k]`` are the input
voltages at which the output code crosses ``k - 1 -> k``. Two ways are provided:

* :func:`transition_levels_analytical` -- closed form for the SAR model:
  with monotone DAC weights and a deterministic comparator, converting a ramp
  yields the maximum code whose actual DAC voltage does not exceed the input.
  Hence ``T[k] = V_DAC(k) - v_off`` (mid-rise, in the shifted domain). This is
  exact for the model and O(2**N).
* :func:`transition_levels_ramp` -- generic measurement by feeding a dense
  ramp and locating the first input that produces code ``>= k``. Used to
  cross-validate the closed form and for the ideal quantizer module.

INL / DNL
---------
Defined per code in LSB units:

* ``DNL[k] = (T[k+1] - T[k])/LSB - 1``        (k = 0 .. 2**N - 2)
* ``INL[k] = (T[k] - k*LSB)/LSB``             (endpoint fit; includes gain/offset)
* ``INL_bf[k]``                                (best-fit line removed)

SNDR / ENOB
-----------
Coherent single-tone records are transformed with an FFT; signal power is the
mainlobe around the fundamental, noise+distortion is everything else (DC
excluded), harmonics are assigned to THD/SFDR. ENOB is derived as
``(SNDR_dB - 1.76)/6.02``, valid when a full-scale sine is used and
quantization noise dominates (see TECHNICAL.md for the assumptions).
"""

from __future__ import annotations

import numpy as np

from .config import ADCConfig
from .dac import BinaryWeightedDAC
from .ideal import IdealADC
from .sar import SARConverter


# ---------------------------------------------------------------------------
# transition levels
# ---------------------------------------------------------------------------

def transition_levels_analytical(
    cfg: ADCConfig,
    dac: BinaryWeightedDAC,
    comparator_offset: float = 0.0,
) -> np.ndarray:
    """Exact transition levels ``T[0 .. 2**N]`` (V) for the SAR model.

    ``T[k] = V_DAC(k) - v_off`` for ``k = 1 .. 2**N - 1``, ``T[0] = 0`` and
    ``T[2**N] = Vref`` (saturation endpoint). Values are in the shifted
    comparison domain ``[0, Vref]``; add ``v_min`` for absolute input volts.
    """
    n_codes = cfg.n_codes
    T = np.empty(n_codes + 1, dtype=float)
    T[0] = 0.0
    T[1:n_codes] = dac.voltage_table()[1:n_codes] - comparator_offset
    T[n_codes] = cfg.v_ref
    return T


def transition_levels_ramp(
    converter: SARConverter | IdealADC,
    cfg: ADCConfig,
    samples_per_code: int = 16,
    v_min: float | None = None,
    v_max: float | None = None,
) -> np.ndarray:
    """Measure transition levels by sweeping a dense ramp through the converter.

    For each code ``k = 1 .. 2**N - 1``, ``T[k]`` is estimated as the input at
    which the code first reaches ``k`` (first-occurrence of each code on a fine
    grid). ``T[0] = v_min``, ``T[2**N] = v_max`` by definition.
    """
    v_lo = v_min if v_min is not None else cfg.v_min
    v_hi = v_max if v_max is not None else cfg.v_max
    n_codes = cfg.n_codes
    n_pts = n_codes * samples_per_code + 1
    v = np.linspace(v_lo, v_hi, n_pts)

    if isinstance(converter, IdealADC):
        codes = converter.code_int(v)
    else:
        codes = converter.convert_bulk_int(v)

    T = np.full(n_codes + 1, np.nan, dtype=float)
    T[0] = v_lo
    T[n_codes] = v_hi
    # first index where code >= k, then linear refinement between neighbours
    for k in range(1, n_codes):
        idx = int(np.argmax(codes >= k))
        if codes[idx] < k:  # never reached (saturation/clip) -- use upper edge
            T[k] = v_hi
            continue
        if idx == 0:
            T[k] = v_lo
            continue
        # linear interpolation between samples idx-1 and idx
        c0, c1 = int(codes[idx - 1]), int(codes[idx])
        v0, v1 = v[idx - 1], v[idx]
        if c1 == c0:
            T[k] = v1
        else:
            T[k] = v0 + (v1 - v0) * (k - c0) / (c1 - c0)
    return T


# ---------------------------------------------------------------------------
# INL / DNL
# ---------------------------------------------------------------------------

def inl_dnl_from_transitions(T: np.ndarray, lsb: float, n_bits: int) -> dict:
    """Compute DNL / INL (LSB) from transition levels ``T[0 .. 2**N]``.

    ``T`` may be in absolute volts (any offset is removed by the definitions).
    Returns endpoint-fit and best-fit-line INL, DNL, and gain/offset errors.
    """
    n_codes = 1 << n_bits
    dnl = np.diff(T[: n_codes + 1]) / lsb - 1.0  # length 2**N
    # endpoint span gain error (LSB): how the endpoint slope differs from ideal
    span = T[n_codes] - T[0]
    gain_lsb_endpoint = span / lsb - n_codes

    # endpoint INL: cumulative DNL with zero endpoints
    inl_ep = np.cumsum(dnl)
    inl_ep = inl_ep - inl_ep[0]
    inl_ep = inl_ep - inl_ep[-1] * np.arange(n_codes) / (n_codes - 1) if n_codes > 1 else inl_ep

    # best-fit line INL (least squares through (k, T[k]-k*LSB))
    k = np.arange(n_codes, dtype=float)
    y = (T[:n_codes] - k * lsb) / lsb
    A = np.vstack([k, np.ones_like(k)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    inl_bf = y - (coef[0] * k + coef[1])
    # best-fit slope of the transition line: T ~ (1 + gain)*LSB * k
    gain_frac_bestfit = float(coef[0])

    return {
        "dnl": dnl,
        "inl_endpoint": inl_ep,
        "inl_bestfit": inl_bf,
        "dnl_max": float(np.max(np.abs(dnl))),
        "inl_max_endpoint": float(np.max(np.abs(inl_ep))),
        "inl_max_bestfit": float(np.max(np.abs(inl_bf))),
        "gain_error_lsb": float(gain_lsb_endpoint),
        "gain_error_frac": gain_frac_bestfit,
    }


# ---------------------------------------------------------------------------
# spectrum / SNDR / ENOB
# ---------------------------------------------------------------------------

def spectral_analysis(
    x: np.ndarray,
    fs: float,
    f_sig: float,
    window: str = "hann",
    n_harmonics: int = 6,
    signal_band: int = 2,
    n_discard: int = 4,
) -> dict:
    """Full spectral analysis of one record.

    Methodology (documented in detail in docs/TECHNICAL.md):

    * the record must be coherent (integer number of cycles of the tone);
    * optional Hann window is applied; the tone's mainlobe (``signal_band``
      bins either side of the fundamental) is counted as *signal*;
    * harmonics 2..n_harmonics are assigned to THD/SFDR (each with the same
      band to absorb window sidelobes);
    * the ``n_discard`` lowest bins are excluded as DC/leakage;
    * everything else is noise+distortion.

    ``SNDR = P_signal / P_(noise+distortion)``; ``ENOB = (SNDR_dB - 1.76)/6.02``.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 64:
        raise ValueError("record too short for spectral analysis")
    x = x - x.mean()

    if window == "hann":
        w = np.hanning(n)
    elif window == "none":
        w = np.ones(n)
    else:
        raise ValueError(f"unknown window {window!r}")

    X = np.fft.rfft(x * w)
    # power per bin (rfft: double positive-freq power except DC/Nyquist)
    P = np.abs(X) ** 2
    P[1:-1] *= 2.0
    P /= (w.sum() ** 2)  # window-energy normalisation (Parseval)

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    nf = len(P)  # rfft length = n//2 + 1
    k_sig = int(np.argmax(P[1:])) + 1

    def band_indices(k: int) -> list[int]:
        return list(range(max(n_discard, k - signal_band), min(nf - 1, k + signal_band) + 1))

    sig_bins = set(band_indices(k_sig))
    harm_bins: dict[int, list[int]] = {}
    k0 = fs / n  # bin width
    for h in range(2, n_harmonics + 1):
        kh = int(round(h * f_sig / k0))
        if kh < nf:
            harm_bins[h] = [b for b in band_indices(kh) if b not in sig_bins]

    all_harm = set()
    for bins in harm_bins.values():
        all_harm.update(bins)

    noise_bins = [k for k in range(n_discard, nf) if k not in sig_bins and k not in all_harm]

    P_sig = float(sum(P[k] for k in sig_bins))
    P_harm = float(sum(P[k] for k in all_harm))
    P_noise = float(sum(P[k] for k in noise_bins))
    P_tot = float(sum(P[n_discard:]))

    sndr_db = 10.0 * np.log10(P_sig / max(P_noise + P_harm, 1e-300))
    snr_db = 10.0 * np.log10(P_sig / max(P_noise, 1e-300)) if P_noise > 0 else float("inf")
    thd_db = 10.0 * np.log10(P_harm / max(P_sig, 1e-300))
    sfdr_db = 10.0 * np.log10(P_sig / max(max((P[k] for k in range(n_discard, nf) if k not in sig_bins), default=0.0), 1e-300))
    enob = (sndr_db - 1.76) / 6.02 if np.isfinite(sndr_db) else float("inf")

    return {
        "n": n,
        "fs": fs,
        "f_sig": f_sig,
        "k_sig": k_sig,
        "window": window,
        "P_sig": P_sig,
        "P_harm": P_harm,
        "P_noise": P_noise,
        "P_total": P_tot,
        "sndr_db": float(sndr_db),
        "snr_db": float(snr_db),
        "thd_db": float(thd_db),
        "sfdr_db": float(sfdr_db),
        "enob": float(enob),
        "freqs": freqs,
        "power": P,
        "power_db": 10.0 * np.log10(P + 1e-300),
    }


def enob(sndr_db: float) -> float:
    """ENOB from measured SNDR (valid under the assumptions in TECHNICAL.md)."""
    return (sndr_db - 1.76) / 6.02


# ---------------------------------------------------------------------------
# conversion latency (independent of any converter instance)
# ---------------------------------------------------------------------------

def conversion_latency(n_bits: int, t_cycle: float, acquisition_time: float = 0.0, fs: float | None = None) -> dict:
    """Latency of a SAR conversion: N comparison cycles + acquisition window."""
    total = acquisition_time + n_bits * t_cycle
    out = {"comparison_cycles": n_bits, "acquisition_time_s": acquisition_time, "t_cycle_s": t_cycle, "total_time_s": total}
    if fs:
        out["latency_sample_periods"] = total * fs
    return out
