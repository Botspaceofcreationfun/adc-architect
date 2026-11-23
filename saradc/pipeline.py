"""End-to-end simulation pipeline: signal -> S/H -> SAR -> decode -> metrics.

A single :class:`Simulator` bundles the ideal reference ADC and the nonideal
SAR model so that the two can be compared on identical test conditions with a
single call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .config import ADCConfig, NonidealityConfig
from .dac import BinaryWeightedDAC
from .frontend import SineWave, apply_sampling
from .ideal import IdealADC
from .metrics import (
    inl_dnl_from_transitions,
    spectral_analysis,
    transition_levels_analytical,
)
from .sar import SARConverter


@dataclass
class SimulationResult:
    """Everything produced by one simulation run."""

    cfg: ADCConfig
    nonideal: NonidealityConfig
    fs: float
    f_sig: float
    amp: float
    n: int
    t: np.ndarray            # nominal sample times
    t_eff: np.ndarray        # jittered sample times
    v_in: np.ndarray         # ideal instantaneous input at t_eff
    v_held: np.ndarray       # held samples fed to the SAR
    codes: np.ndarray        # SAR output codes (configured convention)
    codes_int: np.ndarray    # unsigned offset-binary codes
    recon: np.ndarray        # ideal digital decode (V)
    recon_actual: np.ndarray  # actual-DAC decode (V)
    qerr_sar: np.ndarray     # v_held - recon (V)
    qerr_ideal: np.ndarray   # v_in - ideal-reconstruct (V)
    ideal_codes: np.ndarray  # ideal ADC codes on the same samples
    spectrum: dict           # spectral_analysis output for the SAR record
    metrics: dict            # INL/DNL/latency summary
    transfer: tuple          # (v_grid, codes, recon) of the SAR model
    ideal: IdealADC | None = None  # ideal reference quantizer
    dac: BinaryWeightedDAC | None = None  # DAC model used by the SAR

    def to_dict(self) -> dict:
        return {
            "cfg": {
                "n_bits": self.cfg.n_bits,
                "v_ref": self.cfg.v_ref,
                "v_min": self.cfg.v_min,
                "coding": self.cfg.coding,
                "tread": self.cfg.tread,
                "reconstruction": self.cfg.reconstruction,
            },
            "nonideal": {
                "comparator_offset": self.nonideal.comparator_offset,
                "comparator_noise_rms": self.nonideal.comparator_noise_rms,
                "ref_error": self.nonideal.ref_error,
                "cap_mismatch_sigma": self.nonideal.cap_mismatch_sigma,
                "acquisition_time": self.nonideal.acquisition_time,
                "switch_tau": self.nonideal.switch_tau,
                "sampling_jitter_rms": self.nonideal.sampling_jitter_rms,
                "slew_nl": self.nonideal.slew_nl,
                "seed": self.nonideal.seed,
            },
            "test_signal": {"fs": self.fs, "f_sig": self.f_sig, "amp": self.amp, "n": self.n},
            "spectrum": {
                k: v for k, v in self.spectrum.items() if k in ("sndr_db", "snr_db", "thd_db", "sfdr_db", "enob", "P_sig", "P_noise", "P_harm")
            },
            "metrics": {
                "inl_max_endpoint_lsb": self.metrics.get("inl_max_endpoint_lsb"),
                "inl_max_bestfit_lsb": self.metrics.get("inl_max_bestfit_lsb"),
                "dnl_max_lsb": self.metrics.get("dnl_max_lsb"),
                "gain_error_frac": self.metrics.get("gain_error_frac"),
                "latency": self.metrics.get("latency"),
            },
        }


class Simulator:
    """Run ideal and nonideal simulations side by side.

    Parameters
    ----------
    cfg : ADCConfig
    nonideal : NonidealityConfig
    fs : float
        Sampling frequency (Hz).
    n : int
        Record length (number of samples).
    n_cycles : int
        Integer number of cycles of the test tone (coherent sampling).
    amp : float
        Tone amplitude (V peak). Keep below the input range to avoid clipping.
    phase : float
        Initial phase of the tone.
    dc : float
        DC offset of the tone.
    """

    def __init__(
        self,
        cfg: ADCConfig,
        nonideal: NonidealityConfig | None = None,
        fs: float = 1e6,
        n: int = 16384,
        n_cycles: int = 997,
        amp: float = 0.45,
        phase: float = 0.0,
        dc: float | None = None,
    ) -> None:
        nonideal = nonideal or NonidealityConfig()
        self.cfg = cfg
        self.nonideal = nonideal
        self.fs = float(fs)
        self.n = int(n)
        self.amp = float(amp)
        self.phase = float(phase)
        # Centre the tone in the input range by default so it does not clip.
        if dc is None:
            dc = cfg.v_min + 0.5 * cfg.v_ref
        self.dc = float(dc)

        from .frontend import coherent_freq

        self.f_sig = coherent_freq(self.fs, self.n, n_cycles)
        self.wave = SineWave(self.amp, self.f_sig, phase=self.phase, dc=self.dc)
        self.rng = np.random.default_rng(nonideal.seed)

        self.ideal = IdealADC(cfg)
        self.dac = BinaryWeightedDAC(cfg, nonideal, rng=self.rng)
        self.sar = SARConverter(cfg, nonideal, dac=self.dac, rng=self.rng)

    # ------------------------------------------------------------------
    def run(self, return_arrays: bool = False) -> SimulationResult:
        """Run the full pipeline and compute all metrics."""
        t = np.arange(self.n, dtype=float) / self.fs
        t_eff, v_held = apply_sampling(self.wave, t, self.cfg, self.nonideal, self.rng)
        v_in = np.asarray(self.wave.eval(t_eff), dtype=float)

        codes = self.sar.convert_bulk(v_held)
        codes_int = self.sar.convert_bulk_int(v_held)
        recon = self.sar.decode_ideal(codes_int)
        recon_actual = self.dac.voltage(codes_int) + self.cfg.v_min
        qerr_sar = v_held - recon
        ideal_codes = self.ideal.code(v_in)
        qerr_ideal = v_in - self.ideal.reconstruct(ideal_codes)

        spectrum = spectral_analysis(recon, self.fs, self.f_sig)

        T = transition_levels_analytical(self.cfg, self.dac, self.nonideal.comparator_offset)
        inl = inl_dnl_from_transitions(T, self.cfg.lsb, self.cfg.n_bits)

        metrics = {
            "inl_max_endpoint_lsb": inl["inl_max_endpoint"],
            "inl_max_bestfit_lsb": inl["inl_max_bestfit"],
            "dnl_max_lsb": inl["dnl_max"],
            "gain_error_frac": inl["gain_error_frac"],
            "latency": self.sar.latency(t_cycle=1e-9, fs=self.fs),
        }

        transfer = self.sar.transfer(n_pts=min(self.cfg.n_codes * 8, 65536))

        res = SimulationResult(
            cfg=self.cfg,
            nonideal=self.nonideal,
            fs=self.fs,
            f_sig=self.f_sig,
            amp=self.amp,
            n=self.n,
            t=t,
            t_eff=t_eff,
            v_in=v_in,
            v_held=v_held,
            codes=codes,
            codes_int=codes_int,
            recon=recon,
            recon_actual=recon_actual,
            qerr_sar=qerr_sar,
            qerr_ideal=qerr_ideal,
            ideal_codes=ideal_codes,
            spectrum=spectrum,
            metrics=metrics,
            transfer=transfer,
            ideal=self.ideal,
            dac=self.dac,
        )
        return res
