"""Cycle-by-cycle successive-approximation register (SAR) converter.

This module implements the actual SAR *algorithm*: each conversion walks the
register one bit at a time, setting a trial bit, generating the corresponding
DAC voltage, comparing it against the held sample, and keeping or clearing the
bit. The code is **never** produced by a single rounding operation -- every
step of the binary search is simulated and can be traced.

Per conversion cycle the model exposes:

* the trial bit (MSB first),
* the trial DAC code and the corresponding DAC output voltage,
* the comparator input ``(v_held + v_off + noise) - v_dac``,
* the comparator decision (keep / clear),
* the SAR register after the decision,
* the residual error ``v_held - v_dac(register)``.

Two execution paths share the same model:

* :meth:`SARConverter.convert` / :meth:`convert_traced` -- scalar, records the
  full per-cycle trace (interactive mode);
* :meth:`SARConverter.convert_bulk` -- vectorised over many samples (metrics,
  transfer curves, experiments), producing identical codes when the comparator
  is deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import ADCConfig, NonidealityConfig
from .dac import BinaryWeightedDAC


class Comparator:
    """Ideal-to-real comparator: input offset plus per-decision Gaussian noise.

    Decision rule (keep trial bit)::

        (v_held - v_dac) + v_off + noise >= 0

    With ``noise_rms == 0`` the decision is deterministic; with noise the bit
    error probability follows the Gaussian Q-function, as in a real latch
    comparator.
    """

    def __init__(
        self,
        cfg: ADCConfig,
        nonideal: NonidealityConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        nonideal = nonideal or NonidealityConfig()
        self.v_off = nonideal.comparator_offset
        self.noise_rms = nonideal.comparator_noise_rms
        if rng is None:
            rng = np.random.default_rng(nonideal.seed)
        self.rng = rng
        self.cfg = cfg

    def decide(self, v_held: float | np.ndarray, v_dac: float | np.ndarray) -> bool | np.ndarray:
        arr_v = np.asarray(v_held, dtype=float)
        arr_d = np.asarray(v_dac, dtype=float)
        c = (arr_v - arr_d) + self.v_off
        if self.noise_rms > 0.0:
            shape = arr_v.shape if arr_v.ndim > 0 else ()
            noise = self.rng.normal(0.0, self.noise_rms, size=shape)
            c = c + noise
        result = c >= 0.0
        if arr_v.ndim == 0:
            return bool(result)
        return result

    def margin(self, v_held, v_dac) -> float:
        """Comparator input before noise (V); useful for visualising margin."""
        return float((v_held - v_dac) + self.v_off)


@dataclass
class CycleStep:
    """One SAR comparison cycle."""

    cycle: int          # 0-based cycle number
    bit_index: int      # which bit is being trialled (0 = LSB)
    bit_weight: float   # ideal weight of this bit in volts (LSB * 2**bit_index)
    trial_code: int     # register | (1 << bit_index)
    trial_voltage: float  # DAC output voltage for the trial code (V)
    comparator_input: float  # (v_held - v_dac) + offset, pre-noise (V)
    decision: bool      # True = keep trial bit
    register_code: int  # SAR register after this cycle
    register_voltage: float  # DAC voltage of the register (V)
    residual: float     # v_held - register_voltage (V)


@dataclass
class TraceResult:
    """Full record of one traced conversion."""

    v_input: float          # analog input voltage (V)
    v_held: float           # held sample actually converted (V)
    v_min_shift: float      # held - v_min (comparison domain) (V)
    lsb: float              # ideal LSB (V)
    n_bits: int
    cycles: list[CycleStep] = field(default_factory=list)
    final_code_int: int = 0  # unsigned offset-binary code
    final_code: int = 0      # code in configured convention
    recon_ideal: float = 0.0  # ideal-decode reconstruction (V)
    recon_actual: float = 0.0  # actual-DAC reconstruction (V)
    residual_ideal: float = 0.0  # v_input - recon_ideal (V)
    residual_actual: float = 0.0  # v_held - recon_actual (V)
    saturated: bool = False

    def to_dict(self) -> dict:
        return {
            "v_input": self.v_input,
            "v_held": self.v_held,
            "v_min_shift": self.v_min_shift,
            "lsb": self.lsb,
            "n_bits": self.n_bits,
            "final_code_int": self.final_code_int,
            "final_code": self.final_code,
            "recon_ideal": self.recon_ideal,
            "recon_actual": self.recon_actual,
            "residual_ideal": self.residual_ideal,
            "residual_actual": self.residual_actual,
            "saturated": self.saturated,
            "cycles": [
                {
                    "cycle": s.cycle,
                    "bit_index": s.bit_index,
                    "bit_weight": s.bit_weight,
                    "trial_code": s.trial_code,
                    "trial_voltage": s.trial_voltage,
                    "comparator_input": s.comparator_input,
                    "decision": s.decision,
                    "register_code": s.register_code,
                    "register_voltage": s.register_voltage,
                    "residual": s.residual,
                }
                for s in self.cycles
            ],
        }


class SARConverter:
    """Cycle-by-cycle SAR ADC around a :class:`BinaryWeightedDAC`.

    Parameters
    ----------
    cfg : ADCConfig
    nonideal : NonidealityConfig
    dac : BinaryWeightedDAC, optional
        Provide to reuse weights (e.g. one DAC across experiments).
    rng : np.random.Generator, optional
    """

    def __init__(
        self,
        cfg: ADCConfig,
        nonideal: NonidealityConfig | None = None,
        dac: BinaryWeightedDAC | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        nonideal = nonideal or NonidealityConfig()
        self.cfg = cfg
        self.nonideal = nonideal
        self.n_bits = cfg.n_bits
        self.n_codes = cfg.n_codes
        self.lsb = cfg.lsb
        if rng is None:
            rng = np.random.default_rng(nonideal.seed)
        self.rng = rng
        self.dac = dac or BinaryWeightedDAC(cfg, nonideal, rng=rng)
        self.comparator = Comparator(cfg, nonideal, rng=rng)
        self.half_code = self.n_codes // 2

    # ------------------------------------------------------------------
    # shared conversion internals
    # ------------------------------------------------------------------
    def _shift(self, v_held: np.ndarray | float) -> np.ndarray | float:
        """Map held analog value into the comparator domain [0, Vref]."""
        if self.cfg.tread == "midtread":
            return np.asarray(v_held, dtype=float) - self.cfg.v_min + 0.5 * self.lsb
        return np.asarray(v_held, dtype=float) - self.cfg.v_min

    def _unshift(self, s: np.ndarray | float) -> np.ndarray | float:
        if self.cfg.tread == "midtread":
            return np.asarray(s, dtype=float) - 0.5 * self.lsb + self.cfg.v_min
        return np.asarray(s, dtype=float) + self.cfg.v_min

    def _to_convention(self, code_int: int | np.ndarray) -> int | np.ndarray:
        if self.cfg.coding == "twos_complement":
            return np.asarray(code_int, dtype=np.int64) - self.half_code
        return code_int

    def decode_ideal(self, code_int: int | np.ndarray) -> float | np.ndarray:
        """Ideal digital decode of an unsigned code (V)."""
        c = np.asarray(code_int, dtype=np.int64)
        rec = c * self.lsb
        if self.cfg.tread == "midrise" and self.cfg.reconstruction == "midpoint":
            rec = rec + 0.5 * self.lsb
        rec = rec + self.cfg.v_min
        if c.ndim == 0:
            return float(rec)
        return rec

    # ------------------------------------------------------------------
    # scalar conversion (with optional per-cycle trace)
    # ------------------------------------------------------------------
    def _convert_scalar(self, v_held: float, trace: bool) -> tuple[int, list[CycleStep]]:
        s = float(self._shift(v_held))
        s = max(0.0, min(self.cfg.v_ref, s))  # clamp into comparison domain
        saturated = s >= self.cfg.v_ref or s <= 0.0

        reg = 0
        steps: list[CycleStep] = []
        for i in range(self.n_bits - 1, -1, -1):
            trial = reg | (1 << i)
            v_trial = float(self.dac.voltage(trial))
            keep = bool(self.comparator.decide(s, v_trial))
            if keep:
                reg = trial
            if trace:
                v_reg = float(self.dac.voltage(reg))
                steps.append(
                    CycleStep(
                        cycle=self.n_bits - 1 - i,
                        bit_index=i,
                        bit_weight=self.lsb * (2 ** i),
                        trial_code=trial,
                        trial_voltage=v_trial,
                        comparator_input=self.comparator.margin(s, v_trial),
                        decision=keep,
                        register_code=reg,
                        register_voltage=v_reg,
                        residual=s - v_reg,
                    )
                )
        return reg, steps

    def convert(self, v_held: float) -> int:
        """Convert one held sample; return the code in the configured convention."""
        reg, _ = self._convert_scalar(v_held, trace=False)
        return int(self._to_convention(reg))

    def convert_int(self, v_held: float) -> int:
        """Convert one held sample; return the unsigned offset-binary code."""
        reg, _ = self._convert_scalar(v_held, trace=False)
        return reg

    def convert_traced(self, v_held: float) -> TraceResult:
        """Convert one held sample and record the full bit-by-bit trace."""
        reg, steps = self._convert_scalar(v_held, trace=True)
        s = float(self._shift(v_held))
        s_clamped = max(0.0, min(self.cfg.v_ref, s))
        recon_ideal = self.decode_ideal(reg)
        recon_actual = float(self.dac.voltage(reg)) + self.cfg.v_min
        return TraceResult(
            v_input=float(v_held),
            v_held=float(v_held),
            v_min_shift=s_clamped,
            lsb=self.lsb,
            n_bits=self.n_bits,
            cycles=steps,
            final_code_int=reg,
            final_code=int(self._to_convention(reg)),
            recon_ideal=recon_ideal,
            recon_actual=recon_actual,
            residual_ideal=float(v_held) - recon_ideal,
            residual_actual=s_clamped - float(self.dac.voltage(reg)),
            saturated=bool(s_clamped >= self.cfg.v_ref or s_clamped <= 0.0),
        )

    # ------------------------------------------------------------------
    # vectorised bulk conversion (identical model, no per-cycle record)
    # ------------------------------------------------------------------
    def convert_bulk(self, v_held: np.ndarray) -> np.ndarray:
        """Convert an array of held samples; returns codes in the configured
        convention. Deterministic decisions when comparator noise is off; the
        result matches :meth:`convert` elementwise."""
        s = np.asarray(self._shift(v_held), dtype=float)
        s = np.clip(s, 0.0, self.cfg.v_ref)

        v_lut = self.dac.voltage_table()
        codes = np.zeros(s.shape, dtype=np.int64)
        for i in range(self.n_bits - 1, -1, -1):
            trial = codes | (1 << i)
            v_trial = v_lut[trial]
            keep = self.comparator.decide(s, v_trial)
            codes = np.where(keep, trial, codes)

        if self.cfg.coding == "twos_complement":
            return codes - self.half_code
        return codes

    def convert_bulk_int(self, v_held: np.ndarray) -> np.ndarray:
        """Bulk conversion returning unsigned offset-binary codes."""
        s = np.asarray(self._shift(v_held), dtype=float)
        s = np.clip(s, 0.0, self.cfg.v_ref)
        v_lut = self.dac.voltage_table()
        codes = np.zeros(s.shape, dtype=np.int64)
        for i in range(self.n_bits - 1, -1, -1):
            trial = codes | (1 << i)
            keep = self.comparator.decide(s, v_lut[trial])
            codes = np.where(keep, trial, codes)
        return codes

    # ------------------------------------------------------------------
    # transfer characteristic from the actual model
    # ------------------------------------------------------------------
    def transfer(self, n_pts: int = 4096) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample the real converter over the input range.

        Returns ``(v_grid, codes, reconstruction)`` where reconstruction is the
        ideal digital decode of the produced code.
        """
        v = np.linspace(self.cfg.v_min, self.cfg.v_max, n_pts)
        codes = self.convert_bulk(v)
        rec = self.decode_ideal(codes) if self.cfg.coding == "offset_binary" else self.decode_ideal(codes + self.half_code)
        return v, codes, rec

    # ------------------------------------------------------------------
    # latency
    # ------------------------------------------------------------------
    def latency(self, t_cycle: float = 1e-9, fs: float | None = None) -> dict:
        """Conversion latency model.

        ``t_cycle`` is the simulated duration of one SAR comparison cycle (s).
        A conversion needs ``n_bits`` cycles (one per bit) plus the acquisition
        window. Returns cycles, wall-clock time, and (when ``fs`` is given) the
        latency expressed in sample periods.
        """
        total_time = self.nonideal.acquisition_time + self.n_bits * t_cycle
        out: dict = {
            "comparison_cycles": self.n_bits,
            "acquisition_time_s": self.nonideal.acquisition_time,
            "t_cycle_s": t_cycle,
            "total_time_s": total_time,
        }
        if fs:
            out["latency_sample_periods"] = total_time * fs
        return out
