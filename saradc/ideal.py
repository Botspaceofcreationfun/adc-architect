"""Ideal ADC: pure quantization without any circuit nonideality.

This module defines what the *ideal* converter does -- the theoretical
reference against which the nonideal SAR model is compared. It supports
configurable resolution, input range, signed/unsigned coding and
mid-rise / mid-tread transfer styles, sample-and-hold semantics (instantaneous
in the ideal case), and reconstruction.
"""

from __future__ import annotations

import numpy as np

from .config import ADCConfig


class IdealADC:
    """Instantaneous ideal quantizer with configurable coding conventions.

    The quantizer maps analog input ``v`` to a digital code and back. Input
    outside ``[v_min, v_max]`` saturates (clips). All arithmetic is exact up
    to floating-point rounding.
    """

    def __init__(self, cfg: ADCConfig) -> None:
        self.cfg = cfg
        self.lsb = cfg.lsb
        self.n_bits = cfg.n_bits
        self.n_codes = cfg.n_codes

    # ------------------------------------------------------------------
    # encoding / decoding
    # ------------------------------------------------------------------
    def code_int(self, v: float | np.ndarray) -> int | np.ndarray:
        """Raw unsigned integer code (offset binary, 0 .. 2**N - 1)."""
        arr = np.asarray(v, dtype=float)
        shifted = arr - self.cfg.v_min
        if self.cfg.tread == "midtread":
            shifted = shifted + 0.5 * self.lsb
        c = np.floor(shifted / self.lsb).astype(np.int64)
        c = np.clip(c, 0, self.n_codes - 1)
        if arr.ndim == 0:
            return int(c)
        return c

    def code(self, v: float | np.ndarray) -> int | np.ndarray:
        """Digital code in the configured convention."""
        c = self.code_int(v)
        if self.cfg.coding == "twos_complement":
            half = self.n_codes // 2
            if isinstance(c, int):
                return c - half
            return c - half
        return c

    def reconstruct_int(self, code_int: int | np.ndarray) -> float | np.ndarray:
        """Analog reconstruction of an unsigned code (in volts)."""
        c = np.asarray(code_int, dtype=np.int64)
        rec = c * self.lsb
        if self.cfg.tread == "midrise" and self.cfg.reconstruction == "midpoint":
            rec = rec + 0.5 * self.lsb
        rec = rec + self.cfg.v_min
        if c.ndim == 0:
            return float(rec)
        return rec

    def reconstruct(self, code: int | np.ndarray) -> float | np.ndarray:
        """Analog reconstruction of a code given in the configured convention."""
        c = np.asarray(code, dtype=np.int64)
        if self.cfg.coding == "twos_complement":
            c = c + self.n_codes // 2
        return self.reconstruct_int(c)

    def quant_error(self, v: float | np.ndarray) -> float | np.ndarray:
        """Quantization error ``v - reconstruct(code(v))`` (V)."""
        return np.asarray(v, dtype=float) - np.asarray(self.reconstruct(self.code(v)), dtype=float)

    # ------------------------------------------------------------------
    # transfer characteristic
    # ------------------------------------------------------------------
    def transfer(self, n_pts: int = 4096) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(v_grid, codes, reconstruction)`` for the full input range."""
        v = np.linspace(self.cfg.v_min, self.cfg.v_max, n_pts)
        codes = self.code(v)
        rec = self.reconstruct(codes)
        return v, codes, rec

    # ------------------------------------------------------------------
    # sample-and-hold + quantize (ideal: instantaneous, no jitter/acquisition)
    # ------------------------------------------------------------------
    def convert_samples(self, v: float | np.ndarray) -> int | np.ndarray:
        """Quantize raw analog samples (ideal S/H implied)."""
        return self.code(v)
