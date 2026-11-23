"""Binary-weighted capacitive DAC model.

The DAC is a simplified model of a charge-redistribution capacitive array: N
binary-weighted capacitors (weights ``2**i``) plus one ideal termination
capacitor of unit value. The output voltage of code ``c`` is

.. math::

    V_{DAC}(c) = V_{ref,actual} * ( sum_i b_i(c) * w_i ) / ( w_sum + 1 )

where ``b_i(c)`` is the i-th bit of ``c`` and ``w_i = 2**i * (1 + e_i)``.
Mismatch sets ``e_i ~ N(0, cap_mismatch_sigma)`` (relative to the unit value).
The ``+1`` termination capacitor makes the ideal full-scale ``(2**N - 1)*LSB``
exactly, matching the ideal-quantizer LSB definition ``LSB = Vref/2**N``.

Because the denominator is the *actual* total capacitance, the transfer is
normalised to ``[0, Vref_actual]``; the residual gain error with respect to
the ideal ``2**N`` denominator is included in the reported gain error / INL.
"""

from __future__ import annotations

import numpy as np

from .config import ADCConfig, NonidealityConfig


class BinaryWeightedDAC:
    """Binary-weighted capacitive DAC with optional unit-capacitor mismatch.

    Parameters
    ----------
    cfg : ADCConfig
        Architecture settings (resolution, reference).
    nonideal : NonidealityConfig
        ``cap_mismatch_sigma`` enables mismatch; ``ref_error`` scales the
        reference voltage seen by the DAC.
    rng : np.random.Generator, optional
        RNG for the mismatch draw. A fresh Generator seeded from
        ``nonideal.seed`` is created if none is given.
    """

    def __init__(
        self,
        cfg: ADCConfig,
        nonideal: NonidealityConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        nonideal = nonideal or NonidealityConfig()
        self.cfg = cfg
        self.n_bits = cfg.n_bits
        self.n_codes = cfg.n_codes
        self.nonideal = nonideal

        if rng is None:
            rng = np.random.default_rng(nonideal.seed)
        self.rng = rng

        ideal_weights = 2.0 ** np.arange(self.n_bits, dtype=float)
        if nonideal.cap_mismatch_sigma > 0.0:
            errors = rng.normal(0.0, nonideal.cap_mismatch_sigma, size=self.n_bits)
            self.weights = ideal_weights * (1.0 + errors)
        else:
            self.weights = ideal_weights.copy()

        # Ideal termination capacitor of one unit keeps ideal FS == (2**N - 1)*LSB.
        self.termination_cap = 1.0
        self.w_sum = float(self.weights.sum())

        self.ref_actual = cfg.v_ref * (1.0 + nonideal.ref_error)
        self.lsb_actual = self.ref_actual / self.n_codes

        # Lookup table: DAC voltage for every code. Cheap (2**N floats) and
        # turns bulk SAR conversion into vectorised gathers.
        self.v_lut = np.empty(self.n_codes, dtype=float)
        code_weights = self._code_weights()
        self.v_lut[:] = (code_weights @ self.weights) / (self.w_sum + self.termination_cap) * self.ref_actual

    # ------------------------------------------------------------------
    def _code_weights(self) -> np.ndarray:
        """``(2**N, N)`` matrix: binary digits of every code (LSB first)."""
        return ((np.arange(self.n_codes)[:, None] >> np.arange(self.n_bits)[None, :]) & 1).astype(np.float64)

    def fraction(self, code: int | np.ndarray) -> float | np.ndarray:
        """Normalised output ``sum(b*w)/(w_sum + 1)`` in [0, 1)."""
        c = np.asarray(code)
        if c.ndim == 0:
            bits = np.array([(int(c) >> i) & 1 for i in range(self.n_bits)], dtype=float)
            return float(bits @ self.weights / (self.w_sum + self.termination_cap))
        bits = ((c[..., None] >> np.arange(self.n_bits)[None, :]) & 1).astype(np.float64)
        return bits @ self.weights / (self.w_sum + self.termination_cap)

    def voltage(self, code: int | np.ndarray) -> float | np.ndarray:
        """DAC output voltage for the given code."""
        c = np.asarray(code)
        if c.ndim == 0:
            return float(self.v_lut[int(c)])
        return self.v_lut[c]

    def voltage_table(self) -> np.ndarray:
        """DAC output voltage for every code ``0 .. 2**N - 1``."""
        return self.v_lut.copy()

    # ------------------------------------------------------------------
    @property
    def gain_error(self) -> float:
        """Relative gain error of the DAC vs the ideal binary array (fraction).

        ``w_sum + 1`` would equal ``2**N`` for ideal weights; any deviation is
        a gain error, and ``ref_error`` adds a further (independent) gain error.
        """
        ideal_sum = float(2 ** self.n_bits)
        return (self.w_sum + self.termination_cap) / ideal_sum - 1.0

    def summary(self) -> dict:
        return {
            "weights": self.weights.tolist(),
            "w_sum": self.w_sum,
            "ref_actual": self.ref_actual,
            "lsb_actual": self.lsb_actual,
            "gain_error": self.gain_error,
        }
