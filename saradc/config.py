"""Configuration dataclasses for the SAR ADC simulator.

Everything that defines a simulation instance (resolution, input range,
coding convention, and every switchable nonideality) lives here so that
experiments and tests can construct fully-specified, reproducible runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# ADC core configuration
# ---------------------------------------------------------------------------

#: Valid resolutions (bits)
MIN_BITS = 4
MAX_BITS = 16


@dataclass(frozen=True)
class ADCConfig:
    """Ideal-ADC / SAR architecture settings.

    Attributes
    ----------
    n_bits:
        Resolution in bits (4 .. 16).
    v_ref:
        Reference voltage (V). The conversion span is ``v_max - v_min == v_ref``.
    v_min:
        Lower bound of the analog input range (V).
    coding:
        Digital output convention.
        ``"offset_binary"``  : unsigned, code in [0, 2**N - 1].
        ``"twos_complement"``: signed bipolar, code = offset_binary - 2**(N-1).
    tread:
        Quantizer transfer style.
        ``"midrise"`` : transition at multiples of LSB; natural SAR structure.
        ``"midtread"``: transition at half-integer multiples of LSB; true zero code.
    reconstruction:
        Digital-to-analog reconstruction of the ideal code.
        ``"midpoint"`` : ``(code + 1/2)*LSB`` (minimum-error decode).
        ``"code"``     : ``code*LSB`` (threshold decode).
    """

    n_bits: int = 8
    v_ref: float = 1.0
    v_min: float = 0.0
    coding: str = "offset_binary"
    tread: str = "midrise"
    reconstruction: str = "midpoint"

    def __post_init__(self) -> None:
        if not isinstance(self.n_bits, int) or not (MIN_BITS <= self.n_bits <= MAX_BITS):
            raise ValueError(f"n_bits must be an int in [{MIN_BITS}, {MAX_BITS}], got {self.n_bits!r}")
        if not (isinstance(self.v_ref, (int, float)) and self.v_ref > 0):
            raise ValueError(f"v_ref must be > 0, got {self.v_ref!r}")
        if not isinstance(self.v_min, (int, float)):
            raise ValueError(f"v_min must be numeric, got {self.v_min!r}")
        if self.coding not in ("offset_binary", "twos_complement"):
            raise ValueError(f"coding must be 'offset_binary' or 'twos_complement', got {self.coding!r}")
        if self.tread not in ("midrise", "midtread"):
            raise ValueError(f"tread must be 'midrise' or 'midtread', got {self.tread!r}")
        if self.reconstruction not in ("midpoint", "code"):
            raise ValueError(f"reconstruction must be 'midpoint' or 'code', got {self.reconstruction!r}")

    @property
    def v_max(self) -> float:
        return self.v_min + self.v_ref

    @property
    def n_codes(self) -> int:
        return 1 << self.n_bits

    @property
    def lsb(self) -> float:
        return self.v_ref / (1 << self.n_bits)


# ---------------------------------------------------------------------------
# Nonideality configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NonidealityConfig:
    """Switchable circuit nonidealities.

    Each field defaults to a "perfect circuit" value, so ``NonidealityConfig()``
    reproduces the ideal ADC exactly. Effects are applied independently so the
    ideal and nonideal cases can be compared one factor at a time.

    Attributes
    ----------
    comparator_offset:
        DC offset (V) added to the comparator input before the decision.
    comparator_noise_rms:
        RMS Gaussian noise (V) added to the comparator input each decision.
    ref_error:
        Relative reference-voltage error (fraction, e.g. 0.01 = +1 %).
        ``Vref_actual = v_ref * (1 + ref_error)``.
    cap_mismatch_sigma:
        Relative std.dev. of the binary-weighted capacitor unit values
        (fraction, e.g. 0.01 = 1 %). 0 disables mismatch.
    acquisition_time:
        Sample-and-hold acquisition window (s). The track switch closes this
        long before the sampling instant; finite time causes incomplete settling.
    switch_tau:
        RC time constant of the acquisition switch/capacitor (s). 0 = ideal
        instantaneous track (no acquisition error regardless of time).
    sampling_jitter_rms:
        RMS sampling-clock jitter (s). 0 disables jitter.
    slew_nl:
        Simplified input-frequency-dependent distortion coefficient of the
        sample-and-hold: an error ``slew_nl * (dv/dt)**2`` is added to the held
        value (even-harmonic distortion). 0 disables.
    seed:
        Random seed for all stochastic sources (reproducibility).
    """

    comparator_offset: float = 0.0
    comparator_noise_rms: float = 0.0
    ref_error: float = 0.0
    cap_mismatch_sigma: float = 0.0
    acquisition_time: float = 1e-6
    switch_tau: float = 0.0
    sampling_jitter_rms: float = 0.0
    slew_nl: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("comparator_offset", self.comparator_offset),
            ("comparator_noise_rms", self.comparator_noise_rms),
            ("ref_error", self.ref_error),
            ("cap_mismatch_sigma", self.cap_mismatch_sigma),
            ("acquisition_time", self.acquisition_time),
            ("switch_tau", self.switch_tau),
            ("sampling_jitter_rms", self.sampling_jitter_rms),
            ("slew_nl", self.slew_nl),
        ):
            if not isinstance(value, (int, float)) or value != value:  # NaN guard
                raise ValueError(f"{name} must be a real number, got {value!r}")
        if self.comparator_noise_rms < 0 or self.cap_mismatch_sigma < 0 or self.sampling_jitter_rms < 0:
            raise ValueError("noise/mismatch/jitter RMS values must be >= 0")
        if self.switch_tau < 0 or self.acquisition_time < 0:
            raise ValueError("acquisition_time and switch_tau must be >= 0")
        if not isinstance(self.seed, int):
            raise ValueError("seed must be an int")

    @property
    def is_ideal(self) -> bool:
        return (
            self.comparator_offset == 0.0
            and self.comparator_noise_rms == 0.0
            and self.ref_error == 0.0
            and self.cap_mismatch_sigma == 0.0
            and self.switch_tau == 0.0
            and self.sampling_jitter_rms == 0.0
            and self.slew_nl == 0.0
        )
