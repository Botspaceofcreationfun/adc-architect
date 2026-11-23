"""SAR ADC simulator: ideal quantization vs. explicit nonideal circuit model."""

from .config import ADCConfig, NonidealityConfig
from .dac import BinaryWeightedDAC
from .frontend import SineWave, apply_sampling, coherent_freq, coherent_sine
from .ideal import IdealADC
from .metrics import (
    conversion_latency,
    enob,
    inl_dnl_from_transitions,
    spectral_analysis,
    transition_levels_analytical,
    transition_levels_ramp,
)
from .pipeline import Simulator, SimulationResult
from .sar import Comparator, CycleStep, SARConverter, TraceResult

__version__ = "1.0.0"

__all__ = [
    "ADCConfig",
    "NonidealityConfig",
    "BinaryWeightedDAC",
    "SineWave",
    "apply_sampling",
    "coherent_freq",
    "coherent_sine",
    "IdealADC",
    "SARConverter",
    "Comparator",
    "CycleStep",
    "TraceResult",
    "Simulator",
    "SimulationResult",
    "spectral_analysis",
    "inl_dnl_from_transitions",
    "transition_levels_analytical",
    "transition_levels_ramp",
    "enob",
    "conversion_latency",
]
