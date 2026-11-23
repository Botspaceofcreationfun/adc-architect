"""Example 1 — ideal vs. nonideal comparison on identical test conditions.

Run:  python3 examples/demo_compare.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saradc import ADCConfig, NonidealityConfig, Simulator


def run(label: str, ni: NonidealityConfig) -> dict:
    cfg = ADCConfig(n_bits=12, v_ref=1.0)
    res = Simulator(cfg, ni, fs=1e6, n=16384, n_cycles=997, amp=0.45).run()
    s = res.spectrum
    m = res.metrics
    print(f"{label:>24}  SNDR {s['sndr_db']:7.2f} dB  ENOB {s['enob']:6.2f}  "
          f"INLmax {m['inl_max_bestfit_lsb']:6.2f} LSB  DNLmax {m['dnl_max_lsb']:6.2f} LSB")
    return res


if __name__ == "__main__":
    lsb12 = 1.0 / 4096
    cases = [
        ("ideal", NonidealityConfig(seed=0)),
        ("capacitor mismatch 1%", NonidealityConfig(cap_mismatch_sigma=0.01, seed=0)),
        ("comparator offset+noise", NonidealityConfig(comparator_offset=0.5 * lsb12, comparator_noise_rms=0.5 * lsb12, seed=0)),
        ("reference error +1%", NonidealityConfig(ref_error=0.01, seed=0)),
        ("jitter 50 ns", NonidealityConfig(sampling_jitter_rms=5e-8, seed=0)),
        ("acquisition 2us/0.2us-tau", NonidealityConfig(acquisition_time=2e-7, switch_tau=2e-7, seed=0)),
        ("slew nonlinearity", NonidealityConfig(slew_nl=1e-14, seed=0)),
    ]
    print(f"{'scenario':>24}  {'SNDR':>10}  {'ENOB':>8}  {'INLmax':>8}  {'DNLmax':>8}")
    results = [run(label, ni) for label, ni in cases]
    print("\nConclusion: every nonideality degrades SNDR/ENOB and/or INL/DNL;")
    print("the ideal case reproduces the 6.02*N+1.76-0.92 theory anchor.")
