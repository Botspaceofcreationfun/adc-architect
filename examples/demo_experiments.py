"""Example 3 — the five spec experiments in one script (SNDR/ENOB/INL/DNL).

Run:  python3 examples/demo_experiments.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saradc import ADCConfig, NonidealityConfig, Simulator

fs, n, amp = 1e6, 16384, 0.45

print("=== 1. increasing the number of bits (ideal) ===")
for b in (4, 6, 8, 10, 12, 14, 16):
    res = Simulator(ADCConfig(n_bits=b), NonidealityConfig(seed=0), fs=fs, n=n, n_cycles=997, amp=amp).run()
    print(f"  {b:>3} bit -> SNDR {res.spectrum['sndr_db']:6.2f} dB, ENOB {res.spectrum['enob']:5.2f}")

print("=== 2. increasing input frequency (12-bit, jitter+acquisition) ===")
ni_fe = NonidealityConfig(sampling_jitter_rms=5e-8, acquisition_time=2e-7, switch_tau=1e-7, seed=0)
for frac in (0.03, 0.12, 0.3):
    cycles = max(3, int(frac * 0.4 * n))
    res = Simulator(ADCConfig(n_bits=12), ni_fe, fs=fs, n=n, n_cycles=cycles, amp=amp).run()
    print(f"  fin={res.f_sig/1e3:7.2f} kHz ({frac*100:3.0f}% fs) -> SNDR {res.spectrum['sndr_db']:6.2f} dB")

print("=== 3. adding comparator noise (12-bit) ===")
for nl in (0.0, 0.25, 0.5, 1.0, 2.0):
    res = Simulator(ADCConfig(n_bits=12), NonidealityConfig(comparator_noise_rms=nl / 4096, seed=0),
                    fs=fs, n=n, n_cycles=997, amp=amp).run()
    print(f"  noise={nl:4.2f} LSB -> SNDR {res.spectrum['sndr_db']:6.2f} dB, ENOB {res.spectrum['enob']:5.2f}")

print("=== 4. introducing capacitor mismatch (12-bit) ===")
for sg in (0.0, 0.005, 0.01, 0.02):
    res = Simulator(ADCConfig(n_bits=12), NonidealityConfig(cap_mismatch_sigma=sg, seed=0),
                    fs=fs, n=n, n_cycles=997, amp=amp).run()
    print(f"  sigma={sg:.3f} -> INLmax {res.metrics['inl_max_bestfit_lsb']:6.2f} LSB, "
          f"DNLmax {res.metrics['dnl_max_lsb']:6.2f} LSB, SNDR {res.spectrum['sndr_db']:6.2f} dB")

print("=== 5. reducing acquisition time (12-bit, tau=0.2us) ===")
for ratio in (10.0, 2.0, 1.0, 0.5, 0.25):
    tau = 2e-7
    res = Simulator(ADCConfig(n_bits=12), NonidealityConfig(acquisition_time=ratio * tau, switch_tau=tau, seed=0),
                    fs=fs, n=n, n_cycles=997, amp=amp).run()
    print(f"  tacq/tau={ratio:4.2f} -> SNDR {res.spectrum['sndr_db']:6.2f} dB")
