"""Example 2 — step through a single conversion cycle by cycle and print the
SAR register, trial DAC code, comparator decision and residual at each cycle.

Run:  python3 examples/demo_sar_trace.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saradc import ADCConfig, NonidealityConfig, SARConverter

cfg = ADCConfig(n_bits=8, v_ref=1.0)
ni = NonidealityConfig(cap_mismatch_sigma=0.005, comparator_offset=0.2, seed=3)
sar = SARConverter(cfg, ni, rng=__import__("numpy").random.default_rng(3))

vin = 0.53125
tr = sar.convert_traced(vin)

print(f"input {vin} V -> held {tr.v_held:.6g} V, {cfg.n_bits}-bit SAR, "
      f"mismatch sigma={ni.cap_mismatch_sigma}, cmp offset={ni.comparator_offset} V")
print(f"{'cyc':>3} {'bit':>3} {'trial':>7} {'trial DAC(V)':>12} {'cmp in(V)':>10} "
      f"{'dec':>5} {'reg':>7} {'reg DAC(V)':>10} {'resid(LSB)':>10}")
for c in tr.cycles:
    print(f"{c.cycle:>3} {c.bit_index:>3} {c.trial_code:>7} {c.trial_voltage:>12.6f} "
          f"{c.comparator_input:>10.4f} {'keep' if c.decision else 'clear':>5} "
          f"{c.register_code:>7} {c.register_voltage:>10.6f} {c.residual / cfg.lsb:>10.3f}")

print(f"final code {tr.final_code_int} (ideal recon {tr.recon_ideal:.6g} V, "
      f"residual {tr.residual_ideal / cfg.lsb:.3f} LSB)")
print("Note the binary search: 128 keep, 192 clear, 160 keep, ... — the code is")
print("assembled by 8 real comparator decisions, not by rounding the input.")
