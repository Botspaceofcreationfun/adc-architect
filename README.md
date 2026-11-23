# ADC Architect — Successive-Approximation ADC Simulation

An interactive simulation of analog-to-digital conversion with emphasis on a
**successive-approximation register (SAR) ADC**, distinguishing the ideal
quantizer from an explicit model of circuit nonidealities.

The simulator is a behavioral model: every SAR conversion is performed
**bit-by-bit** (set trial bit → generate DAC voltage → compare → keep/clear),
never by a single rounding operation. All stochastic models (comparator noise,
jitter, capacitor mismatch) are deterministic given a seed.

```
                        ┌───────────────┐    v_held     ┌──────────────────────┐
  v(t) ──► S/H ──► (jitter, ──► (acquisition ──► (slew ──► SAR converter ──► code
  (sine   |  acquisition)      settling,        nonlin.    (N cycles,
   wave)  |                     RC model)        opt.)      comparator+dac)
          └────────── ideal reference quantizer (no nonidealities) ──────────► code
```

## Features

* **Ideal ADC** — resolution 4–16 bits, adjustable reference and input range,
  sample-and-hold semantics, offset-binary and two's-complement coding,
  mid-rise / mid-tread transfer styles, transfer characteristics,
  quantization-error histograms.
* **Cycle-by-cycle SAR model** — per-cycle trial bit, trial DAC code and
  voltage, comparator decision, SAR register, and residual error; bulk
  vectorized path for large records with identical results.
* **Selectable nonidealities** — comparator offset and noise, reference-voltage
  error, capacitor mismatch in a binary-weighted capacitive DAC, finite
  acquisition time, sampling jitter, optional slew-rate nonlinearity
  (input-frequency-dependent distortion).
* **Performance metrics** — INL, DNL (endpoint and best-fit), SNDR with
  documented methodology, THD, SFDR, ENOB, conversion latency.
* **Interface** — CLI, matplotlib figures, and a fully self-contained
  **interactive HTML dashboard** with a bit-by-bit SAR trace stepper.
* **Experiments** — resolution sweep, input-frequency sweep, comparator-noise
  sweep, capacitor-mismatch sweep, acquisition-time sweep.

## Installation

Requirements: **Python ≥ 3.10**, `numpy`, `matplotlib`, `pytest` (tests only).

```bash
cd adc-architect
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
# optional: pip install -e .   (installs the `saradc` package)
```

Verified with Python 3.12.11, numpy 1.26.4, matplotlib 3.11.0, pytest 9.2.0.

## Quick start

```bash
# 1. Full demonstration into output/ (plots + metrics + traces + JSON)
python3 -m saradc.cli demo

# 2. Cycle-by-cycle SAR trace for one input
python3 -m saradc.cli sar --bits 8 --vin 0.63

# 3. Metrics with nonidealities (mismatch + comparator noise), plots + JSON
python3 -m saradc.cli metrics --bits 12 --mismatch 0.01 --cmp-noise 0.0003 --plot --json

# 4. The five spec experiments (SNDR vs bits / fin / noise / mismatch / tacq)
python3 -m saradc.cli experiments

# 5. Interactive dashboard (self-contained HTML; open in any browser)
python3 -m saradc.cli dashboard
#    → output/dashboard.html
```

`python3 -m saradc.cli --help` lists all subcommands and options.

### Example CLI workflows

**Ideal vs. nonideal comparison on the same signal:**

```bash
python3 -m saradc.cli metrics --bits 12 --plot --json --out output/ideal
python3 -m saradc.cli metrics --bits 12 --mismatch 0.01 --jitter 5e-8 \
       --tacq 2e-7 --tau 1e-7 --plot --json --out output/nonideal
```

**Signed (two's complement) bipolar conversion:**

```bash
python3 -m saradc.cli sar --bits 8 --vref 2.0 --vmin -1.0 --coding twos_complement --vin -0.63
```

**Mid-tread transfer style:**

```bash
python3 -m saradc.cli metrics --bits 8 --tread midtread --plot
```

**Trace JSON for programmatic use:**

```bash
python3 -m saradc.cli sar --bits 12 --vin 0.53125 --json --out output/trace.json
```

## Tests

```bash
python3 -m pytest tests/ -q
```

64 tests covering the quantizer, the SAR algorithm, the DAC, nonidealities,
metrics, and the pipeline. Key validations include:

* SAR ≡ `floor(v/LSB)` exactly for an ideal DAC (12-bit, 20 000 samples);
* measured SNDR of an ideal N-bit quantizer matches `6.02·N + 1.76 − backoff`
  within tolerance;
* ramp-measured transition levels match the analytical closed form
  `T[k] = V_DAC(k) − v_off`;
* the JS dashboard SAR engine reproduces the Python model exactly
  (1608 cross-checked cases);
* the SAR is provably *not* a single rounding op (mismatched transitions).

## Project layout

```
adc-architect/
├── saradc/               # the simulator package
│   ├── config.py         # ADC + nonideality configuration dataclasses
│   ├── ideal.py          # ideal quantizer (coding, reconstruction, transfer)
│   ├── dac.py            # binary-weighted capacitive DAC + mismatch
│   ├── frontend.py       # test tones + S/H (jitter, acquisition, slew NL)
│   ├── sar.py            # cycle-by-cycle SAR converter + comparator + trace
│   ├── metrics.py        # INL/DNL/SNDR/THD/SFDR/ENOB/latency
│   ├── pipeline.py       # end-to-end Simulator
│   ├── plotting.py       # matplotlib figures
│   ├── dashboard.py      # interactive HTML dashboard generator
│   └── cli.py            # command-line interface
├── tests/                # pytest suite (64 tests)
├── examples/             # example scripts
├── docs/
│   ├── TECHNICAL.md      # architecture, equations, algorithms, limitations
│   └── VERIFICATION.md   # what was verified, how, and what remains
├── output/               # generated plots, JSON metrics, dashboard
├── requirements.txt
└── README.md
```

## Using the package from Python

```python
from saradc import ADCConfig, NonidealityConfig, Simulator

cfg = ADCConfig(n_bits=12, v_ref=1.0)
ni  = NonidealityConfig(cap_mismatch_sigma=0.01, comparator_noise_rms=0.0003, seed=7)

sim = Simulator(cfg, ni, fs=1e6, n=16384, n_cycles=997, amp=0.45)
res = sim.run()
print(res.spectrum["sndr_db"], res.spectrum["enob"])   # e.g. 49.9, 8.0
print(res.metrics)                                     # INL/DNL/latency
print(res.transfer)                                    # (v, codes, recon)

# bit-by-bit trace of a single conversion
tr = sim.sar.convert_traced(0.63)
for step in tr.cycles:
    print(step.cycle, step.trial_code, step.trial_voltage, step.decision, step.residual)
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: numpy` | `pip install -r requirements.txt` |
| Plots blank / `No module named matplotlib` | install matplotlib |
| Tone looks clipped in metrics | keep `amp < v_ref/2` for a centred sine, or set `dc` so the tone stays inside `[v_min, v_max]`; defaults are safe |
| Dashboard does nothing in the browser | open it via a local file path (works offline); no server needed |
| Slow 16-bit runs | `convert_bulk` is vectorised; avoid scalar `convert_traced` in loops |

## Scope and limits

This is a **behavioral / architectural** simulator, not a transistor-level
circuit simulator (no Spice, no charge-injection, no comparator meta-stability,
no thermal noise in the DAC, no parasitic coupling). See
`docs/TECHNICAL.md` for the exact models and their assumptions.
