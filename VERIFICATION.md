# Verification Report

Date: 2026-10-27 (UTC+0) · Environment: Linux, Python 3.12.11, numpy 1.26.4,
matplotlib 3.11.0, pytest 9.2.0, node v22.23.2 (dashboard JS check), Chromium
(headless dashboard render).

## 1. Implemented features (spec §1–§5)

| Spec item | Status | Where |
|---|---|---|
| Resolution 4–16 bits | ✅ | `ADCConfig.n_bits` (validated) |
| Adjustable input range & reference | ✅ | `v_min`, `v_ref`, bipolar via `v_min=−Vref/2` |
| Sampling & sample-and-hold | ✅ | `frontend.apply_sampling` (ideal: instantaneous) |
| Quantization & reconstruction | ✅ | `ideal.py`, `sar.decode_ideal` (midpoint/code) |
| Signed & unsigned coding | ✅ | offset_binary, twos_complement; mid-rise/mid-tread |
| Transfer-characteristic plots | ✅ | `plot_transfer`, dashboard panel |
| Quantization-error histograms | ✅ | `plot_quant_error_hist`, dashboard panel |
| Resolution ↔ step-size demo | ✅ | `cli ideal` table, experiment 1 |
| Cycle-by-cycle SAR model | ✅ | `sar.convert_traced` / `convert_bulk` |
| Exposed register/trial/cmp/residual per cycle | ✅ | `CycleStep`, trace table, dashboard stepper |
| SAR not a single rounding op | ✅ | test `test_sar_not_single_rounding` |
| Comparator offset & noise | ✅ | `Comparator` |
| Reference-voltage error | ✅ | `ref_error` → gain error |
| Capacitor mismatch (binary-weighted DAC) | ✅ | `BinaryWeightedDAC`, σ_u |
| Finite acquisition time | ✅ | RC settling model |
| Sampling jitter | ✅ | aperture view |
| Input-frequency-dependent distortion | ✅ | `slew_nl` behavioral term (+ frequency-dependent acquisition) |
| INL / DNL | ✅ | analytical + ramp, endpoint + best-fit |
| SNDR with documented methodology | ✅ | `spectral_analysis`, TECHNICAL.md §7.2 |
| ENOB from measured SNDR | ✅ | `(SNDR−1.76)/6.02` |
| Conversion latency | ✅ | `latency()` / `conversion_latency()` |
| Controlled test signals | ✅ | coherent sine generator |
| Interactive trace / transfer / error / spectral panel | ✅ | dashboard (verified headless) |
| Five spec experiments | ✅ | `cli experiments` |

## 2. Automated tests

Command: `python3 -m pytest tests/ -q` → **64 passed, 0 failed** (0.4 s).

Notable numeric validations (all passing):

* **SAR ≡ floor(v/LSB)** exactly for an ideal 12-bit DAC over 20 000 random
  inputs (`np.testing.assert_array_equal`).
* **SNDR vs. theory** for ideal N-bit quantizers (coherent sine, −0.92 dBFS
  backoff, i.e. `amp = 0.45 V`, `Vref = 1 V`):

  | N | measured SNDR (dB) | theory 6.02N+1.76−0.92 |
  |---|---|---|
  | 6 | 37.18 | 36.97 |
  | 8 | 48.90 | 49.00 |
  | 10 | 61.10 | 61.05 |
  | 12 | 73.08 | 73.08 |
  (16 384-point records; the pytest suite repeats this at 32 768 points with
  ±0.5–1.0 dB tolerances and passes.)

* **Transition-level cross-check**: ramp-measured `T[k]` equals the closed
  form `V_DAC(k) − v_off` within 1.5 LSB (8-bit, mismatch + offset).
* **Comparator offset**: `code(v)` with offset `o` ≡ `code(v + o)` without,
  elementwise over 5000 samples.
* **Σ DNL = endpoint gain error** to 1e−6.
* **Best-fit INL ≤ endpoint INL** (mismatch DAC).
* **JS dashboard SAR engine ≡ Python model**: 1608 (bits × scenario × input)
  cases, 0 mismatches.
* **Reproducibility**: identical seed → identical codes and SNDR.
* **Nonideal trends**: jitter lowers SNDR; mismatch raises INL/DNL
  monotonically with σ; comparator noise lowers SNDR > 6 dB at 2 LSB;
  acquisition error grows with frequency and shrinks with `T_acq/τ`;
  `slew_nl` puts measurable power at `2·fin`.

## 3. Manual/end-to-end checks performed

* `cli sar` trace for `v=0.63` (8-bit): correct binary search (128 keep →
  192 clear → 160 keep → … → code 161 = 0.6289 V + 0.22 LSB residual).
* `cli ideal --plot`: quantization error rms = `LSB/√12` exactly; max error =
  `LSB/2` exactly.
* `cli metrics` (12-bit, mismatch 1%, cmp noise 0.0003 V): SNDR 49.7 dB,
  ENOB 8.0, INLmax(bf) 9.28 LSB, DNLmax 13.4 LSB, gain 0.14% — physically
  consistent for an uncalibrated binary-weighted DAC.
* `cli experiments`: all five sweeps monotonic and theory-anchored
  (exp 1: 4→16 bit SNDR 24.6→95.0 dB following 6.02N+1.76−0.92; exp 2:
  26.5 dB @ 40% fs; exp 3: −15 dB @ 2 LSB noise; exp 4: INL 0→47 LSB;
  exp 5: −5 dB at `T_acq/τ=0.25`).
* Dashboard: generated (0.63 MB, self-contained), **headless Chromium render
  with zero JS errors**, controls exercised (scenario/bits/input/step), SAR
  stepper advances cycle-by-cycle; JS syntax checked with node --check.
* 16-bit conversion of 16 384 samples runs in ≪ 1 s (bulk path).

## 4. Known issues / limitations (accepted)

1. Behavioral model, not transistor-level (see TECHNICAL.md §9) — by design
   of the spec ("Do not equate a conceptual SAR simulator with a
   transistor-level circuit simulator").
2. Single-tone acquisition effect shows mostly as amplitude/phase (memory)
   error; strong harmonic distortion from acquisition needs multi-tone or the
   `slew_nl` term. Honest, documented behavior.
3. Closed-form INL assumes monotone DAC weights (valid for the σ ranges
   exercised); the ramp method does not.
4. 4-bit SNDR sits ~0.3 dB below the 6.02N+1.76 anchor (quantization
   distortion is correlated at very low resolution); tolerance covers it.
5. Dashboard INL/DNL/spectrum are precomputed at generation time; live
   parameter changes recompute the SAR trace in JS but not the bulk metrics
   (re-generate the dashboard to refresh).
6. No pipelined-SAR latency modelling (single-conversion latency only).

## 5. Not implemented (out of scope / deferred)

* Transistor-level models (Spice-like): charge injection, meta-stability,
  DAC thermal noise, layout/parasitics.
* Differential/fully-differential ADC architecture.
* Statistical binning / process corners (runs are Monte-Carlo style via
  seeds; a multi-run wrapper is trivial to add).
* Support for arbitrary user waveform files (analytic waves only; API
  extension point exists).

## 6. Next improvements (priority order)

1. Monte-Carlo harness: sweep seeds per nonideality and report
   mean/p99 INL, DNL, ENOB (currently single-seed deterministic runs).
2. IEEE-1241 sine-fit and histogram-based INL/DNL estimators as independent
   third measurement paths (beyond the ramp and closed-form).
3. Differential DAC + common-mode/reference-loading model, and a
   two-tone IMD experiment.
