"""Command-line interface for the SAR ADC simulator.

Subcommands
-----------
demo            Run the full demonstration (plots + metrics + trace) into output/.
sar             Single cycle-by-cycle SAR conversion trace.
ideal           Ideal-quantizer statistics and transfer plot.
metrics         Full metrics run (SNDR/ENOB/INL/DNL) with optional nonidealities.
experiments     The five experiments from the spec (bits, frequency, comparator
                noise, capacitor mismatch, acquisition time).
dashboard       Generate the self-contained interactive HTML dashboard.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import ADCConfig, NonidealityConfig
from .ideal import IdealADC
from .pipeline import Simulator
from .sar import SARConverter

DEFAULT_OUT = Path("output")


def _add_cfg_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--bits", type=int, default=8, help="resolution in bits (4..16)")
    p.add_argument("--vref", type=float, default=1.0, help="reference voltage (V)")
    p.add_argument("--vmin", type=float, default=0.0, help="input range lower bound (V)")
    p.add_argument("--coding", choices=["offset_binary", "twos_complement"], default="offset_binary")
    p.add_argument("--tread", choices=["midrise", "midtread"], default="midrise")
    p.add_argument("--recon", choices=["midpoint", "code"], default="midpoint")


def _add_nonideal_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--cmp-offset", type=float, default=0.0, help="comparator offset (V)")
    p.add_argument("--cmp-noise", type=float, default=0.0, help="comparator noise RMS (V)")
    p.add_argument("--ref-error", type=float, default=0.0, help="relative reference error (fraction)")
    p.add_argument("--mismatch", type=float, default=0.0, help="capacitor mismatch sigma (fraction)")
    p.add_argument("--tacq", type=float, default=1e-6, help="acquisition time (s)")
    p.add_argument("--tau", type=float, default=0.0, help="switch RC tau (s)")
    p.add_argument("--jitter", type=float, default=0.0, help="sampling jitter RMS (s)")
    p.add_argument("--slew-nl", type=float, default=0.0, help="S/H slew-rate nonlinearity coefficient")
    p.add_argument("--seed", type=int, default=0, help="random seed")


def _add_signal_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--fs", type=float, default=1e6, help="sampling frequency (Hz)")
    p.add_argument("--n", type=int, default=16384, help="record length (samples)")
    p.add_argument("--cycles", type=int, default=997, help="integer cycles of test tone (coherent)")
    p.add_argument("--amp", type=float, default=0.45, help="tone amplitude (V peak)")


def _make_cfg(args) -> ADCConfig:
    return ADCConfig(n_bits=args.bits, v_ref=args.vref, v_min=args.vmin,
                     coding=args.coding, tread=args.tread, reconstruction=args.recon)


def _make_ni(args) -> NonidealityConfig:
    return NonidealityConfig(
        comparator_offset=args.cmp_offset,
        comparator_noise_rms=args.cmp_noise,
        ref_error=args.ref_error,
        cap_mismatch_sigma=args.mismatch,
        acquisition_time=args.tacq,
        switch_tau=args.tau,
        sampling_jitter_rms=args.jitter,
        slew_nl=args.slew_nl,
        seed=args.seed,
    )


# ---------------------------------------------------------------------------
# sar: single conversion trace
# ---------------------------------------------------------------------------

def cmd_sar(args) -> int:
    cfg = _make_cfg(args)
    ni = _make_ni(args)
    sar = SARConverter(cfg, ni, rng=np.random.default_rng(args.seed))
    vin = args.vin

    if not (cfg.v_min <= vin <= cfg.v_max):
        print(f"note: input {vin:.6g} V lies outside the range [{cfg.v_min:.6g}, {cfg.v_max:.6g}] -- will be clamped", file=sys.stderr)

    tr = sar.convert_traced(vin)

    print(f"SAR conversion trace -- {cfg.n_bits} bit, Vref={cfg.v_ref:.6g} V, LSB={cfg.lsb:.6g} V")
    print(f"input voltage      : {vin:.10g} V")
    print(f"held sample        : {tr.v_held:.10g} V")
    print(f"coding             : {cfg.coding} ({cfg.tread} / reconstruction={cfg.reconstruction})")
    print()
    hdr = f"{'cyc':>3} {'bit':>3} {'weight(V)':>10} {'trial':>7} {'trial DAC(V)':>12} {'cmp in(V)':>10} {'dec':>5} {'reg':>7} {'reg DAC(V)':>10} {'resid(V)':>10}"
    print(hdr)
    print("-" * len(hdr))
    for c in tr.cycles:
        print(
            f"{c.cycle:>3} {c.bit_index:>3} {c.bit_weight:>10.4g} {c.trial_code:>7} {c.trial_voltage:>12.8g} "
            f"{c.comparator_input:>10.6g} {'keep' if c.decision else 'clear':>5} {c.register_code:>7} "
            f"{c.register_voltage:>10.8g} {c.residual:>10.6g}"
        )
    print("-" * len(hdr))
    print(f"final code (int)   : {tr.final_code_int}")
    print(f"final code ({cfg.coding}) : {tr.final_code}")
    print(f"reconstruction     : ideal {tr.recon_ideal:.10g} V | actual DAC {tr.recon_actual:.10g} V")
    print(f"residual (ideal)   : {tr.residual_ideal:.6g} V = {tr.residual_ideal / cfg.lsb:.4f} LSB")
    print(f"saturated          : {tr.saturated}")

    if args.json:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(tr.to_dict(), indent=2))
        print(f"\ntrace JSON written to {out}")
    return 0


# ---------------------------------------------------------------------------
# ideal: ideal quantizer statistics + plot
# ---------------------------------------------------------------------------

def cmd_ideal(args) -> int:
    cfg = _make_cfg(args)
    adc = IdealADC(cfg)
    out = Path(args.out)

    v = np.linspace(cfg.v_min + 1e-9, cfg.v_max - 1e-9, 200000)
    e = np.asarray(adc.quant_error(v), dtype=float)
    print(f"Ideal ADC -- {cfg.n_bits} bit, range [{cfg.v_min}, {cfg.v_max}] V, LSB={cfg.lsb:.6g} V, "
          f"coding={cfg.coding}, tread={cfg.tread}, recon={cfg.reconstruction}")
    print(f"quantization error: mean {np.mean(e):.4g} V, rms {np.std(e):.4g} V (theory {cfg.lsb / np.sqrt(12):.4g} V)")
    print(f"max |error|: {np.max(np.abs(e)):.4g} V (theory LSB/2 = {cfg.lsb / 2:.4g} V)")

    # resolution vs step size demonstration
    print("\nresolution vs quantization step:")
    print(f"{'bits':>5} {'LSB (V)':>12} {'#codes':>8}")
    for b in (4, 8, 12, 16):
        print(f"{b:>5} {cfg.v_ref / (1 << b):>12.6g} {1 << b:>8}")

    if args.plot:
        from .plotting import plot_transfer, plot_quant_error_hist
        res = Simulator(cfg, NonidealityConfig(seed=args.seed), fs=args.fs, n=args.n, n_cycles=args.cycles, amp=args.amp).run()
        out.mkdir(parents=True, exist_ok=True)
        p1 = plot_transfer(res, out / "ideal_transfer.png")
        p2 = plot_quant_error_hist(res, out / "ideal_qerr_hist.png")
        print(f"\nplots written: {p1}, {p2}")
    return 0


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def cmd_metrics(args) -> int:
    cfg = _make_cfg(args)
    ni = _make_ni(args)
    out = Path(args.out)
    sim = Simulator(cfg, ni, fs=args.fs, n=args.n, n_cycles=args.cycles, amp=args.amp)
    res = sim.run()
    spec, met = res.spectrum, res.metrics

    print(f"Metrics -- {cfg.n_bits}-bit SAR, fs={args.fs/1e6:.3g} MHz, fin={res.f_sig/1e3:.3g} kHz, amp={args.amp:.3g} V")
    print(f"nonidealities: {ni}")
    print(f"  SNDR : {spec['sndr_db']:8.2f} dB")
    print(f"  SNR  : {spec['snr_db']:8.2f} dB")
    print(f"  THD  : {spec['thd_db']:8.2f} dB")
    print(f"  SFDR : {spec['sfdr_db']:8.2f} dB")
    print(f"  ENOB : {spec['enob']:8.3f} bit   (=(SNDR-1.76)/6.02)")
    print(f"  INL max (endpoint): {met['inl_max_endpoint_lsb']:8.3f} LSB")
    print(f"  INL max (bestfit) : {met['inl_max_bestfit_lsb']:8.3f} LSB")
    print(f"  DNL max           : {met['dnl_max_lsb']:8.3f} LSB")
    print(f"  gain error        : {met['gain_error_frac']*100:8.3f} %")
    lat = met["latency"]
    print(f"  latency           : {lat['comparison_cycles']} cycles, {lat['total_time_s']:.3g} s, "
          f"{lat['latency_sample_periods']:.3g} sample periods @ fs")

    if args.plot or args.json:
        out.mkdir(parents=True, exist_ok=True)
        from .plotting import plot_inl_dnl, plot_quant_error_hist, plot_spectrum, plot_transfer
        files = [
            plot_transfer(res, out / "metrics_transfer.png"),
            plot_quant_error_hist(res, out / "metrics_qerr_hist.png"),
            plot_inl_dnl(res, out / "metrics_inl_dnl.png"),
            plot_spectrum(res, out / "metrics_spectrum.png"),
        ]
        print("plots written:")
        for f in files:
            print(f"  {f}")
    if args.json:
        jp = out / "metrics.json"
        jp.write_text(json.dumps(res.to_dict(), indent=2, default=float))
        print(f"metrics JSON written to {jp}")
    return 0


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------

EXPERIMENT_LABELS = {
    "bits": "increasing the number of bits",
    "freq": "increasing input frequency",
    "noise": "adding comparator noise",
    "mismatch": "introducing capacitor mismatch",
    "acq": "reducing acquisition time",
}


def _row(items):
    return "  " + " | ".join(f"{k:>6}={v}" for k, v in items)


def cmd_experiments(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    ax_bits, ax_freq, ax_noise, ax_mis, ax_acq, ax_sum = axes.ravel()

    fs, n = 1e6, 16384
    amp = 0.45

    # ---- 1. resolution sweep (ideal) -------------------------------------
    bits_list = [4, 6, 8, 10, 12, 14, 16]
    sndr_bits, enob_bits = [], []
    print("Experiment 1: resolution vs SNDR (ideal quantizer)")
    for b in bits_list:
        cfg = ADCConfig(n_bits=b)
        res = Simulator(cfg, NonidealityConfig(seed=args.seed), fs=fs, n=n, n_cycles=997, amp=amp).run()
        sndr_bits.append(res.spectrum["sndr_db"])
        enob_bits.append(res.spectrum["enob"])
        print(f"  {b:>3} bit -> SNDR {res.spectrum['sndr_db']:7.2f} dB, ENOB {res.spectrum['enob']:6.2f}")
    ax_bits.plot(bits_list, sndr_bits, "o-")
    theory = [6.02 * b + 1.76 - 20 * np.log10(0.5 / amp) for b in bits_list]
    ax_bits.plot(bits_list, theory, "k--", label="theory 6.02N+1.76-backoff")
    ax_bits.set_xlabel("bits"); ax_bits.set_ylabel("SNDR (dB)"); ax_bits.set_title("Resolution sweep (ideal)")
    ax_bits.legend(fontsize=7)
    results["bits"] = {"bits": bits_list, "sndr_db": sndr_bits, "enob": enob_bits, "theory_db": theory}

    # ---- 2. input frequency sweep (front-end limited: jitter + acquisition)
    print("Experiment 2: input frequency vs SNDR (12-bit, jitter + acquisition)")
    cfg12 = ADCConfig(n_bits=12)
    fin_frac = [0.01, 0.03, 0.06, 0.12, 0.2, 0.3, 0.4]
    sndr_freq = []
    ni_fe = NonidealityConfig(sampling_jitter_rms=5e-8, acquisition_time=2e-7, switch_tau=1e-7, seed=args.seed)
    for frac in fin_frac:
        cycles = int(fin_frac[-1] * n // 2)  # keep below Nyquist
        cycles = max(3, int(frac * 0.4 * n))
        res = Simulator(cfg12, ni_fe, fs=fs, n=n, n_cycles=cycles, amp=amp).run()
        sndr_freq.append(res.spectrum["sndr_db"])
        print(f"  fin={res.f_sig/1e3:7.2f} kHz ({frac*100:3.0f}% fs) -> SNDR {res.spectrum['sndr_db']:7.2f} dB")
    ax_freq.plot([f * fs / 1e3 for f in fin_frac], sndr_freq, "o-")
    ax_freq.set_xlabel("input frequency (kHz)"); ax_freq.set_ylabel("SNDR (dB)")
    ax_freq.set_title("Input-frequency sweep (jitter + acquisition)")
    results["freq"] = {"fin_khz": [f * fs / 1e3 for f in fin_frac], "sndr_db": sndr_freq}

    # ---- 3. comparator noise sweep ----------------------------------------
    print("Experiment 3: comparator noise vs SNDR/ENOB (12-bit)")
    noise_lsb = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
    sndr_noise = []
    for nl in noise_lsb:
        res = Simulator(cfg12, NonidealityConfig(comparator_noise_rms=nl * cfg12.lsb, seed=args.seed),
                        fs=fs, n=n, n_cycles=997, amp=amp).run()
        sndr_noise.append(res.spectrum["sndr_db"])
        print(f"  noise={nl:4.2f} LSB -> SNDR {res.spectrum['sndr_db']:7.2f} dB, ENOB {res.spectrum['enob']:6.2f}")
    ax_noise.plot(noise_lsb, sndr_noise, "o-")
    ax_noise.set_xlabel("comparator noise (LSB)"); ax_noise.set_ylabel("SNDR (dB)")
    ax_noise.set_title("Comparator noise sweep")
    results["noise"] = {"noise_lsb": noise_lsb, "sndr_db": sndr_noise}

    # ---- 4. capacitor mismatch sweep --------------------------------------
    print("Experiment 4: capacitor mismatch vs INL/DNL (12-bit)")
    sigma_list = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05]
    inl_max, dnl_max, sndr_mis = [], [], []
    for sg in sigma_list:
        res = Simulator(cfg12, NonidealityConfig(cap_mismatch_sigma=sg, seed=args.seed),
                        fs=fs, n=n, n_cycles=997, amp=amp).run()
        inl_max.append(res.metrics["inl_max_bestfit_lsb"])
        dnl_max.append(res.metrics["dnl_max_lsb"])
        sndr_mis.append(res.spectrum["sndr_db"])
        print(f"  sigma={sg:.4f} -> INLmax {res.metrics['inl_max_bestfit_lsb']:7.3f} LSB, "
              f"DNLmax {res.metrics['dnl_max_lsb']:7.3f} LSB, SNDR {res.spectrum['sndr_db']:7.2f} dB")
    ax_mis.plot(sigma_list, inl_max, "o-", label="INL max (LSB)")
    ax_mis.plot(sigma_list, dnl_max, "s-", label="DNL max (LSB)")
    ax_mis.set_xlabel("cap mismatch sigma"); ax_mis.set_ylabel("LSB")
    ax_mis.set_title("Capacitor mismatch sweep"); ax_mis.legend(fontsize=7)
    results["mismatch"] = {"sigma": sigma_list, "inl_max_lsb": inl_max, "dnl_max_lsb": dnl_max, "sndr_db": sndr_mis}

    # ---- 5. acquisition time sweep -----------------------------------------
    print("Experiment 5: acquisition time vs SNDR (12-bit, tau fixed)")
    tau = 2e-7
    tacq_list = [1e-5, 2e-6, 8e-7, 4e-7, 2e-7, 1e-7, 5e-8]
    sndr_acq = []
    for ta in tacq_list:
        res = Simulator(cfg12, NonidealityConfig(acquisition_time=ta, switch_tau=tau, seed=args.seed),
                        fs=fs, n=n, n_cycles=997, amp=amp).run()
        sndr_acq.append(res.spectrum["sndr_db"])
        print(f"  tacq={ta*1e6:8.4g} us (tacq/tau={ta/tau:5.2f}) -> SNDR {res.spectrum['sndr_db']:7.2f} dB")
    ax_acq.plot([t / tau for t in tacq_list], sndr_acq, "o-")
    ax_acq.set_xlabel("tacq / tau"); ax_acq.set_ylabel("SNDR (dB)")
    ax_acq.set_title("Acquisition-time sweep")
    results["acq"] = {"tacq_tau_ratio": [t / tau for t in tacq_list], "sndr_db": sndr_acq}

    # ---- summary panel: ENOB vs bits --------------------------------------
    ax_sum.plot(bits_list, enob_bits, "o-")
    ax_sum.plot(bits_list, [6.02 * b / 6.02 for b in bits_list], "k--", label="ideal N")
    ax_sum.set_xlabel("bits"); ax_sum.set_ylabel("ENOB (bit)")
    ax_sum.set_title("ENOB vs resolution"); ax_sum.legend(fontsize=7)

    fig.suptitle("ADC experiments: ideal vs nonideal model (fs = 1 MHz, coherent tone, -0.9 dBFS)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png = out / "experiments.png"
    fig.savefig(png, dpi=120)
    plt.close(fig)
    print(f"\nexperiment plot written to {png}")

    jp = out / "experiments.json"
    jp.write_text(json.dumps(results, indent=2, default=float))
    print(f"experiment data written to {jp}")
    return 0


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------

def cmd_demo(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = ADCConfig(n_bits=args.bits, v_ref=args.vref, v_min=args.vmin,
                    coding=args.coding, tread=args.tread, reconstruction=args.recon)
    ni = _make_ni(args)
    sim = Simulator(cfg, ni, fs=args.fs, n=args.n, n_cycles=args.cycles, amp=args.amp)
    res = sim.run()

    print(f"Demo -- {cfg.n_bits}-bit SAR ADC, nonideal={ni}")
    print(f"SNDR {res.spectrum['sndr_db']:.2f} dB | ENOB {res.spectrum['enob']:.2f} bit | "
          f"INLmax(ep) {res.metrics['inl_max_endpoint_lsb']:.3f} LSB | DNLmax {res.metrics['dnl_max_lsb']:.3f} LSB")

    from .plotting import plot_inl_dnl, plot_quant_error_hist, plot_sar_trace, plot_spectrum, plot_transfer

    files = [
        plot_transfer(res, out / "demo_transfer.png"),
        plot_quant_error_hist(res, out / "demo_qerr_hist.png"),
        plot_inl_dnl(res, out / "demo_inl_dnl.png"),
        plot_spectrum(res, out / "demo_spectrum.png"),
    ]
    # traced conversion at a couple of representative inputs
    sar = sim.sar
    for label, vin in (("0.25", 0.25 * cfg.v_ref + cfg.v_min), ("mid", 0.5 * cfg.v_ref + cfg.v_min), ("0.81", 0.81 * cfg.v_ref + cfg.v_min)):
        tr = sar.convert_traced(vin)
        files.append(plot_sar_trace(tr, out / f"demo_trace_{label}.png"))
        print(f"trace @ {vin:.4g} V -> code {tr.final_code_int} ({tr.final_code}), residual {tr.residual_ideal/cfg.lsb:.3f} LSB")

    # metrics summary JSON
    (out / "demo_metrics.json").write_text(json.dumps(res.to_dict(), indent=2, default=float))
    print("files written:")
    for f in files:
        print(f"  {f}")
    print(f"  {out / 'demo_metrics.json'}")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="saradc", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("sar", help="cycle-by-cycle SAR conversion trace")
    sp.add_argument("--vin", type=float, required=True, help="analog input voltage (V)")
    sp.add_argument("--json", action="store_true", help="also write trace JSON")
    sp.add_argument("--out", type=str, default=str(DEFAULT_OUT / "sar_trace.json"))
    _add_cfg_args(sp); _add_nonideal_args(sp)
    sp.set_defaults(func=cmd_sar)

    sp = sub.add_parser("ideal", help="ideal quantizer statistics and plots")
    sp.add_argument("--plot", action="store_true", help="write transfer + error-histogram plots")
    sp.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    _add_cfg_args(sp); _add_nonideal_args(sp); _add_signal_args(sp)
    sp.set_defaults(func=cmd_ideal)

    sp = sub.add_parser("metrics", help="full metrics run with plots")
    sp.add_argument("--plot", action="store_true", help="write plots")
    sp.add_argument("--json", action="store_true", help="write metrics JSON")
    sp.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    _add_cfg_args(sp); _add_nonideal_args(sp); _add_signal_args(sp)
    sp.set_defaults(func=cmd_metrics)

    sp = sub.add_parser("experiments", help="run the five spec experiments")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    sp.set_defaults(func=cmd_experiments)

    sp = sub.add_parser("demo", help="full demonstration into output/")
    sp.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    _add_cfg_args(sp); _add_nonideal_args(sp); _add_signal_args(sp)
    sp.set_defaults(func=cmd_demo)

    sp = sub.add_parser("dashboard", help="generate interactive HTML dashboard")
    sp.add_argument("--out", type=str, default=str(DEFAULT_OUT / "dashboard.html"))
    sp.add_argument("--bits", type=int, nargs="+", default=[8, 12], help="bit depths to include")
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=cmd_dashboard)

    return p


def cmd_dashboard(args) -> int:
    from .dashboard import build_dashboard

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    build_dashboard(out, bits_list=args.bits, seed=args.seed)
    print(f"dashboard written to {out}")
    print("open it in a browser (no server needed; the file is self-contained)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
