"""Matplotlib plotting helpers for the SAR ADC simulator.

All figures are saved as PNG files (deterministic, reproducible from the same
seed). The interactive HTML dashboard (``saradc.dashboard``) provides the
interactive versions of these same panels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .pipeline import SimulationResult


def _style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 9,
        }
    )


def plot_transfer(res: SimulationResult, path: str | Path) -> Path:
    """Transfer characteristic: ideal vs. SAR model (code vs. input voltage)."""
    _style()
    import matplotlib.pyplot as plt

    v_grid, codes, recon = res.transfer
    n_pts = len(v_grid)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(v_grid, codes, lw=1.2, color="#1f77b4", label=f"SAR model ({res.cfg.n_bits} bit)")
    # ideal reference sampled on the same grid
    v_ref = res.cfg.v_min + np.linspace(0, res.cfg.v_ref, n_pts)
    ideal_codes = res.ideal.code_int(v_ref)
    ax.plot(v_ref, ideal_codes, lw=1.2, ls="--", color="#888888", label="Ideal quantizer")
    ax.set_xlabel(f"Input voltage (V)  [{res.cfg.v_min:.3g}, {res.cfg.v_max:.3g}]")
    ax.set_ylabel("Output code (offset binary)")
    ax.set_title("Transfer characteristic")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def plot_quant_error_hist(res: SimulationResult, path: str | Path) -> Path:
    """Histogram of quantization errors: ideal vs. SAR (same samples)."""
    _style()
    import matplotlib.pyplot as plt

    lsb = res.cfg.lsb
    fig, ax = plt.subplots(figsize=(8, 5))
    lo, hi = -0.7 * lsb, 0.7 * lsb
    bins = np.linspace(lo, hi, 61)
    ax.hist(res.qerr_sar / lsb, bins=bins, alpha=0.55, color="#1f77b4", label="SAR model error (LSB)")
    ax.hist(res.qerr_ideal / lsb, bins=bins, alpha=0.55, color="#ff7f0e", label="Ideal quantizer error (LSB)")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Error (LSB)")
    ax.set_ylabel("Count")
    ax.set_title("Quantization / residual error histogram")
    ax.legend()
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def plot_inl_dnl(res: SimulationResult, path: str | Path) -> Path:
    """INL (endpoint + best-fit) and DNL curves in LSB."""
    _style()
    import matplotlib.pyplot as plt

    lsb = res.cfg.lsb
    n_codes = res.cfg.n_codes
    from .metrics import inl_dnl_from_transitions, transition_levels_analytical

    T = transition_levels_analytical(res.cfg, res.dac, res.nonideal.comparator_offset)
    m = inl_dnl_from_transitions(T, lsb, res.cfg.n_bits)
    k = np.arange(n_codes)

    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax[0].plot(k, m["inl_endpoint"], lw=1.0, color="#1f77b4", label="INL (endpoint fit)")
    ax[0].plot(k, m["inl_bestfit"], lw=1.0, color="#ff7f0e", label="INL (best-fit line)")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_ylabel("INL (LSB)")
    ax[0].set_title(f"INL/DNL — {res.cfg.n_bits}-bit SAR model")
    ax[0].legend(loc="upper left")

    ax[1].plot(k, m["dnl"], lw=1.0, color="#2ca02c")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_ylabel("DNL (LSB)")
    ax[1].set_xlabel("Code")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def plot_spectrum(res: SimulationResult, path: str | Path) -> Path:
    """Output spectrum (dBFS) with signal / harmonics / noise annotation."""
    _style()
    import matplotlib.pyplot as plt

    spec = res.spectrum
    freqs = spec["freqs"]
    db = spec["power_db"]
    fs = res.fs
    k_sig = spec["k_sig"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(freqs / 1e3, db, lw=0.6, color="#1f77b4")
    ax.axvline(freqs[k_sig] / 1e3, color="#ff7f0e", ls="--", lw=1.0, label="fundamental")
    ax.axhline(10 * np.log10(spec["P_noise"] / spec["P_sig"]), color="r", ls=":", lw=1.0,
               label=f"noise floor ≈ {10*np.log10(spec['P_noise']/spec['P_sig']):.1f} dBFS")
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Power (dBFS)")
    ax.set_title(
        f"Output spectrum — SNDR {spec['sndr_db']:.1f} dB, ENOB {spec['enob']:.2f} bit "
        f"(fs={fs/1e6:.3g} MHz, fin={res.f_sig/1e3:.3g} kHz)"
    )
    ax.legend(loc="lower left")
    ax.set_xlim(0, fs / 2e3)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def plot_sar_trace(trace, path: str | Path, lsb: float | None = None) -> Path:
    """Cycle-by-cycle SAR conversion trace figure."""
    _style()
    import matplotlib.pyplot as plt

    n = len(trace.cycles)
    cycles = np.arange(n)
    trial_v = np.array([c.trial_voltage for c in trace.cycles])
    reg_v = np.array([c.register_voltage for c in trace.cycles])
    residual = np.array([c.residual for c in trace.cycles])
    kept = np.array([c.decision for c in trace.cycles])

    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].axhline(trace.v_held, color="k", ls="--", lw=1.0, label=f"held input = {trace.v_held:.5g} V")
    ax[0].plot(cycles, trial_v, "o-", ms=5, color="#1f77b4", label="trial DAC voltage")
    ax[0].plot(cycles, reg_v, "s-", ms=5, color="#2ca02c", label="register DAC voltage")
    for c, k in zip(trace.cycles, kept):
        ax[0].annotate("keep" if k else "clear", (c.cycle, c.trial_voltage), fontsize=7, ha="center", va="bottom")
    ax[0].set_ylabel("Voltage (V)")
    ax[0].set_title("SAR conversion: trial vs. register DAC voltage")
    ax[0].legend(loc="upper right")

    ax[1].bar(cycles, residual / trace.lsb, color=["#d62728" if not k else "#2ca02c" for k in kept], alpha=0.8)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xlabel("Cycle")
    ax[1].set_ylabel("Residual (LSB)")
    ax[1].set_title("Residual error after each decision")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
