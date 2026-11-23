"""Self-contained interactive HTML dashboard.

The dashboard is a single HTML file with all data embedded as JSON and a
small canvas-based plotting engine written in vanilla JavaScript (no CDN,
no server, works offline). It provides:

* scenario selector (ideal / capacitor mismatch / comparator offset+noise /
  acquisition+jitter) and bit-depth selector;
* transfer characteristic (ideal vs. SAR overlay);
* quantization-error histogram (ideal vs. SAR);
* INL/DNL curves in LSB;
* output spectrum with signal and harmonic markers, plus SNDR/ENOB cards;
* an interactive bit-by-bit SAR trace: choose an input voltage and step
  through the conversion cycle by cycle (register, trial DAC code, DAC
  voltage, comparator decision, residual).

The SAR trace is recomputed in JavaScript with the *same* algorithm and DAC
model as the Python simulator (deterministic comparator, i.e. noise disabled
for the trace; noise is reflected in the bulk metrics/spectrum).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .config import ADCConfig, NonidealityConfig
from .metrics import inl_dnl_from_transitions, transition_levels_analytical
from .pipeline import Simulator


def _arr(a: np.ndarray | list, nd: int = 4) -> list:
    """Round an array to ``nd`` decimals and return a plain list (JSON-safe)."""
    a = np.asarray(a, dtype=float)
    out = np.round(a, nd)
    return [float(v) if math.isfinite(v) else 0.0 for v in out]


def _scenarios() -> dict:
    return {
        "ideal": {
            "label": "Ideal",
            "nonideal": NonidealityConfig(),
        },
        "mismatch": {
            "label": "Capacitor mismatch (sigma=1%)",
            "nonideal": NonidealityConfig(cap_mismatch_sigma=0.01),
        },
        "comparator": {
            "label": "Comparator offset + noise",
            "nonideal": NonidealityConfig(comparator_offset=0.4, comparator_noise_rms=0.35),
        },
        "frontend": {
            "label": "Acquisition + jitter",
            "nonideal": NonidealityConfig(sampling_jitter_rms=5e-8, acquisition_time=2e-7, switch_tau=1e-7),
        },
    }


def build_dashboard(out: Path, bits_list: list[int] | None = None, seed: int = 0,
                    fs: float = 1e6, n: int = 16384, n_cycles: int = 997, amp: float = 0.45) -> Path:
    bits_list = bits_list or [8, 12]
    scenarios = _scenarios()

    data: dict = {}
    for bits in bits_list:
        cfg = ADCConfig(n_bits=bits, v_ref=1.0)
        data[str(bits)] = {}
        for sid, sc in scenarios.items():
            # carry the shared seed into every stochastic model
            sc_n = NonidealityConfig(**{**sc["nonideal"].__dict__, "seed": seed})
            sim = Simulator(cfg, sc_n, fs=fs, n=n, n_cycles=n_cycles, amp=amp)
            res = sim.run()

            spec = res.spectrum
            # ---- downsampled spectrum (peak-hold per bucket) ----
            nf = len(spec["freqs"])
            n_pts = 900
            idx = np.linspace(0, nf - 1, n_pts, dtype=int)
            # peak-hold in buckets for faithful spurs
            bounds = np.unique(np.linspace(0, nf, n_pts + 1, dtype=int))
            freq_pts, db_pts = [], []
            for a, b in zip(bounds[:-1], bounds[1:]):
                if b - a <= 0:
                    continue
                j = a + int(np.argmax(spec["power"][a:b]))
                freq_pts.append(spec["freqs"][j] / 1e3)
                db_pts.append(round(float(np.max(spec["power_db"][a:b])), 2))
            harm_peaks = []
            for h in (2, 3, 4, 5, 6):
                k = round(h * spec["k_sig"])
                if k < nf:
                    harm_peaks.append([h, round(float(spec["freqs"][k] / 1e3), 3),
                                       round(float(spec["power_db"][k]), 2)])

            v_grid, codes, _ = res.transfer
            # downsample transfer to <= 2048 points (keep step edges)
            step = max(1, len(v_grid) // 2048)
            T = transition_levels_analytical(cfg, sim.dac, sc_n.comparator_offset)
            inl = inl_dnl_from_transitions(T, cfg.lsb, cfg.n_bits)

            hist_bins = np.linspace(-0.75, 0.75, 65)  # LSB units
            h_sar, _ = np.histogram(res.qerr_sar / cfg.lsb, bins=hist_bins, density=False)
            h_ideal, _ = np.histogram(res.qerr_ideal / cfg.lsb, bins=hist_bins, density=False)
            h_sar = h_sar / max(h_sar.sum(), 1)
            h_ideal = h_ideal / max(h_ideal.sum(), 1)

            data[str(bits)][sid] = {
                "label": sc["label"],
                "fs": fs,
                "fin_khz": round(res.f_sig / 1e3, 3),
                "amp": amp,
                "sndr_db": round(spec["sndr_db"], 2),
                "enob": round(spec["enob"], 3),
                "inl_max": round(inl["inl_max_bestfit"], 4),
                "dnl_max": round(inl["dnl_max"], 4),
                "gain_frac": round(inl["gain_error_frac"], 6),
                "latency_cycles": cfg.n_bits,
                "transfer": {
                    "v": _arr(v_grid[::step], 6),
                    "code": _arr(codes[::step], 0),
                },
                "hist": {"edges": _arr(hist_bins, 4), "sar": _arr(h_sar, 5), "ideal": _arr(h_ideal, 5)},
                "inl": _arr(inl["inl_bestfit"], 4),
                "dnl": _arr(inl["dnl"], 4),
                "spectrum": {"freq_khz": freq_pts, "db": db_pts, "harmonics": harm_peaks,
                             "k_sig": spec["k_sig"]},
                # deterministic SAR trace engine parameters (comparator noise off)
                "trace": {
                    "n_bits": cfg.n_bits,
                    "v_ref": cfg.v_ref,
                    "v_min": cfg.v_min,
                    "lsb": cfg.lsb,
                    "weights": _arr(sim.dac.weights, 8),
                    "w_sum": round(sim.dac.w_sum, 8),
                    "termination": 1.0,
                    "ref_actual": round(sim.dac.ref_actual, 10),
                    "cmp_offset": round(sc_n.comparator_offset, 10),
                    "tread": cfg.tread,
                },
            }

    html = _render(data, seed)
    out.write_text(html, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# HTML / JS template
# ---------------------------------------------------------------------------

def _render(data: dict, seed: int) -> str:
    payload = json.dumps(data, separators=(",", ":"))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SAR ADC Architect — interactive dashboard</title>
<style>
  :root {{ --fg:#222; --line:#ccd; --accent:#1f77b4; }}
  * {{ box-sizing:border-box; }}
  body {{ font:14px/1.45 -apple-system,"Segoe UI",Roboto,sans-serif; color:var(--fg);
          margin:0; background:#f7f8fa; }}
  header {{ background:#12263a; color:#fff; padding:14px 22px; }}
  header h1 {{ margin:0; font-size:19px; }}
  header p {{ margin:4px 0 0; color:#bcd; font-size:12px; }}
  .wrap {{ max-width:1180px; margin:0 auto; padding:16px 22px 40px; }}
  .controls {{ display:flex; flex-wrap:wrap; gap:14px; align-items:center;
               background:#fff; border:1px solid var(--line); border-radius:8px;
               padding:10px 14px; margin-bottom:14px; }}
  .controls label {{ font-weight:600; margin-right:4px; }}
  select, input[type=range] {{ font:inherit; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr));
            gap:10px; margin-bottom:16px; }}
  .card {{ background:#fff; border:1px solid var(--line); border-radius:8px; padding:8px 12px; }}
  .card .k {{ font-size:11px; color:#667; text-transform:uppercase; letter-spacing:.04em; }}
  .card .v {{ font-size:20px; font-weight:700; margin-top:2px; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
  @media (max-width:900px) {{ .grid {{ grid-template-columns:1fr; }} }}
  .panel {{ background:#fff; border:1px solid var(--line); border-radius:8px; padding:10px 12px; }}
  .panel h3 {{ margin:0 0 4px; font-size:14px; }}
  .panel .sub {{ font-size:11px; color:#667; margin-bottom:6px; }}
  canvas {{ width:100%; height:260px; display:block; }}
  .trace {{ background:#fff; border:1px solid var(--line); border-radius:8px; padding:12px; margin-top:14px; }}
  .trace h3 {{ margin:0 0 6px; font-size:14px; }}
  .trace .row {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; }}
  .bitsreg {{ display:flex; gap:3px; }}
  .bitcell {{ width:26px; height:34px; border:1.5px solid #aab; border-radius:4px;
              display:flex; align-items:center; justify-content:center; font-weight:700;
              font-size:13px; background:#f2f3f5; color:#889; }}
  .bitcell.on {{ background:#1f77b4; color:#fff; border-color:#1f77b4; }}
  .bitcell.trial {{ outline:3px solid #ff7f0e; }}
  table.steps {{ border-collapse:collapse; width:100%; margin-top:10px; font-size:12px; }}
  table.steps th, table.steps td {{ border:1px solid var(--line); padding:3px 8px; text-align:right; }}
  table.steps th {{ background:#eef1f4; }}
  .btn {{ border:1px solid #99a; background:#f0f2f5; border-radius:5px; padding:4px 12px;
         cursor:pointer; font:inherit; }}
  .btn:active {{ background:#dfe3e8; }}
  .note {{ font-size:11px; color:#667; margin-top:8px; }}
</style>
</head>
<body>
<header>
  <h1>SAR ADC Architect — interactive simulation</h1>
  <p>Successive-approximation ADC: ideal quantization vs. explicit circuit nonidealities.
     Coherent single-tone test (fs = 1 MHz, -0.9 dBFS). All data computed by the Python
     simulator and embedded in this file.</p>
</header>
<div class="wrap">

  <div class="controls">
    <label>Scenario</label>
    <select id="scenario"></select>
    <label>Bits</label>
    <select id="bitsSel"></select>
    <label>Input voltage</label>
    <input type="range" id="vin" min="0" max="1000" value="630" style="width:200px">
    <span id="vinLbl" style="font-weight:700">0.630 V</span>
    <span style="flex:1"></span>
    <button class="btn" id="stepBack">&#9664; step</button>
    <button class="btn" id="stepFwd">step &#9654;</button>
    <button class="btn" id="playBtn">play</button>
  </div>

  <div class="cards" id="cards"></div>

  <div class="grid">
    <div class="panel"><h3>Transfer characteristic</h3>
      <div class="sub">output code vs. input voltage (SAR model vs. ideal quantizer)</div>
      <canvas id="cTransfer"></canvas></div>
    <div class="panel"><h3>Quantization error histogram</h3>
      <div class="sub">residual/quantization error distribution in LSB (SAR vs. ideal)</div>
      <canvas id="cHist"></canvas></div>
    <div class="panel"><h3>INL / DNL</h3>
      <div class="sub">best-fit INL and DNL in LSB vs. code</div>
      <canvas id="cInlDnl"></canvas></div>
    <div class="panel"><h3>Output spectrum</h3>
      <div class="sub">dBFS vs. frequency; markers on harmonics 2..6</div>
      <canvas id="cSpectrum"></canvas></div>
  </div>

  <div class="trace">
    <h3>Cycle-by-cycle SAR conversion trace</h3>
    <div class="row">
      <span class="bitsreg" id="bitReg"></span>
      <span id="regCode" style="font-weight:700"></span>
      <span id="stepInfo" style="color:#667"></span>
    </div>
    <table class="steps" id="stepTable">
      <thead><tr><th>cycle</th><th>bit</th><th>trial code</th><th>trial DAC (V)</th>
        <th>cmp input (V)</th><th>decision</th><th>register</th><th>reg DAC (V)</th><th>residual (V)</th></tr></thead>
      <tbody></tbody>
    </table>
    <div class="note">Trace uses the same SAR algorithm and DAC model as the simulator with a
      deterministic comparator (noise disabled in the trace; noise/jitter/acquisition effects
      appear in the spectrum and metrics above).</div>
  </div>

</div>
<script>
"use strict";
const DATA = {payload};
const scenarioSel = document.getElementById("scenario");
const bitsSel = document.getElementById("bitsSel");
const vinSlider = document.getElementById("vin");
const vinLbl = document.getElementById("vinLbl");

const SCENARIO_KEYS = Object.keys(DATA[Object.keys(DATA)[0]]);
const BITS = Object.keys(DATA).map(Number);

for (const k of SCENARIO_KEYS) {{
  const o = document.createElement("option"); o.value = k; o.text = DATA[BITS[0]][k].label;
  scenarioSel.appendChild(o);
}}
for (const b of BITS) {{
  const o = document.createElement("option"); o.value = b; o.text = b + " bit";
  bitsSel.appendChild(o);
}}
bitsSel.value = String(BITS[0]);

// ---------------- tiny canvas chart engine ----------------
function setup(canvas) {{
  const dpr = window.devicePixelRatio || 1;
  const r = canvas.getBoundingClientRect();
  canvas.width = Math.max(10, Math.floor(r.width * dpr));
  canvas.height = Math.max(10, Math.floor(r.height * dpr));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return {{ ctx, w: r.width, h: r.height }};
}}
function lineChart(canvas, xs, ys, opts) {{
  opts = opts || {{}};
  const {{ctx, w, h}} = setup(canvas);
  const pad = {{l: 44, r: 10, t: 10, b: 24}};
  ctx.clearRect(0, 0, w, h);
  let yMin = opts.yMin !== undefined ? opts.yMin : Math.min(...ys);
  let yMax = opts.yMax !== undefined ? opts.yMax : Math.max(...ys);
  const yPad = (yMax - yMin) * 0.08 || 1;
  yMin -= yPad; yMax += yPad;
  const x0 = pad.l, x1 = w - pad.r, y0 = h - pad.b, y1 = pad.t;
  const X = v => x0 + (v - Math.min(...xs)) / (Math.max(...xs) - Math.min(...xs) || 1) * (x1 - x0);
  const Y = v => y0 - (v - yMin) / (yMax - yMin) * (y0 - y1);
  // grid + axes
  ctx.strokeStyle = "#dde"; ctx.fillStyle = "#667"; ctx.font = "10px sans-serif"; ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {{
    const y = yMin + (yMax - yMin) * i / 5, yy = Y(y);
    ctx.beginPath(); ctx.moveTo(x0, yy); ctx.lineTo(x1, yy); ctx.stroke();
    ctx.fillText(y.toPrecision(3), 4, yy + 3);
  }}
  const xTicks = 6;
  for (let i = 0; i <= xTicks; i++) {{
    const v = Math.min(...xs) + (Math.max(...xs) - Math.min(...xs)) * i / xTicks;
    ctx.beginPath(); ctx.moveTo(X(v), y0); ctx.lineTo(X(v), y1); ctx.stroke();
    ctx.fillText(v.toPrecision(3), X(v) - 14, h - 6);
  }}
  // data
  const line = opts.color || "#1f77b4";
  ctx.strokeStyle = line; ctx.lineWidth = 1.4; ctx.beginPath();
  for (let i = 0; i < xs.length; i++) {{
    const px = X(xs[i]), py = Y(ys[i]);
    i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
  }}
  ctx.stroke();
  if (opts.yLine !== undefined) {{
    ctx.strokeStyle = "#ff7f0e"; ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(x0, Y(opts.yLine)); ctx.lineTo(x1, Y(opts.yLine)); ctx.stroke();
    ctx.setLineDash([]);
  }}
  if (opts.markers) for (const m of opts.markers) {{
    ctx.fillStyle = m.color || "#d62728"; ctx.beginPath();
    ctx.arc(X(m.x), Y(m.y), 3.2, 0, Math.PI * 2); ctx.fill();
  }}
}}
function barChart(canvas, centers, yss, opts) {{
  opts = opts || {{}};
  const {{ctx, w, h}} = setup(canvas);
  const pad = {{l: 44, r: 10, t: 10, b: 24}};
  ctx.clearRect(0, 0, w, h);
  const all = [].concat(...yss);
  let yMax = Math.max(...all, 1e-9) * 1.1;
  const yMin = 0;
  const x0 = pad.l, x1 = w - pad.r, y0 = h - pad.b, y1 = pad.t;
  const X = v => x0 + (v - centers[0]) / (centers[centers.length-1] - centers[0] || 1) * (x1 - x0);
  const Y = v => y0 - (v - yMin) / (yMax - yMin) * (y0 - y1);
  ctx.strokeStyle = "#dde"; ctx.fillStyle = "#667"; ctx.font = "10px sans-serif";
  for (let i = 0; i <= 4; i++) {{
    const y = yMax * i / 4, yy = Y(y);
    ctx.beginPath(); ctx.moveTo(x0, yy); ctx.lineTo(x1, yy); ctx.stroke();
    ctx.fillText(y.toPrecision(2), 4, yy + 3);
  }}
  const n = centers.length;
  const bw = (x1 - x0) / n;
  const series = yss.length;
  const wb = bw * 0.8 / series;
  const colors = ["#1f77b4", "#ff7f0e"];
  for (let s = 0; s < series; s++) {{
    ctx.fillStyle = colors[s % colors.length]; ctx.globalAlpha = 0.85;
    for (let i = 0; i < n; i++) {{
      const bx = x0 + i * bw + (bw - wb) / 2 + s * wb;
      const bh = (Y(0) - Y(yss[s][i]));
      ctx.fillRect(bx, Y(yss[s][i]), wb - 1, Math.max(0, bh));
    }}
    ctx.globalAlpha = 1;
  }}
  ctx.fillText(centers[0].toPrecision(3), x0 - 8, h - 6);
  ctx.fillText(centers[n-1].toPrecision(3), x1 - 30, h - 6);
}}

// ---------------- state ----------------
let state = {{
  scenario: SCENARIO_KEYS[0],
  bits: BITS[0],
  vin: 0.63,
  steps: [],
  stepIdx: -1,
  playing: false,
  timer: null,
}};

function cur() {{ return DATA[state.bits][state.scenario]; }}

function fmt(v, d) {{ return v.toFixed(d === undefined ? 4 : d); }}

// ---------------- render ----------------
function renderCards() {{
  const c = cur();
  const cards = [
    ["SNDR", fmt(c.sndr_db, 2) + " dB"],
    ["ENOB", fmt(c.enob, 3) + " bit"],
    ["INL max", fmt(c.inl_max, 3) + " LSB"],
    ["DNL max", fmt(c.dnl_max, 3) + " LSB"],
    ["Gain err", fmt(c.gain_frac * 100, 3) + " %"],
    ["Latency", c.latency_cycles + " cycles"],
    ["f in", fmt(c.fin_khz, 2) + " kHz"],
  ];
  document.getElementById("cards").innerHTML = cards.map(c2 =>
    `<div class="card"><div class="k">${{c2[0]}}</div><div class="v">${{c2[1]}}</div></div>`).join("");
}}

function renderCharts() {{
  const c = cur();
  // transfer: SAR (line) vs ideal (step, recomputed on the same grid)
  const tv = c.transfer.v, tc = c.transfer.code;
  const bits = state.bits;
  const idealC = tc.map(v => {{
    const s = (v - c.trace.v_min) / c.trace.v_ref;
    return Math.max(0, Math.min(Math.floor(s * Math.pow(2, bits)), Math.pow(2, bits) - 1));
  }});
  // draw the transfer curves (SAR line + ideal dashed) below
  {{
    const canvas = document.getElementById("cTransfer");
    const {{ctx, w, h}} = setup(canvas);
    const pad = {{l: 44, r: 10, t: 10, b: 24}};
    const x0 = pad.l, x1 = w - pad.r, y0 = h - pad.b, y1 = pad.t;
    const X = v => x0 + v / c.trace.v_ref * (x1 - x0);
    const Y = c2 => y0 - c2 / (Math.pow(2, bits) - 1) * (y0 - y1);
    ctx.strokeStyle = "#dde"; ctx.fillStyle = "#667"; ctx.font = "10px sans-serif";
    for (let i = 0; i <= 5; i++) {{
      const y = y0 - (y0 - y1) * i / 5;
      ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
      ctx.fillText(((Math.pow(2, bits) - 1) * i / 5).toFixed(0), 4, y + 3);
      const xv = x0 + (x1 - x0) * i / 5;
      ctx.beginPath(); ctx.moveTo(xv, y0); ctx.lineTo(xv, y1); ctx.stroke();
      ctx.fillText((c.trace.v_ref * i / 5).toFixed(2), xv - 12, h - 6);
    }}
    ctx.lineWidth = 1.4; ctx.strokeStyle = "#1f77b4"; ctx.beginPath();
    for (let i = 0; i < tv.length; i++) {{
      const px = X(tv[i]), py = Y(tc[i]);
      i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    }}
    ctx.stroke();
    ctx.strokeStyle = "#ff7f0e"; ctx.setLineDash([4, 3]); ctx.beginPath();
    for (let i = 0; i < tv.length; i++) {{
      const px = X(tv[i]), py = Y(idealC[i]);
      i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    }}
    ctx.stroke(); ctx.setLineDash([]);
  }}

  // histogram
  {{
    const canvas = document.getElementById("cHist");
    const {{ctx, w, h}} = setup(canvas);
    const pad = {{l: 44, r: 10, t: 10, b: 24}};
    const x0 = pad.l, x1 = w - pad.r, y0 = h - pad.b, y1 = pad.t;
    const n = c.hist.edges.length - 1;
    const centers = c.hist.edges.slice(0, n).map((e, i) => (e + c.hist.edges[i+1]) / 2);
    const yMax = Math.max(...c.hist.sar, ...c.hist.ideal, 1e-9) * 1.1;
    const Y = v => y0 - v / yMax * (y0 - y1);
    const X = v => x0 + (v - centers[0]) / (centers[n-1] - centers[0]) * (x1 - x0);
    const bw = (x1 - x0) / n;
    ctx.strokeStyle = "#dde"; ctx.fillStyle = "#667"; ctx.font = "10px sans-serif";
    for (let i = 0; i <= 4; i++) {{
      const y = Y(yMax * i / 4);
      ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
      ctx.fillText((yMax * i / 4).toPrecision(2), 4, y + 3);
    }}
    const colors = ["#1f77b4", "#ff7f0e"]; const series = [c.hist.sar, c.hist.ideal];
    for (let s = 0; s < 2; s++) {{
      ctx.fillStyle = colors[s]; ctx.globalAlpha = 0.8;
      for (let i = 0; i < n; i++) {{
        const bx = X(centers[i]) - bw / 2 + s * bw * 0.35;
        ctx.fillRect(bx, Y(series[s][i]), bw * 0.3, Math.max(0, Y(0) - Y(series[s][i])));
      }}
      ctx.globalAlpha = 1;
    }}
    ctx.fillText("−0.75", x0 - 14, h - 6); ctx.fillText("+0.75 LSB", x1 - 46, h - 6);
  }}

  // INL / DNL (dual color on one canvas)
  {{
    const canvas = document.getElementById("cInlDnl");
    const {{ctx, w, h}} = setup(canvas);
    const pad = {{l: 44, r: 10, t: 10, b: 24}};
    const x0 = pad.l, x1 = w - pad.r, y0 = h - pad.b, y1 = pad.t;
    const nCodes = c.inl.length;
    const xs = Array.from({{length: nCodes}}, (_, i) => i);
    const allY = [].concat(c.inl, c.dnl);
    let yMin = Math.min(...allY), yMax = Math.max(...allY);
    const yp = (yMax - yMin) * 0.08 || 1; yMin -= yp; yMax += yp;
    const Y = v => y0 - (v - yMin) / (yMax - yMin) * (y0 - y1);
    const X = v => x0 + v / (nCodes - 1) * (x1 - x0);
    ctx.strokeStyle = "#dde"; ctx.fillStyle = "#667"; ctx.font = "10px sans-serif";
    for (let i = 0; i <= 5; i++) {{
      const y = Y(yMin + (yMax - yMin) * i / 5);
      ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
      ctx.fillText((yMin + (yMax - yMin) * i / 5).toPrecision(2), 4, y + 3);
    }}
    ctx.strokeStyle = "#1f77b4"; ctx.lineWidth = 1.2; ctx.beginPath();
    for (let i = 0; i < nCodes; i++) {{
      const px = X(xs[i]), py = Y(c.inl[i]);
      i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    }}
    ctx.stroke();
    ctx.strokeStyle = "#2ca02c"; ctx.beginPath();
    for (let i = 0; i < nCodes - 1; i++) {{
      const px = X(xs[i]), py = Y(c.dnl[i]);
      i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    }}
    ctx.stroke();
    ctx.fillText("0", x0 - 8, h - 6); ctx.fillText((nCodes - 1).toFixed(0), x1 - 20, h - 6);
  }}

  // spectrum
  {{
    const canvas = document.getElementById("cSpectrum");
    const {{ctx, w, h}} = setup(canvas);
    const pad = {{l: 44, r: 10, t: 10, b: 24}};
    const x0 = pad.l, x1 = w - pad.r, y0 = h - pad.b, y1 = pad.t;
    const f = c.spectrum.freq_khz, db = c.spectrum.db;
    let yMin = Math.min(...db), yMax = Math.max(...db);
    const yp = (yMax - yMin) * 0.05 || 1; yMin -= yp; yMax += yp;
    const X = v => x0 + v / Math.max(...f) * (x1 - x0);
    const Y = v => y0 - (v - yMin) / (yMax - yMin) * (y0 - y1);
    ctx.strokeStyle = "#dde"; ctx.fillStyle = "#667"; ctx.font = "10px sans-serif";
    for (let i = 0; i <= 5; i++) {{
      const y = Y(yMin + (yMax - yMin) * i / 5);
      ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
      ctx.fillText((yMin + (yMax - yMin) * i / 5).toFixed(0), 4, y + 3);
    }}
    ctx.strokeStyle = "#1f77b4"; ctx.lineWidth = 0.9; ctx.beginPath();
    for (let i = 0; i < f.length; i++) {{
      const px = X(f[i]), py = Y(db[i]);
      i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    }}
    ctx.stroke();
    for (const [h, hf, hdb] of c.spectrum.harmonics) {{
      ctx.fillStyle = "#d62728"; ctx.beginPath(); ctx.arc(X(hf), Y(hdb), 3, 0, 2 * Math.PI); ctx.fill();
      ctx.font = "9px sans-serif"; ctx.fillText("x" + h, X(hf) + 4, Y(hdb) - 4);
    }}
    ctx.fillText("0", x0 - 8, h - 6); ctx.fillText(Math.max(...f).toFixed(0) + " kHz", x1 - 34, h - 6);
  }}
}}

// ---------------- SAR trace engine (mirrors the Python model) ----------------
function sarTrace(c, vin) {{
  // returns {{steps:[...], finalCode, reconIdeal}}
  const t = c.trace;
  const vShift = (t.tread === "midtread") ? vin - t.v_min + 0.5 * t.lsb : vin - t.v_min;
  const s = Math.max(0, Math.min(t.v_ref, vShift));
  const denom = t.w_sum + t.termination;
  const dacV = code => {{
    let sum = 0;
    for (let i = 0; i < t.n_bits; i++) if (code & (1 << i)) sum += t.weights[i];
    return sum / denom * t.ref_actual;
  }};
  let reg = 0;
  const steps = [];
  for (let i = t.n_bits - 1; i >= 0; i--) {{
    const trial = reg | (1 << i);
    const vt = dacV(trial);
    const cmpIn = (s - vt) + t.cmp_offset;
    const keep = cmpIn >= 0;
    if (keep) reg = trial;
    steps.push({{
      cycle: t.n_bits - 1 - i, bit: i, trialCode: trial, trialV: vt, cmpIn: cmpIn,
      decision: keep, reg: reg, regV: dacV(reg), residual: s - dacV(reg),
    }});
  }}
  const recon = reg * t.lsb + (t.tread === "midrise" ? 0.5 * t.lsb : 0) + t.v_min;
  return {{ steps: steps, finalCode: reg, recon: recon, s: s }};
}}

function renderTrace() {{
  const c = cur();
  const {{steps, finalCode, recon, s}} = sarTrace(c, state.vin);
  state.steps = steps; state.stepIdx = -1;
  // bit cells
  const cells = [];
  for (let i = 0; i < c.trace.n_bits; i++) cells.push(`<div class="bitcell" id="bit${{i}}">0</div>`);
  document.getElementById("bitReg").innerHTML = cells.join("");
  document.getElementById("regCode").textContent =
    "code " + finalCode + " (recon " + recon.toFixed(5) + " V, residual " + (s - recon).toFixed(5) + " V)";
  document.getElementById("vinLbl").textContent = state.vin.toFixed(3) + " V";
  // seed table with initial state
  updateStep();
}}

function updateStep() {{
  const t = state.steps;
  const body = document.getElementById("stepTable").querySelector("tbody");
  const rows = [];
  if (state.stepIdx < 0) {{
    rows.push(`<tr><td colspan="9" style="text-align:center;color:#667">press step / play to walk the conversion</td></tr>`);
  }} else {{
    const shown = t.slice(0, state.stepIdx + 1);
    for (const st of shown) {{
      rows.push(`<tr>
        <td>${{st.cycle}}</td><td>${{st.bit}}</td><td>${{st.trialCode}}</td>
        <td>${{st.trialV.toFixed(6)}}</td><td>${{st.cmpIn.toFixed(6)}}</td>
        <td>${{st.decision ? "keep" : "clear"}}</td><td>${{st.reg}}</td>
        <td>${{st.regV.toFixed(6)}}</td><td>${{st.residual.toFixed(6)}}</td></tr>`);
    }}
  }}
  body.innerHTML = rows.join("");
  // bit cells
  const last = state.stepIdx >= 0 ? t[state.stepIdx] : null;
  for (let i = 0; i < state.bits; i++) {{
    const cell = document.getElementById("bit" + i);
    let on = false, trial = false;
    if (last) {{
      on = (last.reg >> i) & 1;
      trial = (i === last.bit);
    }}
    cell.className = "bitcell" + (on ? " on" : "") + (trial ? " trial" : "");
    cell.textContent = on ? "1" : "0";
  }}
  const info = state.stepIdx >= 0
    ? "cycle " + (state.stepIdx + 1) + " / " + state.bits
    : "register cleared, ready";
  document.getElementById("stepInfo").textContent = info;
}}

function stepOnce() {{
  if (state.stepIdx < state.steps.length - 1) state.stepIdx++;
  updateStep();
}}
function stepBackOnce() {{
  if (state.stepIdx > 0) state.stepIdx--;
  updateStep();
}}

function play() {{
  state.playing = !state.playing;
  document.getElementById("playBtn").textContent = state.playing ? "pause" : "play";
  if (state.playing) {{
    state.timer = setInterval(() => {{
      if (state.stepIdx >= state.steps.length - 1) {{
        clearInterval(state.timer); state.playing = false;
        document.getElementById("playBtn").textContent = "play";
        return;
      }}
      stepOnce();
    }}, 600);
  }} else {{
    clearInterval(state.timer);
  }}
}}

// ---------------- wiring ----------------
scenarioSel.addEventListener("change", () => {{ state.scenario = scenarioSel.value; renderAll(); }});
bitsSel.addEventListener("change", () => {{ state.bits = Number(bitsSel.value); renderAll(); }});
vinSlider.addEventListener("input", () => {{
  state.vin = (Number(vinSlider.value) / 1000) * DATA[state.bits][state.scenario].trace.v_ref;
  renderTrace();
}});
document.getElementById("stepFwd").addEventListener("click", stepOnce);
document.getElementById("stepBack").addEventListener("click", stepBackOnce);
document.getElementById("playBtn").addEventListener("click", play);

function renderAll() {{
  renderCards(); renderCharts(); renderTrace();
}}
renderAll();
window.addEventListener("resize", renderAll);
</script>
</body>
</html>
"""


def _main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--out", default="dashboard.html")
    p.add_argument("--bits", type=int, nargs="+", default=[8, 12])
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    build_dashboard(Path(a.out), bits_list=a.bits, seed=a.seed)
    print(f"written {a.out}")


if __name__ == "__main__":
    _main()
