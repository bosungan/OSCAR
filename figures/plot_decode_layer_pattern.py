#!/usr/bin/env python
# ============================================================================
# plot_decode_layer_pattern.py  (dependency-free, emits SVG)
# Visualize the per-layer kernel launch sequence of the OSCAR INT2 decode step,
# extracted directly from an nsys cuda_gpu_trace CSV (one decode layer period).
#
# Usage:
#   python figures/plot_decode_layer_pattern.py [GPU_TRACE_CSV]
# Default CSV: /tmp/oscar_gpu_trace.csv
#   (produced by: nsys stats --report cuda_gpu_trace --format csv <rep>)
# Output: figures/decode_layer_kernel_pattern.svg
# ============================================================================
import csv, re, sys, os
from html import escape

CSV = sys.argv[1] if len(sys.argv) > 1 else "/tmp/oscar_gpu_trace.csv"
OUT = os.path.join(os.path.dirname(__file__), "decode_layer_kernel_pattern.svg")

# ---- load trace ------------------------------------------------------------
lines = open(CSV).read().splitlines()
hi = next(i for i, l in enumerate(lines) if l.startswith("Start (ns)"))
rows = []
for r in csv.DictReader(lines[hi:]):
    try:
        s = int(r["Start (ns)"]); d = int(r["Duration (ns)"])
    except (ValueError, TypeError):
        continue
    rows.append((s, d, r["Name"].strip().strip('"')))
rows.sort()
t0 = rows[0][0]

# measured decode window; extract ONE layer period (input_layernorm -> next).
A = 15.340
seq = [(s, d, n) for s, d, n in rows if A <= (s - t0) / 1e9 <= 15.376]
period = seq[21:42]
p0 = period[0][0]

def classify(n):
    if "fused_add_rmsnorm" in n:           return ("RMSNorm (+residual)", "norm")
    if "wmma" in n and "16x16" in n:       return ("QKV proj GEMM", "proj")
    if "splitKreduce" in n:                return ("splitK reduce", "proj")
    if "fused_qknorm" in n:                return ("q/k-norm", "norm")
    if "fused_rope" in n:                  return ("RoPE", "misc")
    if "CUDAFunctorOnSelf_add" in n:       return ("cache idx update", "misc")
    if "store_kvcache" in n:               return ("store K/V -> HP win", "misc")
    if "wmma" in n and "32x32" in n and "_nn_" in n: return ("[OSCAR] rotate Q/K (R)", "oscar")
    if "FillFunctor" in n:                 return ("init attn buf", "misc")
    if "stage1_quant_int2" in n:           return ("[OSCAR] INT2-KV attn", "oscar")
    if "_fwd_grouped_kernel_stage1" in n:  return ("HP-window attn", "attn")
    if "stage2_unified" in n:              return ("[OSCAR] merge HP+INT2", "oscar")
    if "wmma" in n and "32x32" in n and "_tn_" in n: return ("o_proj prologue (V-rot absorb)", "oscar")
    if "64x64" in n and "stages_64x5" in n and "sliced" not in n: return ("gate+up proj GEMM", "proj")
    if "act_and_mul" in n:                 return ("SwiGLU (silu*up)", "misc")
    if "64x64" in n and "sliced1x2" in n:  return ("proj GEMM (o / down)", "proj")
    return (re.sub(r"<.*", "", n)[:24], "misc")

COL = {"norm": "#9e9e9e", "proj": "#4878cf", "attn": "#7fb069",
       "oscar": "#d1495b", "misc": "#cccccc"}

period_us = (period[-1][0] + period[-1][1] - p0) / 1000.0

# ---- SVG canvas ------------------------------------------------------------
W, H = 1500, 760
ML, MR = 60, 40
plot_w = W - ML - MR
PAD_US = 30
total_us = period_us + 2 * PAD_US
def X(us):  # us-from-layer-start -> svg x
    return ML + (us + PAD_US) / total_us * plot_w

svg = []
svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'font-family="DejaVu Sans, Arial, sans-serif" viewBox="0 0 {W} {H}">')
svg.append(f'<rect width="{W}" height="{H}" fill="white"/>')

# title
svg.append(f'<text x="{W/2}" y="30" text-anchor="middle" font-size="19" '
           f'font-weight="bold">OSCAR INT2 decode — per-layer kernel launch sequence '
           f'(Qwen3-8B)</text>')
svg.append(f'<text x="{W/2}" y="52" text-anchor="middle" font-size="13" fill="#444">'
           f'1 decoder layer ≈ {period_us:.0f} µs wall · '
           f'one decode step = this block × 36 layers · bars are duration-proportional</text>')

# ---- main gantt row --------------------------------------------------------
BAR_Y, BAR_H = 250, 46
# phase bands
PHASES = [(0, 9,  "A · Attention input"),
          (9, 12, "B · Attention core"),
          (12, 14, "C · Attn output"),
          (14, 18, "D · MLP")]
for a0, a1, lab in PHASES:
    x0 = X((period[a0][0] - p0) / 1000.0)
    x1 = X((period[a1-1][0] + period[a1-1][1] - p0) / 1000.0)
    svg.append(f'<rect x="{x0:.1f}" y="{BAR_Y-70}" width="{x1-x0:.1f}" height="{BAR_H+90}" '
               f'fill="#000" opacity="0.035"/>')
    svg.append(f'<text x="{(x0+x1)/2:.1f}" y="{BAR_Y-52}" text-anchor="middle" '
               f'font-size="13" font-weight="bold" fill="#333">{escape(lab)}</text>')

# axis line
svg.append(f'<line x1="{X(0):.1f}" y1="{BAR_Y+BAR_H+6}" x2="{X(period_us):.1f}" '
           f'y2="{BAR_Y+BAR_H+6}" stroke="#999" stroke-width="1"/>')
for tick in range(0, int(period_us) + 1, 100):
    xt = X(tick)
    svg.append(f'<line x1="{xt:.1f}" y1="{BAR_Y+BAR_H+6}" x2="{xt:.1f}" '
               f'y2="{BAR_Y+BAR_H+12}" stroke="#999"/>')
    svg.append(f'<text x="{xt:.1f}" y="{BAR_Y+BAR_H+26}" text-anchor="middle" '
               f'font-size="11" fill="#666">{tick}</text>')
svg.append(f'<text x="{X(period_us/2):.1f}" y="{BAR_Y+BAR_H+44}" text-anchor="middle" '
           f'font-size="12" fill="#444">time within one decoder layer (µs)</text>')

for i, (s, d, n) in enumerate(period):
    label, phase = classify(n)
    x = X((s - p0) / 1000.0)
    w = max(1.4, d / 1000.0 / total_us * plot_w)
    svg.append(f'<rect x="{x:.1f}" y="{BAR_Y}" width="{w:.1f}" height="{BAR_H}" '
               f'fill="{COL[phase]}" stroke="black" stroke-width="0.6"/>')
    cx = x + w / 2
    svg.append(f'<text x="{cx:.1f}" y="{BAR_Y-6}" text-anchor="middle" '
               f'font-size="10" fill="#222">{i+1}</text>')
    # leader label, alternating below
    ly = BAR_Y + BAR_H + 60 + (i % 3) * 30
    svg.append(f'<line x1="{cx:.1f}" y1="{BAR_Y+BAR_H}" x2="{cx:.1f}" y2="{ly-11:.1f}" '
               f'stroke="#bbb" stroke-width="0.6"/>')
    bold = ' font-weight="bold"' if phase == "oscar" else ''
    fill = COL["oscar"] if phase == "oscar" else "#222"
    svg.append(f'<text x="{cx:.1f}" y="{ly:.1f}" text-anchor="middle" font-size="10" '
               f'fill="{fill}"{bold}>{escape(label)}</text>')
    svg.append(f'<text x="{cx:.1f}" y="{ly+12:.1f}" text-anchor="middle" font-size="9" '
               f'fill="#777">{d/1e3:.1f}µs</text>')

# legend
LEG = [("proj", "Dense GEMM (model — not OSCAR)"),
       ("attn", "HP-window attention"),
       ("oscar", "OSCAR-specific (rotation / INT2 / merge)"),
       ("norm", "Norm"), ("misc", "RoPE / store / misc")]
lx, ly = ML, 78
for key, txt in LEG:
    svg.append(f'<rect x="{lx}" y="{ly}" width="16" height="16" fill="{COL[key]}" '
               f'stroke="black" stroke-width="0.6"/>')
    svg.append(f'<text x="{lx+22}" y="{ly+13}" font-size="12" fill="#333">{escape(txt)}</text>')
    lx += 30 + len(txt) * 7.0

# ---- footnote --------------------------------------------------------------
osc = sum(d for s, d, n in period if classify(n)[1] == "oscar") / 1000.0
tot = sum(d for s, d, n in period) / 1000.0
svg.append(f'<text x="{ML}" y="{H-40}" font-size="12" fill="#333">'
           f'Per layer: total GPU busy ≈ {tot:.0f} µs · '
           f'OSCAR-specific kernels (red) ≈ {osc:.0f} µs ({osc/tot*100:.0f}%) across '
           f'{sum(1 for s,d,n in period if classify(n)[1]=="oscar")} extra kernels.</text>')
svg.append(f'<text x="{ML}" y="{H-22}" font-size="12" fill="#333">'
           f'Key inefficiency: OSCAR splits ONE attention into 3 kernels '
           f'(HP-window + INT2-KV + merge) and adds Q/K rotation GEMMs — '
           f'more kernel launches vs. a single fused BF16 decode attention.</text>')

svg.append('</svg>')
open(OUT, "w").write("\n".join(svg))
print("wrote", OUT, f"({os.path.getsize(OUT)} bytes)")
print(f"period={period_us:.1f}us  kernels={len(period)}  "
      f"oscar_time={osc:.1f}us/{tot:.1f}us")
