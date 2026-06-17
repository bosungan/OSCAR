#!/usr/bin/env python
# ============================================================================
# plot_decode_layer_pattern.py  (dependency-free, emits SVG)
# Per-layer kernel launch sequence of the OSCAR INT2 decode step, extracted from
# a sglang torch-profiler Chrome trace (.trace.json.gz) produced by
#   profiling/server/profile_oscar_server.sh   (server path, b=1, seq=16k)
#
# Reflects GRAPH=ON (CUDA graph replay): per-step kernels run back-to-back (host
# launch gaps removed) = realistic serving layout. Single panel, duration-
# proportional. ALL kernels are listed in the legend; only IMPORTANT kernels
# (OSCAR-specific / attention / big GEMMs) get a number badge on the bar to keep
# the timeline uncluttered.
#
# Usage:  python figures/plot_decode_layer_pattern.py [DECODE.trace.json.gz]
# Output: figures/decode_layer_kernel_pattern.svg
# ============================================================================
import json, gzip, re, sys, os, glob
from html import escape

DEFAULT = sorted(glob.glob(os.path.join(
    os.path.dirname(__file__), "..", "profiling", "server", "traces",
    "*s16384*", "*", "*DECODE.trace.json.gz")))
TRACE = sys.argv[1] if len(sys.argv) > 1 else (DEFAULT[-1] if DEFAULT else None)
OUT = os.path.join(os.path.dirname(__file__), "decode_layer_kernel_pattern.svg")
assert TRACE, "no DECODE trace found"

opener = gzip.open if TRACE.endswith(".gz") else open
ev = json.load(opener(TRACE))["traceEvents"]
ks = sorted((e["ts"], e["dur"], e["name"]) for e in ev
            if e.get("cat") == "kernel" and "ts" in e and "dur" in e)
graph_on = any(e.get("cat") == "cuda_runtime" and "GraphLaunch" in e.get("name", "")
               for e in ev)

# ---- full decode-STEP accounting (figure draws ONE layer; step = 36 layers + per-step work)
_step_span = (max(t + d for t, d, n in ks) - ks[0][0]) / 1000.0
_anch = [t for t, d, n in ks if "stage1_quant_int2" in n]
_per_layer = (_anch[-1] - _anch[0]) / (len(_anch) - 1) / 1000.0
_layers36 = 36 * _per_layer
_lm_head = sum(d for t, d, n in ks if "gemv2T" in n) / 1000.0
_overhead = _step_span - _layers36

def classify(n, d):
    s = n
    if "fused_add_rmsnorm" in s or ("rmsnorm" in s.lower() and "qknorm" not in s.lower()):
        return ("RMSNorm (+residual)", "norm")
    if "fused_qknorm" in s:                      return ("q/k-norm", "norm")
    if "splitKreduce" in s:                      return ("splitK reduce", "proj")
    if "fused_rope" in s:                         return ("RoPE", "misc")
    if "store_kvcache" in s:                      return ("store K/V -> HP win", "misc")
    if "stage1_quant_int2" in s:                  return ("[OSCAR] INT2-KV attn", "oscar")
    if "_fwd_grouped_kernel_stage1" in s:         return ("[OSCAR] HP-window attn", "oscar")
    if "stage2_unified" in s:                     return ("[OSCAR] merge HP+INT2", "oscar")
    if "act_and_mul" in s:                        return ("SwiGLU (silu*up)", "misc")
    if "memcpy" in s.lower():                     return ("memcpy", "misc")
    if "memset" in s.lower():                     return ("memset", "misc")
    if "cutlass::Kernel2" in s or ("wmma" in s and "Kernel2" in s):
        return ("QKV proj GEMM", "proj") if d > 40 else ("[OSCAR] rotate / V-absorb GEMM", "oscar")
    if "ampere" in s and "gemm" in s.lower():
        if d > 200: return ("gate+up proj GEMM", "proj")
        if d > 100: return ("down proj GEMM", "proj")
        return ("o_proj GEMM", "proj")
    if "vectorized_elementwise" in s:             return ("elementwise (idx/init)", "misc")
    return (re.sub(r"<.*", "", n)[:22], "misc")

# ---- one layer via INT2-attn anchor (exactly 36 = 1/layer), rotated to start at input RMSNorm
anchors = [i for i, (t, d, n) in enumerate(ks) if "stage1_quant_int2" in n]
a, b = anchors[1], anchors[2]
wall_us = ks[b][0] - ks[a][0]
seg = ks[a:b]
def is_qkv(n, d): return ("cutlass::Kernel2" in n or ("wmma" in n and "Kernel2" in n)) and d > 40
rot = next((j for j in range(len(seg) - 1)
            if "fused_add_rmsnorm" in seg[j][2] and is_qkv(seg[j + 1][2], seg[j + 1][1])), 0)
seg = seg[rot:] + seg[:rot]

COL = {"norm": "#9e9e9e", "proj": "#4878cf", "attn": "#7fb069",
       "oscar": "#d1495b", "misc": "#cccccc"}
items = []; x = 0.0
for t, d, n in seg:
    lab, cat = classify(n, d)
    items.append((x, d, lab, cat)); x += d
period_us = x; busy = x

# every kernel is numbered 1..N (the legend lists them all); only IMPORTANT ones
# get a number badge drawn on the bar.
def important(d, cat): return cat in ("oscar", "attn") or d > 40

# ---- SVG -------------------------------------------------------------------
W, H = 1500, 480
ML, MR = 60, 40
plot_w = W - ML - MR
PAD = period_us * 0.01
span = period_us + 2 * PAD
def X(us): return ML + (us + PAD) / span * plot_w

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'font-family="DejaVu Sans, Arial, sans-serif" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>']
svg.append(f'<text x="{W/2}" y="28" text-anchor="middle" font-size="19" font-weight="bold">'
           f'OSCAR INT2 decode — per-layer kernel sequence (Qwen3-8B), GRAPH {"ON" if graph_on else "OFF"}</text>')
svg.append(f'<text x="{W/2}" y="49" text-anchor="middle" font-size="13" fill="#444">'
           f'batch 1 · seq 16k · CUDA-graph replay (launch gaps removed) · 1 layer ≈ {wall_us:.0f} µs · '
           f'this figure shows ONE layer (the step repeats it × 36) · bars duration-proportional</text>')

# colour legend
LEG = [("proj", "Dense GEMM (model — not OSCAR)"),
       ("oscar", "OSCAR-specific (rotation / INT2 / HP-window / merge)"), ("norm", "Norm"), ("misc", "RoPE / store / misc")]
lx = ML
for k, txt in LEG:
    svg.append(f'<rect x="{lx}" y="66" width="15" height="15" fill="{COL[k]}" stroke="black" stroke-width="0.6"/>')
    svg.append(f'<text x="{lx+21}" y="78" font-size="12" fill="#333">{escape(txt)}</text>')
    lx += 28 + len(txt) * 7.0

# ---- the bar row (badges only on important kernels, staggered + leaders) ----
BAR_Y, BAR_H = 170, 44
svg.append(f'<line x1="{X(0):.1f}" y1="{BAR_Y+BAR_H+6}" x2="{X(period_us):.1f}" y2="{BAR_Y+BAR_H+6}" stroke="#999"/>')
for tk in range(0, int(period_us) + 1, 100):
    svg.append(f'<line x1="{X(tk):.1f}" y1="{BAR_Y+BAR_H+6}" x2="{X(tk):.1f}" y2="{BAR_Y+BAR_H+11}" stroke="#999"/>')
    svg.append(f'<text x="{X(tk):.1f}" y="{BAR_Y+BAR_H+25}" text-anchor="middle" font-size="11" fill="#777">{tk}</text>')
svg.append(f'<text x="{X(period_us/2):.1f}" y="{BAR_Y+BAR_H+43}" text-anchor="middle" font-size="12" fill="#444">time within one decoder layer (µs) — kernels packed back-to-back under graph replay</text>')

seen = 0
for i, (off, d, label, cat) in enumerate(items):
    gx = X(off); w = max(1.3, d / span * plot_w)
    svg.append(f'<rect x="{gx:.1f}" y="{BAR_Y}" width="{w:.1f}" height="{BAR_H}" fill="{COL[cat]}" stroke="black" stroke-width="0.6"/>')
    if important(d, cat):
        cx = gx + w / 2
        lvl = seen % 3
        ly = BAR_Y - 8 - lvl * 16          # 3 staggered rows above the bar
        col = COL["oscar"] if cat == "oscar" else "#222"
        svg.append(f'<line x1="{cx:.1f}" y1="{BAR_Y}" x2="{cx:.1f}" y2="{ly+3:.1f}" stroke="#bbb" stroke-width="0.5"/>')
        svg.append(f'<text x="{cx:.1f}" y="{ly:.1f}" text-anchor="middle" font-size="12" fill="{col}" font-weight="bold">{i+1}</text>')
        seen += 1

# ---- full step legend: ALL kernels (3 columns) -----------------------------
LEG_Y = 280
svg.append(f'<text x="{ML}" y="{LEG_Y-6}" font-size="13" font-weight="bold" fill="#333">'
           f'Kernel legend (all kernels) — numbered ones are marked on the bar; small misc/norm kernels are bars only</text>')
col_x = [ML, 540, 1010]; rows_per = (len(items) + 2) // 3
for i, (off, d, label, cat) in enumerate(items):
    c = i // rows_per; r = i % rows_per
    gx = col_x[c]; gy = LEG_Y + 12 + r * 19
    badge = important(d, cat)
    bold = ' font-weight="bold"' if cat == "oscar" else ''
    fill = COL["oscar"] if cat == "oscar" else ("#333" if badge else "#888")
    num = f"{i+1}." if badge else "· "          # bullet for un-badged kernels
    svg.append(f'<rect x="{gx}" y="{gy-10}" width="11" height="11" fill="{COL[cat]}" stroke="black" stroke-width="0.5"/>')
    svg.append(f'<text x="{gx+17}" y="{gy}" font-size="12" fill="{fill}"{bold}>{num}  {escape(label)}  ·  {d:.1f}µs</text>')

# ---- footnotes -------------------------------------------------------------
osc = sum(d for _, d, _, c in items if c == "oscar")
svg.append(f'<text x="{ML}" y="{H-52}" font-size="12" fill="#333" font-weight="bold">'
           f'Full decode STEP ≈ {_step_span:.1f} ms = 36 × layer ({_layers36:.1f} ms) + {_overhead:.1f} ms per-step '
           f'(LM head {_lm_head:.2f} ms + sampling/embed) — the LM head runs ONCE/step, not per layer.</text>')
svg.append(f'<text x="{ML}" y="{H-34}" font-size="12" fill="#333">'
           f'Per layer: GPU busy ≈ {busy:.0f} µs ≈ wall {wall_us:.0f} µs (util ~{busy/wall_us*100:.0f}%). '
           f'vs eager ~1032 µs/layer: graph replay removed ~{1032-wall_us:.0f} µs of host launch gaps.</text>')
svg.append(f'<text x="{ML}" y="{H-16}" font-size="12" fill="#333">'
           f'OSCAR-specific kernels (red) ≈ {osc:.0f} µs ({osc/busy*100:.0f}% of busy): rotation GEMMs + INT2 attn + HP-window attn + merge + V-absorb '
           f'— under graph replay, the honest serving cost (no launch overhead).</text>')
svg.append('</svg>')
open(OUT, "w").write("\n".join(svg))
print(f"wrote {OUT}")
print(f"graph_on={graph_on} layer={period_us:.0f}us kernels={len(items)} "
      f"numbered_on_bar={sum(1 for o,d,l,c in items if important(d,c))} "
      f"step={_step_span:.2f}ms lm_head={_lm_head:.2f}ms oscar={osc:.0f}us")
