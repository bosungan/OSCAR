#!/usr/bin/env python
# ============================================================================
# plot_decode_scaling.py  (dependency-free, emits SVG)
# OSCAR vs GEMM decode-time scaling with context length (b=1), from the
# server-path DECODE traces. Data source: profiling/server/DECODE_SCALING_ANALYSIS.md
# (graph ON, Qwen3-8B, INT2 KV). Two panels:
#   (A) share of decode GPU time: OSCAR% vs GEMM% with the ~200k crossover
#   (B) absolute ms: GEMM flat (weight-bound) vs OSCAR/INT2-attn linear (KV read)
# Output: figures/decode_oscar_vs_seq.svg
# ============================================================================
import os
from html import escape

# seq in K tokens, evenly spaced on log2 (each ×2)
SEQ   = [16, 32, 64, 128, 256, 512]
OSCARp= [10.9, 16.5, 25.3, 38.9, 53.3, 69.4]
GEMMp = [85.3, 80.0, 71.8, 58.6, 44.9, 29.5]
OSCARm= [2.79, 4.50, 7.67, 14.48, 25.78, 51.18]
GEMMm = [21.84, 21.84, 21.78, 21.80, 21.75, 21.72]
INT2m = [1.82, 3.45, 6.47, 12.92, 23.66, 47.65]
BUSYm = [25.6, 27.3, 30.4, 37.2, 48.4, 73.7]
RED, BLUE, GRAY, DGRAY = "#d1495b", "#4878cf", "#9e9e9e", "#555"
N = len(SEQ)

W, H = 1040, 760
ML, MR = 80, 40
plot_w = W - ML - MR
def XI(i): return ML + i / (N - 1) * plot_w          # even log2 spacing

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'font-family="DejaVu Sans, Arial, sans-serif" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>']
svg.append(f'<text x="{W/2}" y="30" text-anchor="middle" font-size="19" font-weight="bold">'
           f'OSCAR INT2 decode: where the time goes vs context length (Qwen3-8B, b=1, graph ON)</text>')

def xaxis(y):
    svg.append(f'<line x1="{ML}" y1="{y}" x2="{ML+plot_w}" y2="{y}" stroke="#999"/>')
    for i, s in enumerate(SEQ):
        svg.append(f'<line x1="{XI(i):.1f}" y1="{y}" x2="{XI(i):.1f}" y2="{y+5}" stroke="#999"/>')
        svg.append(f'<text x="{XI(i):.1f}" y="{y+20}" text-anchor="middle" font-size="12" fill="#555">{s}k</text>')

def line(vals, ymap, color, dash=""):
    pts = " ".join(f"{XI(i):.1f},{ymap(v):.1f}" for i, v in enumerate(vals))
    da = f' stroke-dasharray="{dash}"' if dash else ""
    svg.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.4"{da}/>')
    for i, v in enumerate(vals):
        svg.append(f'<circle cx="{XI(i):.1f}" cy="{ymap(v):.1f}" r="3.5" fill="{color}"/>')

# ---- Panel A: percentage share + crossover --------------------------------
AY0, AY1 = 90, 320            # top, bottom of plot area
def yA(p): return AY1 - p / 100.0 * (AY1 - AY0)
svg.append(f'<text x="{ML}" y="{AY0-14}" font-size="14" font-weight="bold" fill="#333">A · Share of decode GPU time (%)</text>')
for g in (0, 25, 50, 75, 100):
    svg.append(f'<line x1="{ML}" y1="{yA(g):.1f}" x2="{ML+plot_w}" y2="{yA(g):.1f}" stroke="#eee"/>')
    svg.append(f'<text x="{ML-8}" y="{yA(g)+4:.1f}" text-anchor="end" font-size="11" fill="#888">{g}</text>')
# crossover: OSCAR%==GEMM% between 128k(i=3) and 256k(i=4)
import bisect
def crossfrac():
    # linear interp on the index axis where OSCARp-GEMMp = 0
    for i in range(N-1):
        d0 = OSCARp[i]-GEMMp[i]; d1 = OSCARp[i+1]-GEMMp[i+1]
        if d0 < 0 <= d1:
            f = -d0/(d1-d0); return i+f, (OSCARp[i]+f*(OSCARp[i+1]-OSCARp[i]))
    return None
cx, cy = crossfrac()
cxx = ML + cx/(N-1)*plot_w
svg.append(f'<line x1="{cxx:.1f}" y1="{AY0}" x2="{cxx:.1f}" y2="{AY1}" stroke="#333" stroke-width="1" stroke-dasharray="4,3"/>')
svg.append(f'<text x="{cxx:.1f}" y="{AY0-1}" text-anchor="middle" font-size="12" font-weight="bold" fill="#333">crossover ≈ 200k</text>')
xaxis(AY1)
line(GEMMp, yA, BLUE)
line(OSCARp, yA, RED)
# value labels
for i,(o,g) in enumerate(zip(OSCARp,GEMMp)):
    svg.append(f'<text x="{XI(i):.1f}" y="{yA(o)-8:.1f}" text-anchor="middle" font-size="10" fill="{RED}" font-weight="bold">{o:.0f}</text>')
    svg.append(f'<text x="{XI(i):.1f}" y="{yA(g)+15:.1f}" text-anchor="middle" font-size="10" fill="{BLUE}" font-weight="bold">{g:.0f}</text>')
# legend A — placed in the top-right header band, above the plot (no overlap with lines/crossover label)
LX = ML + plot_w - 300
svg.append(f'<rect x="{LX}" y="46" width="12" height="12" fill="{RED}"/><text x="{LX+16}" y="56" font-size="12" fill="#333">OSCAR (rotation+INT2+HP+merge+bookkeep)</text>')
svg.append(f'<rect x="{LX}" y="64" width="12" height="12" fill="{BLUE}"/><text x="{LX+16}" y="74" font-size="12" fill="#333">GEMM (QKV/o/gate-up/down/LM-head)</text>')

# ---- Panel B: absolute ms --------------------------------------------------
BY0, BY1 = 440, 670
YMAX = 80
def yB(m): return BY1 - m / YMAX * (BY1 - BY0)
svg.append(f'<text x="{ML}" y="{BY0-14}" font-size="14" font-weight="bold" fill="#333">B · Absolute decode-step GPU time (ms)</text>')
for g in (0, 20, 40, 60, 80):
    svg.append(f'<line x1="{ML}" y1="{yB(g):.1f}" x2="{ML+plot_w}" y2="{yB(g):.1f}" stroke="#eee"/>')
    svg.append(f'<text x="{ML-8}" y="{yB(g)+4:.1f}" text-anchor="end" font-size="11" fill="#888">{g}</text>')
xaxis(BY1)
line(BUSYm, yB, DGRAY, dash="2,2")
line(GEMMm, yB, BLUE)
line(OSCARm, yB, RED)
line(INT2m, yB, RED, dash="5,3")
# annotate GEMM flat + OSCAR linear
svg.append(f'<text x="{XI(N-1):.1f}" y="{yB(GEMMm[-1])+16:.1f}" text-anchor="end" font-size="11" fill="{BLUE}">GEMM ≈ 21.8 ms (flat = weight-bound)</text>')
svg.append(f'<text x="{XI(N-1):.1f}" y="{yB(OSCARm[-1])-8:.1f}" text-anchor="end" font-size="11" fill="{RED}" font-weight="bold">OSCAR {OSCARm[-1]:.0f} ms</text>')
svg.append(f'<text x="{XI(N-1):.1f}" y="{yB(INT2m[-1])+14:.1f}" text-anchor="end" font-size="10" fill="{RED}">INT2 attn (∝ seq, drives OSCAR)</text>')
svg.append(f'<text x="{XI(N-2):.1f}" y="{yB(BUSYm[-2])-6:.1f}" text-anchor="middle" font-size="10" fill="{DGRAY}">total busy</text>')

# ---- footnotes -------------------------------------------------------------
svg.append(f'<text x="{ML}" y="{H-44}" font-size="12" fill="#333" font-weight="bold">'
           f'Decode bottleneck shifts weight-GEMM → KV-attention at ~200k: GEMM time is constant (weight-bound), '
           f'OSCAR grows ∝ seq (INT2 attn reads the whole KV).</text>')
svg.append(f'<text x="{ML}" y="{H-26}" font-size="12" fill="#333">'
           f'OSCAR-specific OVERHEAD (rotation+merge+HP) stays a flat ~0.8 ms/step (≈1% → 0.4% of decode); '
           f'the growth is the KV read, fundamental to any KV-cache method (8× larger in BF16).</text>')
svg.append(f'<text x="{ML}" y="{H-9}" font-size="10.5" fill="#888">'
           f'b=1 · graph ON · source: profiling/server/DECODE_SCALING_ANALYSIS.md · x-axis is log2 (each tick ×2)</text>')
svg.append('</svg>')

OUT = os.path.join(os.path.dirname(__file__), "decode_oscar_vs_seq.svg")
open(OUT, "w").write("\n".join(svg))
print("wrote", OUT, f"| crossover at index {cx:.2f} (~{16*2**cx:.0f}k), {cy:.0f}%")
