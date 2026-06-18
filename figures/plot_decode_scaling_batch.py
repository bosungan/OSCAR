#!/usr/bin/env python
# ============================================================================
# plot_decode_scaling_batch.py  (dependency-free, emits SVG)
# OSCAR vs GEMM decode-time scaling with BATCH (seq=16k fixed), companion to
# plot_decode_scaling.py (which sweeps context). Data: profiling/server batch
# sweep (b=1/4/16/32 @ seq16k, graph OFF, 4 decode steps).
#   (A) share of decode GPU time: OSCAR% vs GEMM% with the crossover ≈ B 20
#   (B) absolute ms: GEMM flat (weight amortized) vs OSCAR/INT2 ∝ batch
# Output: figures/decode_oscar_vs_batch.svg
# ============================================================================
import os, math
from html import escape

B     = [1, 4, 16, 32]                       # placed at log2(B) on x
OSCARp= [10.2, 18.8, 44.1, 60.9]
GEMMp = [86.6, 77.6, 53.2, 37.1]
OSCARm= [10.22, 20.98, 74.16, 151.38]
GEMMm = [87.00, 86.81, 89.37, 92.24]
INT2m = [6.49, 16.31, 65.56, 137.43]
BUSYm = [100.49, 111.88, 168.06, 248.68]
RED, BLUE, GRAY, DGRAY = "#d1495b", "#4878cf", "#9e9e9e", "#555"
LOG = [math.log2(b) for b in B]; XMAX = LOG[-1]

W, H = 1040, 760
ML, MR = 80, 40
plot_w = W - ML - MR
def XL(lg): return ML + lg / XMAX * plot_w
def XI(i): return XL(LOG[i])

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'font-family="DejaVu Sans, Arial, sans-serif" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>']
svg.append(f'<text x="{W/2}" y="30" text-anchor="middle" font-size="19" font-weight="bold">'
           f'OSCAR INT2 decode: where the time goes vs BATCH SIZE (Qwen3-8B, seq=16k, graph OFF)</text>')

def xaxis(y):
    svg.append(f'<line x1="{ML}" y1="{y}" x2="{ML+plot_w}" y2="{y}" stroke="#999"/>')
    for i, b in enumerate(B):
        svg.append(f'<line x1="{XI(i):.1f}" y1="{y}" x2="{XI(i):.1f}" y2="{y+5}" stroke="#999"/>')
        svg.append(f'<text x="{XI(i):.1f}" y="{y+20}" text-anchor="middle" font-size="12" fill="#555">{b}</text>')
    svg.append(f'<text x="{ML+plot_w/2:.1f}" y="{y+38}" text-anchor="middle" font-size="12" fill="#444">batch size (log scale)</text>')

def line(vals, ymap, color, dash=""):
    pts = " ".join(f"{XI(i):.1f},{ymap(v):.1f}" for i, v in enumerate(vals))
    da = f' stroke-dasharray="{dash}"' if dash else ""
    svg.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.4"{da}/>')
    for i, v in enumerate(vals):
        svg.append(f'<circle cx="{XI(i):.1f}" cy="{ymap(v):.1f}" r="3.5" fill="{color}"/>')

# ---- Panel A: % share + crossover ----
AY0, AY1 = 90, 320
def yA(p): return AY1 - p / 100.0 * (AY1 - AY0)
svg.append(f'<text x="{ML}" y="{AY0-14}" font-size="14" font-weight="bold" fill="#333">A · Share of decode GPU time (%)</text>')
for g in (0, 25, 50, 75, 100):
    svg.append(f'<line x1="{ML}" y1="{yA(g):.1f}" x2="{ML+plot_w}" y2="{yA(g):.1f}" stroke="#eee"/>')
    svg.append(f'<text x="{ML-8}" y="{yA(g)+4:.1f}" text-anchor="end" font-size="11" fill="#888">{g}</text>')
# crossover on log2 axis where OSCAR%-GEMM%==0 (between B16 and B32)
for i in range(len(B)-1):
    d0=OSCARp[i]-GEMMp[i]; d1=OSCARp[i+1]-GEMMp[i+1]
    if d0<0<=d1:
        f=-d0/(d1-d0); clg=LOG[i]+f*(LOG[i+1]-LOG[i]); cb=2**clg; break
cxx=XL(clg)
svg.append(f'<line x1="{cxx:.1f}" y1="{AY0}" x2="{cxx:.1f}" y2="{AY1}" stroke="#333" stroke-width="1" stroke-dasharray="4,3"/>')
svg.append(f'<text x="{cxx:.1f}" y="{AY0-1}" text-anchor="middle" font-size="12" font-weight="bold" fill="#333">crossover ≈ B {cb:.0f}</text>')
xaxis(AY1)
line(GEMMp, yA, BLUE); line(OSCARp, yA, RED)
for i,(o,g) in enumerate(zip(OSCARp,GEMMp)):
    svg.append(f'<text x="{XI(i):.1f}" y="{yA(o)-8:.1f}" text-anchor="middle" font-size="10" fill="{RED}" font-weight="bold">{o:.0f}</text>')
    svg.append(f'<text x="{XI(i):.1f}" y="{yA(g)+15:.1f}" text-anchor="middle" font-size="10" fill="{BLUE}" font-weight="bold">{g:.0f}</text>')
# legend top-right header band
LX = ML + plot_w - 300
svg.append(f'<rect x="{LX}" y="46" width="12" height="12" fill="{RED}"/><text x="{LX+16}" y="56" font-size="12" fill="#333">OSCAR (rotation+INT2+HP+merge+bookkeep)</text>')
svg.append(f'<rect x="{LX}" y="64" width="12" height="12" fill="{BLUE}"/><text x="{LX+16}" y="74" font-size="12" fill="#333">GEMM (QKV/o/gate-up/down/LM-head)</text>')

# ---- Panel B: absolute ms ----
BY0, BY1 = 440, 670
YMAX = 260
def yB(m): return BY1 - m / YMAX * (BY1 - BY0)
svg.append(f'<text x="{ML}" y="{BY0-14}" font-size="14" font-weight="bold" fill="#333">B · Absolute decode GPU time (ms, summed over 4 steps)</text>')
for g in (0, 50, 100, 150, 200, 250):
    svg.append(f'<line x1="{ML}" y1="{yB(g):.1f}" x2="{ML+plot_w}" y2="{yB(g):.1f}" stroke="#eee"/>')
    svg.append(f'<text x="{ML-8}" y="{yB(g)+4:.1f}" text-anchor="end" font-size="11" fill="#888">{g}</text>')
xaxis(BY1)
line(BUSYm, yB, DGRAY, dash="2,2")
line(GEMMm, yB, BLUE)
line(OSCARm, yB, RED)
line(INT2m, yB, RED, dash="5,3")
svg.append(f'<text x="{XI(len(B)-1):.1f}" y="{yB(GEMMm[-1])-8:.1f}" text-anchor="end" font-size="11" fill="{BLUE}">GEMM ≈ 90 ms (flat = weight read amortized)</text>')
svg.append(f'<text x="{XI(len(B)-1):.1f}" y="{yB(OSCARm[-1])-8:.1f}" text-anchor="end" font-size="11" fill="{RED}" font-weight="bold">OSCAR {OSCARm[-1]:.0f} ms (∝ batch)</text>')
svg.append(f'<text x="{XI(len(B)-1):.1f}" y="{yB(INT2m[-1])+14:.1f}" text-anchor="end" font-size="10" fill="{RED}">INT2 attn (drives OSCAR)</text>')
svg.append(f'<text x="{XI(len(B)-2):.1f}" y="{yB(BUSYm[-2])-6:.1f}" text-anchor="middle" font-size="10" fill="{DGRAY}">total busy</text>')

# ---- footnotes ----
svg.append(f'<text x="{ML}" y="{H-44}" font-size="12" fill="#333" font-weight="bold">'
           f'Batching shifts decode weight-GEMM → KV-attention at just B≈20 (seq 16k): GEMM absolute time is flat '
           f'(per-token GEMM drops 30×), OSCAR grows ∝ batch.</text>')
svg.append(f'<text x="{ML}" y="{H-26}" font-size="12" fill="#333">'
           f'Q/K rotation GEMM stays a flat ~1.4 ms even under batch (1.4% → 0.5% of decode); '
           f'the growth is INT2/HP attention (KV read), not rotation. → "rotation inefficiency" not supported.</text>')
svg.append(f'<text x="{ML}" y="{H-9}" font-size="10.5" fill="#888">'
           f'seq=16k · graph OFF · companion to figures/decode_oscar_vs_seq.svg · source: profiling/server/DECODE_SCALING_ANALYSIS.md §5-7</text>')
svg.append('</svg>')

OUT = os.path.join(os.path.dirname(__file__), "decode_oscar_vs_batch.svg")
open(OUT, "w").write("\n".join(svg))
print("wrote", OUT, f"| crossover ≈ B {cb:.1f}")
