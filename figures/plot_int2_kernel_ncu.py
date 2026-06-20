#!/usr/bin/env python
# ============================================================================
# plot_int2_kernel_ncu.py  (dependency-free, emits SVG)
# NCU hardware-counter view of OSCAR's INT2 decode-attention kernel
# (_fwd_grouped_kernel_stage1_quant_int2) on A6000 (peak BW 768 GB/s).
# Data: profiling/ncu-kernel/INT2_KERNEL_ANALYSIS.md  (b1@128k GEMV, b8@16k batched).
#   Panel A: utilization % — MBU / Compute / Occupancy, vs the 70-85% ideal MBU band.
#   Panel B: the "why" — scheduler warp fill (of 12) and grid waves/SM (need >=2).
# Output: figures/int2_kernel_ncu.svg
# ============================================================================
import os

RED, BLUE, GRN, GRAY, DK = "#d1495b", "#4878cf", "#3a923a", "#9e9e9e", "#444"
W, H = 1100, 760
ML, MR = 78, 40
pw = W - ML - MR

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'font-family="DejaVu Sans, Arial, sans-serif" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>']
svg.append(f'<text x="{W/2}" y="28" text-anchor="middle" font-size="18" font-weight="bold">'
           f'OSCAR INT2 decode-attention kernel — NCU on A6000: bandwidth-bound at 27–43% of peak</text>')
svg.append(f'<text x="{W/2}" y="48" text-anchor="middle" font-size="12.5" fill="#666">'
           f'_fwd_grouped_kernel_stage1_quant_int2 · Qwen3-8B · per-layer launch (avg of 3) · peak 768 GB/s</text>')

# ----- legend (config colors) -----
LX = ML
svg.append(f'<rect x="{LX}" y="60" width="13" height="13" fill="{RED}"/>'
           f'<text x="{LX+18}" y="71" font-size="12" fill="#333">b=1 · 128k ctx (GEMV / long-context)</text>')
svg.append(f'<rect x="{LX+290}" y="60" width="13" height="13" fill="{BLUE}"/>'
           f'<text x="{LX+308}" y="71" font-size="12" fill="#333">b=8 · 16k ctx (batched)</text>')

# ============ Panel A: utilization % ============
AY0, AY1 = 110, 360
def yA(p): return AY1 - p/100.0*(AY1-AY0)
svg.append(f'<text x="{ML}" y="{AY0-12}" font-size="14" font-weight="bold" fill="#333">'
           f'A · GPU utilization (% of peak) — the kernel leaves &gt;half the bandwidth unused</text>')
for g in (0,25,50,75,100):
    svg.append(f'<line x1="{ML}" y1="{yA(g):.1f}" x2="{ML+pw}" y2="{yA(g):.1f}" stroke="#eee"/>')
    svg.append(f'<text x="{ML-8}" y="{yA(g)+4:.1f}" text-anchor="end" font-size="11" fill="#888">{g}</text>')
# ideal MBU band 70-85
svg.append(f'<rect x="{ML}" y="{yA(85):.1f}" width="{pw}" height="{yA(70)-yA(85):.1f}" '
           f'fill="{GRN}" opacity="0.13"/>')
svg.append(f'<text x="{ML+pw-6}" y="{yA(85)+13:.1f}" text-anchor="end" font-size="11" fill="{GRN}">'
           f'ideal mem-bound kernel: MBU 70–85%</text>')

groups = [("DRAM throughput\n(= MBU)", 26.8, 42.8, True),
          ("Compute (SM)\nthroughput", 34.8, 54.2, False),
          ("Achieved\noccupancy", 8.3, 13.8, False)]
gw = pw/len(groups); bw = 52
for i,(lab,v1,v8,hl) in enumerate(groups):
    cx = ML + gw*i + gw/2
    x1 = cx - bw - 6; x8 = cx + 6
    for x,v,c in ((x1,v1,RED),(x8,v8,BLUE)):
        svg.append(f'<rect x="{x:.1f}" y="{yA(v):.1f}" width="{bw}" height="{AY1-yA(v):.1f}" fill="{c}"/>')
        svg.append(f'<text x="{x+bw/2:.1f}" y="{yA(v)-5:.1f}" text-anchor="middle" font-size="12" '
                   f'font-weight="bold" fill="{c}">{v:.1f}</text>')
    for j,ln in enumerate(lab.split("\n")):
        svg.append(f'<text x="{cx:.1f}" y="{AY1+16+j*13:.1f}" text-anchor="middle" font-size="11.5" fill="#444">{ln}</text>')
# gap arrow on MBU
svg.append(f'<line x1="{ML+gw*0+gw/2:.1f}" y1="{yA(42.8):.1f}" x2="{ML+gw*0+gw/2:.1f}" y2="{yA(75):.1f}" '
           f'stroke="{DK}" stroke-width="1" stroke-dasharray="3,3"/>')
svg.append(f'<text x="{ML+gw*0+gw/2+8:.1f}" y="{yA(60):.1f}" font-size="10.5" fill="{DK}">~2× gap</text>')

# ============ Panel B: WHY — starved occupancy / grid ============
BY0, BY1 = 430, 600
svg.append(f'<text x="{ML}" y="{BY0-12}" font-size="14" font-weight="bold" fill="#333">'
           f'B · Why: too few warps in flight to hide DRAM latency</text>')
def bar(x,y,wmax,frac,c,h=15):
    svg.append(f'<rect x="{x}" y="{y}" width="{wmax}" height="{h}" fill="#eee" stroke="#ccc"/>')
    svg.append(f'<rect x="{x}" y="{y}" width="{wmax*frac:.1f}" height="{h}" fill="{c}"/>')
bx = ML+250; bwmax = 360
rows = [
 ("active warps / scheduler", 1.00/12, 1.67/12, "1.00 / 1.67 of 12 max"),
 ("grid waves per SM",        0.38/2.0, 1.52/2.0, "0.38 / 1.52 (need ≥2)"),
 ("occupancy (achieved)",     8.3/100, 13.8/100, "8.3% / 13.8% (reg-capped 16.7%)"),
]
for i,(lab,f1,f8,note) in enumerate(rows):
    y = BY0 + i*44
    svg.append(f'<text x="{ML}" y="{y+12:.1f}" font-size="12" fill="#333">{lab}</text>')
    bar(bx, y, bwmax, min(f1,1), RED)
    bar(bx, y+18, bwmax, min(f8,1), BLUE)
    svg.append(f'<text x="{bx+bwmax+8}" y="{y+22:.1f}" font-size="10.5" fill="#666">{note}</text>')

# ============ causes box ============
cy = 640
svg.append(f'<text x="{ML}" y="{cy}" font-size="12.5" font-weight="bold" fill="#333">Root causes (NCU rules):</text>')
causes = [
 "1. 255 registers/thread → only 2 blocks/SM → occupancy capped at 16.7% (achieved 8–14%).",
 "2. b=1 grid = 64 blocks &lt; 84 SMs (0.38 waves): can't even fill the GPU once. Batching b1→b8 lifts MBU 27→43%.",
 "3. 56% uncoalesced global loads: strided INT2+scale KV layout wastes &gt;half the L1TEX sectors.",
 "4. No tensor cores — ALU (integer dequant) is the busiest pipe (37–51%), FP32-peak ~0%  → CUDA-core path.",
]
for i,c in enumerate(causes):
    svg.append(f'<text x="{ML}" y="{cy+18+i*17:.1f}" font-size="11.5" fill="#333">{c}</text>')

svg.append(f'<text x="{ML}" y="{H-10}" font-size="10" fill="#999">'
           f'config proxies for server-feasible (1,256k)/(32,16k) — NCU needs in-process bench_one_batch (one-shot prefill OOMs b16/b32). '
           f'source: profiling/ncu-kernel/INT2_KERNEL_ANALYSIS.md</text>')
svg.append('</svg>')

OUT = os.path.join(os.path.dirname(__file__), "int2_kernel_ncu.svg")
open(OUT,"w").write("\n".join(svg))
print("wrote", OUT)
