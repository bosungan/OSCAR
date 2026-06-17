# OSCAR decode kernel-time scaling vs context length (b=1)

How the decode-step GPU time splits across **OSCAR machinery / dense GEMM / others**
as sequence length grows, from the server-path DECODE traces
(`profiling/server/traces/oscar_b1_s*/.../*-DECODE.trace.json.gz`, graph **ON**,
Qwen3-8B, INT2 KV).

Categories:
- **OSCAR** = mixed-KV machinery: INT2 attn + HP-window attn + merge + Q/K rotation GEMMs + KV bookkeeping/flush.
- **GEMM** = dense model matmuls: QKV / o_proj / gate-up / down proj + LM head (`gemv2T`).
- **OTHER** = RMSNorm / q-k-norm / RoPE / SwiGLU / elementwise / sampling.

> Note: the "256k" point is the `s252144` run (≈246k tokens). Numbers are GPU-busy
> time summed over the recorded decode step(s); ratios are within one decode step.

## 1. Three-way split (the headline)

| seq | decode busy (ms) | **OSCAR %** | **GEMM %** | OTHER % | OSCAR (ms) | GEMM (ms) |
|----:|-----:|-----:|-----:|-----:|-----:|-----:|
| 16k  | 25.6 | **10.9%** | 85.3% | 3.9% | 2.79 | 21.84 |
| 32k  | 27.3 | **16.5%** | 80.0% | 3.5% | 4.50 | 21.84 |
| 64k  | 30.4 | **25.3%** | 71.8% | 3.0% | 7.67 | 21.78 |
| 128k | 37.2 | **38.9%** | 58.6% | 2.4% | 14.48 | 21.80 |
| 256k | 48.4 | **53.3%** | 44.9% | 1.8% | 25.78 | 21.75 |
| 512k | 73.7 | **69.4%** | 29.5% | 1.1% | 51.18 | 21.72 |

**Read-out:**
1. **GEMM time is constant ≈ 21.7–21.8 ms** at every seq from 16k to 512k — the
   dense matmuls are **weight-bandwidth bound** (read the 15 GB of weights each
   step), independent of context length.
2. **OSCAR time scales ~linearly with seq** (2.8 → 51.2 ms; 32× seq → ~18× time,
   slightly sublinear).
3. **Crossover ≈ 200k**: OSCAR overtakes GEMM between **128k (39% vs 59%)** and
   **256k (53% vs 45%)**, then keeps climbing to **69% at 512k** (GEMM down to 30%).
   Below ~200k decode is GEMM(weight)-bound; above it, OSCAR(KV-attention)-bound.
4. **OTHER shrinks as a fraction** (3.9% → 1.1%) — small constant overhead diluted
   by growing OSCAR.

## 2. What inside OSCAR grows (sub-breakdown, ms)

| OSCAR sub-kernel | 16k | 32k | 64k | 128k | 256k | 512k | scaling |
|---|---:|---:|---:|---:|---:|---:|---|
| **INT2 attn** (`stage1_quant_int2`) | 1.82 | 3.45 | 6.47 | 12.92 | 23.66 | **47.65** | **∝ seq** (reads whole KV) |
| HP-window attn (`stage1`) | 0.24 | 0.24 | 0.24 | 0.24 | 0.23 | 0.23 | constant (320-tok window) |
| merge (`stage2_unified`) | 0.21 | 0.21 | 0.21 | 0.21 | 0.19 | 0.20 | constant (per-layer) |
| Q/K rotation GEMM | 0.36 | 0.35 | 0.34 | 0.35 | 0.34 | 0.33 | **constant** (per-layer) |
| KV bookkeep / flush | 0.16 | 0.24 | 0.41 | 0.76 | 1.36 | 2.78 | grows mildly (index/page work) |

**Read-out:** OSCAR's entire growth is the **INT2 attention** (it reads the full KV
cache each step, so ∝ context). Everything else in OSCAR — **the Q/K rotation
(~0.35 ms), HP-window attn (~0.24 ms), merge (~0.20 ms) — is a flat per-layer
constant** that does NOT scale with seq. Only the KV-index bookkeeping grows
(modestly, with the number of KV pages).

## 3. Implications for the research question (rotation inefficiency)

- At **b=1**, the **rotation overhead is a small fixed cost** (~0.33–0.36 ms/step,
  ≈1.4% of decode at 16k → **~0.45% at 512k** — it *shrinks* as a fraction as
  context grows). It is **not** the cost that scales — so "rotation makes
  long-context decode slow" is **not** supported at b=1; the rotation tax is
  constant and minor.
- The cost that scales is the **INT2 attention = KV read**, which is fundamental to
  *any* KV-cache attention (it would be even larger in BF16, ~8× the bytes). So
  long-context decode becomes **KV-bound around ~200k**, and OSCAR's INT2 is what
  makes that KV read cheap (2-bit).
- Net: at b=1, the decode bottleneck **shifts from weight-GEMM (short ctx) to
  KV-attention (long ctx) ~200k**; OSCAR-specific *overhead* (rotation/merge/
  bookkeeping) stays small. To expose rotation/quant overhead more, **increase
  batch** (amortizes the weight GEMM, raises attention's share at shorter ctx) or
  compare **OSCAR vs BF16** at matched seq.

## 4. Caveats
- graph **ON** (CUDA-graph replay): kernels still itemized by the torch profiler;
  GPU times are real, launch gaps removed (serving-realistic).
- b=1 only; one decode step (`PROFILE_STEPS`). LM head (`gemv2T`, ~1.66 ms) counts
  under GEMM and is per-step constant.
- Source: `profiling/server/profile_oscar_server.sh` traces; regenerate the table
  by re-parsing the `*-DECODE.trace.json.gz` files with the categories above.
