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

---

# OSCAR decode kernel-time scaling vs BATCH (seq=16k)

Same categorization, but now sweeping **batch size at fixed seq=16k** (graph **OFF**,
4 decode steps). Ratios are GPU-busy-time based, so graph on/off is comparable
(kernel durations identical; only host launch gaps differ, which we don't sum).

## 5. Three-way split vs batch

| batch | decode busy (ms) | **OSCAR %** | **GEMM %** | OTHER % | OSCAR (ms) | GEMM (ms) |
|----:|-----:|-----:|-----:|-----:|-----:|-----:|
| 1  | 100.5 | **10.2%** | 86.6% | 3.3% | 10.2 | 87.0 |
| 4  | 111.9 | **18.8%** | 77.6% | 3.7% | 21.0 | 86.8 |
| 16 | 168.1 | **44.1%** | 53.2% | 2.7% | 74.2 | 89.4 |
| 32 | 248.7 | **60.9%** | 37.1% | 2.0% | 151.4 | 92.2 |

**Read-out:**
1. **GEMM absolute time is ~CONSTANT (87 → 92 ms) across B=1→32** — even though 32× more
   tokens are processed. b=1 decode is already weight-bound, so adding batch just
   **amortizes the same weight read** over more tokens (per-token GEMM: **21,749 →
   720 µs/token, a 30× drop**). This is the textbook batching win.
2. **OSCAR grows ~linearly with batch** (10 → 151 ms): each sequence's attention is
   independent.
3. **OSCAR% rises 10% → 61%; crossover (OSCAR > GEMM) ≈ batch 20** — *at just 16k
   context*. Compare: at b=1 the same crossover needs **~200k seq**. So **batch 32 @
   16k ≈ b=1 @ 512k** in OSCAR dominance — batching reaches the attention-bound
   regime far more cheaply than context length does.

## 6. OSCAR sub-breakdown vs batch (ms, summed over 4 steps)

| OSCAR sub-kernel | B=1 | B=4 | B=16 | B=32 | scaling |
|---|---:|---:|---:|---:|---|
| **INT2 attn** | 6.49 | 16.31 | 65.56 | **137.43** | **∝ batch** (per-seq KV read) |
| HP-window attn | 0.92 | 1.63 | 5.09 | 9.87 | grows w/ batch (per-seq window) |
| merge | 0.81 | 0.98 | 1.18 | 2.00 | grows slowly |
| **Q/K rotation GEMM** | 1.42 | 1.47 | 1.64 | **1.32** | **FLAT** (≈1.4 ms, batch-invariant) |
| KV bookkeep/flush | 0.58 | 0.59 | 0.68 | 0.76 | ~flat |

**Read-out (key for the rotation-overhead question):** even under batching — the regime
where the hypothesis could bite — the **Q/K rotation GEMM stays a flat ~1.4 ms**, so as a
fraction it *shrinks* (1.4% at B=1 → **0.5% at B=32**). What grows is the **INT2 attn (KV
read) + HP-window attn**, i.e. the *attention* work, not the rotation. The OSCAR-specific
machinery beyond plain INT2 (rotation + HP-window + merge) totals ~13 ms ≈ **5% at B=32**,
and the rotation portion of that is negligible and batch-invariant.

## 7. Combined picture (both axes)

Decode reaches **attention(OSCAR)-bound** via EITHER long context (b=1, ~200k) OR
moderate batch (B≈20 at 16k). In both axes:
- **GEMM = weight-bound, flat** (per-token cost drops with batch).
- **OSCAR growth = INT2/HP attention (KV read)** — fundamental to any KV-cache method.
- **Rotation overhead = small, flat** in BOTH seq and batch → the "rotation induces
  inefficiency" hypothesis is **not supported**; the rotation tax is ≤1.5% and shrinking.

Figures: `figures/decode_oscar_vs_seq.svg` (context axis) and
`figures/decode_oscar_vs_batch.svg` (batch axis).
