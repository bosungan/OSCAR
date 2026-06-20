# Is OSCAR's INT2 decode-attention kernel *efficient*? — NCU hardware-counter verdict

Nsight Compute (`--set full`) on `_fwd_grouped_kernel_stage1_quant_int2` (the GQA grouped
INT2 decode kernel), Qwen3-8B, **A6000 (peak BW 768 GB/s)**. Reports under
`profiling/ncu-kernel/reports/`. Per-launch numbers averaged over 3 steady-state layer launches.

> **Why these two configs and not (1,256k)/(32,16k):** NCU must attach to the kernel-launching
> process, so we use the in-process `bench_one_batch` path (the server runs the model in a child
> NCU/nsys can't follow). That path does the whole prefill in ONE forward → it OOMs above ~130k
> tokens of `B*SEQ`, and bench skips `B*SEQ > KV-pool`. So `b16/b32` and `b1@256k` are infeasible
> here (3× OOM / batch-skip confirmed). We profile the **largest fitting proxy of each regime**:
> `b=1 @ 128k` (long-context, GEMV) and `b=8 @ 16k` (batched). NCU counters characterize the
> **kernel shape** (occupancy, coalescing, pipe mix, MBU), which is config-robust — the absolute
> ms at the real configs already come from `profiling/server/`.

## 1. Headline counters

| metric | **b1 · 128k** (GEMV) | **b8 · 16k** (batched) | ideal (well-tuned mem-bound) |
|---|---:|---:|---:|
| **DRAM throughput = MBU** | **26.8 %** | **42.8 %** | 70–85 % |
| achieved BW (= MBU × 768) | **206 GB/s** | **329 GB/s** | 540–650 GB/s |
| Compute (SM) throughput | 34.8 % | 54.2 % | — |
| **Achieved occupancy** | **8.3 %** | 13.8 % | ≥50 % |
| active warps / scheduler (max 12) | **1.00** | 1.67 | ≥8 |
| eligible warps / scheduler | 0.40 | 0.71 | — |
| **grid blocks (device has 84 SMs)** | **64** | 512 | ≥168 (2+ waves) |
| waves per SM | 0.38 | 1.52 | ≥2 |
| registers / thread | 255 | 242 | <128 |
| occupancy limiter | **registers** (block-limit 2) | registers (block-limit 4) | — |
| L1 / L2 hit rate | 32 % / 41 % | 32 % / 39 % | — |
| top pipeline | **ALU 37 %** (int) | **ALU 51 %** (int) | — |
| FP32-peak achieved | ~0 % | ~1 % | — |
| **uncoalesced global loads** | **56 % excess sectors** | 56 % excess | 0 % |
| duration / layer | 423 µs | 272 µs | — |

## 2. Verdict — **yes, the INT2 kernel is bandwidth-inefficient, and NCU says exactly why**

It reaches only **27 % (b=1) / 43 % (b=8) of A6000 peak BW** — a well-tuned memory-bound kernel
hits 70–85 %. Four concrete, compounding causes:

1. **Register-limited occupancy (the dominant cause).** The kernel uses **255 registers/thread**
   (the hardware max is 255) → only **2 thread-blocks fit per SM** → theoretical occupancy 16.7 %,
   **achieved 8.3 % (b=1) / 13.8 % (b=8)**. With only **1.0 active warp per scheduler** (of 12),
   there are nowhere near enough warps in flight to **hide DRAM latency** → the memory pipe sits
   idle → 27 % MBU. This is a *latency-hiding* failure, not a "too many bytes" failure.

2. **Grid underfill at low batch.** At b=1 the kernel launches **64 blocks on 84 SMs (0.38 waves)** —
   it can't even cover the GPU once; ~20 SMs do nothing. NCU's #1 rule: *"grid too small to fill
   the device."* Batching to b=8 → 512 blocks (1.52 waves) is what lifts MBU 27 %→43 % (and
   occupancy 8 %→14 %). So **the b=1 penalty is structural underutilization**, exactly matching the
   roofline finding that b=1 needs ~512k context to look busy.

3. **56 % uncoalesced global loads.** Over **half** the L1TEX load sectors are wasted — the INT2/
   scale KV layout is strided, so each warp pulls partial sectors. This inflates on-chip traffic
   and issue slots, compounding (1)–(2). (DRAM bytes themselves ≈ the roofline model — 26.8 %×768×
   423 µs ≈ **87 MB/layer** vs modeled **84 MB/layer**, so the waste is at L1TEX/issue level, not
   extra DRAM.)

4. **No tensor cores; ALU (integer) is the busiest pipe (37–51 %).** The dequant `(q−zero)·scale`
   runs as integer/FP32 ALU ops on CUDA cores; FP32-peak utilization is ~0 %, and **no Tensor pipe
   appears in the counters at all**. This confirms the **CUDA-core path** for INT2 decode attention —
   the premise of the cross-GPU thesis (§4).

## 3. What this means for the research question

- The growth cost we saw (INT2 attn ∝ seq/batch) is **real KV-read work**, but the kernel executes
  it at **~⅓–½ of the bandwidth it should**. So there is a **genuine ~2–2.6× kernel-efficiency gap**
  on top of the unavoidable KV traffic — and it is an **implementation** gap (occupancy, coalescing),
  not a fundamental one. Concretely fixable: cut register pressure (split the kernel / fewer live
  temps), fix the KV load layout for coalescing, raise grid parallelism (more split-K along context).
- **Batching is the cheap win already visible**: b=1→b=8 nearly doubles MBU (27→43 %) purely by
  filling the grid and adding warps. But it **plateaus** because register pressure still caps
  occupancy at ~14 % — so batching alone won't reach 70 %+. The register/coalescing fixes are the
  ceiling-raisers.

## 4. Link to the cross-GPU (compute-bound) thesis

NCU confirms the load-bearing assumption: **INT2 decode attention runs on CUDA cores, not tensor
cores** (ALU is top pipe, FP32-peak ~0 %, no Tensor counters). Combined with AI ≈ 25.6 FLOP/byte
(see roofline notes), this means on a CUDA-core ridge the op is memory-bound on A6000 (ridge 50.4)
but would cross into **compute(ALU)-bound on A100/H100/B200** (ridge 9.7–20). The dequant ALU work
quantified here (ALU 37–51 % even while memory-bound on A6000) is precisely what would dominate once
a faster-BW GPU removes the memory wall.

## 5. Caveats

- **Config proxy:** b=1@128k and b=8@16k stand in for the server-feasible (1,256k)/(32,16k). Counters
  are shape-robust; absolute ms come from `profiling/server/`. `b16/b32` one-shot prefill OOMs the
  in-process path (3 attempts) — they require the server path, which NCU can't attach to.
- **`--set full`, 3 launches/config**, steady-state layers (NCU `--launch-skip 40`). Tight spread
  across launches (MBU 26.5–27.0 / 42.5–43.0), so 3 is enough.
- **Tensor-core absence is inferred** from ALU being top pipe + FP32-peak ~0 % + no Tensor counters
  in the `--set full` collection (not a direct `=0` read). Consistent with a Triton flash-decode at
  small M.
- NCU locks clocks (base) → absolute µs are NCU-clocked, not wall-clock; **ratios/percentages
  (MBU, occupancy) are what we use** and are clock-independent.

## 6. Reproduce

```bash
bash profiling/ncu-kernel/profile_int2_ncu.sh 1 131072 8 1 b1_128k   # long-ctx
bash profiling/ncu-kernel/profile_int2_ncu.sh 8 16384  8 3 b8_16k    # batched
bash profiling/ncu-kernel/extract_int2_ncu.sh reports/oscar_b1_128k.ncu-rep
```

Figure: `figures/int2_kernel_ncu.svg`.
