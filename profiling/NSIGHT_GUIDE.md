# Dissecting OSCAR prefill/decode with Nsight (long-context, single batch)

Goal: for an arbitrary fixed shape `(INPUT_LEN, OUTPUT_LEN)`, capture **one prefill** and the
**steady-state decode** under Nsight and read where time goes — to test the hypothesis that at
long context the **KV-cache read** (not weights) dominates. Single batch, you pick the shape.
Tool: `sglang.bench_one_batch` + `nsys`. No serving.

---

## 0. Why long context is the interesting regime (the byte math)

Per-token KV bytes for Qwen3-8B (num_kv_heads=8, head_dim=128, layers=36, K+V):
- **BF16**: `8·128·2·36·2 B ≈ 144 KB/token`  → at 128K ctx = **18 GB > 16 GB weights**.
- **INT2** (÷8, +scales): `≈ 18-20 KB/token` → at 128K ctx = **~2.4 GB ≪ 16 GB weights**.

So your intuition ("KV > weights at long context") holds for **BF16**, but OSCAR's INT2 KV is so
small that the *bytes* only overtake weights around **~900K tokens**. The point of profiling is to
find where the **decode attention kernel time** (which reads all past KV and does inline dequant +
the HP/INT2 split) starts to rival the constant weight-GEMM time — that crossover is the real
"KV-bound" onset for OSCAR, and it can arrive earlier than the raw byte math suggests because the
INT2 attention kernel does extra work per KV element.

---

## 1. Run it

```bash
# args: INPUT_LEN OUTPUT_LEN [GPU] [TAG]
bash profiling/profile_oscar.sh 4096   64 3 ctx4k     # short baseline
bash profiling/profile_oscar.sh 32768  64 3 ctx32k
bash profiling/profile_oscar.sh 131072 32 3 ctx128k   # long context
```
Each call writes, under `profiling/traces/`:
`oscar_<TAG>_prefill.nsys-rep`, `oscar_<TAG>_decode.nsys-rep` (+ `.log` with latency).

Key knobs (prepend as env):
- `GRAPH=off` (default) → **eager**, so *every* kernel appears in the trace (use this to SEE ops).
  `GRAPH=on` → realistic latency (CUDA-graph replay hides launch overhead) but graph-internal
  kernels don't itemize in the kernel summary. Run both; compare.
- `MEM_FRAC=0.6` → raise if the KV pool can't hold `INPUT_LEN+OUTPUT_LEN` (log prints
  `max_total_num_tokens`; it must exceed your context). At 0.6 you get ~450K INT2 tokens.
- `CHUNKED_PREFILL=4096` → prefill runs in 4096-token chunks (you'll see the prefill kernels
  repeat per chunk). Set `CHUNKED_PREFILL=$INPUT_LEN` to force a **single-shot** prefill.
- `MAX_RUNNING=1` → single sequence; keeps the BF16 HP sink/recent pools tiny.
- `QUANT=oscar|plain_int2|bf16` → A/B the same shape against baselines.
- `PROFILE_STEPS=8` → number of decode steps captured.

Long-context note: `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` and `--context-length` are set by
the script so you can push past the model's native window. Outputs are garbage at that point —
fine, we only want realistic compute/memory traffic.

---

## 2. Read it — without the GUI

```bash
bash profiling/report.sh profiling/traces/oscar_ctx128k_decode.nsys-rep \
                         profiling/traces/oscar_ctx128k_prefill.nsys-rep
```
This prints, per trace: GPU kernel time by op (`cuda_gpu_kern_sum`) and the CPU API summary
(`cuda_api_sum`: `cudaLaunchKernel`, `cudaDeviceSynchronize`, `cudaGraphLaunch`).

Most useful single command — the **per-instance timeline** of the decode attention kernel, to watch
it grow with context:
```bash
nsys stats --report cuda_gpu_trace --format table profiling/traces/oscar_ctx128k_decode.nsys-rep \
  | grep _fwd_grouped_kernel_stage1_quant_int2
```

## 2b. Read it — in the Nsight Systems GUI (the visual timeline you want)
```bash
nsys-ui profiling/traces/oscar_ctx128k_decode.nsys-rep   # needs a display (X-forward or copy to laptop)
```
- The **CUDA HW > Kernels** row is the GPU timeline. Zoom to one decode step (between two
  `cudaDeviceSynchronize`); you'll see the per-layer pattern repeat 36×.
- Hover a kernel for exact duration; box-select a step and use the **"Stats System View"** /
  right-click "Show in Events View" to get per-region kernel sums.
- Look for **gaps** between kernels (CPU launch-bound) vs back-to-back kernels (GPU-bound).

---

## 3. Map kernel names → OSCAR ops

| Kernel substring in trace | What it is | OSCAR-only? |
|---|---|---|
| `ampere_*gemm*`, `cutlass*gemm*`, `gemv2T` | model weight matmuls (qkv/o/gate-up/down) + LM head | no (fundamental) |
| `cutlass*flash*` (prefill) | FA3 prefill attention | no |
| `_fwd_grouped_kernel_stage1_quant_int2` | **decode INT2 attention + inline 2-bit dequant** (reads KV body) | yes |
| `_fwd_grouped_kernel_stage1` | decode HP(BF16) attention over sink+recent window | yes |
| `_fwd_kernel_stage2_unified` | LSE merge of HP+INT2 splits | yes |
| `_pretransformed_int2_set_kv_clip_single_kernel` | prefill/decode INT2 clip+quant+pack store | yes |
| `_kv_oscar_rotate_k_clip_single_kernel` | fused K-rotate+clip+quant+pack store | yes |
| small `cutlass_*wmma*128` (decode) | Q@R_k / O@R_v.T rotation GEMM | yes |
| `_count_mixed_hp_lens` / `_scatter_mixed_kv_indices` | per-step HP/INT2 tier classification | yes |
| `_flush_plan_kernel` / `_fused_flush_quant_kernel` / `_flush_remap_kernel` | recent→INT2 flush (every N_Q steps) | yes |
| `store_kvcache`, `fused_rope`, `fused_qknorm`, `rmsnorm`, `act_and_mul` | stock per-layer ops | no |
| `DeviceRadixSort`/`DeviceScan`/`DeviceSelect`/`reduce` | torch ops behind classification/flush/kv-splits | yes (driven by mixed-KV) |

(Full kernel catalog + source lines: see the per-file inventory in
`sglang-research/python/sglang/QuantKernel/` and `mem_cache/kv_quant_kernels.py`.)

---

## 4. The experiment that answers your hypothesis

Fix `OUTPUT_LEN`, sweep `INPUT_LEN` ∈ {4096, 16384, 65536, 131072(, 262144)}. For each, from the
**decode** trace record:
- `T_weight`  = Σ time of `ampere_*gemm*` + `cutlass*gemm*` + `gemv2T`  (≈ constant per step)
- `T_kv_attn` = `_fwd_grouped_kernel_stage1_quant_int2` + `_fwd_grouped_kernel_stage1` + `_fwd_kernel_stage2_unified`  (grows with ctx)
- `T_bookkeep`= classification + flush + sort/scan/select  (grows with ctx)

Then plot each vs `INPUT_LEN` and find the **crossover** where `T_kv_attn` overtakes `T_weight`.
Compare `QUANT=oscar` vs `QUANT=bf16` (same shape): BF16 KV is 8× larger, so its attention should
cross over much earlier — quantifying exactly what INT2 buys at long context, and how much of the
saving the OSCAR machinery (dequant + split + bookkeeping) gives back.

Use `GRAPH=on` for the headline latency numbers and `GRAPH=off` for the per-op attribution.

---

## 5. Gotchas
- **Eager vs graph**: `GRAPH=off` adds CPU launch overhead (≈ +25–30% decode wall) but is the only
  way to itemize graph-internal kernels. Never compare an eager latency to a graphed one.
- **First decode step is an outlier** (includes capture/flush warmup). The bench reports a median;
  for the trace, skip step 0 (use `--profile-start-step` via PROFILE_STEPS / ignore the first).
- **KV pool sizing**: if boot fails or truncates, the log line `max_total_num_tokens=` must exceed
  your `INPUT_LEN+OUTPUT_LEN`; raise `MEM_FRAC`.
- **Chunked prefill**: default 4096 → long prefill is several chunked forwards (realistic). Set
  `CHUNKED_PREFILL=$INPUT_LEN` to see one big prefill instead.
- **nsys overhead**: keep `PROFILE_STEPS` small (8–16); the `--capture-range=cudaProfilerApi`
  ensures only the chosen phase is recorded.
- **GPU choice**: profile on a fully-idle GPU (`nvidia-smi`); a shared GPU perturbs timings and the
  weight-GEMM (bandwidth) numbers especially.
