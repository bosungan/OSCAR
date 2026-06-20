# NCU kernel-level profiling — OSCAR INT2 decode attention

This tier answers a question the other two profilers can't: **is OSCAR's INT2 decode
attention kernel an *efficient* use of the GPU, or is it leaving performance on the table?**

- `profiling/single-batch/` (nsys) → one whole-run timeline (prefill+decode kernels).
- `profiling/server/` (torch profiler) → per-stage GPU-**time** splits (OSCAR/GEMM/other).
- **`profiling/ncu-kernel/` (this) → hardware counters INSIDE one kernel**: achieved DRAM
  bandwidth %, real DRAM bytes, tensor-core vs CUDA-core path, occupancy, warp-stall reasons.

## Why a separate tier (and why bench_one_batch, not the server)

NCU must attach to the process that **launches** the kernel and replay it with counters on.
The server path runs the model in a **child process** (nsys already failed to follow it), so we
use the **in-process** `sglang.bench_one_batch` driver — same one `single-batch/` uses.

**Trade-off / constraint:** `bench_one_batch` bypasses the Scheduler, so `--chunked-prefill-size`
is **ignored** → the whole prefill is ONE forward → activation OOMs at very large `B*SEQ`.
So we cannot directly run the server-feasible `(32,16k)` / `(1,256k)`. Instead we profile the
**largest `(B,SEQ)` that fits one-shot prefill** as a proxy. This is sound because NCU counters
(TC usage, DRAM-BW %, occupancy, stall mix) characterize the **kernel shape**, not wall-time —
they are config-robust. The absolute ms at the big configs already come from `server/`.

## Target kernel

Qwen3-8B is GQA (8 KV / 32 Q heads) → decode uses the **grouped** variant:
`_fwd_grouped_kernel_stage1_quant_int2` (regex `.*stage1_quant_int2.*`).

## Usage

```bash
# smoke (validates permissions + lock + kernel match)
bash profiling/ncu-kernel/profile_int2_ncu.sh 1 4096 8 1 smoke

# the two regimes we care about:
bash profiling/ncu-kernel/profile_int2_ncu.sh 1 <maxfit-seq> 8 1 b1_long   # b=1 long ctx  (GEMV / CUDA-core regime)
bash profiling/ncu-kernel/profile_int2_ncu.sh <maxfit-b> 16384 8 3 bN_16k  # batched       (TC-entry regime)
```

Env knobs: `QUANT` (oscar|plain_int2|bf16), `MEM_FRAC` (lower frees prefill activation room),
`SKIP`/`COUNT` (NCU `--launch-skip`/`--launch-count` over matched int2 launches), `NCU_SET`.

## Gotchas discovered on this box

- **Interprocess lock:** NCU's `/tmp/nsight-compute-lock` is owned by another user (sticky `/tmp`
  → can't recreate) → `InterprocessLockFailed`. **Fixed** by relocating the lock via
  `TMPDIR=/tmp/ncu_$USER` (the script does this). Safe because per-GPU runs use distinct free GPUs.
- **Kernel replay is slow** (`--set full` saves/restores device memory between passes). Keep
  `COUNT` small (2–4). For very large KV, drop to `COUNT=2`.
- **One-shot prefill OOM:** if it OOMs, lower `SEQ`/`BATCH` or `MEM_FRAC` (bench_one_batch ignores
  chunked prefill, so peak activation ~ `B*SEQ` tokens, not the chunk size).

## Extracting the numbers

```bash
bash profiling/ncu-kernel/extract_int2_ncu.sh reports/<name>.ncu-rep
# or open the full report in the Nsight Compute GUI:
ncu-ui reports/<name>.ncu-rep
```

Headline metrics (see `extract_int2_ncu.sh` for the full list):
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` → **MBU** (validates the roofline ~46–48%).
- `dram__bytes.sum` → **real** DRAM bytes (validates the hand-modeled 377 MB/seq byte count).
- `sm__pipe_tensor_op_hmma_cycles_active...` / `sm__inst_executed_pipe_tensor.sum` → **TC vs CUDA-core**
  (0 ⇒ the GEMV runs on CUDA cores ⇒ the cross-GPU compute-bound thesis holds on A100/H100).
- `sm__warps_active...` → achieved occupancy; `launch__occupancy_limit_*` → what caps it.
- `smsp__...stalled_long_scoreboard...` (memory) vs `...short_scoreboard...` (MIO/dequant) vs
  `...math_pipe_throttle...` → **why** BW tops out below peak.

Results + figures live in `INT2_KERNEL_ANALYSIS.md` (written after the runs).
