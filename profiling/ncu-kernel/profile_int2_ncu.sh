#!/usr/bin/env bash
# ============================================================================
# profile_int2_ncu.sh — Nsight Compute (NCU) kernel-level profile of OSCAR's
#                       INT2 decode attention kernel (_fwd_grouped_kernel_stage1_quant_int2).
#
# WHY THIS EXISTS:
#   The server-path profiler (profiling/server) gives PER-STAGE GPU-time splits but
#   NOT kernel-internal counters. To answer "is INT2 attention an *efficient* kernel?"
#   we need NCU's hardware counters: achieved DRAM bandwidth %, real DRAM bytes,
#   tensor-core usage (TC vs CUDA-core), achieved occupancy, and warp-stall reasons.
#
#   NCU must attach to the process that LAUNCHES the kernel. The server runs the model
#   in a CHILD process (nsys already failed to follow it). So we use the IN-PROCESS
#   `sglang.bench_one_batch` path (same one profiling/single-batch uses). Trade-off:
#   bench_one_batch IGNORES --chunked-prefill-size (bypasses the Scheduler) -> the whole
#   prefill is ONE forward -> activation OOMs at very large B*SEQ. So we profile the
#   LARGEST (B,SEQ) that FITS one-shot prefill as a proxy for the user's (32,16k)/(1,256k):
#   kernel counters (TC usage, DRAM%, occupancy, stalls) are config-robust — they
#   characterize the kernel shape, not the absolute wall-time.
#
# TARGET KERNEL: GQA (8 KV / 32 Q heads) -> grouped variant
#   _fwd_grouped_kernel_stage1_quant_int2  (regex: .*stage1_quant_int2.*)
#
# Usage:
#   bash profiling/ncu-kernel/profile_int2_ncu.sh BATCH SEQ [OUTPUT_LEN] [GPU] [TAG]
# Examples:
#   bash profiling/ncu-kernel/profile_int2_ncu.sh 1 4096   8 1 smoke    # smoke test
#   bash profiling/ncu-kernel/profile_int2_ncu.sh 1 131072 8 1 b1_128k  # long-ctx (GEMV regime)
#   bash profiling/ncu-kernel/profile_int2_ncu.sh 8 16384  8 3 b8_16k   # batched (TC-entry regime)
#
# Env overrides (prepend VAR=value):
#   QUANT      oscar | plain_int2 | bf16        (default oscar)
#   MEM_FRAC   static-pool fraction             (default 0.55; lower frees prefill activation room)
#   SKIP       NCU --launch-skip (matched int2 launches to skip past warmup) (default 40)
#   COUNT      NCU --launch-count (int2 kernels to fully profile)            (default 4)
#   NCU_SET    NCU metric set                   (default full)
#   CHUNKED_PREFILL  --chunked-prefill-size (IGNORED by bench_one_batch, kept for parity) (default SEQ)
#
# Output: profiling/ncu-kernel/reports/<QUANT>_<TAG>.ncu-rep (+ .log).
#   Inspect:  ncu -i reports/<...>.ncu-rep --page details        (human)
#             ncu -i reports/<...>.ncu-rep --csv --page raw       (extract)
# ============================================================================
set -uo pipefail

BATCH="${1:?usage: profile_int2_ncu.sh BATCH SEQ [OUTPUT_LEN] [GPU] [TAG]}"
SEQ="${2:?need SEQ}"
OUTPUT_LEN="${3:-8}"
GPU="${4:-1}"
TAG="${5:-b${BATCH}_s${SEQ}}"

QUANT="${QUANT:-oscar}"
MEM_FRAC="${MEM_FRAC:-0.55}"
SKIP="${SKIP:-40}"
COUNT="${COUNT:-4}"
NCU_SET="${NCU_SET:-full}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-$SEQ}"

REPO=/home/bosungan/OSCAR
ENVBIN=/home/bosungan/.conda/envs/oscar/bin
ROT="$REPO/rotation/qwen3-8B/GPQA/seq30000_prompt122_group128/rotations"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
OUTDIR="${OUTDIR:-$REPO/profiling/ncu-kernel/reports}"
CTX_LEN=$(( SEQ + OUTPUT_LEN + 256 ))
mkdir -p "$OUTDIR"
out="$OUTDIR/${QUANT}_${TAG}"

# NCU finalize/export can choke writing to /home (XFS) — write to /tmp then copy (mirrors nsys script).
scratch="${NCU_SCRATCH:-/tmp/oscar_ncu_scratch}"; mkdir -p "$scratch"
srep="$scratch/${QUANT}_${TAG}"

export PATH="$ENVBIN:$PATH"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export CUDA_VISIBLE_DEVICES="$GPU"
# NCU's interprocess lock defaults to /tmp/nsight-compute-lock, which another user owns
# (sticky /tmp -> we can't recreate it) -> InterprocessLockFailed. Relocate the lock into a
# dir WE own via TMPDIR. (Per-GPU runs target distinct free GPUs, so skipping the shared
# cross-user lock is safe here.)
export TMPDIR="${NCU_TMPDIR:-/tmp/ncu_${USER}}"; mkdir -p "$TMPDIR"
export PYTHONPATH="$REPO/rotation/_triton_per_rank:$REPO/sglang-research/python"
export PYTHONUNBUFFERED=1
# Triton must NOT use a CUDA-graph capture path here; bench_one_batch eager so kernels are individually launched.

SRV=(--model-path "$MODEL" --trust-remote-code --tensor-parallel-size 1
     --mem-fraction-static "$MEM_FRAC" --max-running-requests "$BATCH"
     --chunked-prefill-size "$CHUNKED_PREFILL"
     --context-length "$CTX_LEN"
     --disable-cuda-graph --disable-piecewise-cuda-graph)
OSCAR_ENV=(SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1)

case "$QUANT" in
  oscar)
    SRV+=(--kv-cache-dtype int2 --kv-cache-quant-group-size 128
          --prefill-attention-backend fa3 --decode-attention-backend triton)
    OSCAR_ENV+=(SGLANG_ENABLE_MIXED_KV_WINDOWS=1 SGLANG_OSCAR_ABSORB_V_ROTATION=1
      SGLANG_MIXED_KV_HP_MAX_SPLITS=8 SGLANG_MIXED_KV_PREFIX_TOKENS=64 SGLANG_MIXED_KV_RECENT_TOKENS=256
      SGLANG_MIXED_KV_HP_DTYPE=bfloat16 SGLANG_MIXED_KV_SCALE_DTYPE=float32
      SGLANG_OSCAR_K_ROTATION_PATH="$ROT/k_rotation_qqt_r_h_pbr.pt"
      SGLANG_OSCAR_V_ROTATION_PATH="$ROT/v_rotation_sst_r_h_pbr.pt"
      SGLANG_OSCAR_K_CLIP_RATIO=0.96 SGLANG_OSCAR_V_CLIP_RATIO=0.92
      SGLANG_OSCAR_FUSED_ROTATE_CLIP_QUANT=1) ;;
  plain_int2)
    SRV+=(--kv-cache-dtype int2 --kv-cache-quant-group-size 128
          --prefill-attention-backend fa3 --decode-attention-backend triton) ;;
  bf16)
    SRV+=(--prefill-attention-backend fa3 --decode-attention-backend triton) ;;
  *) echo "unknown QUANT=$QUANT"; exit 1 ;;
esac

base=(python -m sglang.bench_one_batch "${SRV[@]}"
      --batch-size "$BATCH" --input-len "$SEQ" --output-len "$OUTPUT_LEN")

NCU=(ncu --target-processes all
     --kernel-name "regex:.*stage1_quant_int2.*"
     --launch-skip "$SKIP" --launch-count "$COUNT"
     --set "$NCU_SET"
     --export "$srep" --force-overwrite)

echo ">>> [NCU/$QUANT] batch=$BATCH seq=$SEQ out=$OUTPUT_LEN GPU=$GPU mem_frac=$MEM_FRAC skip=$SKIP count=$COUNT set=$NCU_SET"
echo "    kernel regex: .*stage1_quant_int2.*   -> ${out}.ncu-rep"
env "${OSCAR_ENV[@]}" "${NCU[@]}" "${base[@]}" > "${out}.log" 2>&1
rc=$?

echo
if grep -qiE "ERR_NVGPUCTRPERM|profiling permission|not have permission" "${out}.log"; then
  echo "    !! NCU PERMISSION DENIED. Need profiling counters enabled. Options:"
  echo "       sudo (run ncu as root), OR set kernel module flag NVreg_RestrictProfilingToAdminUsers=0."
fi
if grep -qiE "OutOfMemoryError|CUDA out of memory" "${out}.log"; then
  echo "    !! OOM during one-shot prefill — lower SEQ/BATCH or MEM_FRAC. (bench_one_batch ignores chunked prefill.)"
fi
if [[ -f "${srep}.ncu-rep" ]]; then
  cp -f "${srep}.ncu-rep" "${out}.ncu-rep" && echo "    report -> ${out}.ncu-rep"
  echo "    profiled kernels:"; ncu -i "${out}.ncu-rep" --page raw --csv 2>/dev/null | head -1 >/dev/null \
    && ncu -i "${out}.ncu-rep" --print-summary per-kernel 2>/dev/null | grep -iE "stage1_quant_int2" | head -5 | sed 's/^/      /'
else
  echo "    WARNING: no .ncu-rep produced (rc=$rc). Tail of log:"; tail -25 "${out}.log" | sed 's/^/      /'
fi
echo "Done."
