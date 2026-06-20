#!/usr/bin/env bash
# ============================================================================
# profile_int2_ncu_server.sh — NCU kernel profile of the INT2 decode-attention
#   kernel via the SERVER path (bench_one_batch_server), so CHUNKED PREFILL is
#   honored and there is NO one-shot activation OOM -> b16/b32 @ 16k and b1@256k
#   become feasible (which the in-process bench_one_batch path could not do).
#
# THE RISK THIS SCRIPT TESTS: the server runs the model in a CHILD process
#   (nsys couldn't follow it). NCU's `--target-processes all` injects into the
#   whole process tree and intercepts kernel launches per-process, so it CAN
#   (hopefully) profile the int2 decode kernel inside the scheduler child.
#   -> ALWAYS smoke-test a tiny config first; if no kernel is captured, the
#      child-follow failed and we fall back to the in-process proxies.
#
# Usage:
#   bash profiling/ncu-kernel/profile_int2_ncu_server.sh BATCH SEQ [OUTPUT_LEN] [GPU] [PORT] [TAG]
# Examples:
#   bash .../profile_int2_ncu_server.sh 1 16384  20 1 31001 smoke_srv
#   bash .../profile_int2_ncu_server.sh 1 262144 20 2 31002 b1_256k
#   bash .../profile_int2_ncu_server.sh 32 16384 20 3 31003 b32_16k
#
# Env (prepend VAR=value):
#   CHUNK     --chunked-prefill-size  (default 2048; THE point — bounds activation)
#   SKIP/COUNT  NCU --launch-skip / --launch-count over matched int2 launches (default 72 / 3)
#   NCU_SET   NCU metric set          (default full)
#   MEM_FRAC  static pool fraction    (default 0.85; chunked prefill removes the spike -> raise)
#   QUANT     oscar | plain_int2 | bf16   (default oscar)
#   OSCAR_BENCH_TIMEOUT  client read timeout sec (default 7200; NCU replay is slow)
#   NCU_TMPDIR  relocate the NCU interprocess lock (default /tmp/ncu_$USER)
#
# Output: profiling/ncu-kernel/reports/oscar_<TAG>.ncu-rep (+ .log)
# ============================================================================
set -uo pipefail

BATCH="${1:?usage: BATCH SEQ [OUTPUT_LEN] [GPU] [PORT] [TAG]}"
SEQ="${2:?need SEQ}"
OUTPUT_LEN="${3:-20}"
GPU="${4:-1}"
PORT="${5:-31001}"
TAG="${6:-b${BATCH}_s${SEQ}}"

CHUNK="${CHUNK:-2048}"
SKIP="${SKIP:-72}"
COUNT="${COUNT:-3}"
NCU_SET="${NCU_SET:-full}"
MEM_FRAC="${MEM_FRAC:-0.85}"
QUANT="${QUANT:-oscar}"
export OSCAR_BENCH_TIMEOUT="${OSCAR_BENCH_TIMEOUT:-7200}"

REPO=/home/bosungan/OSCAR
ENVBIN=/home/bosungan/.conda/envs/oscar/bin
ROT="$REPO/rotation/qwen3-8B/GPQA/seq30000_prompt122_group128/rotations"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
OUTDIR="${OUTDIR:-$REPO/profiling/ncu-kernel/reports}"
CTX_LEN=$(( SEQ + OUTPUT_LEN + 256 ))
mkdir -p "$OUTDIR"
out="$OUTDIR/oscar_${TAG}"
scratch="${NCU_SCRATCH:-/tmp/oscar_ncu_scratch}"; mkdir -p "$scratch"
srep="$scratch/oscar_${TAG}"

export PATH="$ENVBIN:$PATH"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$REPO/rotation/_triton_per_rank:$REPO/sglang-research/python"
export PYTHONUNBUFFERED=1
export TMPDIR="${NCU_TMPDIR:-/tmp/ncu_${USER}}"; mkdir -p "$TMPDIR"   # relocate NCU lock (foreign-owned /tmp lock)

# ---- server flags (mirror profile_oscar_server.sh) — GRAPH OFF for clean eager launches ----
SRV=(--model-path "$MODEL" --trust-remote-code --tensor-parallel-size 1
     --mem-fraction-static "$MEM_FRAC" --max-running-requests "$BATCH"
     --chunked-prefill-size "$CHUNK"
     --context-length "$CTX_LEN" --port "$PORT"
     --skip-server-warmup
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

base=(python "$REPO/profiling/ncu-kernel/_run_bench_server_ncu.py" "${SRV[@]}"
      --batch-size "$BATCH" --input-len "$SEQ" --output-len "$OUTPUT_LEN" --skip-warmup)

NCU=(ncu --target-processes all
     --kernel-name "regex:.*stage1_quant_int2.*"
     --launch-skip "$SKIP" --launch-count "$COUNT"
     --set "$NCU_SET"
     --export "$srep" --force-overwrite)

echo ">>> [NCU/server/$QUANT] batch=$BATCH seq=$SEQ chunk=$CHUNK out=$OUTPUT_LEN GPU=$GPU port=$PORT skip=$SKIP count=$COUNT"
echo "    -> ${out}.ncu-rep  (child-process capture test)"
env "${OSCAR_ENV[@]}" "${NCU[@]}" "${base[@]}" > "${out}.log" 2>&1
rc=$?

echo
grep -qiE "ERR_NVGPUCTRPERM|InterprocessLockFailed" "${out}.log" && echo "    !! NCU permission/lock error (see log)"
grep -qiE "OutOfMemory|CUDA out of memory" "${out}.log" && echo "    !! OOM (raise MEM_FRAC or lower CHUNK)"
grep -qiE "No kernels were profiled" "${out}.log" && echo "    !! NO KERNELS — NCU did NOT capture the child's int2 kernel (child-follow failed)."
if [[ -f "${srep}.ncu-rep" ]]; then
  cp -f "${srep}.ncu-rep" "${out}.ncu-rep" && echo "    report -> ${out}.ncu-rep  (CHILD CAPTURE WORKS)"
  ncu -i "${out}.ncu-rep" --print-summary per-kernel 2>/dev/null | grep -iE "stage1_quant_int2" | head -3 | sed 's/^/      /'
else
  echo "    no .ncu-rep (rc=$rc). Tail:"; tail -20 "${out}.log" | sed 's/^/      /'
fi
echo "Done."
