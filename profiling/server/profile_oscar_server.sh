#!/usr/bin/env bash
# ============================================================================
# profile_oscar_server.sh — SERVER-backed profile for OSCAR INT2 KV.
#                           ONE (batch, seq) per invocation.
#
# WHY THIS EXISTS (read this):
#   The static microbench `profiling/single-batch/profile_oscar.sh` drives
#   `sglang.bench_one_batch`, which calls the model's low-level `extend()`
#   DIRECTLY and BYPASSES the Scheduler -> `--chunked-prefill-size` is IGNORED,
#   the whole prefill runs in ONE forward, and activation OOMs past ~256k tokens.
#
#   This script drives `sglang.bench_one_batch_server`, which LAUNCHES A REAL
#   SERVER (HTTP + Scheduler) and benchmarks one batch through it. With ZERO
#   source changes that gives, vs the static path:
#     * MULTI-BATCH      : --batch-size goes through the real scheduler.
#     * CHUNKED PREFILL  : --chunked-prefill-size is HONORED -> activation bounded
#                          by the chunk size, not B*SEQ -> 256k/512k/1M feasible.
#
#   PROFILER: we use sglang's BUILT-IN per-stage profiler (`--profile-by-stage`),
#   NOT nsys. The server runs the model in a child process; nsys could not follow
#   that child (capture came back empty). The built-in profiler runs torch's
#   profiler INSIDE the server via the /start_profile endpoint, so it reliably
#   captures the GPU work and SPLITS prefill vs decode. Output is a Chrome/torch
#   trace (.trace.json.gz) -> open in Perfetto (https://ui.perfetto.dev).
#   Still a WRAPPER: nothing under sglang-research/ is modified.
#
# Usage:
#   bash profiling/server/profile_oscar_server.sh BATCH SEQ [OUTPUT_LEN] [GPU]
# Examples:
#   bash profiling/server/profile_oscar_server.sh 1 16384            # smoke test
#   bash profiling/server/profile_oscar_server.sh 1 524288 8 3       # 512k on GPU 3
#   QUANT=bf16 bash profiling/server/profile_oscar_server.sh 1 262144
#
# Writes a per-run dir under profiling/server/traces/<QUANT>_<TAG>/<timestamp>/
# containing the Chrome traces (one per stage: prefill + decode) and a .log.
#
# Env overrides (prepend as VAR=value):
#   CHUNK     --chunked-prefill-size (THE point)   (default 8192)
#   OSCAR_BENCH_TIMEOUT  client HTTP read timeout, sec (default 7200=2h). Long-ctx
#               prefill (512k/1M) takes many minutes; the stock sglang client caps
#               this at 600s and would ReadTimeout. The _run_bench_server.py
#               launcher raises it. Bump higher for 1M+.
#   PROFILE_STEPS  decode steps to RECORD           (default 4; must be <= OUTPUT_LEN)
#   START_STEP  decode step to START recording at   (default unset = from step 0,
#               which includes the warmup/flush outlier; set e.g. 8 to skip to
#               steady-state. Needs OUTPUT_LEN >= START_STEP + PROFILE_STEPS.)
#   QUANT     oscar | bf16                         (default oscar)
#   GRAPH     on | off                             (default off; off => eager, kernels visible)
#   MEM_FRAC  static-pool fraction                 (default 0.85; RAISE it — chunked
#             prefill removes the activation spike, so the KV pool is the limit.)
#   MAX_RUNNING  --max-running-requests            (default = BATCH)
#   PORT      server port                          (default 31000)
#   MODEL / OUTDIR / TAG                            (sensible defaults)
#
# MEMORY: with chunked prefill peak activation ~ CHUNK tokens (not B*SEQ). 512k
#   decode needs ~ weights(15 GB) + INT2 KV(~11 GB) + small activation ≈ 27 GB.
#   The KV POOL is the binding constraint now -> RAISE MEM_FRAC. After boot the
#   log prints `max_total_num_tokens=`; it must exceed BATCH*(SEQ+OUTPUT_LEN).
# ============================================================================
set -uo pipefail

BATCH="${1:?usage: profile_oscar_server.sh BATCH SEQ [OUTPUT_LEN] [GPU]}"
SEQ="${2:?need SEQ (input/context length)}"
OUTPUT_LEN="${3:-8}"
GPU="${4:-0}"

CHUNK="${CHUNK:-2048}"
PROFILE_STEPS="${PROFILE_STEPS:-4}"
START_STEP="${START_STEP:-}"          # empty => record from decode step 0
QUANT="${QUANT:-oscar}"
GRAPH="${GRAPH:-off}"
MEM_FRAC="${MEM_FRAC:-0.80}"
MAX_RUNNING="${MAX_RUNNING:-$BATCH}"
PORT="${PORT:-31000}"

REPO=/home/bosungan/OSCAR
ENVBIN=/home/bosungan/.conda/envs/oscar/bin
ROT="$REPO/rotation/qwen3-8B/GPQA/seq30000_prompt122_group128/rotations"
OUTDIR="${OUTDIR:-$REPO/profiling/server/traces}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
TAG="${TAG:-b${BATCH}_s${SEQ}_chunk${CHUNK}}"
CTX_LEN=$(( SEQ + OUTPUT_LEN + 256 ))
runroot="$OUTDIR/${QUANT}_${TAG}"      # run_profile appends a /<timestamp>/ inside this
log="$runroot/run.log"
mkdir -p "$runroot"

export PATH="$ENVBIN:$PATH"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$REPO/rotation/_triton_per_rank:$REPO/sglang-research/python"
export PYTHONUNBUFFERED=1

# ---- server flags + OSCAR env (mirrors the single-batch script) ------------
SRV=(--model-path "$MODEL" --trust-remote-code --tensor-parallel-size 1
     --mem-fraction-static "$MEM_FRAC" --max-running-requests "$MAX_RUNNING"
     --chunked-prefill-size "$CHUNK"          # <-- HONORED here (goes through Scheduler)
     --context-length "$CTX_LEN" --port "$PORT")
OSCAR_ENV=(SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1)
[[ "$GRAPH" == "off" ]] && SRV+=(--disable-cuda-graph --disable-piecewise-cuda-graph)

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
  bf16)
    SRV+=(--prefill-attention-backend fa3 --decode-attention-backend triton) ;;
  *) echo "unknown QUANT=$QUANT"; exit 1 ;;
esac

if [[ -n "$START_STEP" && $(( START_STEP + PROFILE_STEPS )) -gt "$OUTPUT_LEN" ]]; then
  echo "WARNING: START_STEP($START_STEP)+PROFILE_STEPS($PROFILE_STEPS) > OUTPUT_LEN($OUTPUT_LEN)"
  echo "         -> not enough decode steps to record; raise OUTPUT_LEN (>= $((START_STEP+PROFILE_STEPS)))."
fi

# bench_one_batch_server self-launches the server (no --base-url) and drives one
# batch through it. --profile --profile-by-stage turns on torch's profiler INSIDE
# the server, dumping per-stage Chrome traces under --profile-output-dir.
base=(python "$REPO/profiling/server/_run_bench_server.py" "${SRV[@]}"
      --batch-size "$BATCH" --input-len "$SEQ" --output-len "$OUTPUT_LEN"
      --skip-warmup
      --profile --profile-by-stage --profile-steps "$PROFILE_STEPS"
      --profile-output-dir "$runroot")
# Optional: skip warmup/outlier decode steps and record steady-state instead.
[[ -n "$START_STEP" ]] && base+=(--profile-start-step "$START_STEP")

echo ">>> [$QUANT/server+profiler] batch=$BATCH seq=$SEQ chunk=$CHUNK out=$OUTPUT_LEN steps=$PROFILE_STEPS start=${START_STEP:-0} GPU=$GPU graph=$GRAPH"
echo "    profile dir root -> $runroot"
env "${OSCAR_ENV[@]}" "${base[@]}" > "$log" 2>&1
rc=$?

echo
if grep -qiE "OutOfMemoryError|CUDA out of memory" "$log"; then
  echo "    OOM — check max_total_num_tokens in the log; raise MEM_FRAC or lower CHUNK."
fi
echo "    max_total_num_tokens:"; grep -iE "max_total_num_tokens" "$log" | tail -1 | sed 's/^/      /'
echo "    bench latency:"; grep -iE "^latency:|throughput:|ttft" "$log" | sed 's/^/      /'

# Locate the dumped Chrome traces (run_profile makes a <timestamp>/ subdir).
echo
echo "    Chrome/torch traces (open in Perfetto -> https://ui.perfetto.dev):"
traces=$(find "$runroot" -type f \( -name "*.trace.json.gz" -o -name "*.json.gz" -o -name "*.pt.trace.json" \) 2>/dev/null | sort)
if [[ -n "$traces" ]]; then
  echo "$traces" | sed 's/^/      /'
else
  echo "      (none found — rc=$rc; see $log. If the server failed to start, check the port/OOM.)"
fi
echo
echo "Done. View: open https://ui.perfetto.dev and drag-drop a .trace.json.gz above"
echo "      (prefill stage = chunked prefill forwards; decode stage = per-step decode kernels)."
