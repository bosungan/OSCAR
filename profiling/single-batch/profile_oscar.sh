#!/usr/bin/env bash
# ============================================================================
# profile_oscar.sh — single-batch Nsight microbenchmark for OSCAR INT2 KV.
#
# Drives sglang.bench_one_batch under nsys to dissect ONE prefill and the
# steady-state decode for arbitrary (INPUT_LEN, OUTPUT_LEN) at long context.
# NOT serving — one fixed batch, you choose the shape.
#
# Usage:
#   bash profiling/single-batch/profile_oscar.sh INPUT_LEN OUTPUT_LEN [GPU] [TAG]
# Examples:
#   bash profiling/single-batch/profile_oscar.sh 4096   64            # short ctx baseline
#   bash profiling/single-batch/profile_oscar.sh 65536  64 3 ctx64k   # long ctx on GPU 3
#   bash profiling/single-batch/profile_oscar.sh 131072 32 3 ctx128k
#
# OUTPUT_LEN controls how many decode steps run (=2 => prefill + a single decode step).
# Env overrides (prepend): MEM_FRAC=, MAX_RUNNING=, QUANT=oscar|plain_int2|bf16,
#   GRAPH=on|off (default off so every op is visible in the trace), CHUNKED_PREFILL=, OUTDIR=.
# One nsys run -> one whole-run trace containing BOTH prefill and decode (no per-phase split here).
# ============================================================================
set -euo pipefail

INPUT_LEN="${1:?usage: profile_oscar.sh INPUT_LEN OUTPUT_LEN [GPU] [TAG]}"
OUTPUT_LEN="${2:?need OUTPUT_LEN}"
GPU="${3:-3}"
TAG="${4:-ctx${INPUT_LEN}}"

REPO=/home/bosungan/OSCAR
ENVBIN=/home/bosungan/.conda/envs/oscar/bin
ROT="$REPO/rotation/qwen3-8B/GPQA/seq30000_prompt122_group128/rotations"
OUTDIR="${OUTDIR:-$REPO/profiling/single-batch/traces}"
mkdir -p "$OUTDIR"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
MEM_FRAC="${MEM_FRAC:-0.6}"            # raise if KV pool can't hold INPUT_LEN+OUTPUT_LEN
MAX_RUNNING="${MAX_RUNNING:-1}"        # single sequence -> keep HP pools tiny
QUANT="${QUANT:-oscar}"               # oscar | plain_int2 | bf16
GRAPH="${GRAPH:-off}"                 # off => eager => every kernel shows in trace
CHUNKED_PREFILL="${CHUNKED_PREFILL:-4096}"  # set = INPUT_LEN for one-shot prefill
# (No PHASE/PROFILE_STEPS: one nsys run traces the whole thing; OUTPUT_LEN controls # decode steps,
#  e.g. OUTPUT_LEN=2 => prefill + a single decode step.)
CTX_LEN="${CTX_LEN:-$(( INPUT_LEN + OUTPUT_LEN + 256 ))}"

export PATH="$ENVBIN:$PATH"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$REPO/rotation/_triton_per_rank:$REPO/sglang-research/python"
export PYTHONUNBUFFERED=1

# ---- per-QUANT server flags + OSCAR env ------------------------------------
SRV=(--model-path "$MODEL" --trust-remote-code --tensor-parallel-size 1
     --mem-fraction-static "$MEM_FRAC" --max-running-requests "$MAX_RUNNING"
     --chunked-prefill-size "$CHUNKED_PREFILL"
     --context-length "$CTX_LEN")
OSCAR_ENV=(SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1)   # allow long ctx for the microbench
# GRAPH=off => fully eager: disable BOTH the full CUDA graph AND the piecewise (torch.compile) graph.
# Without --disable-piecewise-cuda-graph, sglang still spends ~30-40s capturing 50 token-bucket graphs
# (replaying the model forward 50x) — that shows up as a huge GEMM region in the trace that is NOT prefill.
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

# ONE nsys run = ONE whole-run trace that contains BOTH prefill and decode. (We can't reliably split
# them into separate nsys traces here: --capture-range=cudaProfilerApi, the flag that would scope the
# capture per-stage, crashes nsys report finalize on this box. So there is no per-phase command to run.)
out="$OUTDIR/${QUANT}_${TAG}"
base=(python -m sglang.bench_one_batch "${SRV[@]}"
      --batch-size 1 --input-len "$INPUT_LEN" --output-len "$OUTPUT_LEN")
# nsys report finalize FAILS writing to /home (XFS): finalize on local /tmp, then copy into OUTDIR.
scratch="${NSYS_SCRATCH:-/tmp/oscar_nsys_scratch}"; mkdir -p "$scratch"
srep="$scratch/${QUANT}_${TAG}"
echo ">>> [$QUANT/nsys] IN=$INPUT_LEN OUT=$OUTPUT_LEN GPU=$GPU graph=$GRAPH -> ${out}.nsys-rep (via $scratch)"
env "${OSCAR_ENV[@]}" nsys profile --force-overwrite=true -o "$srep" --trace=cuda,nvtx \
    "${base[@]}" > "${out}.log" 2>&1
if [[ -f "${srep}.nsys-rep" ]]; then
  cp -f "${srep}.nsys-rep" "${out}.nsys-rep" && echo "    report -> ${out}.nsys-rep"
else
  echo "    WARNING: nsys produced no report at ${srep}.nsys-rep (see ${out}.log)"
fi
echo "    latency:"; grep -iE "Prefill\.|Decode\.  median" "${out}.log" | sed 's/^/      /'
echo "Done. Open ${out}.nsys-rep in Nsight Systems (the prefill+decode you want = the LAST few seconds of the timeline)."
