#!/usr/bin/env bash
# ============================================================================
# profile_oscar_multi.sh — MULTI-batch Nsight sweep for OSCAR INT2 KV.
#
# Wraps sglang.bench_one_batch (no project source is modified) and runs it once
# per (BATCH, SEQ) combination under nsys, producing a SEPARATE trace + log for
# each combo. This is the multi-batch counterpart of
#   profiling/single-batch/profile_oscar.sh
# Each combo is its own process -> its own self-contained nsys-rep (you can open
# any one in the GUI and see exactly that (batch, seq) prefill+decode).
#
# Usage:
#   BATCHES="1 4 8" SEQS="4096 16384" bash profiling/multi-batch/profile_oscar_multi.sh
#   BATCHES="1 2 4" SEQS="8192 32768" OUTPUT_LEN=8 QUANT=oscar GPU=3 \
#       bash profiling/multi-batch/profile_oscar_multi.sh
#
# It runs the CARTESIAN PRODUCT of BATCHES x SEQS. For each (B,S) it writes, in
# profiling/multi-batch/traces/:
#   <QUANT>_b<B>_s<S>.nsys-rep   (whole-run trace: prefill + decode)
#   <QUANT>_b<B>_s<S>.log        (sglang stdout + Prefill./Decode. median latency)
#
# Env overrides (all optional; prepend as VAR=value):
#   BATCHES   batch sizes to sweep            (default "1 4 8")
#   SEQS      input (context) lengths         (default "4096 16384 65536")
#   OUTPUT_LEN  decode steps+1 (2 => prefill + 1 decode step)  (default 2)
#   GPU       CUDA device index               (default 3)
#   QUANT     oscar | bf16                    (default oscar)
#   GRAPH     on | off (off => eager, every kernel visible)    (default off)
#   MEM_FRAC  static-pool fraction            (default 0.8; see KV/activation note)
#   MODEL     HF model path                   (default Qwen/Qwen3-8B)
#   OUTDIR    where traces/logs land          (default profiling/multi-batch/traces)
#   SKIP_EXISTING  1 => skip a combo whose .nsys-rep already exists  (default 0)
#   DRY_RUN   1 => print the planned combos and exit (no GPU work)   (default 0)
#
# MEMORY NOTE (read before long/large sweeps):
#   * KV pool must hold B*(SEQ+OUTPUT_LEN) tokens. After boot, each log prints
#     `max_total_num_tokens=` — it MUST exceed B*(SEQ+OUTPUT_LEN) or the combo
#     errors. Raise MEM_FRAC to grow the pool.
#   * Prefill ACTIVATION scales with the total prefill tokens B*SEQ (one forward
#     over the whole batch). Large B*SEQ can OOM in torch (e.g. the MLP SwiGLU
#     buffer) even when the KV pool fits. LOWER MEM_FRAC frees room for activation
#     but shrinks the pool — the two pull opposite ways, so very large B*SEQ may
#     simply not fit on one GPU. Tune MEM_FRAC per sweep; a failing combo is
#     logged and the sweep continues to the next one.
# ============================================================================
set -uo pipefail   # NOTE: no -e — one failing combo must not abort the whole sweep.

# ---- sweep grid ------------------------------------------------------------
BATCHES="${BATCHES:-1 4 8}"
SEQS="${SEQS:-4096 16384 65536}"
OUTPUT_LEN="${OUTPUT_LEN:-2}"
GPU="${GPU:-3}"
QUANT="${QUANT:-oscar}"
GRAPH="${GRAPH:-off}"
MEM_FRAC="${MEM_FRAC:-0.8}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
DRY_RUN="${DRY_RUN:-0}"

REPO=/home/bosungan/OSCAR
ENVBIN=/home/bosungan/.conda/envs/oscar/bin
ROT="$REPO/rotation/qwen3-8B/GPQA/seq30000_prompt122_group128/rotations"
OUTDIR="${OUTDIR:-$REPO/profiling/multi-batch/traces}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-4096}"
mkdir -p "$OUTDIR"

export PATH="$ENVBIN:$PATH"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$REPO/rotation/_triton_per_rank:$REPO/sglang-research/python"
export PYTHONUNBUFFERED=1

# nsys report finalize FAILS writing to /home (XFS): finalize on local /tmp, then copy.
scratch="${NSYS_SCRATCH:-/tmp/oscar_nsys_scratch}"; mkdir -p "$scratch"

# ---- plan / announce -------------------------------------------------------
read -ra _B <<<"$BATCHES"; read -ra _S <<<"$SEQS"
n_combos=$(( ${#_B[@]} * ${#_S[@]} ))
echo "============================================================"
echo " OSCAR multi-batch sweep : QUANT=$QUANT GRAPH=$GRAPH GPU=$GPU OUTPUT_LEN=$OUTPUT_LEN MEM_FRAC=$MEM_FRAC"
echo " BATCHES = [$BATCHES]   SEQS = [$SEQS]   => $n_combos combos"
echo " OUTDIR  = $OUTDIR"
echo "============================================================"
if [[ "$DRY_RUN" == "1" ]]; then
  for B in "${_B[@]}"; do for S in "${_S[@]}"; do
    printf "   would run: batch=%-4s seq=%-8s -> %s/%s_b%s_s%s.nsys-rep\n" "$B" "$S" "$OUTDIR" "$QUANT" "$B" "$S"
  done; done
  echo "(DRY_RUN=1 — nothing executed)"; exit 0
fi

# ---- per-combo server flags + OSCAR env (mirrors single-batch script) ------
build_srv() {   # $1=BATCH $2=SEQ ; sets global SRV[] and OSCAR_ENV[]
  local B="$1" S="$2"
  local ctx=$(( S + OUTPUT_LEN + 256 ))
  SRV=(--model-path "$MODEL" --trust-remote-code --tensor-parallel-size 1
       --mem-fraction-static "$MEM_FRAC" --max-running-requests "$B"
       --chunked-prefill-size "$CHUNKED_PREFILL"
       --context-length "$ctx")
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
}

# ---- sweep -----------------------------------------------------------------
declare -a SUMMARY=()
i=0
for B in "${_B[@]}"; do
  for S in "${_S[@]}"; do
    i=$((i+1))
    tag="b${B}_s${S}"
    out="$OUTDIR/${QUANT}_${tag}"
    srep="$scratch/${QUANT}_${tag}"
    echo
    echo ">>> [$i/$n_combos] $QUANT batch=$B seq=$S out=$OUTPUT_LEN -> ${out}.nsys-rep"

    if [[ "$SKIP_EXISTING" == "1" && -f "${out}.nsys-rep" ]]; then
      echo "    SKIP (exists)"; SUMMARY+=("b=$B s=$S : SKIPPED"); continue
    fi

    build_srv "$B" "$S"
    base=(python -m sglang.bench_one_batch "${SRV[@]}"
          --batch-size "$B" --input-len "$S" --output-len "$OUTPUT_LEN")

    env "${OSCAR_ENV[@]}" nsys profile --force-overwrite=true -o "$srep" --trace=cuda,nvtx \
        "${base[@]}" > "${out}.log" 2>&1
    rc=$?

    if [[ -f "${srep}.nsys-rep" ]]; then
      cp -f "${srep}.nsys-rep" "${out}.nsys-rep"
    fi

    # classify outcome from the log
    if grep -qiE "OutOfMemoryError|CUDA out of memory" "${out}.log"; then
      echo "    OOM (see ${out}.log) — try lower MEM_FRAC or smaller B*SEQ"
      SUMMARY+=("b=$B s=$S : OOM")
    elif [[ ! -f "${out}.nsys-rep" ]]; then
      echo "    FAILED rc=$rc, no report (see ${out}.log)"
      SUMMARY+=("b=$B s=$S : FAILED(rc=$rc)")
    else
      lat=$(grep -iE "Decode\.  median" "${out}.log" | head -1 | sed 's/.*latency: //')
      echo "    OK -> ${out}.nsys-rep   ${lat:+decode median: $lat}"
      SUMMARY+=("b=$B s=$S : OK ${lat:+(decode $lat)}")
    fi
  done
done

# ---- summary ---------------------------------------------------------------
echo
echo "============================================================"
echo " SWEEP SUMMARY ($QUANT, GRAPH=$GRAPH)"
echo "============================================================"
for line in "${SUMMARY[@]}"; do echo "  $line"; done
echo
echo "Summarize any trace:  bash profiling/single-batch/report.sh ${OUTDIR}/${QUANT}_b<B>_s<S>.nsys-rep"
