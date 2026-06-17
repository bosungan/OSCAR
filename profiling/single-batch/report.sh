#!/usr/bin/env bash
# report.sh — summarize one or more .nsys-rep traces without the GUI.
# Usage: bash profiling/single-batch/report.sh traces/oscar_ctx64k.nsys-rep [...]
#   (generic .nsys-rep summarizer — also works on multi-batch traces)
set -euo pipefail
export PATH="/home/bosungan/.conda/envs/oscar/bin:$PATH"
for rep in "$@"; do
  echo "################################################################"
  echo "# $rep"
  echo "################################################################"
  echo "=== GPU kernel time by op (Time% | Instances | Name) ==="
  nsys stats --report cuda_gpu_kern_sum --format table "$rep" 2>/dev/null \
    | awk -F'|' 'NF>5{print $2"|"$4"|"substr($10,1,72)}' | head -30
  echo
  echo "=== CPU API (launches / sync / graph) ==="
  nsys stats --report cuda_api_sum --format table "$rep" 2>/dev/null \
    | grep -iE "Name|cudaLaunchKernel|Synchronize|GraphLaunch|Memcpy|Memset" | head -10
  echo
done
# Tip: per-instance timeline of one kernel (see how attention grows with ctx):
#   nsys stats --report cuda_gpu_trace --format table <rep> | grep _fwd_grouped_kernel_stage1_quant_int2
