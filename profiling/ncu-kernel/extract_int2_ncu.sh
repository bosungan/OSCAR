#!/usr/bin/env bash
# ============================================================================
# extract_int2_ncu.sh — pull the decision-relevant NCU counters out of a
# .ncu-rep into a compact CSV/table. Averages across the profiled int2 launches.
#
# Usage: bash profiling/ncu-kernel/extract_int2_ncu.sh reports/<name>.ncu-rep
#
# Metrics (the ones that answer "is INT2 attention efficient?"):
#   gpu__time_duration.sum                                   kernel time (ns)
#   dram__bytes.sum                                          REAL DRAM bytes moved (validate roofline)
#   dram__throughput.avg.pct_of_peak_sustained_elapsed       <-- MBU (BW utilization %)  *** headline
#   sm__throughput.avg.pct_of_peak_sustained_elapsed         compute SOL %
#   gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed  overall SOL %
#   sm__warps_active.avg.pct_of_peak_sustained_active        achieved occupancy %
#   sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active  TENSOR-CORE usage % (0 => CUDA-core path)
#   sm__inst_executed_pipe_tensor.sum                        tensor-core instr count (0 => no TC)
#   l1tex__t_sector_hit_rate.pct / lts__t_sector_hit_rate.pct  L1 / L2 hit %
#   smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio  memory-stall (cycles/issue)
#   smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio short-scoreboard (MIO/dequant)
#   smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio  math-pipe stall
# ============================================================================
set -uo pipefail
REP="${1:?usage: extract_int2_ncu.sh <report.ncu-rep>}"
export PATH="/home/bosungan/.conda/envs/oscar/bin:$PATH"

METRICS="gpu__time_duration.sum,\
dram__bytes.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,\
sm__inst_executed_pipe_tensor.sum,\
l1tex__t_sector_hit_rate.pct,\
lts__t_sector_hit_rate.pct,\
launch__occupancy_limit_registers,\
launch__occupancy_limit_shared_mem,\
launch__occupancy_limit_warps,\
launch__registers_per_thread,\
launch__shared_mem_per_block_static,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio"

echo "### NCU extract: $REP"
echo "--- per-launch CSV (raw) ---"
ncu -i "$REP" --csv --metrics "$METRICS" 2>/dev/null
echo
echo "--- SpeedOfLight section (human) ---"
ncu -i "$REP" --page details --section SpeedOfLight 2>/dev/null | sed -n '1,40p'
