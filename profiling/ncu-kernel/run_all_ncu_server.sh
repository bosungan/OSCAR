#!/usr/bin/env bash
# Orchestrate the 5 server-path NCU runs across GPUs {1,2,3}, max 3 concurrent.
# Each run gets a distinct GPU, port, and NCU lock dir (TMPDIR) so they don't collide.
set -uo pipefail
cd /home/bosungan/OSCAR

# BATCH SEQ TAG MEMFRAC
CONFIGS=(
  "1 131072 b1_128k 0.85"
  "1 262144 b1_256k 0.85"
  "8 16384  b8_16k  0.85"
  "16 16384 b16_16k 0.85"
  "32 16384 b32_16k 0.92"
)
GPUS=(1 2 3)
i=0
for cfg in "${CONFIGS[@]}"; do
  set -- $cfg; B=$1; S=$2; TAG=$3; MF=$4
  gpu=${GPUS[$(( i % ${#GPUS[@]} ))]}
  port=$(( 31010 + i ))
  # throttle: keep at most ${#GPUS[@]} concurrent server+ncu runs
  while [ "$(pgrep -af 'ncu .*stage1_quant_int2'|grep -v pgrep|wc -l)" -ge "${#GPUS[@]}" ]; do sleep 10; done
  echo ">>> launching $TAG (b=$B s=$S) on GPU$gpu port$port memfrac$MF"
  MEM_FRAC=$MF NCU_TMPDIR=/tmp/ncu_g${gpu} SKIP=72 COUNT=3 \
    timeout 3000 bash profiling/ncu-kernel/profile_int2_ncu_server.sh "$B" "$S" 24 "$gpu" "$port" "$TAG" \
    > /tmp/ncu_srv_${TAG}.out 2>&1 &
  sleep 25   # stagger server starts so GPU mem/port alloc don't race
  i=$(( i + 1 ))
done
wait
echo "=== ALL 5 DONE ==="
ls -la profiling/ncu-kernel/reports/oscar_b*_*.ncu-rep 2>/dev/null
