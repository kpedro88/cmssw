#!/bin/bash
# Memory-scaling-vs-threads test for the AOTInductor GlobalCache producer.
#
# Runs torchAOTProducer_cfg.py with several thread counts and extracts the peak
# RSS reported by the Timing/SimpleMemoryCheck services. Because the model is a
# single shared, read-only GlobalCache resource, RSS should stay roughly flat as
# threads increase (only per-thread activations/stacks grow), demonstrating that
# the weights are NOT duplicated per thread.
#
# Usage: scan_threads.sh [nevents] [ninfer] "[thread list]"
#   e.g. scan_threads.sh 4000 100 "1 2 4 8"
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="$HERE/torchAOTProducer_cfg.py"

NEV="${1:-4000}"
NINFER="${2:-100}"
THREADS="${3:-1 2 4 8}"

printf "%-8s %-12s %-14s %-12s %-10s\n" threads events RSS_MB VSIZE_MB realtime_s
echo "-------------------------------------------------------------------"

for nt in $THREADS; do
  log="/tmp/torchAOT_nt${nt}.log"
  cmsRun "$CFG" nthreads=$nt nevents=$NEV ninfer=$NINFER > "$log" 2>&1 || { echo "cmsRun failed for nt=$nt (see $log)"; continue; }

  # SimpleMemoryCheck prints a final summary line:
  #   "MemoryReport> Peak rss size <RSS> Mbytes (VSIZE <VSIZE>)"
  peak=$(grep -E "MemoryReport> Peak rss size" "$log" | tail -1)
  rss=$(echo "$peak"   | grep -oE "Peak rss size [0-9.]+" | awk '{print $4}')
  vsize=$(echo "$peak" | grep -oE "VSIZE [0-9.]+"        | awk '{print $2}')
  # Total event-loop real time per event from the Timing summary
  rt=$(grep -E "event loop Real/event" "$log" | grep -oE "[0-9.]+" | tail -1)

  printf "%-8s %-12s %-14s %-12s %-10s\n" "$nt" "$NEV" "${rss:-NA}" "${vsize:-NA}" "${rt:-NA}"
done
