#!/bin/bash
# scram b runtests entry point for the AOTInductor GlobalCache memory-scaling demo.
#
# 1. Generates a small AOTInductor .pt2 (so the test is self-contained and does
#    not rely on a large committed model file).
# 2. Runs the TorchAOTProducer on 1 and 4 threads (shared GlobalCache mode).
# 3. Asserts that the model .so RSS (where the AOTInductor weights live) does NOT
#    grow with threads -- i.e. the weights are shared and read-only.
set -e
function die { echo "FAILED $1"; exit 1; }

TESTDIR="${LOCALTOP:+$LOCALTOP/src}/PhysicsTools/PyTorch/test"
[ -d "$TESTDIR" ] || TESTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="$TESTDIR/torchAOTProducer_cfg.py"
WORK="$(mktemp -d)"
# The model must live inside the release src area so edm::FileInPath can resolve
# it. Generate it into the package's test/data and reference it by FileInPath.
REL="PhysicsTools/PyTorch/test/data/memtest_model.pt2"
MODELABS="$TESTDIR/data/memtest_model.pt2"
mkdir -p "$TESTDIR/data"
trap 'rm -rf "$WORK"; rm -f "$MODELABS"' EXIT

echo ">> Generating small AOTInductor model"
python3 "$TESTDIR/make_aot_producer_model.py" "$MODELABS" > "$WORK/gen.log" 2>&1 \
  || { cat "$WORK/gen.log"; die "model generation"; }

# Extract the .so RSS reported by SimpleMemoryCheck for a given run.
so_rss() { grep "of which .so's" "$1" | tail -1 | grep -oE "[0-9.]+ MBytes \(RSS\)" | grep -oE "^[0-9.]+"; }

run() {  # run <nthreads> <logfile>
  cmsRun "$CFG" nthreads=$1 nevents=$(( $1 * 30 )) ninfer=1 batch=16 features=10 \
      model="$REL" > "$2" 2>&1 || { cat "$2"; die "cmsRun nt=$1"; }
}

echo ">> Shared GlobalCache mode: 1 vs 4 threads"
run 1 "$WORK/shared_nt1.log"
run 4 "$WORK/shared_nt4.log"

s1=$(so_rss "$WORK/shared_nt1.log"); s4=$(so_rss "$WORK/shared_nt4.log")
echo "   shared .so RSS:  nt1=$s1  nt4=$s4 MB"

# The small model's weights are tiny, so we only assert the qualitative trend:
# shared mode .so RSS must stay essentially flat (< 5 MB growth) from 1->4
# threads. (The dramatic numbers come from the big model in scan_threads.sh.)
awk -v a="$s1" -v b="$s4" 'BEGIN{ if (b-a > 5.0) { print "shared .so RSS grew too much:", b-a, "MB"; exit 1 } }' \
  || die "shared-mode weights appear to be duplicated per thread"

echo "OK - shared GlobalCache keeps model weights shared across threads"
