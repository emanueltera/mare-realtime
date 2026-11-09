#!/usr/bin/env bash
# Start one model-A training in the background on the box, logging to <out>/train.log:
#   bash e2e/run_train.sh <out dir> [e2e.train options]
set -euo pipefail
out=$1; shift
cd "$(dirname "$0")/.."
mkdir -p "$out"
ulimit -n 65536
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup /workspace/venv/bin/python -m e2e.train --out "$out" "$@" > "$out/train.log" 2>&1 < /dev/null &
echo "started pid $! -> $out/train.log"
