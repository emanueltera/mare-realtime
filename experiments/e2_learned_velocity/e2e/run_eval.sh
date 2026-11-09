#!/usr/bin/env bash
# Predict every window of the given sets with a frozen checkpoint and score each set with the
# shared harness (on the box, from experiments/e2_learned_velocity):
#   [GPU_MEM_GB=8] bash e2e/run_eval.sh <checkpoint> <method> <set>...
# → /workspace/data/e2/preds/<method>/<set>.csv (+ .npz) and /workspace/data/e2/eval/<method>/<set>/
set -euo pipefail
ckpt=$1; method=$2; shift 2
cd "$(dirname "$0")/.."
ulimit -n 65536
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for s in "$@"; do
  /workspace/venv/bin/python -m e2e.predict --ckpt "$ckpt" --set "$s" --out "/workspace/data/e2/preds/$method/$s.csv" --gpu-mem-gb "${GPU_MEM_GB:-8}"
  /workspace/venv/bin/python -m e2.evaluate --pred "/workspace/data/e2/preds/$method/$s.csv" --sets /workspace/data/e2/sets \
    --out "/workspace/data/e2/eval/$method/$s" --name "$method" --require-all
done
/workspace/venv/bin/python -m e2e.calib --preds "/workspace/data/e2/preds/$method" --out "/workspace/data/e2/eval/$method"
