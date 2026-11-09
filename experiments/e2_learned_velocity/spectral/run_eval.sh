#!/bin/bash
# Predict every window of the given sets with a frozen checkpoint and score them with the
# shared harness. Run on the box from anywhere:
#   bash spectral/run_eval.sh <method> <checkpoint> <set> [<set> ...]
# Predictions: /workspace/data/e2/preds/<method>/<set>.csv; harness: /workspace/data/e2/eval/<method>/<set>/.
set -euo pipefail
method=$1; ckpt=$2; shift 2
cd /workspace/mare-realtime/experiments/e2_learned_velocity
for s in "$@"; do
  pred=/workspace/data/e2/preds/$method/$s.csv
  /workspace/venv/bin/python -m spectral.predict --ckpt "$ckpt" --set "$s" --out "$pred" --workers ${WORKERS:-8}
  /workspace/venv/bin/python -m e2.evaluate --pred "$pred" --sets /workspace/data/e2/sets \
    --out /workspace/data/e2/eval/$method/$s --name "$method" --require-all
done
