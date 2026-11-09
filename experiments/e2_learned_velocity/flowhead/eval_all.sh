#!/usr/bin/env bash
# Predict every window of the given sets with a frozen flowhead checkpoint and score each with
# the shared harness. Run on the box from anywhere:
#   bash experiments/e2_learned_velocity/flowhead/eval_all.sh <method> <ckpt> <set> [<set> …]
# → /workspace/data/e2/preds/<method>/<set>.csv (+ .diag.csv), /workspace/data/e2/eval/<method>/<set>/
set -euo pipefail
method=$1; ckpt=$2; shift 2
cd "$(dirname "$0")/.."
py=/workspace/venv/bin/python
export HF_HOME=/workspace/.hf_home
for s in "$@"; do
  pred=/workspace/data/e2/preds/$method/$s.csv
  $py -W ignore -m flowhead.predict --ckpt "$ckpt" --set "$s" --out "$pred"
  $py -m e2.evaluate --pred "$pred" --sets /workspace/data/e2/sets --out /workspace/data/e2/eval/$method/$s --name "$method"
done
