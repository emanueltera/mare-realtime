#!/bin/bash
# Pair velocities → K = 2 / 8 / 16 window predictions → the shared harness, for B0 or B1.
#   bash baselines/eval_windows.sh b0-phasecorr phasecorr      (pairs from e2.baselines)
#   bash baselines/eval_windows.sh b1-searaft searaft          (pairs from baselines.flowpairs)
set -euo pipefail
METHOD=$1; PAIRS=$2; shift 2
SETS=${*:-val test-iid test-sea test-alt test-rate e1-core}
cd /workspace/mare-realtime/experiments/e2_learned_velocity
PY=/workspace/venv/bin/python
for s in $SETS; do
  $PY -m baselines.windows --pairs /workspace/data/e2/baselines/pairs/$PAIRS/$s.csv --set-root /workspace/data/e2/sets/$s \
    --out /workspace/data/e2/preds/$METHOD/$s.csv --K 2 8 16
  $PY -m e2.evaluate --pred /workspace/data/e2/preds/$METHOD/$s.csv --sets /workspace/data/e2/sets \
    --out /workspace/data/e2/eval/$METHOD/$s --name $METHOD --require-all
done
