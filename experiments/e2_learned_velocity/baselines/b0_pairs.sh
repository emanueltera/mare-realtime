#!/bin/bash
# B0 pair velocities: masked phase correlation between consecutive tiles (K = 2), e2.baselines
# unchanged, for every set the baselines score (CPU only, 16 processes).
set -euo pipefail
cd /workspace/mare-realtime/experiments/e2_learned_velocity
mkdir -p /workspace/data/e2/baselines/pairs/phasecorr
for s in test-iid test-sea test-alt val test-rate e1-core; do
  start=$(date +%s)
  OMP_NUM_THREADS=1 /workspace/venv/bin/python -m e2.baselines --set-root /workspace/data/e2/sets/$s \
    --out /workspace/data/e2/baselines/pairs/phasecorr/$s.csv --workers 16
  echo "$s $(( $(date +%s) - start )) s"
done
echo done
