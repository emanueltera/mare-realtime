#!/bin/bash
# B2: E1's shell fit on E2 tiles, every set the baselines score (dispersive and static models,
# K = 32 and 64, windows every 16 frames). Then the kmax ablation on e1-core: E1's physical
# wavelength range (E1's cell is ~1.69x E2's, so kmax_frac 0.8 / 1.69 = 0.475).
set -euo pipefail
cd /workspace/mare-realtime/experiments/e2_learned_velocity
OUT=/workspace/data/e2/baselines/shell
export OMP_NUM_THREADS=2
for s in e1-core test-iid test-sea test-alt test-rate val; do
  start=$(date +%s)
  /workspace/venv/bin/python -m baselines.shell --set-root /workspace/data/e2/sets/$s --out-dir $OUT/$s --jobs 6 --chunk 256 > $OUT/$s.log 2>&1
  echo "$s $(( $(date +%s) - start )) s"
done
start=$(date +%s)
/workspace/venv/bin/python -m baselines.shell --set-root /workspace/data/e2/sets/e1-core --out-dir $OUT/e1-core-kmax0475 \
  --kmax-frac 0.475 --jobs 6 --chunk 256 > $OUT/e1-core-kmax0475.log 2>&1
echo "e1-core-kmax0475 $(( $(date +%s) - start )) s"
echo done
