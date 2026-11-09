#!/bin/bash
# B1 pair velocities: SEA-RAFT-M on consecutive tiles, every set the baselines score, plus the
# first 1500 train runs (B3's training data). On the box, ~5 GB of GPU memory.
set -euo pipefail
cd /workspace/mare-realtime/experiments/e2_learned_velocity
OUT=/workspace/data/e2/baselines/pairs/searaft
mkdir -p $OUT
export HF_HOME=/workspace/.hf_home
for s in val train test-iid test-sea test-alt test-rate e1-core; do
  lim=""; [ "$s" = train ] && lim="--limit 1500"
  start=$(date +%s)
  /workspace/venv/bin/python -m baselines.flowpairs --set-root /workspace/data/e2/sets/$s --out $OUT/$s.csv $lim
  echo "$s $(( $(date +%s) - start )) s"
done
echo done
