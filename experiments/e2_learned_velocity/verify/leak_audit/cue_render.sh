#!/bin/bash
# render the cue job with each renderer variant in turn, then rectify
A=/workspace/data/e2/audit_leak
port=5291
for v in ctrl gate0 gaterev foam0 cloud0; do
  cd $A/mare-$v
  echo "=== $v start $(date -Is)"
  node scripts/generate.mjs $A/cue_job.json --out $A/cue_render/$v --port $port --concurrency 16 > $A/logs/render-$v.log 2>&1
  echo "=== $v render exit $? $(date -Is)"
  port=$((port+1))
done
cd $A
for v in ctrl gate0 gaterev foam0 cloud0; do
  OMP_NUM_THREADS=1 /workspace/venv/bin/python cue_rectify.py $v > $A/logs/rectify-$v.log 2>&1
  echo "=== $v rectify exit $? $(date -Is)"
done
echo ALLDONE
