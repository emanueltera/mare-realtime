#!/bin/bash
# One set of renders, each run shrunk to (N, 512, 288) luma as it finishes (e3.shrink), PNGs deleted.
#   bash experiments/e4_varied_scenes/render_all.sh <e3|e4> <set> [<generate.mjs options>]
# e3 re-renders one of E3's own job files (its sets live on a volume this box cannot reach).
set -u
REPO=/workspace/mare-realtime
PY=/workspace/venv/bin/python
exp=$1; set_name=$2; shift 2
DATA=/workspace/data/$exp
case $exp in e3) JOBS=experiments/e3_relative_pose/jobs ;; e4) JOBS=experiments/e4_varied_scenes/jobs ;; *) echo "unknown experiment $exp"; exit 2 ;; esac
mkdir -p $DATA/render/$set_name $DATA/sets $DATA/logs
rm -f $DATA/render/$set_name/DONE
cd $REPO/experiments/e3_relative_pose
$PY -m e3.shrink --render $DATA/render/$set_name/$exp-$set_name --out $DATA/sets/$set_name \
  --workers 12 --done-flag $DATA/render/$set_name/DONE > $DATA/logs/shrink-$set_name.log 2>&1 &
SHR=$!
cd $REPO
echo "=== render $exp $set_name start $(date -Is)"
for attempt in 1 2; do
  node scripts/generate.mjs $JOBS/$set_name.json --out $DATA/render/$set_name \
    --port 5311 --concurrency ${CONCURRENCY:-32} "$@" >> $DATA/logs/render-$set_name.log 2>&1
  code=$?
  echo "=== render $exp $set_name attempt $attempt exit $code $(date -Is)"
  [ $code -eq 0 ] && break
done
touch $DATA/render/$set_name/DONE
wait $SHR
echo "=== shrink $exp $set_name done $(date -Is)"
