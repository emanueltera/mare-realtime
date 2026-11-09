#!/bin/bash
# E3 renders, each run shrunk to (N, 512, 288) luma as it finishes (python -m e3.shrink), PNGs deleted.
#   bash experiments/e3_relative_pose/render_all.sh <set> [<generate.mjs options>]
set -u
REPO=/workspace/mare-realtime
DATA=/workspace/data/e3
PY=/workspace/venv/bin/python
set_name=$1; shift
mkdir -p $DATA/render/$set_name $DATA/sets $DATA/logs
rm -f $DATA/render/$set_name/DONE
cd $REPO/experiments/e3_relative_pose
$PY -m e3.shrink --render $DATA/render/$set_name/e3-$set_name --out $DATA/sets/$set_name \
  --workers 16 --done-flag $DATA/render/$set_name/DONE > $DATA/logs/shrink-$set_name.log 2>&1 &
SHR=$!
cd $REPO
echo "=== render $set_name start $(date -Is)"
for attempt in 1 2; do
  node scripts/generate.mjs experiments/e3_relative_pose/jobs/$set_name.json --out $DATA/render/$set_name \
    --port 5311 --concurrency ${CONCURRENCY:-32} "$@" >> $DATA/logs/render-$set_name.log 2>&1
  code=$?
  echo "=== render $set_name attempt $attempt exit $code $(date -Is)"
  [ $code -eq 0 ] && break
done
touch $DATA/render/$set_name/DONE
wait $SHR
echo "=== shrink $set_name done $(date -Is)"
