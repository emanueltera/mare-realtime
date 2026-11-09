#!/bin/bash
# E2 renders, each set rectified as it renders (python -m e2.rectify stream), PNGs deleted
# once a run's tiles are verified (the first 5 runs of each set keep theirs).
#   bash experiments/e2_learned_velocity/render_all.sh <set> [<generate.mjs options>] ...
# e.g. render_all.sh val; render_all.sh train --limit 4200
set -u
REPO=/workspace/mare-realtime
DATA=/workspace/data/e2
PY=/workspace/venv/bin/python
set_name=$1; shift
mkdir -p $DATA/render/$set_name $DATA/sets $DATA/logs
rm -f $DATA/render/$set_name/DONE
cd $REPO/experiments/e2_learned_velocity
$PY -m e2.rectify stream --render $DATA/render/$set_name/e2-$set_name --out $DATA/sets/$set_name --set $set_name \
  --keep 5 --workers 12 --done-flag $DATA/render/$set_name/DONE > $DATA/logs/rectify-$set_name.log 2>&1 &
RECT=$!
cd $REPO
echo "=== render $set_name start $(date -Is)"
for attempt in 1 2; do
  node scripts/generate.mjs experiments/e2_learned_velocity/jobs/$set_name.json --out $DATA/render/$set_name \
    --port 5211 --concurrency ${CONCURRENCY:-32} "$@" >> $DATA/logs/render-$set_name.log 2>&1
  code=$?
  echo "=== render $set_name attempt $attempt exit $code $(date -Is)"
  [ $code -eq 0 ] && break
done
touch $DATA/render/$set_name/DONE
wait $RECT
echo "=== rectify $set_name done $(date -Is)"
