#!/bin/bash
# After the renders: train the network and its sensor-only twin, predict every test set, score.
#   nohup bash experiments/e3_relative_pose/run_e3.sh > /workspace/data/e3/logs/run.log 2>&1 &
set -u
DATA=/workspace/data/e3
PY=/workspace/venv/bin/python
S=$DATA/sets
cd /workspace/mare-realtime/experiments/e3_relative_pose
until grep -q ALL_DONE $DATA/logs/chain.log; do sleep 60; done
ulimit -n 65536 2>/dev/null || ulimit -n $(ulimit -Hn)
echo "=== train start $(date -Is)"
$PY -m e3.train --train $S/train --val $S/val --out $DATA/blind --blind --epochs 16 --workers 16 --batch 256 > $DATA/logs/train-blind.log 2>&1 &
$PY -m e3.train --train $S/train --val $S/val --out $DATA/net --epochs 16 --hours 4 > $DATA/logs/train-net.log 2>&1
wait
echo "=== train done $(date -Is)"
for m in net blind; do [ -f $DATA/$m/best.pt ] || { echo "=== $m did not train; see logs/train-$m.log"; echo "=== E3 FAILED $(date -Is)"; exit 1; }; done
for set in val test-long test-rate; do
  for m in net blind; do $PY -m e3.predict --ckpt $DATA/$m/best.pt --set $S/$set --out $DATA/preds/$m-$set.npz; done
  $PY -m e3.evaluate --set $S/$set --out $DATA/eval/$set net=$DATA/preds/net-$set.npz blind=$DATA/preds/blind-$set.npz > $DATA/logs/eval-$set.log 2>&1
done
echo "=== E3 done $(date -Is)"
