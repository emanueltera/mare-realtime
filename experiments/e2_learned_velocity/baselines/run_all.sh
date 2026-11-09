#!/bin/bash
# E2 baselines B0–B3, end to end, on the box (/workspace/mare-realtime, /workspace/venv).
# Steps can run concurrently where noted; none renders, so none needs the GPU to itself.
# Outputs: pair velocities and B2/B3 working files under /workspace/data/e2/baselines/,
# predictions under /workspace/data/e2/preds/<method>/<set>.csv, harness outputs under
# /workspace/data/e2/eval/<method>/<set>/.
set -euo pipefail
cd /workspace/mare-realtime/experiments/e2_learned_velocity
PY=/workspace/venv/bin/python
SETS="val test-iid test-sea test-alt test-rate e1-core"
B=/workspace/data/e2/baselines
mkdir -p $B/logs

# B0: phase-correlation pairs (CPU, 16 processes, ~8 min for all six sets) → K = 2/8/16 windows → harness.
bash baselines/b0_pairs.sh                                   # writes $B/pairs/phasecorr/<set>.csv
bash baselines/eval_windows.sh b0-phasecorr phasecorr $SETS

# B1: SEA-RAFT-M pairs (GPU, ~5 GB; val, the first 1500 train runs, the test sets) → windows → harness.
bash baselines/b1_pairs.sh                                   # writes $B/pairs/searaft/<set>.csv
bash baselines/eval_windows.sh b1-searaft searaft $SETS

# B2: shell fit (6 processes, ~1.5 GB GPU each), then the harness on each model's predictions.
bash baselines/b2_shell.sh                                   # writes $B/shell/<set>/pred-{dispersive,static}.csv
for m in dispersive static; do
  for s in $SETS e1-core-kmax0475; do
    [ -f $B/shell/$s/pred-$m.csv ] || continue
    meth=b2-shell-$m; [ $s = e1-core-kmax0475 ] && meth=b2-shell-$m-kmax0475
    set=${s%-kmax0475}
    mkdir -p /workspace/data/e2/preds/$meth
    cp $B/shell/$s/pred-$m.csv /workspace/data/e2/preds/$meth/$set.csv
    $PY -m e2.evaluate --pred /workspace/data/e2/preds/$meth/$set.csv --sets /workspace/data/e2/sets \
      --out /workspace/data/e2/eval/$meth/$set --name $meth --require-all
  done
done

# B3: B1's K = 8 windows on the train subset, smoke test, grid on val, freeze, predict, harness.
$PY -m baselines.windows --pairs $B/pairs/searaft/train.csv --set-root /workspace/data/e2/sets/train \
  --out /workspace/data/e2/preds/b1-searaft/train.csv --K 8
$PY -m baselines.correction smoke --preds /workspace/data/e2/preds/b1-searaft --out $B/b3 | tee $B/logs/b3_smoke.log
$PY -m baselines.correction train --preds /workspace/data/e2/preds/b1-searaft --out $B/b3 | tee $B/logs/b3_train.log
$PY -m baselines.correction predict --preds /workspace/data/e2/preds/b1-searaft --ckpt $B/b3/best.pt \
  --out /workspace/data/e2/preds/b3-mlp --sets $SETS
$PY -m baselines.correction predict --preds /workspace/data/e2/preds/b1-searaft --ckpt $B/b3/best.pt --const \
  --out /workspace/data/e2/preds/b3-const --sets $SETS
for f in full radial; do   # ablation: the best of each parameterisation on val
  $PY -m baselines.correction predict --preds /workspace/data/e2/preds/b1-searaft --ckpt $B/b3/best-$f.pt \
    --out /workspace/data/e2/preds/b3-mlp-$f --sets $SETS
done
for m in b3-mlp b3-const b3-mlp-full b3-mlp-radial; do
  for s in $SETS; do
    $PY -m e2.evaluate --pred /workspace/data/e2/preds/$m/$s.csv --sets /workspace/data/e2/sets \
      --out /workspace/data/e2/eval/$m/$s --name $m --require-all
  done
done

# Cross-check against E1 on e1-core, and the tables NOTES.md quotes.
$PY -m baselines.compare_e1 --eval /workspace/data/e2/eval --out $B/e1_crosscheck
$PY -m baselines.tables --eval /workspace/data/e2/eval --out $B/tables
