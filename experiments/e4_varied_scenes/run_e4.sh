#!/bin/bash
# After the renders: train on the widened scenes, then score every model on every test set.
#   nohup bash experiments/e4_varied_scenes/run_e4.sh > /workspace/data/e4/logs/run.log 2>&1 &
# Models: var6k (all 6000 runs), var3k and var3k-s1 (3000 runs, as E3 had; two seeds, so the
# run-to-run spread shows), blind (sensors only), against e3 (E3's network, not retrained).
# The last line of the log is `=== E4 done`, `=== E4 done, failed: <what>` or `=== E4 FAILED: <why>`.
set -u
D=/workspace/data/e4
E3=/workspace/data/e3
PY=/workspace/venv/bin/python
REPO=/workspace/mare-realtime
S=$D/sets
die() { echo "=== E4 FAILED: $* $(date -Is)"; exit 1; }
[ -f $E3/net/best.pt ] || die "no E3 checkpoint at $E3/net/best.pt"
$PY -c "import tabulate" || die "tabulate is not installed"
until grep -q ALL_DONE $D/logs/chain.log 2>/dev/null; do
  pgrep -f e4_varied_scenes/chain.sh > /dev/null || { grep -q ALL_DONE $D/logs/chain.log 2>/dev/null || die "the render chain stopped before ALL_DONE"; }
  sleep 60
done
# Every set whole, or nothing trains on it: generate and shrink record failures without stopping.
check_set() {  # job file, set directory
  local want have
  want=$($PY -c "import json,sys; print(len(json.load(open(sys.argv[1]))['runs']))" $1)
  have=$(ls $2/*/frames.npy 2>/dev/null | wc -l)
  echo "=== $2: $have of $want runs"
  [ "$have" = "$want" ]
}
J3=$REPO/experiments/e3_relative_pose/jobs
J4=$REPO/experiments/e4_varied_scenes/jobs
check_set $J3/test-long.json $E3/sets/test-long || die "test-long incomplete"
check_set $J3/test-rate.json $E3/sets/test-rate || die "test-rate incomplete"
for set in val-varied test-varied train; do check_set $J4/$set.json $S/$set || die "$set incomplete"; done
cd $REPO/experiments/e3_relative_pose
ulimit -n 65536 2>/dev/null || ulimit -n $(ulimit -Hn)
echo "=== train start $(date -Is)"
T="--train $S/train --val $S/val-varied --fx --epochs 16"
declare -A PID
train() {  # name, options
  local name=$1; shift
  $PY -m e3.train $T --out $D/$name "$@" > $D/logs/train-$name.log 2>&1 &
  PID[$name]=$!
}
train blind --blind --workers 4 --batch 256
train var3k --limit 3000 --hours 16 --workers 10
train var3k-s1 --limit 3000 --seed 1 --hours 16 --workers 10
train var6k --hours 24 --workers 12
GOOD=(e3); BAD=()
for name in "${!PID[@]}"; do
  if wait ${PID[$name]}; then
    echo "=== trained $name $(date -Is)"; GOOD+=($name)
  else
    echo "=== TRAIN FAILED $name (exit $?), see logs/train-$name.log $(date -Is)"; BAD+=(train:$name)
  fi
done
echo "=== train done $(date -Is); scoring ${GOOD[*]}"
ckpt() { case $1 in e3) echo $E3/net/best.pt ;; *) echo $D/$1/best.pt ;; esac; }
score() {  # name, set directory, predict options
  local name=$1 set=$2; shift 2
  local args=()
  for m in "${GOOD[@]}"; do
    if $PY -m e3.predict --ckpt $(ckpt $m) --set $set --out $D/preds/$m-$name.npz "$@"; then
      args+=("$m=$D/preds/$m-$name.npz")
    else
      echo "=== PREDICT FAILED $m on $name"; BAD+=(predict:$m:$name)
    fi
  done
  $PY -m e3.evaluate --set $set --out $D/eval/$name "${args[@]}" > $D/logs/eval-$name.log 2>&1 || { echo "=== EVAL FAILED $name"; BAD+=(eval:$name); }
}
score test-long $E3/sets/test-long
score test-rate $E3/sets/test-rate
score test-varied $S/test-varied
score test-varied-fx $S/test-varied --fx
if [ ${#BAD[@]} -gt 0 ]; then echo "=== E4 done, failed: ${BAD[*]} $(date -Is)"; else echo "=== E4 done $(date -Is)"; fi
