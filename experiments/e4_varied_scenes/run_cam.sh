#!/bin/bash
# E4's camera-format part: render the cam-* sets (the recorded flights' camera, see make_jobs.mjs),
# train cam3k (3000 runs, as var3k), cam3k-noair (the same without the airspeed inputs: the
# recorded flights carry no airspeed sensor, so this is the model that can be run on them) and
# blind-cam (sensors only, over the same ranges), then score all on test-cam, plain and through
# the camera effects.
#   nohup bash experiments/e4_varied_scenes/render_cam.sh >> /workspace/data/e4/logs/cam-render.log 2>&1 &
#   nohup bash experiments/e4_varied_scenes/run_cam.sh > /workspace/data/e4/logs/cam.log 2>&1 &
# It waits for render_cam.sh's CAM_RENDERS_DONE line, then checks every set is whole.
# E3-format networks cannot take these frames (e3.predict refuses a camera mismatch), so the
# comparison with var3k is on each model's own test set, not on one shared set.
# The last line of the log is `=== CAM done`, `=== CAM done, failed: <what>` or `=== CAM FAILED: <why>`.
set -u
D=/workspace/data/e4
PY=/workspace/venv/bin/python
REPO=/workspace/mare-realtime
S=$D/sets
J=$REPO/experiments/e4_varied_scenes/jobs
die() { echo "=== CAM FAILED: $* $(date -Is)"; exit 1; }
$PY -c "import tabulate" || die "tabulate is not installed"
until grep -q CAM_RENDERS_DONE $D/logs/cam-render.log 2>/dev/null; do
  pgrep -f e4_varied_scenes/render_cam.sh > /dev/null || { grep -q CAM_RENDERS_DONE $D/logs/cam-render.log 2>/dev/null || die "render_cam.sh stopped before CAM_RENDERS_DONE"; }
  sleep 60
done
check_set() {  # job file, set directory
  local want have
  want=$($PY -c "import json,sys; print(len(json.load(open(sys.argv[1]))['runs']))" $1)
  have=$(for r in $2/*/; do [ -f $r/frames.npy ] && [ -f $r/frames.csv ] && echo; done | wc -l)
  echo "=== $2: $have of $want runs"
  [ "$have" = "$want" ]
}
for set in val-cam test-cam cam-train; do check_set $J/$set.json $S/$set || die "$set incomplete"; done
cd $REPO/experiments/e3_relative_pose
ulimit -n 65536 2>/dev/null || ulimit -n $(ulimit -Hn)
echo "=== cam train start $(date -Is)"
T="--train $S/cam-train --val $S/val-cam --fx --epochs 16"
declare -A PID
train() {  # name, options
  local name=$1; shift
  $PY -m e3.train $T --out $D/$name "$@" > $D/logs/train-$name.log 2>&1 &
  PID[$name]=$!
}
train blind-cam --blind --workers 4 --batch 256
train cam3k --hours 16 --workers 10
train cam3k-noair --no-airspeed --hours 16 --workers 10
GOOD=(); BAD=()
for name in "${!PID[@]}"; do
  if wait ${PID[$name]}; then
    echo "=== trained $name $(date -Is)"; GOOD+=($name)
  else
    echo "=== TRAIN FAILED $name (exit $?), see logs/train-$name.log $(date -Is)"; BAD+=(train:$name)
  fi
done
[ ${#GOOD[@]} -gt 0 ] || die "no model trained: ${BAD[*]}"
echo "=== cam train done $(date -Is); scoring ${GOOD[*]}"
score() {  # name, set directory, predict options
  local name=$1 set=$2; shift 2
  local args=()
  for m in "${GOOD[@]}"; do
    if $PY -m e3.predict --ckpt $D/$m/best.pt --set $set --out $D/preds/$m-$name.npz "$@"; then
      args+=("$m=$D/preds/$m-$name.npz")
    else
      echo "=== PREDICT FAILED $m on $name"; BAD+=(predict:$m:$name)
    fi
  done
  [ ${#args[@]} -gt 0 ] || { echo "=== EVAL SKIPPED $name: no predictions"; BAD+=(eval:$name); return; }
  $PY -m e3.evaluate --set $set --out $D/eval/$name "${args[@]}" > $D/logs/eval-$name.log 2>&1 || { echo "=== EVAL FAILED $name"; BAD+=(eval:$name); }
}
score test-cam $S/test-cam
score test-cam-fx $S/test-cam --fx
if [ ${#BAD[@]} -gt 0 ]; then echo "=== CAM done, failed: ${BAD[*]} $(date -Is)"; else echo "=== CAM done $(date -Is)"; fi
