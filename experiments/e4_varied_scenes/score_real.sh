#!/bin/bash
# After run_cam.sh: run cam3k-noair on a recorded flight's over-water stretches (extract_real.py's
# output) and score it against the flight's GNSS, with the camera's gyro and with the attitude's
# rotations as the gyro input (e3/real.py).
#   nohup bash experiments/e4_varied_scenes/score_real.sh /workspace/data/real/<flight> > /workspace/data/e4/logs/real.log 2>&1 &
# The last line is `=== REAL done` or `=== REAL FAILED: <why>`.
set -u
REAL=$1
D=/workspace/data/e4
PY=/workspace/venv/bin/python
die() { echo "=== REAL FAILED: $* $(date -Is)"; exit 1; }
[ -f $REAL/index.json ] || die "no index.json in $REAL"
until grep -q "=== CAM" $D/logs/cam.log 2>/dev/null; do
  pgrep -f e4_varied_scenes/run_cam.sh > /dev/null || { grep -q "=== CAM" $D/logs/cam.log 2>/dev/null || die "run_cam.sh stopped without a final line"; }
  sleep 60
done
grep -q "=== CAM FAILED" $D/logs/cam.log && die "run_cam.sh failed"
[ -f $D/cam3k-noair/best.pt ] || die "no cam3k-noair checkpoint"
cd /workspace/mare-realtime/experiments/e3_relative_pose
name=$(basename $REAL)
for g in gyro att; do
  $PY -m e3.real --ckpt $D/cam3k-noair/best.pt --real $REAL --out $D/eval/real-$name-$g --gyro $g > $D/logs/real-$name-$g.log 2>&1 || die "e3.real --gyro $g, see logs/real-$name-$g.log"
  echo "=== scored $name, gyro $g $(date -Is)"
done
echo "=== REAL done $(date -Is)"
