#!/bin/bash
# The camera-format sets' renders (see make_jobs.mjs), test sets first so a failure shows early.
#   nohup bash experiments/e4_varied_scenes/render_cam.sh >> /workspace/data/e4/logs/cam-render.log 2>&1 &
# CONCURRENCY (default 24) is lower than the portrait sets' 32: E4's models may still be on the GPU.
set -u
cd /workspace/mare-realtime
export CONCURRENCY=${CONCURRENCY:-24}
R=experiments/e4_varied_scenes/render_all.sh
bash $R e4 val-cam
bash $R e4 test-cam
bash $R e4 cam-train
echo "CAM_RENDERS_DONE $(date -Is)"
