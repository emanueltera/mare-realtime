#!/bin/bash
# Every render E4 needs, the test sets first so a failure shows early.
#   nohup bash experiments/e4_varied_scenes/chain.sh >> /workspace/data/e4/logs/chain.log 2>&1 &
set -u
cd /workspace/mare-realtime
R=experiments/e4_varied_scenes/render_all.sh
bash $R e3 test-long
bash $R e3 test-rate
bash $R e4 val-varied
bash $R e4 test-varied
bash $R e4 train
echo "ALL_DONE $(date -Is)"
