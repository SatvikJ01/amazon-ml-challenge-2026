#!/usr/bin/env bash
# E030 test inference: pass 1 (India, US) -> GPU rescore -> anchors -> stage-3 pass -> day2_s1 files.
set -u
cd "$(dirname "$0")/.."
CAP="systemd-run --user --scope -q -p MemoryMax=5G -p MemorySwapMax=0"
while kill -0 101694 2>/dev/null; do sleep 20; done          # India pass 1 already running
for c in India US; do
  for a in 1 2 3; do $CAP .venv/bin/python -u -m src.infer_v3 pass1 --country $c && break; echo "RETRY pass1 $c"; sleep 30; done
done
echo PASS1_DONE
until [ -f experiments/E030_collective/oof_p2.parquet ] && [ -f experiments/E030_stage2x/model.json ]; do sleep 30; done
for c in France India US; do $CAP .venv/bin/python -u -m src.infer_v3 rescore --country $c; echo "EXIT $? rescore $c"; done
for c in France India US; do $CAP .venv/bin/python -u -m src.infer_v3 anchors --country $c; echo "EXIT $? anchors $c"; done
until grep -q ALL_DONE logs/E030b.log; do sleep 30; done
S3=$(python3 -c "import json;print(json.load(open('experiments/E030_stage3/report.json'))['best_f05'])")
S2=$(python3 -c "import json;print(json.load(open('experiments/E030_stage2/report.json'))['best_f05'])")
echo "holdout: stage2 $S2  stage3 $S3"
if python3 -c "import sys; sys.exit(0 if $S3 >= $S2 else 1)"; then
  for c in France India US; do
    for a in 1 2 3; do $CAP .venv/bin/python -u -m src.infer_v3 pass2 --country $c && break; echo "RETRY pass2 $c"; sleep 20; done
  done
  $CAP .venv/bin/python -u -m src.infer_v3 write --sub-id day2_s1 --note "v3 stage 3 (holdout $S3)"; echo "EXIT $? write"
else
  $CAP .venv/bin/python -u -m src.infer_v3 write --sub-id day2_s1 --from-pass1 --note "v3 stage 2 (holdout $S2)"; echo "EXIT $? write"
fi
echo ALL_DONE
