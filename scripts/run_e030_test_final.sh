#!/usr/bin/env bash
# After the E030 training chain and test pass 1: stage-3 test pass (if stage 3 wins on
# holdout) and write the day2_s1 candidate submission. Uploading is a manual decision.
set -u
cd "$(dirname "$0")/.."
until grep -q ALL_DONE logs/E030.log && grep -q ALL_DONE logs/test_v3_pass1.log; do sleep 30; done
CAP="systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0"
S3=$(python3 -c "import json;print(json.load(open('experiments/E030_stage3/report.json'))['best_f05'])")
S2=$(python3 -c "import json;print(json.load(open('experiments/E030_stage2/report.json'))['best_f05'])")
echo "holdout: stage2 $S2  stage3 $S3"
if python3 -c "import sys; sys.exit(0 if $S3 >= $S2 else 1)"; then
  for c in France India US; do
    for a in 1 2 3; do $CAP .venv/bin/python -u -m src.infer_v3 pass2 --country $c && break; echo "RETRY pass2 $c"; sleep 20; done
  done
  $CAP .venv/bin/python -u -m src.infer_v3 write --sub-id day2_s1 --note "v3 stage 3 (holdout $S3)"; echo "EXIT $? write"
else
  $CAP .venv/bin/python -u -m src.infer_v3 write --sub-id day2_s1 --from-pass1 --note "v3 stage 2 (holdout $S2); stage 3 did not beat it"; echo "EXIT $? write"
fi
echo ALL_DONE
