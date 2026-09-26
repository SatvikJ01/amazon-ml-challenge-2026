#!/usr/bin/env bash
# Test side of stage 3: GPU rescore of pass-1 p2 (E030_stage2x, best iteration) -> anchor retrieval.
# pass2 + write run separately once the stage-3 model is chosen on the holdout.
set -u
cd "$(dirname "$0")/.."
CAP="systemd-run --user --scope -q -p MemoryMax=5500M -p MemorySwapMax=0"
for c in ${RESCORE:-France India US}; do
  $CAP .venv/bin/python -u -m src.infer_v3 rescore --country $c; rc=$?; echo "EXIT $rc rescore $c"
  [ $rc -eq 0 ] || { echo FAILED; exit 1; }
done
for c in France India US; do
  $CAP .venv/bin/python -u -m src.infer_v3 anchors --country $c; rc=$?; echo "EXIT $rc anchors $c"
  [ $rc -eq 0 ] || { echo FAILED; exit 1; }
done
echo ANCHORS_DONE
