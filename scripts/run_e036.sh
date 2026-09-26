#!/usr/bin/env bash
# E036 stage 4: cross-fitted stage-3 probabilities (GPU) -> collective features from p3 -> LightGBM.
set -u
cd "$(dirname "$0")/.."
step() { local cap=$1; shift; echo ">>> $*"; systemd-run --user --scope -q -p MemoryMax=$cap -p MemorySwapMax=0 .venv/bin/python -u -m "$@"; local rc=$?; echo "EXIT $rc $1 $2"; [ $rc -eq 0 ] || { echo FAILED; exit 1; }; }
step 7G src.collective oof --tag v3c_ancz --exp E035_stage3 --out E036_oof --model xgb
for c in India US; do step 5500M src.stage4 build --country $c; done
step 7G src.train --tag v3c_s4 --exp E036_stage4
echo ALL_DONE
