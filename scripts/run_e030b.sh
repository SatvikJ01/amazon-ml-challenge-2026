#!/usr/bin/env bash
# E030 training remainder with GPU cross-fitting (the LightGBM version needed ~3 h under contention).
set -u
cd "$(dirname "$0")/.."
CAP="systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0"
step() { echo ">>> $*"; $CAP .venv/bin/python -u -m "$@"; local rc=$?; echo "EXIT $rc $1 $2"; return $rc; }
step src.train --tag v3c --exp E030_stage2x --model xgb || exit 1
step src.collective oof --tag v3c --exp E030_stage2 --out E030_collective --model xgb || exit 1
for c in India US; do step src.collective build --tag v3c --out E030_collective --country $c || exit 1; done
for c in India US; do step src.anchor_pass retrieve --split train --country $c --oof-dir E030_collective --hits-dir E030_anchor || exit 1; done
for c in India US; do step src.anchor_pass build --country $c --old-tag v3c --out-tag v3c_anc --entities entities_v3c.npy --hits-dir E030_anchor || exit 1; done
step src.train --tag v3c_anc --exp E030_stage3 || exit 1
echo ALL_DONE
