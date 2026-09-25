#!/usr/bin/env bash
# E030 v3 pipeline on 300k training entities (old 30k holdout included for comparison).
set -u
cd "$(dirname "$0")/.."
until grep -qE "EXIT [0-9]+ train US S3" logs/keys.log; do sleep 15; done
CAP="systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0"
PY=".venv/bin/python -u -m"
step() { echo ">>> $*"; $CAP $PY "$@"; local rc=$?; echo "EXIT $rc $1 $2 ${4:-}"; return $rc; }
for c in India US; do step src.v3 s1data --country $c --entities data/processed/entities_v3.npy || exit 1; done
cp data/processed/entities_v3.npy data/processed/entities_v3s1.npy
step src.stage1 --tag v3s1 --out E030_stage1 --v3 || exit 1
for c in India US; do step src.v3 s2data --country $c --stage1 E030_stage1 --entities data/processed/entities_v3.npy || exit 1; done
step src.train --tag v3c --exp E030_stage2 || exit 1
step src.collective oof --tag v3c --exp E030_stage2 --out E030_collective || exit 1
for c in India US; do step src.collective build --tag v3c --out E030_collective --country $c || exit 1; done
for c in India US; do step src.anchor_pass retrieve --split train --country $c --oof-dir E030_collective --hits-dir E030_anchor || exit 1; done
for c in India US; do step src.anchor_pass build --country $c --old-tag v3c --out-tag v3c_anc --entities entities_v3c.npy --hits-dir E030_anchor || exit 1; done
step src.train --tag v3c_anc --exp E030_stage3 || exit 1
echo ALL_DONE
