#!/usr/bin/env bash
# E013: cascade = cross-fitted stage 1 -> survivor training set -> stage-2 LightGBM.
set -u
cd "$(dirname "$0")/.."
CAP="systemd-run --user --scope -q -p MemoryMax=5000M -p MemorySwapMax=0"
# stage 1 already trained (E013_stage1)
$CAP .venv/bin/python -u -m src.make_stage2 --tag trn2 --stage1 E013_stage1 --out-tag trn2c; echo "EXIT $? make_stage2"
$CAP .venv/bin/python -u -m src.train --tag trn2c --exp E013_cascade_stage2; echo "EXIT $? train"
echo ALL_DONE
