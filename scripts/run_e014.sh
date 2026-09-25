#!/usr/bin/env bash
# E014: leave-one-country-out at full density (France proxy) + exclusivity test.
# Both cascade stages are trained on US only; all 883k India train entities are scored.
set -u
cd "$(dirname "$0")/.."
CAP="systemd-run --user --scope -q -p MemoryMax=5000M -p MemorySwapMax=0"
$CAP .venv/bin/python -u -m src.stage1 --tag trn2 --out E014_stage1_US --countries US; echo "EXIT $? stage1"
$CAP .venv/bin/python -u -m src.make_stage2 --tag trn2 --stage1 E014_stage1_US --out-tag trn2cUS --countries US; echo "EXIT $? make_stage2"
$CAP .venv/bin/python -u -m src.train --tag trn2cUS --exp E014_loco_US --countries US; echo "EXIT $? train"
$CAP .venv/bin/python -u -m src.evaluate_full --exp E014_loco_US --stage1 E014_stage1_US --country India --depth 30; echo "EXIT $? eval_full"
echo ALL_DONE
