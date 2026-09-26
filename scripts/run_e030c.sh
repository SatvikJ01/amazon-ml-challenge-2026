#!/usr/bin/env bash
# E030 remainder: the US anchor build was OOM-killed at 6G; rebuild it with more headroom
# (nothing else heavy is running), then train stage 3.
set -u
cd "$(dirname "$0")/.."
step() { local cap=$1; shift; echo ">>> $*"; systemd-run --user --scope -q -p MemoryMax=$cap -p MemorySwapMax=0 .venv/bin/python -u -m "$@"; local rc=$?; echo "EXIT $rc $1 $2"; return $rc; }
step 9G src.anchor_pass build --country US --old-tag v3c --out-tag v3c_anc --entities entities_v3c.npy --hits-dir E030_anchor || { echo FAILED; exit 1; }
step 7G src.train --tag v3c_anc --exp E030_stage3 || { echo FAILED; exit 1; }
echo ALL_DONE
