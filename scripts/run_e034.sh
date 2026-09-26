#!/usr/bin/env bash
# E034: name-substitution + shared-address features on top of E033; starts after E033 training.
set -u
cd "$(dirname "$0")/.."
until grep -qE "EXIT [0-9]+ train E033" logs/E033.log; do sleep 20; done
for c in India US; do
  systemd-run --user --scope -q -p MemoryMax=5500M -p MemorySwapMax=0 .venv/bin/python -u -m src.patch_extra --country $c --tag v3c_ancx --out-tag v3c_ancy --sets name
  rc=$?; echo "EXIT $rc patch $c"; [ $rc -eq 0 ] || { echo FAILED; exit 1; }
done
systemd-run --user --scope -q -p MemoryMax=7G -p MemorySwapMax=0 .venv/bin/python -u -m src.train --tag v3c_ancy --exp E034_stage3
rc=$?; echo "EXIT $rc train E034_stage3"; [ $rc -eq 0 ] && echo ALL_DONE || echo FAILED
