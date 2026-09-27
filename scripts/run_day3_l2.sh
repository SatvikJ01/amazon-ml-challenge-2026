#!/usr/bin/env bash
# day3_l2 = day3_e039 + residual level-2 (ALL_nc, holdout +0.00085) on India/US; France keeps prob3.
# Starts after the E039 test chain has written day3_e039.
set -u
cd "$(dirname "$0")/.."
until grep -qE "ALL_DONE|FAILED" logs/E039_test.log; do sleep 30; done
grep -q "archived" logs/E039_test.log || { echo "E039 write did not archive"; echo FAILED; exit 1; }
for c in India US France; do
  systemd-run --user --scope -q -p MemoryMax=5G -p MemorySwapMax=0 nice -n 10 env OMP_NUM_THREADS=2 \
    .venv/bin/python scripts/apply_l2.py --run E039_test --countries India US --process $c
  rc=$?; echo "EXIT $rc apply_l2 $c"; [ $rc -eq 0 ] || { echo FAILED; exit 1; }
done
tail -3 experiments/E039_test/apply_l2.log 2>/dev/null
.venv/bin/python scripts/apply_l2.py --run E039_test --countries India US --process India US France --no-gates 2>&1 | tail -3 | tee /tmp/claude-1000/-home-alphatron-PADHAI-LIKHAI-Semester-IX-Amazon-ML-Challenge-ML-Challenge/3c84b878-f755-4117-a9de-8f906729ee2b/scratchpad/l2_check.txt
grep -q "ALL PARTS COMPLETE" /tmp/claude-1000/-home-alphatron-PADHAI-LIKHAI-Semester-IX-Amazon-ML-Challenge-ML-Challenge/3c84b878-f755-4117-a9de-8f906729ee2b/scratchpad/l2_check.txt || { echo "L2 parts incomplete"; echo FAILED; exit 1; }
systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 .venv/bin/python -u -m src.infer_v3 write --run E039_test --split testT --tag tstT --scores scores_l2 --sub-id day3_l2 --note "E039 + residual level-2 ALL_nc (holdout +0.00085) on India/US; France raw"
echo "EXIT $? write"; echo ALL_DONE
