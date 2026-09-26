#!/usr/bin/env bash
# Evening queue on the laptop (one heavy job at a time): after day2_s2 ->
# retrieval-v4 pilot (E037) -> stage 4 (E036).
set -u
cd "$(dirname "$0")/.."
until grep -qE "ALL_DONE|FAILED" logs/day2_s2.log; do sleep 30; done
for c in India US; do for s in 2 3; do
  systemd-run --user --scope -q -p MemoryMax=6500M -p MemorySwapMax=0 .venv/bin/python -u -m src.exp_retrieval_v4 search --country $c --source $s
  echo "EXIT $? search $c $s"
done; done
systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 .venv/bin/python -u -m src.exp_retrieval_v4 report
echo "EXIT $? report"
echo PILOT_DONE
bash scripts/run_e036.sh
