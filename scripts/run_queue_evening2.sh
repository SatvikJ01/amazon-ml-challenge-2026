#!/usr/bin/env bash
# Retrieval pilot (E037), all channels -> report -> stage 4 (E036, waits for day2_s2).
set -u
cd "$(dirname "$0")/.."
while kill -0 1919385 2>/dev/null; do sleep 10; done
S="systemd-run --user --scope -q -p MemoryMax=6500M -p MemorySwapMax=0 .venv/bin/python -u -m src.exp_retrieval_v4 search"
$S --country India --source 2 --channels cname,caddr,bm25; echo "EXIT $? India 2"
$S --country India --source 3 --channels cname,caddr,bm25; echo "EXIT $? India 3"
$S --country US --source 2 --channels name,addr,cname,caddr,bm25; echo "EXIT $? US 2"
$S --country US --source 3 --channels name,addr,cname,caddr,bm25; echo "EXIT $? US 3"
systemd-run --user --scope -q -p MemoryMax=5G -p MemorySwapMax=0 .venv/bin/python -u -m src.exp_retrieval_v4 report
echo "EXIT $? report"
echo PILOT_DONE
bash scripts/run_e036.sh
