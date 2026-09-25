#!/usr/bin/env bash
# Exact-key channel (caps S1<=10, T<=60), one capped process per (split, country, source).
set -u
cd "$(dirname "$0")/.."
CAP="systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0"
for c in India US; do for s in 2 3; do $CAP .venv/bin/python -u -m src.key_channel --split train --tag trnall --country $c --source $s --max-s1 10 --max-t 60; echo "EXIT $? train $c S$s"; done; done
for c in France India US; do for s in 2 3; do $CAP .venv/bin/python -u -m src.key_channel --split test --tag testall --country $c --source $s --max-s1 10 --max-t 60; echo "EXIT $? test $c S$s"; done; done
echo ALL_DONE
