#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
until grep -q ALL_DONE logs/E023.log; do sleep 15; done
CAP="systemd-run --user --scope -q -p MemoryMax=5000M -p MemorySwapMax=0"
for c in India US; do $CAP .venv/bin/python -u -m src.anchor_pass build --country $c; echo "EXIT $? buildB $c"; done
rm -rf experiments/E023B_anchor_union
$CAP .venv/bin/python -u -m src.train --tag trn2c_anc --exp E023B_anchor_union; echo "EXIT $? trainB"
echo ALL_DONE
