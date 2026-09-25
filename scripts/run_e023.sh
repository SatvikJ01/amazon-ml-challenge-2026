#!/usr/bin/env bash
# E023: predicted-anchor retrieval holdout experiment (baseline E013/E021p untouched).
set -u
cd "$(dirname "$0")/.."
CAP="systemd-run --user --scope -q -p MemoryMax=5000M -p MemorySwapMax=0"
for c in India US; do $CAP .venv/bin/python -u -m src.anchor_pass retrieve --split train --country $c; echo "EXIT $? retrieve $c"; done
for c in India US; do
  $CAP .venv/bin/python -u -m src.anchor_pass build --country $c --no-new; echo "EXIT $? buildA $c"
  $CAP .venv/bin/python -u -m src.anchor_pass build --country $c; echo "EXIT $? buildB $c"
done
$CAP .venv/bin/python -u -m src.train --tag trn2c_ancA --exp E023A_anchor_feats_only; echo "EXIT $? trainA"
$CAP .venv/bin/python -u -m src.train --tag trn2c_anc --exp E023B_anchor_union; echo "EXIT $? trainB"
$CAP .venv/bin/python -u -m src.anchor_pass retrieve --split test --country France; echo "EXIT $? retrieve test France"
echo ALL_DONE
