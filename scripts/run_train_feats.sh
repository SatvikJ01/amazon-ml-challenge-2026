#!/usr/bin/env bash
# Training features for the 150k-entity sample, one capped process per country.
set -u
cd "$(dirname "$0")/.."
CAP="systemd-run --user --scope -q -p MemoryMax=5500M -p MemorySwapMax=0"
for c in ${COUNTRIES:-India US}; do
  $CAP .venv/bin/python -u -m src.build_features --split train --tag trnall --depth 30 \
      --entities data/processed/entities_trn.npy --out-tag trn --countries $c
  echo "EXIT $? $c"
done
echo ALL_DONE
