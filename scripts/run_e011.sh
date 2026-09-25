#!/usr/bin/env bash
# E011: v2 features (canonical digits + name ambiguity) -> train -> holdout eval.
set -u
cd "$(dirname "$0")/.."
until grep -q ALL_DONE logs/blocking_test.log; do sleep 10; done
CAP="systemd-run --user --scope -q -p MemoryMax=5500M -p MemorySwapMax=0"
for c in India US; do
  $CAP .venv/bin/python -u -m src.build_features --split train --tag trnall --depth 30 \
      --entities data/processed/entities_trn.npy --out-tag trn2 --countries $c
  echo "EXIT $? feats $c"
done
cp data/processed/entities_trn.npy data/processed/entities_trn2.npy
$CAP .venv/bin/python -u -m src.train --tag trn2 --exp E011_lgb_v2feats
echo "EXIT $? train"
echo ALL_DONE
