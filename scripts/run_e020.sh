#!/usr/bin/env bash
# E020: forward ∪ reverse candidates -> features -> stage 1 (rev feats) -> survivors -> stage 2.
set -u
cd "$(dirname "$0")/.."
until grep -q ALL_DONE logs/reverse.log; do sleep 20; done
CAP="systemd-run --user --scope -q -p MemoryMax=5500M -p MemorySwapMax=0"
for c in India US; do
  $CAP .venv/bin/python -u -m src.build_features --split train --tag trnall --depth 30 --reverse \
      --entities data/processed/entities_trn3.npy --out-tag trn3 --countries $c; echo "EXIT $? feats $c"
done
$CAP .venv/bin/python -u -m src.stage1 --tag trn3 --out E020_stage1 --reverse; echo "EXIT $? stage1"
$CAP .venv/bin/python -u -m src.make_stage2 --tag trn3 --stage1 E020_stage1 --out-tag trn3c; echo "EXIT $? make_stage2"
$CAP .venv/bin/python -u -m src.train --tag trn3c --exp E020_reverse_stage2; echo "EXIT $? train"
echo ALL_DONE
