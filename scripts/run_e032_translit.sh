#!/usr/bin/env bash
# E032: v3 pipeline on the native-script-dictionary data version (trainT/testT).
# India candidates are rebuilt (forward, reverse, key); US/France candidate files are
# hard-linked unchanged.  Starts after the E030 test pass 1 frees the machine.
set -u
cd "$(dirname "$0")/.."
until grep -q ALL_DONE logs/test_v3_pass1.log 2>/dev/null; do sleep 30; done
CAP="systemd-run --user --scope -q -p MemoryMax=7G -p MemorySwapMax=0"
PY=".venv/bin/python -u -m"
step() { echo ">>> $*"; $CAP $PY "$@"; local rc=$?; echo "EXIT $rc $*" | cut -c1-120; return $rc; }
P=data/processed
# --- blocking: India rebuilt on translated names ---
for sp in "trainT trnT" "testT tstT"; do set -- $sp
  for s in 2 3; do
    step src.candidates --split $1 --tag $2 --topk 30 --countries India --sources $s --skip-existing
    step src.candidates --split $1 --tag $2 --topk 3 --countries India --sources $s --reverse --skip-existing
    step src.key_channel --split $1 --tag $2 --country India --source $s --max-s1 10 --max-t 60
  done
done
for s in 2 3; do for sfx in "" "_rev" "_key"; do
  ln -f $P/cands_trnall_US_s${s}${sfx}.parquet $P/cands_trnT_US_s${s}${sfx}.parquet
  for c in US France; do ln -f $P/cands_testall_${c}_s${s}${sfx}.parquet $P/cands_tstT_${c}_s${s}${sfx}.parquet; done
done; done
echo "BLOCKING_DONE"
# --- training chain (same steps as E030) ---
for c in India US; do step src.v3 s1data --split trainT --tag trnT --prefix v3T --country $c --entities $P/entities_v3.npy || exit 1; done
cp $P/entities_v3.npy $P/entities_v3Ts1.npy
step src.stage1 --tag v3Ts1 --out E032_stage1 --v3 --neg-frac 0.25 || exit 1
for c in India US; do step src.v3 s2data --split trainT --tag trnT --prefix v3T --country $c --stage1 E032_stage1 --entities $P/entities_v3.npy || exit 1; done
step src.train --tag v3Tc --exp E032_stage2 || exit 1
step src.collective oof --tag v3Tc --exp E032_stage2 --out E032_collective || exit 1
for c in India US; do step src.collective build --tag v3Tc --out E032_collective --country $c --split trainT || exit 1; done
for c in India US; do step src.anchor_pass retrieve --split trainT --country $c --oof-dir E032_collective --hits-dir E032_anchor || exit 1; done
for c in India US; do step src.anchor_pass build --country $c --old-tag v3Tc --out-tag v3Tc_anc --entities entities_v3Tc.npy --hits-dir E032_anchor --train-split trainT --train-tag trnT || exit 1; done
step src.train --tag v3Tc_anc --exp E032_stage3 || exit 1
echo ALL_DONE
