#!/usr/bin/env bash
# E032 on the laptop: native-script dictionary data version (trainT/testT) through the full
# current pipeline (v3 -> stage 2 -> anchors -> stage 3 with E033-E035 features) and test
# inference -> submission candidate day3_e032.  One heavy step at a time, each memory-capped.
set -u
cd "$(dirname "$0")/.."
P=data/processed
step() { local cap=$1; shift; echo ">>> $(date +%H:%M) $*"; systemd-run --user --scope -q -p MemoryMax=$cap -p MemorySwapMax=0 .venv/bin/python -u -m "$@"; local rc=$?; echo "EXIT $rc $1 $2"; [ $rc -eq 0 ] || { echo FAILED; exit 1; }; }
# --- blocking: only India changes (US/France files are hard links of the originals) ---
for sp in "trainT trnT" "testT tstT"; do set -- $sp
  for s in 2 3; do
    step 6500M src.candidates --split $1 --tag $2 --topk 30 --countries India --sources $s --skip-existing
    step 6500M src.candidates --split $1 --tag $2 --topk 3 --countries India --sources $s --reverse --skip-existing
    step 6500M src.key_channel --split $1 --tag $2 --country India --source $s --max-s1 10 --max-t 60
  done
done
for s in 2 3; do for sfx in "" "_rev" "_key"; do
  ln -f $P/cands_trnall_US_s${s}${sfx}.parquet $P/cands_trnT_US_s${s}${sfx}.parquet
  for c in US France; do ln -f $P/cands_testall_${c}_s${s}${sfx}.parquet $P/cands_tstT_${c}_s${s}${sfx}.parquet; done
done; done
echo BLOCKING_DONE
# --- training chain ---
step 7G src.v3 s1data --split trainT --tag trnT --prefix v3T --country India --entities $P/entities_v3.npy
ln -f $P/feats_v3s1_US.parquet $P/feats_v3Ts1_US.parquet        # US inputs are identical
cp $P/entities_v3.npy $P/entities_v3Ts1.npy
step 7G src.stage1 --tag v3Ts1 --out E032_stage1 --v3 --neg-frac 0.25
for c in India US; do step 7G src.v3 s2data --split trainT --tag trnT --prefix v3T --country $c --stage1 E032_stage1 --entities $P/entities_v3.npy; done
step 7G src.train --tag v3Tc --exp E032_stage2
step 7G src.train --tag v3Tc --exp E032_stage2x --model xgb
step 7500M src.collective oof --tag v3Tc --exp E032_stage2 --out E032_collective --model xgb
for c in India US; do step 6500M src.collective build --tag v3Tc --out E032_collective --country $c --split trainT; done
for c in India US; do step 6500M src.anchor_pass retrieve --split trainT --country $c --oof-dir E032_collective --hits-dir E032_anchor; done
for c in India US; do step 9G src.anchor_pass build --country $c --old-tag v3Tc --out-tag v3Tc_anc --entities entities_v3Tc.npy --hits-dir E032_anchor --train-split trainT --train-tag trnT; done
for c in India US; do step 6G src.patch_extra --country $c --tag v3Tc_anc --out-tag v3Tc_ancz --sets num,name,ctx --split trainT; done
step 7G src.train --tag v3Tc_ancz --exp E032_stage3
echo TRAIN_DONE
# --- test inference ---
T="--run E032_test --split testT --tag tstT"
for c in France India US; do
  for a in 1 2 3; do systemd-run --user --scope -q -p MemoryMax=5500M -p MemorySwapMax=0 .venv/bin/python -u -m src.infer_v3 pass1 --country $c $T --stage1 E032_stage1 --stage2 E032_stage2 && break; echo "RETRY pass1 $c"; sleep 20; done
done
for c in France India US; do step 5500M src.infer_v3 rescore --country $c $T --stage2x E032_stage2x; done
for c in France India US; do step 6G src.infer_v3 anchors --country $c $T; done
for c in France India US; do
  for a in 1 2 3; do systemd-run --user --scope -q -p MemoryMax=7G -p MemorySwapMax=0 .venv/bin/python -u -m src.infer_v3 pass2 --country $c $T --stage3 E032_stage3 --scores scores && break; echo "RETRY pass2 $c"; sleep 20; done
done
.venv/bin/python - <<'PY' || { echo "INCOMPLETE"; echo FAILED; exit 1; }
import math, pathlib, pyarrow.parquet as pq
for sub in ("pass1_feats", "scores"):
    d = pathlib.Path("experiments/E032_test") / sub
    for c in ("France", "India", "US"):
        n = pq.ParquetFile(f"data/processed/testT_s1_{c}.parquet").metadata.num_rows
        miss = [i for i in range(math.ceil(n / 200_000)) if not (d / f"{c}_p{i}.parquet").exists()]
        print(sub, c, "missing", miss); assert not miss
PY
step 6G src.infer_v3 write $T --scores scores --sub-id day3_e032 --note "E032 native-script dictionary + stage 3 E033-E035 features"
echo ALL_DONE
