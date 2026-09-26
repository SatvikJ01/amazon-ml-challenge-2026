#!/usr/bin/env bash
# Laptop: E039 test inference.  Stage 1+2 models exist already -> pass 1 and anchors start now;
# pass 2 waits for E039 stage 3 on EC2 (pulled when ready).  LightGBM p2 everywhere (OOF was
# LightGBM), so no rescore step.
set -u
cd "$(dirname "$0")/.."
T="--run E039_test --split testT --tag tstT"
guard() { .venv/bin/python - "$1" <<'PY' || { echo "INCOMPLETE $1"; echo FAILED; exit 1; }
import math, pathlib, sys, pyarrow.parquet as pq
d = pathlib.Path("experiments/E039_test") / sys.argv[1]
for c in ("France", "India", "US"):
    n = pq.ParquetFile(f"data/processed/testT_s1_{c}.parquet").metadata.num_rows
    miss = [i for i in range(math.ceil(n / 200_000)) if not (d / f"{c}_p{i}.parquet").exists()]
    print(sys.argv[1], c, "missing", miss); assert not miss
PY
}
for c in France India US; do
  for a in 1 2 3; do systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 .venv/bin/python -u -m src.infer_v3 pass1 --country $c $T --stage1 E039_stage1 --stage2 E039_stage2 && break; echo "RETRY pass1 $c"; sleep 20; done
done
guard pass1_feats
for c in France India US; do systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 .venv/bin/python -u -m src.infer_v3 anchors --country $c $T; echo "EXIT $? anchors $c"; done
echo ANCHORS_DONE
until timeout 40 ssh -o BatchMode=yes -o ConnectTimeout=20 ml 'grep -q ALL_DONE amazon-ml-challenge-2026/logs/E039.log' 2>/dev/null; do
  timeout 40 ssh -o BatchMode=yes ml 'grep -q FAILED amazon-ml-challenge-2026/logs/E039.log' 2>/dev/null && { echo "E039 training FAILED"; echo FAILED; exit 1; }
  sleep 120
done
rsync -a --exclude '*.parquet' ml:amazon-ml-challenge-2026/experiments/E039_stage3 experiments/ && echo "stage 3 pulled"
rsync -a ml:amazon-ml-challenge-2026/experiments/E039_stage3/holdout_pred.parquet experiments/E039_stage3/
for c in France India US; do
  for a in 1 2 3; do systemd-run --user --scope -q -p MemoryMax=7G -p MemorySwapMax=0 .venv/bin/python -u -m src.infer_v3 pass2 --country $c $T --stage3 E039_stage3 --scores scores && break; echo "RETRY pass2 $c"; sleep 20; done
done
guard scores
systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 .venv/bin/python -u -m src.infer_v3 write $T --scores scores --sub-id day3_e039 --note "E039: E032 pipeline on 3x training entities"
echo "EXIT $? write"; echo ALL_DONE
