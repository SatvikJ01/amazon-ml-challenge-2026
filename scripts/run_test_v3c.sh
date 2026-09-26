#!/usr/bin/env bash
# E030 test inference (corrected): complete ALL pass-1 parts (each part in its own child
# process) -> verify completeness -> GPU rescore -> anchors -> stage-3 pass -> day2_s1 files.
set -u
cd "$(dirname "$0")/.."
CAP="systemd-run --user --scope -q -p MemoryMax=5500M -p MemorySwapMax=0"
US_PID=$(cat /tmp/claude-1000/-home-alphatron-PADHAI-LIKHAI-Semester-IX-Amazon-ML-Challenge-ML-Challenge/3c84b878-f755-4117-a9de-8f906729ee2b/scratchpad/us_pid)
while kill -0 $US_PID 2>/dev/null; do sleep 20; done
for c in France India US; do
  for a in 1 2 3 4; do $CAP .venv/bin/python -u -m src.infer_v3 pass1 --country $c && break; echo "RETRY pass1 $c"; sleep 20; done
done
# completeness guard: every part of every country must exist
python3 - <<'PY' || { echo "PASS1 INCOMPLETE"; exit 1; }
import math, pathlib, pyarrow.parquet as pq
d = pathlib.Path("experiments/E030_test/pass1_feats")
for c in ("France", "India", "US"):
    n = pq.ParquetFile(f"data/processed/test_s1_{c}.parquet").metadata.num_rows
    parts = math.ceil(n / 200_000)
    missing = [i for i in range(parts) if not (d / f"{c}_p{i}.parquet").exists()]
    print(c, "parts", parts, "missing", missing)
    assert not missing
PY
echo PASS1_DONE
for c in France India US; do $CAP .venv/bin/python -u -m src.infer_v3 rescore --country $c; echo "EXIT $? rescore $c"; done
for c in France India US; do $CAP .venv/bin/python -u -m src.infer_v3 anchors --country $c; echo "EXIT $? anchors $c"; done
until grep -q ALL_DONE logs/E030b.log; do sleep 30; done
S3=$(python3 -c "import json;print(json.load(open('experiments/E030_stage3/report.json'))['best_f05'])")
S2=$(python3 -c "import json;print(json.load(open('experiments/E030_stage2/report.json'))['best_f05'])")
echo "holdout: stage2 $S2  stage3 $S3"
if python3 -c "import sys; sys.exit(0 if $S3 >= $S2 else 1)"; then
  for c in France India US; do
    for a in 1 2 3; do $CAP .venv/bin/python -u -m src.infer_v3 pass2 --country $c && break; echo "RETRY pass2 $c"; sleep 20; done
  done
  $CAP .venv/bin/python -u -m src.infer_v3 write --sub-id day2_s1 --note "v3 stage 3 (holdout $S3)"; echo "EXIT $? write"
else
  $CAP .venv/bin/python -u -m src.infer_v3 write --sub-id day2_s1 --from-pass1 --note "v3 stage 2 (holdout $S2)"; echo "EXIT $? write"
fi
echo ALL_DONE
