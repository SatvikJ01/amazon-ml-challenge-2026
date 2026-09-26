#!/usr/bin/env bash
# day2_s2: stage 3 with E033 + E034 features (holdout 0.97942) on test.
set -u
cd "$(dirname "$0")/.."
until grep -qE "ANCHORS_DONE|FAILED" logs/test_v3d.log; do sleep 30; done
grep -q ANCHORS_DONE logs/test_v3d.log || { echo "anchors failed"; echo FAILED; exit 1; }
CAP="systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0"
for c in France India US; do
  ok=0
  for a in 1 2 3; do $CAP .venv/bin/python -u -m src.infer_v3 pass2 --country $c --stage3 E034_stage3 --scores scores_e034 && { ok=1; break; }; echo "RETRY pass2 $c"; sleep 20; done
  [ $ok -eq 1 ] || { echo FAILED; exit 1; }
done
.venv/bin/python - <<'PY' || { echo "PASS2 INCOMPLETE"; echo FAILED; exit 1; }
import math, pathlib, pyarrow.parquet as pq
d = pathlib.Path("experiments/E030_test/scores_e034")
for c in ("France", "India", "US"):
    n = pq.ParquetFile(f"data/processed/test_s1_{c}.parquet").metadata.num_rows
    miss = [i for i in range(math.ceil(n / 200_000)) if not (d / f"{c}_p{i}.parquet").exists()]
    print(c, "missing", miss); assert not miss
PY
$CAP .venv/bin/python -u -m src.infer_v3 write --sub-id day2_s2 --scores scores_e034 --note "stage 3 + E033/E034 features (holdout 0.97942)"
rc=$?; echo "EXIT $rc write"; [ $rc -eq 0 ] && echo ALL_DONE || echo FAILED
