#!/usr/bin/env bash
# day2_s2 retry: pass 2 in entity chunks (India parts exceeded 8 GB), E035 stage 3.
set -u
cd "$(dirname "$0")/.."
CAP="systemd-run --user --scope -q -p MemoryMax=7G -p MemorySwapMax=0"
OUT=scores_e035_stage3
for c in France India US; do
  ok=0
  for a in 1 2 3; do $CAP .venv/bin/python -u -m src.infer_v3 pass2 --country $c --stage3 E035_stage3 --scores $OUT && { ok=1; break; }; echo "RETRY pass2 $c"; sleep 20; done
  [ $ok -eq 1 ] || { echo FAILED; exit 1; }
done
OUT=$OUT .venv/bin/python - <<'PY' || { echo "PASS2 INCOMPLETE"; echo FAILED; exit 1; }
import math, os, pathlib, pyarrow.parquet as pq
d = pathlib.Path("experiments/E030_test") / os.environ["OUT"]
for c in ("France", "India", "US"):
    n = pq.ParquetFile(f"data/processed/test_s1_{c}.parquet").metadata.num_rows
    miss = [i for i in range(math.ceil(n / 200_000)) if not (d / f"{c}_p{i}.parquet").exists()]
    print(c, "missing", miss); assert not miss
PY
$CAP .venv/bin/python -u -m src.infer_v3 write --sub-id day2_s2 --scores $OUT --note "E035 stage 3 (holdout 0.97989)"
rc=$?; echo "EXIT $rc write"; [ $rc -eq 0 ] && echo ALL_DONE || echo FAILED
