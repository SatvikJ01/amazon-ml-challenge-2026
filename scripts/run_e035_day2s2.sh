#!/usr/bin/env bash
# E035 (house-number distance, empty-address context, name duplication) on top of E034,
# then day2_s2 test pass with whichever of E034 / E035 is better on the holdout (expF).
set -u
cd "$(dirname "$0")/.."
for c in India US; do
  systemd-run --user --scope -q -p MemoryMax=5500M -p MemorySwapMax=0 .venv/bin/python -u -m src.patch_extra --country $c --tag v3c_ancy --out-tag v3c_ancz --sets ctx
  rc=$?; echo "EXIT $rc patch $c"; [ $rc -eq 0 ] || { echo FAILED; exit 1; }
done
systemd-run --user --scope -q -p MemoryMax=7G -p MemorySwapMax=0 .venv/bin/python -u -m src.train --tag v3c_ancz --exp E035_stage3
echo "EXIT $? train E035_stage3"
BEST=$(.venv/bin/python -c "
import json, os
s = {e: json.load(open(f'experiments/{e}/report.json'))['decision_rules']['expF_m0.0'] for e in ('E034_stage3', 'E035_stage3') if os.path.exists(f'experiments/{e}/report.json')}
print(max(s, key=s.get))")
SC=$(.venv/bin/python -c "import json; print(round(json.load(open('experiments/$BEST/report.json'))['decision_rules']['expF_m0.0'], 5))")
echo "CHOSEN $BEST holdout $SC"
until grep -qE "ANCHORS_DONE|FAILED" logs/test_v3d.log; do sleep 30; done
grep -q ANCHORS_DONE logs/test_v3d.log || { echo "anchors failed"; echo FAILED; exit 1; }
CAP="systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0"
OUT=scores_$(echo $BEST | tr 'A-Z' 'a-z')
for c in France India US; do
  ok=0
  for a in 1 2 3; do $CAP .venv/bin/python -u -m src.infer_v3 pass2 --country $c --stage3 $BEST --scores $OUT && { ok=1; break; }; echo "RETRY pass2 $c"; sleep 20; done
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
$CAP .venv/bin/python -u -m src.infer_v3 write --sub-id day2_s2 --scores $OUT --note "$BEST stage 3 (holdout $SC)"
rc=$?; echo "EXIT $rc write"; [ $rc -eq 0 ] && echo ALL_DONE || echo FAILED
