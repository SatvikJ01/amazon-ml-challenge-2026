#!/usr/bin/env bash
# day3_l2b = day3_e039 + E044 residual (trained on 702k training entities via cross-fitted stage-3 OOF,
# holdout +0.00123) on India/US; France raw.  Also day3_l2b_india (India corrected, US/France raw).
set -u
cd "$(dirname "$0")/.."
for c in India US France; do
  systemd-run --user --scope -q -p MemoryMax=5G -p MemorySwapMax=0 nice -n 5 env OMP_NUM_THREADS=4 \
    .venv/bin/python scripts/apply_l2.py --run E039_test --countries India US --process $c --model-dir experiments/L2_E044 --out-dir scores_l2b --threads 4
  rc=$?; echo "EXIT $rc apply $c"; [ $rc -eq 0 ] || { echo FAILED; exit 1; }
done
.venv/bin/python scripts/apply_l2.py --run E039_test --countries India US --process India US France --model-dir experiments/L2_E044 --out-dir scores_l2b --no-gates 2>&1 | tail -2 > /tmp/claude-1000/-home-alphatron-PADHAI-LIKHAI-Semester-IX-Amazon-ML-Challenge-ML-Challenge/3c84b878-f755-4117-a9de-8f906729ee2b/scratchpad/l2b_check.txt
grep -q "ALL PARTS COMPLETE" /tmp/claude-1000/-home-alphatron-PADHAI-LIKHAI-Semester-IX-Amazon-ML-Challenge-ML-Challenge/3c84b878-f755-4117-a9de-8f906729ee2b/scratchpad/l2b_check.txt || { echo "incomplete"; echo FAILED; exit 1; }
systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 .venv/bin/python -u -m src.infer_v3 write --run E039_test --split testT --tag tstT --scores scores_l2b --sub-id day3_l2b --note "E039 + E044 residual (702k-entity training, 600 rounds, holdout +0.00123) on India/US; France raw"
echo "EXIT $? write l2b"
# hedge: India from scores_l2b, US/France raw (copies from scores_l2in, which keeps them at prob3)
R=experiments/E039_test; mkdir -p $R/scores_l2b_in
cp $R/scores_l2b/India_p*.parquet $R/scores_l2b_in/ && cp $R/scores_l2in/US_p*.parquet $R/scores_l2in/France_p*.parquet $R/scores_l2b_in/
ls $R/scores_l2b_in | wc -l
systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 .venv/bin/python -u -m src.infer_v3 write --run E039_test --split testT --tag tstT --scores scores_l2b_in --sub-id day3_l2b_india --note "E039 + E044 residual on India only; US/France raw (hedge)"
echo "EXIT $? write l2b_india"; echo ALL_DONE
