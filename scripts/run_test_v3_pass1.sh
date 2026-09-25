#!/usr/bin/env bash
# v3 test pass 1 (stage 1 + stage 2 on survivors) then anchor retrieval, per country.
# Resumable: finished parts are skipped. Runs alongside the training chain.
set -u
cd "$(dirname "$0")/.."
CAP="systemd-run --user --scope -q -p MemoryMax=5G -p MemorySwapMax=0"
until grep -q "EXIT .* pass1 France" logs/test_v3_France.log; do sleep 15; done
for c in France India US; do
  for attempt in 1 2 3; do
    $CAP .venv/bin/python -u -m src.infer_v3 pass1 --country $c && { echo "EXIT 0 pass1 $c"; break; }
    echo "RETRY pass1 $c (attempt $attempt)"; sleep 30
  done
done
for c in France India US; do
  $CAP .venv/bin/python -u -m src.infer_v3 anchors --country $c; echo "EXIT $? anchors $c"
done
echo ALL_DONE
