#!/usr/bin/env bash
# Usage: scripts/run_inference.sh <exp> <stage1_exp> <sub_id> [decision args...]
# One capped process per country (fresh heap, checkpointed parts), then a light
# final step that reuses all checkpoints, decides, writes, validates, archives.
set -u
cd "$(dirname "$0")/.."
EXP=$1; S1=$2; SUB=$3; shift 3
CAP="systemd-run --user --scope -q -p MemoryMax=5500M -p MemorySwapMax=0"
for c in France India US; do
  for attempt in 1 2; do
    $CAP .venv/bin/python -u -m src.inference --exp "$EXP" --stage1 "$S1" --sub-id "$SUB" --depth 30 --countries $c && break
    echo "RETRY $c (attempt $attempt failed)"
  done
done
$CAP .venv/bin/python -u -m src.inference --exp "$EXP" --stage1 "$S1" --sub-id "$SUB" --depth 30 "$@"
echo "EXIT $?"
