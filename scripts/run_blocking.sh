#!/usr/bin/env bash
# Full blocking, one capped process per (split, country, source). Resumable.
set -u
cd "$(dirname "$0")/.."
CAP="systemd-run --user --scope -q -p MemoryMax=5500M -p MemorySwapMax=0"
PY=.venv/bin/python
run() {  # split tag country source
  $CAP $PY -u -m src.candidates --split "$1" --tag "$2" --topk 30 --countries "$3" --sources "$4" --skip-existing
  echo "EXIT $? $1 $3 S$4"
}
for c in India US; do for s in 2 3; do run train trnall $c $s; done; done
for c in France India US; do for s in 2 3; do run test testall $c $s; done; done
echo ALL_DONE
