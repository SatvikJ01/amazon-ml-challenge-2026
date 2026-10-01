#!/usr/bin/env bash
# Download the score files and logs (not the model weights) of an E046 Kaggle run:
#   bash kaggle/e046_ce/fetch.sh <kernel-slug> <out-dir>
set -euo pipefail
KAGGLE=${KAGGLE:-kaggle}
mkdir -p "$2"
$KAGGLE kernels output "sjaiswal0/$1" -p "$2" --file-pattern '^(ce[0-9]_(hold|test)\.parquet|log[0-9]\.json|.*\.log)$' -o
ls -la "$2"
