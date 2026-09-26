#!/usr/bin/env bash
# Cloud -> laptop: experiment reports, DAG logs and finished submissions (no data).
set -eu
H=${1:?ssh host alias}; R=${2:-ml}
cd "$(dirname "$0")/../.."
rsync -a --include='*/' --include='report.json' --include='features.json' --include='feature_importance.csv' \
      --include='holdout_pred.parquet' --exclude='*' "$H:$R/experiments/" experiments/
rsync -a "$H:$R/logs/dag/" logs/dag/
rsync -a "$H:$R/submissions/" submissions/
