#!/usr/bin/env bash
# Laptop -> cloud: code (git-tracked files + official validator) and the raw dataset
# (zstd-compressed stream, ~0.7 GB on the wire instead of 2.5 GB).
# Usage: scripts/aws/push.sh <ssh-host-alias> [remote-dir]
set -eu
H=${1:?ssh host alias}; R=${2:-ml}
cd "$(dirname "$0")/../.."
ssh "$H" "mkdir -p $R/data/raw $R/student_resource/utils"
git ls-files -z | rsync -a --from0 --files-from=- ./ "$H:$R/"
rsync -a student_resource/utils/ "$H:$R/student_resource/utils/"
if ! ssh "$H" "test -f $R/data/raw/.dataset_ok"; then
  tar -C data/raw -cf - dataset | zstd -T0 -3 | ssh "$H" "zstd -d | tar -C $R/data/raw -xf - && touch $R/data/raw/.dataset_ok"
fi
ssh "$H" "du -sh $R/data/raw/dataset && ls $R"
