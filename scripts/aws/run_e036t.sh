#!/usr/bin/env bash
# EC2: stage 4 on top of E032 (native-script data): cross-fitted stage-3 p3 (LightGBM, same family
# as the E032 stage-3 model that scores the test) -> collective features from p3 -> stage 4.
set -u
cd "$(dirname "$0")/../.."
step() { echo ">>> $(date +%H:%M) $*"; /usr/bin/time -f "PEAK_RSS_KB %M  WALL %e s" .venv/bin/python -u -m "$@"; local rc=$?; echo "EXIT $rc $1 $2"; [ $rc -eq 0 ] || { echo FAILED; exit 1; }; }
step src.collective oof --tag v3Tc_ancz --exp E032_stage3 --out E036T_oof --model lgb
for c in India US; do step src.stage4 build --country $c --tag v3Tc_ancz --out-tag v3Tc_s4 --oof E036T_oof --split trainT; done
step src.train --tag v3Tc_s4 --exp E036T_stage4
echo ALL_DONE
