#!/usr/bin/env bash
# EC2 (8 vCPU / 61 GB): E039 = E032 pipeline (native-script data) on 3x the training entities
# (the 300k E032 entities + 600k new), LightGBM everywhere (OOF and stage 2) so the test side needs
# no rescore.  Training only; test inference runs on the laptop with the resulting models.
set -u
cd "$(dirname "$0")/../.."
P=data/processed
step() { echo ">>> $(date +%H:%M) $*"; /usr/bin/time -f "PEAK_RSS_KB %M  WALL %e s" .venv/bin/python -u -m "$@"; local rc=$?; echo "EXIT $rc $*" | cut -c1-140; return $rc; }
par() { "$@" & }                       # used as: par step ...; par step ...; waitall
waitall() { local ok=0; for p in $(jobs -p); do wait $p || ok=1; done; [ $ok -eq 0 ] || { echo FAILED; exit 1; }; }
one() { step "$@" || { echo FAILED; exit 1; }; }
rm -f $P/train_s1_India.parquet $P/train_s1_US.parquet          # symlinks from the stage-4 job
one src.audit_gt
one src.audit_sources
one src.prep
one src.translit build
one src.translit apply --split train
blk() { for s in 2 3; do
  step src.candidates --split trainT --tag trnT --topk 30 --countries $1 --sources $s --skip-existing || return 1
  step src.candidates --split trainT --tag trnT --topk 3 --countries $1 --sources $s --reverse --skip-existing || return 1
  step src.key_channel --split trainT --tag trnT --country $1 --source $s --max-s1 10 --max-t 60 || return 1
done; }
par blk India; par blk US; waitall
echo BLOCKING_DONE
.venv/bin/python - <<'PY' || { echo FAILED; exit 1; }
import numpy as np
from src.candidates import sample_entities
old = np.load("data/processed/entities_v3.npy")
pool = sample_entities("train", 1_400_000, 7)
extra = pool[~np.isin(pool, old)][:600_000]
e = np.unique(np.concatenate([old, extra]))
np.save("data/processed/entities_v5.npy", e); print(f"entities_v5: {len(e):,} ({len(old):,} E032 + {len(extra):,} new)")
PY
par step src.v3 s1data --split trainT --tag trnT --prefix v5 --country India --entities $P/entities_v5.npy
par step src.v3 s1data --split trainT --tag trnT --prefix v5 --country US --entities $P/entities_v5.npy
waitall
cp $P/entities_v5.npy $P/entities_v5s1.npy
one src.stage1 --tag v5s1 --out E039_stage1 --v3 --neg-frac 0.25
par step src.v3 s2data --split trainT --tag trnT --prefix v5 --country India --stage1 E039_stage1 --entities $P/entities_v5.npy
par step src.v3 s2data --split trainT --tag trnT --prefix v5 --country US --stage1 E039_stage1 --entities $P/entities_v5.npy
waitall
one src.train --tag v5c --exp E039_stage2
one src.collective oof --tag v5c --exp E039_stage2 --out E039_collective --model lgb
par step src.collective build --tag v5c --out E039_collective --country India --split trainT
par step src.collective build --tag v5c --out E039_collective --country US --split trainT
waitall
par step src.anchor_pass retrieve --split trainT --country India --oof-dir E039_collective --hits-dir E039_anchor
par step src.anchor_pass retrieve --split trainT --country US --oof-dir E039_collective --hits-dir E039_anchor
waitall
for c in India US; do one src.anchor_pass build --country $c --old-tag v5c --out-tag v5c_anc --entities entities_v5c.npy --hits-dir E039_anchor --train-split trainT --train-tag trnT; done
par step src.patch_extra --country India --tag v5c_anc --out-tag v5c_ancz --sets num,name,ctx --split trainT
par step src.patch_extra --country US --tag v5c_anc --out-tag v5c_ancz --sets num,name,ctx --split trainT
waitall
one src.train --tag v5c_ancz --exp E039_stage3
echo ALL_DONE
