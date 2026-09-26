#!/usr/bin/env bash
# EC2: E041 = E039 stage 3 + hard anti-match flags (is_neighbor_number, mismatch_token_max_idf).
# Waits for E039, patches its stage-3 tables, retrains, compares on the identical holdout.
set -u
cd "$(dirname "$0")/../.."
until grep -qE "ALL_DONE|FAILED" logs/E039.log; do sleep 60; done
grep -q ALL_DONE logs/E039.log || { echo "E039 failed; E041 not started"; echo FAILED; exit 1; }
step() { echo ">>> $(date +%H:%M) $*"; /usr/bin/time -f "PEAK_RSS_KB %M  WALL %e s" .venv/bin/python -u -m "$@"; local rc=$?; echo "EXIT $rc $*" | cut -c1-140; return $rc; }
step src.patch_extra --country India --tag v5c_ancz --out-tag v5c_ancw --sets anti --split trainT & p1=$!
step src.patch_extra --country US --tag v5c_ancz --out-tag v5c_ancw --sets anti --split trainT & p2=$!
wait $p1 && wait $p2 || { echo FAILED; exit 1; }
step src.train --tag v5c_ancw --exp E041_stage3 || { echo FAILED; exit 1; }
.venv/bin/python - <<'PY'
import json
for e in ("E039_stage3", "E041_stage3"):
    r = json.load(open(f"experiments/{e}/report.json"))
    print(e, "expF", round(r["decision_rules"]["expF_m0.0"], 5), {c: round(v["f05"], 4) for c, v in r["by_country"].items()},
          {k: round(v["f05"], 4) for k, v in r["by_n_true"].items()},
          {k: v for k, v in r["micro"].items() if k in ("fp", "fn")})
PY
for e in E039_stage3 E041_stage3; do echo "== $e on the original 60k holdout"; .venv/bin/python -m src.eval_subset --exp $e --entities data/processed/entities_v3c.npy 2>/dev/null | grep -A5 '"rules"' | head -6; done
echo ALL_DONE
