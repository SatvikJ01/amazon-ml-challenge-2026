#!/usr/bin/env bash
# E016: which feature groups hurt transfer to an unseen country?
# Train stage 2 on one country, evaluate on the other's holdout fold (tag trn2c).
set -u
cd "$(dirname "$0")/.."
CAP="systemd-run --user --scope -q -p MemoryMax=4500M -p MemorySwapMax=0"
COUNTS="s1_name_count s1_core_count cand_core_count_in_s1 comp_n n_cands"
ABSRET="blk_score blk_rank comp_best_other"
SCRIPT="name_nonascii_c"
run() { # name train eval drops...
  local name=$1 tr=$2 ev=$3; shift 3
  $CAP .venv/bin/python -u -m src.train --tag trn2c --exp E016_$name --train-countries $tr --eval-countries $ev --drop-features "$@" > logs/E016_$name.log 2>&1
  echo "$name EXIT $? $(python3 -c "import json;r=json.load(open('experiments/E016_$name/report.json'));print(r['best_rule'],round(r['best_f05'],5))" 2>/dev/null)"
}
run A0_us2in US India __none__
run A1_counts_us2in US India $COUNTS
run A2_absret_us2in US India $ABSRET
run A3_script_us2in US India $SCRIPT
run A4_all_us2in US India $COUNTS $ABSRET $SCRIPT
run A0_in2us India US __none__
run A4_all_in2us India US $COUNTS $ABSRET $SCRIPT
echo ALL_DONE
