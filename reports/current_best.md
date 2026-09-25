# Current Best

| Slot | Value |
|---|---|
| CURRENT BEST MODEL | E013 cascade: stage-1 pruning + LightGBM stage 2 (63 feats) |
| CURRENT BEST CV | **0.9614** macro F0.5, holdout fold 0 (30,007 entities), threshold 0.7 · US .9661 · India .9545 |
| CURRENT BEST ENSEMBLE | — (single model) |
| BEST ROBUST VALIDATION (LOCO) | — pending (US→India full-density run) |
| BEST SUBMISSION | day1_s1 (E013 cascade, thr 0.7): public LB **0.952** (CV 0.9614) |
| BLOCKING CEILING | oracle F0.5 .9809, pair recall .946 at 30/source |
| LOSS BUDGET | FP .0074 · matcher FN .0157 · blocking FN .0191 |
| NEXT EXPERIMENT | day1_s1 (E011, thr 0.7) in progress; then E012 LOCO US→India full density + exclusivity |

Updated 2026-09-25 08:25 IST.
