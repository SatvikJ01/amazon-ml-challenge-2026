# Current Best

| Slot | Value |
|---|---|
| CURRENT BEST MODEL (holdout) | **E034 stage 3**: v3 stage 2 → GPU cross-fitted p2 → anchors → LightGBM stage 3 with sibling, number-consensus (E033) and name-substitution / shared-address (E034) features |
| CURRENT BEST CV | **0.97942** (60k holdout, expF) · E033 0.97711 · E030 stage 3 0.97522 · stage 2 0.97180 (expF) · US .9829 · India .9742 |
| CANDIDATE RECALL / ORACLE | .9785 / .9918 (with anchor candidates) |
| BEST SUBMISSION (LB) | day2_s2 = E035 stage 3: **0.9705** (day2_s1 0.9634, day1_s2 0.953); holdout→LB gap 0.0084 → 0.0094 |
| IN PROGRESS | E032 native-script rebuild on laptop (→ day3_e032, ~03:00); cloud v4 (translit + 1 M entities + stage 4) pending AWS instance |
| ROLLBACK | E013 cascade (day1 submissions), all artefacts untouched |
| LOSS BUDGET (E030 s2) | blocking .0114 · matcher .0116 · false positives .0067 |
| RISKS | France unseen (LB-implied ~0.91 for E013); test has ~40 % distractors vs 26 % train (label-free estimate) → precision harder on test |
| NEXT | more stage-3 feature iterations from the remaining-error sample; E032 native-script rebuild |

Updated 2026-09-26 15:00 IST.
