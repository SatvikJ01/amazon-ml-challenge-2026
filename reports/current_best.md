# Current Best

| Slot | Value |
|---|---|
| CURRENT BEST MODEL (holdout) | **E030 v3 stage 2**: forward ∪ reverse ∪ exact-key candidates → cheap stage 1 → LightGBM (73 feats), 300k training entities |
| CURRENT BEST CV | **0.9722** (60k holdout) · 0.9720 on the original 30k holdout (E023B 0.9669, E013 0.9614) · US .9775 · India .9641 |
| CANDIDATE RECALL / ORACLE | .9717 / .9901 |
| BEST SUBMISSION (LB) | day1_s2 = E013 + expF + exclusivity: **0.953** (day1_s1: 0.952) |
| IN PROGRESS | E030 stage 3 (OOF p2 → anchors → sibling stage), test inference E030 (auto-writes day2_s1), E032 native-script dictionary rebuild |
| ROLLBACK | E013 cascade (day1 submissions), all artefacts untouched |
| LOSS BUDGET (E030 s2) | blocking .0114 · matcher .0116 · false positives .0067 |
| RISKS | France unseen (LB-implied ~0.91 for E013); test has ~40 % distractors vs 26 % train (label-free estimate) → precision harder on test |
| NEXT | day2_s1 = E030 (stage 3 if it beats stage 2); day2_s2 = E032; E031 self-training for France |

Updated 2026-09-26 05:10 IST.
