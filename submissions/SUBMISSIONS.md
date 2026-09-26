# Submission Register

| ID | Time (IST) | Model | Decision | Local CV (holdout F0.5) | LB score | Purpose | Result / decision |
|---|---|---|---|---|---|---|---|
| day1_s1 | 09-25 12:44 | E013 cascade (stage1 p≥1e-3 → LGB 63 feats) | thr 0.7, no exclusivity | 0.9614 | **0.952** (public) | first valid submission; CV↔LB calibration | gap −0.009; see E014 / shift analysis |
| day1_s2 | 09-25 20:35 | same scores as day1_s1 | **expected-F0.5 + exclusivity** | 0.9613 (in-dist.), LOCO best | **0.953** (public) | does the decision rule help on LB? (only change vs s1) | +0.001 vs s1, same direction as local (tie in-dist., best under LOCO) → keep expF + exclusivity. diff: 2.54 % of entities (France 5.29 %); removes 9,554 multi-claimed records |
| day2_s1 | 09-26 14:04 | E030 v3 stage 2 (LightGBM p2, 73 feats) | expF + exclusivity | 0.9718 (expF) / 0.9722 (thr 0.7) | **0.9634** (public) | first v3 submission: does the +0.010 holdout gain transfer? | 11.4 % of entities differ from day1_s2 (+0.11 matches/entity, −0.02); empty 5.93 % → 5.60 %; France changes most (17.3 %). LB +0.0104 vs day1_s2 = holdout gain +0.0104 → holdout→LB gap stable at 0.0084: holdout gains transfer 1:1 |
| day2_s2 | 09-26 17:11 | E035 stage 3: v3 stage 2 → anchors → LightGBM with sibling, number-consensus, name-substitution, shared-address, house-number-distance features | expF + exclusivity | **0.97989** | **0.9705** (public; exp. 0.9715) | does the stage-3 feature stack transfer? | vs day2_s1: 11.9 % entities differ; US/India +0.09–0.10 added, −0.03 removed per entity (holdout-like); **France differs: +0.092 added / −0.104 removed, empty 4.72 %→5.39 %** — watch: if LB < ≈0.970, France is losing on the new features. Result: gap 0.0094 (vs 0.0084); label-free review of 18 sampled France removals: mostly correct distractor removals (one-word substitutions, other addresses), no systematic failure → France kept on stage 3 |
| day3_e032 | 09-27 01:41 | E032: native-script dictionary (testT) + v3 + anchors + stage 3 (E033–E035 features) | expF + exclusivity | **0.98265** | pending (exp. ≈0.973–0.974) | does the India native-script fix transfer? | vs day2_s2: 6.3 % entities differ; India +0.061 added / −0.022 removed per entity, empty 6.20 %→5.88 %; US ±0.025; France +0.042 / −0.054 |

day1_s1 stats: 1,732,544 rows; 107,075 empty (6.2 %); 5.57 M matches (3.21/entity);
28.8 M candidates (16.6/entity). Per country mean matches / empty rate:
France 3.27 / 5.3 %, India 3.15 / 6.6 %, US 3.27 / 6.0 %. Official validator: PASS.
