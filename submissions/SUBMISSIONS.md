# Submission Register

| ID | Time (IST) | Model | Decision | Local CV (holdout F0.5) | LB score | Purpose | Result / decision |
|---|---|---|---|---|---|---|---|
| day1_s1 | 09-25 12:44 | E013 cascade (stage1 p≥1e-3 → LGB 63 feats) | thr 0.7, no exclusivity | 0.9614 | **0.952** (public) | first valid submission; CV↔LB calibration | gap −0.009; see E014 / shift analysis |
| day1_s2 | 09-25 20:35 | same scores as day1_s1 | **expected-F0.5 + exclusivity** | 0.9613 (in-dist.), LOCO best | *pending upload* | does the decision rule help on LB? (only change vs s1) | diff: 2.54 % of entities (France 5.29 %); removes 9,554 multi-claimed records |

day1_s1 stats: 1,732,544 rows; 107,075 empty (6.2 %); 5.57 M matches (3.21/entity);
28.8 M candidates (16.6/entity). Per country mean matches / empty rate:
France 3.27 / 5.3 %, India 3.15 / 6.6 %, US 3.27 / 6.0 %. Official validator: PASS.
