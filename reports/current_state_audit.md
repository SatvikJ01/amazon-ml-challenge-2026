# Current State Audit — 2026-09-25 21:25 IST

## Champion
| Item | Value |
|---|---|
| Model | E013 cascade: stage-1 LightGBM (10 retrieval/competition feats, cross-fitted, keep p ≥ 1e-3) → stage-2 LightGBM (63 feats) |
| Validation split | 150k sampled train S1 entities (60k India / 90k US), entity-hash fold 0 = 30,007 holdout entities; scored against **full** ground truth (blocking misses count) |
| Holdout macro F0.5 | **0.9614** (thr 0.7) · expF 0.9613 |
| Precision / recall (India) | 0.979 / 0.903 |
| Per country | US 0.966 · India 0.955 |
| Singletons | 0.969 · 1-match entities 0.892 |
| Blocking pair recall (holdout, final candidate set) | 0.945 |
| Oracle F0.5 ceiling (final candidate set) | 0.9806 |
| Test candidates (final set) | 28.8 M (16.6 / entity) after stage 1; 104 M before |

## Leaderboard
| Submission | Model / decision | CV | Public LB |
|---|---|---|---|
| day1_s1 | E013, thr 0.7 | 0.9614 | **0.952** |
| day1_s2 | E013, expF + exclusivity | 0.9613 | not yet uploaded |

User-reported top of leaderboard ≈ 0.987 (unverified). That is **above our oracle ceiling (0.9806)** and above our in-distribution CV, so the gap is not only France.

## Completed experiments (all numbers recovered from `experiments/*/report.json`)
| ID | Result |
|---|---|
| E010 | 0.9578 baseline LightGBM (56 feats) |
| E011 | 0.9614 (+ canonical digits, name ambiguity) |
| E012/E013 | cascade: 25 % of pairs, −0.00004 F0.5; stage 2 = 0.96142 |
| E014 | LOCO US→India, full density (883k entities): **0.884** (P 0.954, R 0.806, singletons 0.749); exclusivity +0.0005 |
| E016 | density-feature ablations under LOCO: none helps (A0 .8897, counts −.0026, abs-retrieval +.0018, script +.0013, all −.0027); India→US 0.9409 vs 0.966 in-dist. **Hypothesis rejected.** |
| E017 (today) | file-order leakage: none (Spearman +0.004 / +0.001) |
| E017b | reverse retrieval (India S2, all 2.02 M targets): recall .9464 → .9536 (top-1, +0.1 cand/ent) → .9583 (top-3, +1.8) → .9608 (top-5, +4.2) |
| E018 | char n-gram channel: +0.0033 recall at +20 cands/ent. **Rejected.** |
| E019 | residual misses (after fwd ∪ rev3): 93 % have a retrieved sibling; 51 % have a sibling with sim ≥ 80; 33 % are ≥ 15 pts closer to a sibling than to S1 |

## Unfinished
None running. `reports/shift_check.json` never produced (low priority).

## Reusable artefacts (do not regenerate)
- `data/processed/{split}_s{1,2,3}_{country}.parquet` normalised text
- `data/processed/cands_{trnall,testall}_{country}_s{2,3}.parquet` forward top-30 (185 M train, 104 M test)
- `data/processed/feats_trn2_*` (full-list features, 9 M pairs), `feats_trn2c_*` (survivors)
- `experiments/E013_*` models; `experiments/E013_cascade_stage2/test_scores_d30_casc/` test scores

## Git
Private repo `SatvikJ01/amazon-ml-challenge-2026`, branch `main`, first commit `2fb1534`.
Data, caches, submissions, model binaries excluded by `.gitignore`.
