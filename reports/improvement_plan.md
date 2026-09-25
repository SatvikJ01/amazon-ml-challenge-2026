# Improvement Plan — 2026-09-25 21:25 IST

## Bottleneck
Loss budget on the in-distribution holdout (F0.5 0.9614): blocking misses ≈ 0.019, retrievable
misses ≈ 0.016, false positives ≈ 0.007. A ~0.987 leaderboard is above our oracle ceiling
(0.9806), so **both recall stages must improve**, and the unseen-country drop (LOCO 0.884–0.941)
compounds it on France.

## Gap analysis (evidence, not assumption)
| Hypothesis | Evidence | Verdict |
|---|---|---|
| France alone explains LB gap | implied France ≈ 0.91 fits LB 0.952, but leader > our ceiling | partial |
| Density-dependent features break transfer | E016 ablations: no gain | rejected |
| File-order / id leakage used by others | Spearman ≈ 0, id diffs random | rejected |
| Blocking recall caps us | ceiling 0.9806 < 0.987 | **confirmed** |
| Missed records are reachable via siblings | E019: 93 % have a retrieved sibling, 51 % sim ≥ 80 | **strong** |
| Reverse retrieval recovers generic-name misses | E017b: +1.2 recall pts at +1.8 cands | **confirmed** |
| Char n-gram channel | E018: +0.3 pts | rejected |

## Ranked plan
| # | Change | Expected gain | Cost | Risk | Pilot / stop rule |
|---|---|---|---|---|---|
| 1 | **Reverse retrieval (target → top-3 S1)**, union with forward; features fwd/rev rank, mutual-NN flag | +0.004–0.008 (recall, generic names) | ~1 h blocking + rebuild | low | done (E017b). Promote if holdout F0.5 ≥ +0.002 |
| 2 | **Sibling / profile features (collective ER)**: for each candidate, max similarity to the entity's confident matches (OOF p ≥ 0.9), sibling-address agreement, count of confident siblings | +0.005–0.010 (retrievable misses, precision) | 5-fold OOF stage 2 (~40 min) + features | medium (label leakage if not OOF) | OOF only; stop if < +0.002 on holdout |
| 3 | **Sibling-query retrieval**: confident matches re-query the pools; union new hits | recall on the 51 % sibling-reachable residual | ~1 h | medium (error propagation) | pilot on India S2; stop if precision on added pairs < 20 % |
| 4 | Query-side (record → S1) normalisation / top-2 margin as features | precision on contested records | low | low | ablate under LOCO |
| 5 | Decision: expF + exclusivity (already built as day1_s2) | +0.000–0.002 | none | low | LB check |
| 6 | Model diversity (CatBoost) + ensemble | +0.001–0.003 | 1 h | low | only after 1–3 |

Deferred with reasons: multilingual embeddings (E018 shows spelling-level misses are only part
of the residual; siblings explain more for less compute on a 4 GB GPU); Fellegi-Sunter (LightGBM
already learns agreement weights; low marginal information); adaptive K (subsumed by reverse +
sibling channels).

## Execution order
Tonight: implement reverse channel in production blocking and launch train + test reverse runs
(~50 min, checkpointed per country/source). Day 2 morning: union features + stage-1/2 retrain
(→ day2_s1), then OOF sibling features (→ day2_s2), then sibling-query retrieval.

## Update 22:10 — evidence after pilots
| Lever | Evidence | Status |
|---|---|---|
| Reverse retrieval top-3 | recall .946 → .958 (India S2) | production blocking running; E020 queued |
| Sibling features (stage 3) | +0.0023 over matched control (E021 vs E021c) | needs full stage-2 features at test time → day-2 rebuild |
| Anchor-query retrieval top-3 | +29 % of residual misses → recall ≈ .970 (upper bound) | implement as pass 2 |
| Light stage 3 on saved scores | +0.0003 | rejected; no day-1 submission from it |

Day-2 architecture (two-pass):
1. Pass 1: forward ∪ reverse candidates → stage 1 → stage 2 → p2; **save stage-2 features per part**.
2. Anchors = p2 ≥ 0.9 → query the target pools (top-3) → new pairs not in pass 1 → stage-2 features + p2.
3. Stage 3 (collective) over pass-1 ∪ pass-2 pairs → expF + exclusivity.
Training mirrors it exactly with cross-fitted p2 on the 150k-entity sample.
Stop rule: promote only if holdout F0.5 beats E021 (0.9633) by ≥ 0.002.
