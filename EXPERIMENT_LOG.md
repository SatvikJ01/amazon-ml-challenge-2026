# Experiment Log

Every experiment answers one question. Scores are **macro F0.5** (`src/metrics.py`) unless marked.
Blocking experiments report pair recall on 5k India S1 queries against the full 2.02 M India S2 pool.

| ID | Date/time (IST) | Question | Setup | Result | Kept? |
|---|---|---|---|---|---|
| E000 | 09-25 05:20 | What structure does the data have? | GT + source audits | H1/H2/H3 all true exactly; singleton 5.6 %; France only in test | — |
| E001 | 09-25 ~06:00 | Is relative-DF TF-IDF blocking viable? | TF-IDF word+char, max_df 2 %, 4k-row chunks | **OOM-killed (exit 137)**, crashed app | No |
| E002 | 09-25 ~07:30 | Recall/cost of bounded rare-term blocking? | hashing, abs max_df 2000, uni+bigram+nc keys | R@5 .897, R@30 .938, 3.8 GB, 1 s/5k queries | Yes |
| E002b | 09-25 ~07:40 | Is DF pruning limiting recall? | max_df 10000 | R@30 .939 (+0.001) at 5x cost | No (keep 2000) |
| E002c | 09-25 ~07:55 | Do consonant-skeleton keys fix transliteration misses? | + k:/kb:/kc:/ak: terms | R@30 **.945**, R@50 .956 | Yes |
| E003 | 09-25 ~07:00 | Full blocking, all 2.2 M train S1 | per-source processes, top-30/source | 185 M pairs; lean index peak 3.4 GB | Yes |
| E010 | 09-25 07:31 | First matcher: how good is LightGBM on v1 features? | 150k-entity sample, depth 30, 56 feats, holdout fold 0 (30k ent.) | **F0.5 0.9578** (thr .75: .9579; expF: .9575); US .9618 / India .9520; ceiling .9809 | **Champion** |
| E011 | 09-25 08:14 | Do canonical-digit + name-ambiguity features fix the E010 miss patterns? | E010 + 7 feats (63), same sample/holdout | **F0.5 0.9614** (+0.0036 paired); US .9661 / India .9545; all n_true buckets up | **Champion** |
| E012 | 09-25 10:50 | Can a cheap stage-1 model prune pairs before string features? | 10 blocking/competition feats, LGB | p1≥1e-3 keeps 25.1 % pairs, 99.89 % positives, ΔF0.5 −0.00004 | Yes |
| E013 | 09-25 11:00 | Does a stage-2 model on survivors match E011? | cross-fitted stage 1 → survivors → LGB 63 feats | **F0.5 0.96142** (E011 .96139); 15 cands/entity vs 60; ceiling .9806 | **Champion (cascade)** |
## Details

### E001 (failed)
Relative `max_df=0.02` on a 2 M-document block keeps terms with DF up to 40k; each 4k-query chunk
product had billions of non-zeros. Fix: absolute DF cap + chunk sizing from predicted nnz (E002).
Lesson recorded in memory: all heavy jobs now run under `systemd-run -p MemoryMax=5G`.

### E002c
Misses before: 76 % had zero name-token overlap (phonetic transliteration, e.g. `praaprttiis`).
The skeleton maps `properties`/`praaprttiis` to `prprts`. Remaining misses: generic names with
truncated addresses competing against many similar distractors. Next lever there is a learned
reranker, not more keys.

### E010 — first matcher
- Early-stop logloss 0.0080 (prior-only ≈ 0.21). 1,800+ rounds.
- Calibration is good (predicted bin means match observed rates), so expected-F0.5 and a
  0.65–0.8 threshold tie within ±0.0009 (SE on 30k entities). The decision layer is not the bottleneck.
- A constant `missed` term is harmful (0.908): it removes the option to abstain and kills singletons.
- Loss decomposition (30k holdout entities): FP 0.0074 (837 pairs), **retrievable FN 0.0157**
  (4,332), **blocking FN 0.0191** (5,621). Recall dominates the loss despite the F0.5 weighting.
- Single-feature AUCs (India): comp_margin **.989**, blk_score .963, comp_is_top .962, addr_tset .924,
  name_tset .760. P(match | not top claimant) = 0.29 %.
- Missed-match patterns: (a) empty candidate address (34 % India / 42 % US of misses vs ~4 % of
  positives) with generic names; (b) digit corruption treated as mismatch: leading zeros
  (`2818→02818`), dropped end digits (`324→32`, `18704→8704`), ranges.
- Next: E011 = v2 features (canonical digits + name-ambiguity counts).

### E011 — v2 features
- Added `digz_*` (leading-zero-stripped digit Jaccard / first-number equality / best and first
  number similarity with 1-digit truncation and 1-edit tolerance) and name-ambiguity counts
  (`s1_name_count`, `s1_core_count`, `cand_core_count_in_s1`, from S1 only).
- 0.9578 → **0.9614** on identical holdout entities (paired; SE of a single score ≈ 0.0009).
  Singletons .9637→.9685, 1-match .8813→.8916, US .9618→.9661, India .9520→.9545.
- `digz_first_sim` is the #3 feature by gain after `comp_margin` and `comp_is_top`.
- Decision rules still tie (thr 0.7: .96139, expF: .96122).
- Test blocking finished: 1,732,544 entities, 103.95 M pairs (France incl.).
- Next: day1_s1 submission (E011, thr 0.7, no exclusivity); then E012 = US-only model scored on
  all 883k India entities (LOCO proxy for France + exclusivity at full density).

### E012/E013 — cascade
- Real-data feature throughput was 10.8k pairs/s (synthetic benchmark said 72k), i.e. ~2.7 h per
  test inference at 104 M pairs. The first day1_s1 run was stopped after France for this reason.
- Profiling: uncached `skeleton_tokens` regex was 48 % of feature time -> lru_cache + per-unique-
  string computation: 9.5k -> 20k pairs/s (warm). Refactor verified exact on 44 columns x 20k pairs.
- Stage 1 (cross-fitted, 5 entity folds): OOF keeps 25.2 % of pairs, 99.88 % of positives.
- Stage 2 on survivors: **0.96142** vs 0.96139 (E011) on the same holdout -> no accuracy cost,
  4x fewer pairs to featurise, candidate_pairs.tsv 4x smaller (~26 M ids instead of ~104 M).
