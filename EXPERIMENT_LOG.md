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
| E014 | 09-25 20:20 | How does the model transfer to an unseen country (France proxy)? Does exclusivity help at full density? | both stages trained on US only, all 883k India entities scored | **0.884** (P .954, R .806, singletons .749); exclusivity +0.0005 | informs decisions |
| E016 | 09-25 21:00 | Do density-dependent features cause the transfer loss? | LOCO ablations of count / absolute-retrieval / script features, both directions | no group helps (best +0.0018, within noise); India→US .941 | Rejected |
| E017 | 09-25 21:05 | File-order leakage? | Spearman of row positions S1 vs matches | +0.004 / +0.001: none | — |
| E017b | 09-25 21:15 | Does reverse retrieval recover blocking misses? | India S2, every S2 record → top-k S1 | recall .9464 → .9536 (k1) / .9583 (k3) / .9608 (k5) | **Yes (k=3)** |
| E018 | 09-25 21:20 | Char n-gram channel for near-miss spellings? | skeleton 3-grams + name 4-grams, 20k queries | +0.0033 recall at +20 cands/entity | Rejected |
| E019 | 09-25 21:25 | Are residual misses reachable via sibling records? | residual misses vs retrieved true siblings | 93 % have a retrieved sibling; 51 % sim ≥ 80; 33 % ≥15 pts closer to sibling than to S1 | **Pursue (collective ER)** |
| E021 | 09-25 22:00 | Do sibling (collective) features help the matcher? | stage 3 = stage-2 feats + OOF p2 + 9 sibling feats (anchors p2 ≥ 0.9), same holdout | **0.96332** (+0.0019 vs E013); multi-match up, singletons −0.003 | **Keep (day-2 rebuild)** |
| E021c | 09-25 22:03 | Is E021's gain just stacking? | stage 3 without sibling feats | 0.96102 (−0.0004) → gain is from sibling feats (+0.0023) | control |
| E021L | 09-25 22:03 | Can a light stage 3 (16 feats) run on saved test scores? | p2 + sibling + retrieval feats only | 0.96176 (+0.0003, noise) → not worth a submission | Rejected |
| E022 | 09-25 22:06 | Do anchors retrieve what S1 text misses? | true retrieved siblings query the S2 pool (India), top-k | recovers 16.8 % (k1) / **29.3 % (k3)** / 34.0 % (k5) of residual misses → pair recall ≈ .958 → .970 (upper bound) | **Pursue** |
| — | 09-25 21:40 | LB check of decision rule | day1_s2 (expF + exclusivity) | public **0.953** (s1: 0.952) | kept |
| E020 | 09-26 ~00:30 | Reverse-union retrain | features with reverse channel | India built; **US feature build OOM (5.5 GB)** → chain aborted; not evaluated | pending fix |
| E023A | 09-26 03:20 | Do anchor-retrieval features help on existing candidates? | predicted anchors (OOF p2 ≥ 0.9), anchor feats on E021p survivors | 0.9631 (E021p 0.9633) → no gain without new candidates | No |
| E023B | 09-26 03:29 | Do **predicted**-anchor candidates help? | + anchor top-3 new pairs (+4.8/entity), full feats, same holdout | **0.9669** (+0.0036 vs E021p, +0.0055 vs E013); cand recall .945→.9605; oracle .9806→.9846; FP 527→571; FN 9045→7566; singletons .9661→.9727; US .9705 / India .9616 | **Champion (holdout)** |
| E024 | 09-26 04:12 | Does a GPU XGBoost second model ensemble with LightGBM? | XGB (CUDA) on E023B data, same holdout | pred corr 0.9989; avg +0.00013 | Rejected (no diversity) |
| E025 | 09-26 03:50 | Loss budget of E023B by error type | counterfactual fixes | blocking misses 0.018 (mostly *easy* pairs lost to top-30 saturation); matcher other .0064; FP .0054; ambiguous ≈.005 | drives v3 |
| E026 | 09-26 03:52 | Which channel recovers the easy blocking misses? | reverse top-3 / exact keys on E023B misses | reverse: 65 % US, 25 % India of 'other'; keys (capped 10/60): +0.45 pt recall on India S2 | both into v3 |
| E030s2 | 09-26 04:35 | v3: forward ∪ reverse ∪ key candidates, cheap stage 1, 300k entities | stage 2 only | **0.9722** (60k holdout) / **0.9720** (original 30k); cand recall .9717; oracle .9901; US .9775 / India .9641 | **Champion** |
| E027 | 09-26 03:40 | Is native-script transliteration deterministic (dictionary-fixable)? | align Indic-script S2/S3 names with S1 names in train GT | 18.2 % of India S2/S3 names Indic; mapping 96.6 % deterministic over 1,347 tokens; test coverage 96.4 % | **Yes → E032** |
| E030lb | 09-26 04:55 | Loss budget of v3 stage 2 | counterfactual fixes, 60k holdout | blocking .0114 (native-script .0037, empty-addr .0026, other .0051); matcher .0116 (empty-addr .0049, other .0060); FP .0067 | drives E032 + stage 3 |
| E030s3 | 09-26 15:05 | Stage 3 on v3: GPU cross-fitted p2 → anchors (p2 ≥ 0.9) retrieve top-3 → sibling features | 60k holdout, same decision rule (expF) | **0.97522** vs stage 2 0.97180 (+0.0034); cand recall .9785 (from .9717); oracle .9918; US .9794 / India .9690 | **Champion** (day2_s2 candidate) |
| E033 | 09-26 15:10 | Do alphanumeric house-number agreement + within-entity number consensus fix the dominant FP/FN pattern? | 11 new stage-3 features (`src/extra_features.py`), same LightGBM, 60k holdout | **0.97711** vs 0.97522 (+0.0019); US .9808 / India .9715; singletons .9679→.9723, |T|=1 .9091→.9146; `nsup_extra_p` 3rd by gain | **Champion** |
| E034 | 09-26 14:40 | Name-token substitution (label-free filler-word score) + shared-address counts on top of E033 | 8 more stage-3 features (`src/extra_features2.py`) | running | — |
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

### E023 — predicted-anchor retrieval
- Anchors = cross-fitted stage-2 predictions (p2 ≥ 0.9); no ground truth used. Oracle (true-sibling)
  pilot E022 kept separately as the upper bound: India S2 residual misses recovered 29 % at k=3
  (pair recall ≈ .958 → .970 on that slice, measured after forward ∪ reverse).
- The gain is entirely from the added candidates (E023A ≈ E021p).
- France (no labels) — label-free: 94.2 % of entities have an anchor (train 92.9–93.4 %),
  3.15 anchors/entity (train 3.05–3.13), 3.6 new candidates/entity (train 4.8): mechanism transfers.

### E030 / E032 — v3 pipeline and native-script dictionary
- v3 candidates = forward top-30 ∪ reverse top-3 ∪ exact keys (caps 10/60). Stage 1 on cheap
  channel/competition features (negative-sampled 25 %, weighted, cross-fitted) keeps 21.2 % of
  24.6 M union pairs with 99.84 % of positives; string features only for survivors; 300k entities.
- Stage 2 alone: **0.9722** (60k holdout), 0.9720 on the original 30k (E023B 0.9669, E013 0.9614).
- Dictionary (E032, `src/translit.py`): 1,316 native tokens kept (count ≥ 3, share ≥ 0.8), learned
  from 551k aligned training pairs only. Applied as a parallel data version (`trainT`/`testT`):
  e.g. `raam maarketting praaivett limittedd` → `ram marketing private limited`;
  1.6 M train+test India names rewritten. Original files untouched (rollback).

### E030 stage 3 / E033 — error sample that motivated the number features (2026-09-26)
- 30 sampled stage-2 holdout false positives: ~half are generator near-copies of a true record with
  the house / unit number replaced (`1002→1004`, `72→93C`, `12→14B`, `224→226A`, `A201→A214`,
  `1A2→1A7`, `56TH→77TH`); alphanumeric tokens are invisible to the digit-only features.
  ~1/3 belong to another S1 with the same name (often empty address) or the same address.
- Missed true matches often carry a corrupted number shared by several records of the cluster
  (`671` in four records vs S1 `471`; `29` in two vs `49`), while a near-copy's number is unique.
- India training pairs: `anum_c_extra` 0 → 50.7 % positive, 1 → 5.7 %, 2 → 0.7 %, ≥3 → 0 %.
- France test (label-free look): no postcodes (0.4 % of addresses); names from a small generic
  vocabulary → address numbers carry most of the evidence there.
