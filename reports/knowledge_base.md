# Knowledge base — Amazon ML Challenge 2026 (Business Entity Resolution)

State as of 2026-09-27 00:40 IST. Measured results are marked with their experiment id; anything
without one is a hypothesis. Holdout = 60k S1 entities (entity-hash fold 0 of the 300k training
sample), scored with the official metric against the **full** ground truth (retrieval misses count).

---

## 1. Task, metric, constraints (verified from the problem statement)

- For every Source-1 (S1) business record, output all Source-2/Source-3 records of the same business
  (possibly none). Fields: `business_name`, `business_address`, `country` only.
- Metric: F0.5 per S1 entity, macro-averaged. `F = 1.25·TP / (0.25·|T| + |P|)`; empty/empty = 1,
  anything else with an empty side = 0. Precision counts 2× recall, but recall still matters.
- Train countries: US, India. Test: US 663k, India 810k, **France 259k (15 %, no labels)** S1 records.
- Rules: no external data / APIs / geocoding; final model MIT/Apache-2.0, ≤ 8 B parameters.
  5 submissions per day; public LB is a subset, private LB decides.
- Outputs: `matching_results.tsv` (scored), `candidate_pairs.tsv` (final set fed to the matcher).

## 2. Verified data structure (E000, E017, audits)

| Fact | Consequence |
|---|---|
| Ground truth lists every train S1 (incl. 5.6 % singletons) | singletons are real, abstaining is sometimes right |
| Every S2/S3 record belongs to **at most one** S1 (strict many-to-one) | one-owner exclusivity is valid |
| No cross-country matches | country is a lossless blocking partition |
| ~26 % of train S2/S3 records have no owner (distractors); label-free estimate for test ≈ 40 % | precision is harder on test |
| Mean 3.46 matches per S1; ≤5 per source (S2), ≤6 (S3); S2/S3 counts ~independent | weak count prior only |
| Record ids random w.r.t. clusters (corr 0.0001); no file-order leakage (E017) | no id/order leak exists |
| 18.2 % of India S2/S3 names in Indic scripts; generator transliterates each word **consistently** (96.6 % deterministic mapping, E027) | dictionary fix (E032) |
| France: no postcodes; names from a small generic vocabulary (86 % core-name sharing vs US 51 %, India 65 %) | address numbers carry the evidence |

### Generator behaviour seen in error samples
- **True-record noise**: case/punctuation, abbreviations (St/Street/Saint, Dr, Ave), state name ↔ code
  ↔ native script, city variants (Bombay/Mumbai), `CITY` appended, NULL tokens (`N/A`, `<NULL>`),
  component drops/reorders, OCR-like typos (`5urya`, `PU8LIC`, `B1ass`), filler words added
  (`Services`, `Partners`, `Center`, `LLC`, `Shri`, `M/s`, `The`), aliases (`X formerly Y`,
  `X a/k/a Y`, `doing business as`), domain forms (`xyz.com`, `@handle`), appended ids
  (`#56422`, `- 9351981258`), **number corruption** (drop first/last digit `1002→002`, `603→60`;
  zero-pad `0600`; digit substitution `471→671` often **shared by several records of the cluster**),
  empty addresses, random brand names at the right address (`ZEPHHALO`).
- **Distractors**: near-copies with a **neighbouring house/unit number** (`1002→1004`, `629→631`,
  `12→14B`, `224→226A`, `A201→A214`), near-copies with **one distinctive name word replaced**
  (`Imperial Care→Imperial Cole`, `Logistics→Global`, `Private→Public`), records of **another S1 with
  the same name** (often empty address) or **the same address** (different business, same building).

## 3. Current production pipeline (E032 = best, holdout 0.98265)

```
normalise (unidecode, punctuation, NULL tokens, consonant skeletons, canonical digits)
 + native-script dictionary for India (trainT/testT data version, 1,316 tokens learned from train GT)
   │
   ├─ forward retrieval: rare-term IDF cosine over name+address terms (words, bigrams, skeleton keys,
   │  name-prefix key), hashed 2^23, absolute DF cap 2000, top-30 per (S1, source)
   ├─ reverse retrieval: each S2/S3 record → top-3 S1
   └─ exact-key channel: name-skeleton × house-number keys, frequency caps (S1 ≤10, target ≤60)
        │ union per (country, source)
        ▼
 stage 1  cheap LightGBM (channel scores/ranks, competition features from the full forward table),
          negative-sampled 25 % (weighted), cross-fitted; keep p1 ≥ 1e-3 → candidate_pairs.tsv
 stage 2  LightGBM, 73 string/address/context features → p2 (XGBoost GPU twin for cross-fitting)
 anchors  predicted matches (cross-fitted p2 ≥ 0.9) query the target pools → top-3 new candidates
 stage 3  LightGBM, 114 features: stage-2 features + p2 + sibling features (similarity to the
          entity's anchors) + E033 number consensus + E034 name substitution / shared address +
          E035 number distance / empty-address context / name duplication
 decision expected-F0.5 prefix selection per entity + one-owner exclusivity
```
Key files: `src/v3.py`, `src/key_channel.py`, `src/stage1.py`, `src/train.py`, `src/collective.py`,
`src/anchor_pass.py`, `src/extra_features{,2,3}.py`, `src/translit.py`, `src/infer_v3.py`,
`src/decision.py`. Training sample: 300k S1 entities (240k train / 60k holdout).

## 4. Score history

| Model | Holdout | Public LB | Gap |
|---|---|---|---|
| E013 cascade (v1) | 0.9614 | day1_s1 0.952 (thr 0.7), day1_s2 **0.953** (expF + exclusivity) | 0.0084 |
| E030 v3 stage 2 | 0.9718 | day2_s1 **0.9634** | 0.0084 |
| E035 stage 3 + features | 0.9799 | day2_s2 **0.9705** | 0.0094 |
| **E032** native-script + stage 3 | **0.9827** | day3_e032 pending (≈ 0.973–0.974 if gap 0.0084–0.0094) | — |
| Leaderboard top | — | 0.990556 | ≈ 0.999 holdout-equivalent |

Holdout gains have transferred ~1:1 to the LB so far. Repeat-run noise on the holdout: ±0.0002 (E038).

## 5. Everything tried

### Retrieval / candidate generation
| Id | What | Result | Status |
|---|---|---|---|
| E001 | TF-IDF word+char, relative max_df | OOM, crashed the app | rejected |
| E002 | Hashed rare-term IDF, absolute DF cap 2000 | R@30 .938 (India S2 5k queries) | kept |
| E002b | DF cap 10000 | +0.001 recall at 5× cost | rejected |
| E002c | Consonant-skeleton keys | R@30 .945 | kept |
| E017b | Reverse retrieval (target → top-k S1) | recall .9464 → .9583 (k=3) | kept (k=3) |
| E018 | Char n-gram channel | +0.0033 recall at +20 cands/entity | rejected |
| E026 | Exact capped keys | +0.45 pt recall India S2 | kept |
| E022 | Oracle anchors (true siblings) query pools | 29 % of residual misses at k=3 (upper bound) | → E023 |
| E023B | **Predicted** anchors (p2 ≥ 0.9) → top-3 new candidates | +0.0036 F0.5; recall .945 → .9605 | kept |
| E030 | v3 union forward ∪ reverse ∪ key + cheap stage 1 | stage 2 0.9722, recall .9717, oracle .9901 | kept |
| E037 | Similarity channels **incremental over the current union**: name-only / address-only IDF, char 3-gram TF-IDF name/address, BM25, RRF | all channels unioned: depth 5 → 3.9 % of misses (+0.0008 recall, +21 cands/ent); depth 20 → 11.8 % (+0.0025 recall, +114 cands/ent); best single = address (5.5 % @20); RRF top-10/source 2.2 % | deprioritised |
| E032c/E032 | Native-script dictionary → rebuild | native misses 886 → 106; recall .9785 → .9821; oracle .9918 → .9939 | kept |

### Matcher features and stages
| Id | What | Result | Status |
|---|---|---|---|
| E010 | First LightGBM (56 feats) | 0.9578 | superseded |
| E011 | Canonical digits + name-ambiguity counts | 0.9614 (+0.0036) | kept |
| E012/13 | Stage-1 pruning cascade | same F0.5, 4× fewer pairs | kept |
| E016 | Remove density-dependent features (LOCO) | no gain | rejected |
| E021 | Sibling (collective) features, stage 3 | +0.0019 (control without siblings −0.0004) | kept |
| E021L | Light stage 3 (16 feats) | +0.0003 (noise) | rejected |
| E023A | Anchor features without new candidates | no gain | rejected |
| E030s3 | Stage 3 on v3 | 0.97180 → 0.97522 (+0.0034) | kept |
| E033 | Alphanumeric house-number agreement + within-entity number consensus | +0.0019 | kept |
| E034 | Name substitution (label-free filler-word score) + shared-address counts | +0.0023 | kept |
| E035 | Number distance/parity, empty-address context, name duplication | +0.0005 | kept |
| E036 / E036T | Stage 4: collective features from cross-fitted stage-3 p3 | laptop OOM; EC2: 0.98218 vs 0.98265 (−0.0005) | rejected |

### Models, decision, data
| Id | What | Result | Status |
|---|---|---|---|
| E010 | Calibration check; expF vs threshold | tie within ±0.0009; constant "missed" term harmful | expF kept |
| day1_s2 | expF + exclusivity on LB | +0.001 LB | kept |
| E024 | LightGBM + XGBoost ensemble | corr 0.9989, +0.00013 | rejected |
| E014 | Transfer to unseen country (US → India) | 0.884 (singletons .749) | informs France risk |
| E038 | Learning curve 25/50/100 % training entities | 0.98159 / 0.98200 / 0.98267 | more data helps modestly |
| E031 | Self-training for France (LOCO proxy) | never ran (memory, then deprioritised) | open |

## 6. Remaining loss (E032 stage 3, loss 0.0173) — counterfactual "fix this bucket only"

| Bucket | Pairs | F0.5 if fixed | Notes |
|---|---|---|---|
| Retrieval miss — empty address | 1,452 | +0.0024 | name-only record never retrieved |
| Retrieval miss — other | 2,149 | +0.0039 | aliases, changed numbers, heavy corruption; not lexically close (E037) |
| Retrieval miss — native script | 106 | +0.0003 | mostly solved |
| Matcher miss — empty address, **shared S1 name** | 2,270 | +0.0036 | likely near-irreducible (identical names, no address) |
| Matcher miss — empty address, unique name | 852 | +0.0013 | reducible in principle |
| Matcher miss — other | 1,698 | +0.0030 | corrupted numbers, heavy name noise |
| False positives | ≈700 | +0.0032 | neighbour-number / one-word near-copies, same-name other S1 |

Micro precision 0.9967, micro recall 0.9583. Worst group: one-match entities (0.9315).
Empty-address records ≈ 42 % of the loss.

## 7. Running now (00:40 IST)

| Where | Job | ETA | Expected |
|---|---|---|---|
| Laptop | E032 test inference (pass 2 → write) → `day3_e032` | ~02:00 | LB ≈ 0.973–0.974 (holdout 0.9827 − observed gap) |
| EC2 r7i.2xlarge | **E039**: E032 pipeline on 3× training entities (900k), LightGBM OOF | holdout ~06:00–06:30 | ≈ +0.001 (extrapolated from E038) |

Then: E039 test inference on the laptop (~5 h) if it beats E032.

## 8. Compute facts
- Laptop: i5-1240P 12C/16T, 15.7 GB RAM (~9 usable with desktop apps), RTX 2050 4 GB. One heavy job
  at a time; GPU XGBoost cross-fitting peaks ~9.7 GB on stage-3 data (OOMs with apps open).
- EC2 r7i.2xlarge: 8 vCPU, 61 GB, no GPU; LightGBM 1M×114×200 rounds 17 s, XGBoost hist 26 s;
  stage-3 LightGBM training 4–5 min; 5-fold LightGBM cross-fit 32 min at 7.7 GB. Upload from the
  laptop ≈ 1.6 MB/s.
- Timings (laptop): test pass 1 ≈ 2.3 h (11 parts), pass 2 ≈ 1–1.5 h, write + validator ≈ 15 min.

## 9. Open directions (not yet tried) — seeds for brainstorming

| Idea | Targets | Evidence / risk |
|---|---|---|
| Empty-address **ownership model** across S1s that share a name (listwise: which S1 owns this name-only record; use each S1's other matches, its count of empty-address matches per source, name-variant fidelity) | 0.0049 matcher + part of 0.0024 retrieval | biggest bucket; shared identical names may be irreducible; needs all-S1 scoring at train time (only 300k/2.2M sampled → skew) |
| Deep reverse retrieval only for empty-address targets (target name → top-20 S1) | 0.0024 retrieval | recovered records would mostly fall into the ambiguous bucket |
| Global assignment (Hungarian / min-cost flow) over candidate → S1 with expected-F0.5 utilities instead of greedy exclusivity | FPs, ambiguous records | exclusivity already greedy; unclear gain |
| Cluster-level reasoning over S2↔S3 records (records of one cluster should match each other) | matcher other, FP | stage 3 sibling features capture part of it; stage 4 did not help |
| Seed / bagging average of stage-3 LightGBM | noise ±0.0002 | cheap; small |
| Decision-layer calibration by segment (country, empty address, n candidates) | FP/FN trade-off | risk of overfitting the holdout |
| France adaptation: self-training on confident France pairs, LOCO validated (US→India) | 15 % of test | LB gap moved only 0.001 so far; no labels |
| Neural cross-encoder (small multilingual, MIT) on uncertain pairs only | name typos, aliases | GPU needed; structurally different model → real ensemble diversity |
| Name-alias parsing (`X formerly Y`, `a/k/a`, `dba`) into two names; domain-name splitting (`bangaloresouthinfracon.com`) | matcher other | cheap feature work |
| Larger training population (E039 running) | all | measured slope +0.0004–0.0007 per doubling |

## 10. Lessons
- Error sampling + counterfactual loss budgets found every real gain (E011, E023, E032, E033–E035).
- Similarity retrieval over the existing union recovers little (E037); the remaining misses are not
  lexically close in name or address.
- Structural collective features help once (stage 3); a second round does not (stage 4).
- Near-identical model families do not ensemble (corr 0.999).
- Every heavy job needs a memory cap, streaming I/O, per-part checkpoints and a watcher that reports
  failures as well as success.

## 11. E046 — pretrained text cross-encoders (09-28, after the deadline)

**Idea.** A pretrained transformer reads both records as one sequence
(`S1 name | S1 address </s></s> candidate name | candidate address`), so every token of one record can
attend to every token of the other; it is fine-tuned end to end as a binary "same business?" classifier.
It sees the raw strings (typos, casing, `<NULL>`, reordered address parts, native script), which our
hand-built similarity features only summarise.

**Where it is applied.** Only the band 1e-3 ≤ pf < 0.999 of the shipped probability: the oracle says the
whole recoverable F0.5 is there (0.98518 → 0.99399 if those 323k holdout pairs were perfect; pairs outside
the band are worth < 0.0003). Test band: 3.74M pairs (US 2.4, India 1.7, France 3.1 per entity).

**How it is combined.** A LightGBM residual on the holdout band, `init_score = logit(pf)`, features = the
CE logit(s) + within-entity CE context (rank, best other, gap) + pf context; 2-fold entity CV on the holdout
for evaluation, the whole holdout for test. Claimant (same-candidate) features were dropped: the holdout
holds only ~8 % of all S1, the test 100 %, so their distribution shifts.

**Results so far.**
- ELECTRA-small (14M, English, 600k pairs, 14 min on the laptop GPU): band AUC 0.970 (pf 0.9835), yet
  stacked **+0.0013** holdout F0.5 (control without CE +0.0001): the CE is weaker alone but complementary.
- What it catches (holdout examples): distractor names one edit away with the identical address
  (`Chandraksh` vs `Chandrakoro`), neighbour house numbers (6622 vs 6629A), wrong city; and it rescues true
  matches with OCR-style noise (`8irnbaum`/`Birnbaum`, `Westem Hovnnaann`/`Western Hovnanian`).
- **Unseen country (LOCO, France proxy):** CE trained on US only → India band AUC 0.79 (ELECTRA) / 0.77
  (multilingual e5-small) vs 0.97 in-country; stacked it *hurts* India (−0.0021 / −0.0025). The stacker
  itself transfers (+0.0014 on India from a US-only stacker when the CE saw India). ⇒ **no CE correction on
  France**; the CE only helps countries it was trained on.
- Test pool composition (record counts): unowned decoys per S1 ×1.9 in test, other businesses ×0.5 (US);
  weighting the stacker's negatives accordingly did not help (E046c), and the CE gain holds on a
  decoy-weighted holdout metric.

**Model choice (research 09-28, licences checked on the HF cards).** MIT/Apache, ≤ 8B, T4 (fp16, no bf16,
no FlashAttention-2): mDeBERTa-v3-base (MIT) is the best value; bge-reranker-v2-m3 (Apache-2.0, XLM-R-large
reranker) the strongest cross-lingual option; XLM-R-large + mDeBERTa-v3-base won Kaggle Foursquare Location
Matching (multilingual POI name/address matching). 7B decoders (Qwen2.5-7B, Mistral-7B, Apache-2.0) need
18–30 h of pure inference for 4M pairs on 2×T4 — not feasible. Excluded by licence: Jina rerankers,
Jellyfish (CC-BY-NC), Qwen2.5-3B (research licence); mmBERT uses the Gemma tokenizer (licence caveat).

**Kaggle runs.** `e046-ce`: XLM-R-base (GPU0) + mDeBERTa-v3-base (GPU1), 2.05M pairs, 1 epoch, fp16.
`e046-ce-large`: bge-reranker-v2-m3 on each GPU, bagged halves (entity parity), frozen word embeddings,
lr 1e-5. Data = private Kaggle dataset `e046-ce-pairs` (text pairs only).

**Final E046 numbers (09-28 14:50).** Band AUC: mDeBERTa-v3-base 0.9813, LaBSE 0.9807, XLM-R-base 0.9795,
bge-reranker-v2-m3 halves 0.9784/0.9782 (time-capped at ~0.85M pairs each), ELECTRA-small 0.9764 (pf 0.9835).
Best stack = mDeBERTa + LaBSE + 10 pairwise stage-3 features: holdout 0.985184 → **0.987643 (+0.00246;
India +0.0028, US +0.0022)**; more models add nothing. Lessons: data volume beat model size on a T4 budget;
LaBSE (translation-pair pretraining, frozen embeddings) was the best value (75 min); mDeBERTa checkpoints load
as fp16 → cast to fp32 before AMP training. Candidates day4_mdl_C (US full) / day4_mdl_D (US down-only),
France raw in both; expected LB ≈ 0.9795.

## 12. E047 — dense-retrieval rescue (10-02)
Fine-tuned multilingual-e5-small bi-encoder (2.35M ground-truth pairs, 55 min on a T4) + exact GPU kNN in both
directions per country; pairs already in our candidate set are dropped with a Bloom filter; the new pairs are scored by
the LaBSE cross-encoder and a LightGBM rescue model. The new holdout pairs contain **76 % of the true matches our
lexical retrieval missed**; holdout F0.5 0.987717 → **0.991184 (+0.0035)**. This is the component the published
0.988–0.990 solutions shared and we lacked: after the cross-encoders, retrieval recall was the largest remaining loss.
Post-deadline total: 0.985184 → 0.991184 on the holdout (France unchanged).
