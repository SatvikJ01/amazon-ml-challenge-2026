# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** {{TEAM_NAME}}  
**Team Members:** {{TEAM_MEMBERS}}  
**Final submission:** `{{FINAL_SUB_ID}}` ({{FINAL_RULE}})  
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

We built a blocking-plus-classifier cascade. The candidate set comes from three complementary retrieval channels: rare-term IDF top-30, reverse top-3 and exact capped keys. For India, a native-script dictionary learned from the training ground truth is applied first. A cheap LightGBM (stage 1) prunes the union, and a pairwise LightGBM (stage 2) scores the survivors. A **collective** stage 3 then uses each entity's own predicted matches ("anchors") in two ways: to retrieve missed records and as sibling evidence. A residual level-2 LightGBM corrects stage 3. The last step is a per-entity **expected-F0.5-optimal** decision with one-owner exclusivity. Every change was chosen from a counterfactual loss budget on a held-out split that is scored exactly like the leaderboard. The holdout macro F0.5 rose from 0.9578 (first model) to **0.985458** for the best holdout model (the final submission `{{FINAL_SUB_ID}}` holds {{FINAL_HOLDOUT}} on the holdout). The public LB rose from 0.952 (first submission) to {{BEST_LB}} ({{BEST_LB_SUB}}, the best score confirmed when this package was built); the final file `{{FINAL_SUB_ID}}` scores {{FINAL_LB}}. The final system uses no external data, no pre-trained or neural model and no API calls. All learned models are LightGBM (MIT licence), trained from scratch.

---

## 2. Methodology

### 2.1 Problem Analysis

**Validation protocol.**
- The holdout is entity-hash fold 0, so all pairs of one S1 entity sit in the same fold.
- It is scored with the official macro F0.5 against the **full** ground truth, so records the blocker never retrieved still count as misses.
- Early stopping of stages 2 and 3 and of the residual uses an inner 10 % split of the training entities only.
- The holdout-to-public-LB gap stayed at 0.0082–0.0096 across the eight scored submissions (SUBMISSIONS.md). Model changes transferred about 1:1 (native-script dictionary, 3× data, India residual: holdout-predicted +0.00069, LB +0.00073), with one exception: the full E044 correction on US, worth +0.00107 per US holdout entity, lost 0.00062 on the LB (day3_l2b 0.975885 vs day3_l2b_india 0.976505). Its test-time US changes were 2.7× the holdout rate (38.7 vs 14.1 per 1,000 entities), so the US correction is a distribution-shift failure, not noise.

| Verified fact (source) | Consequence |
|---|---|
| Train: 2.21 M S1 (US 1.32 M, India 0.88 M), 5.03 M S2, 5.29 M S3. Test: 1,732,544 S1 (US 663,106 · India 809,986 · **France 259,452 = 15 %, unlabelled**) and 9.97 M S2+S3 | the pipeline must be country-agnostic; France cannot be validated |
| The ground truth lists every S1; 5.6 % are singletons (E000) | abstaining must be possible; a singleton scores 1.0 or 0 |
| Every S2/S3 record belongs to **at most one** S1, and there are no cross-country matches (E000) | one-owner exclusivity is valid; country is a lossless partition |
| About 26 % of train S2/S3 records have no owner (distractors); the label-free estimate for test is about 40 % | precision is harder on test than on the holdout |
| 3.46 matches per S1 on average, at most 5 per S2 and 6 per S3; ids are random and there is no file-order leak (E017) | only a weak count prior is available (e.g. a remaining-capacity feature) |
| 18.2 % of India S2/S3 names are in Indic scripts, and the generator transliterates each word **consistently**: a 96.6 % deterministic mapping over 1,347 tokens that covers 96.4 % of test tokens (E027) | a learned dictionary fixes them (E032) |
| France has no postcodes (0.4 % of addresses) and draws names from a small generic vocabulary (core-name sharing 86 % vs US 51 %, India 65 %) | address numbers carry most of the evidence |
| Transfer to an unseen country is costly: US-only → India scores 0.884 (E014) and 0.92324 vs 0.97593 in-country (E043) | France has no labels, so any correction there is unvalidated (`{{FINAL_SUB_ID}}`: France = {{FINAL_FRANCE}}) |

**Generator noise seen in error samples.** True records show case and punctuation changes, abbreviations (St/Street/Saint), state name ↔ code ↔ native script, city variants (Bombay/Mumbai), `N/A`/`<NULL>` tokens, dropped or reordered components, OCR typos (`5urya`, `PU8LIC`), filler words (`Services`, `LLC`, `Shri`, `M/s`), aliases (`X formerly Y`, `a/k/a`), domain forms, appended ids and empty addresses. They also show **number corruption** (`1002→002`, `603→60`, `0600`, `471→671`), and the same corrupted number is often shared by several records of one cluster.

**Distractors** are near-copies with a **neighbouring house or unit number** (`1002→1004`, `12→14B`, `A201→A214`), near-copies with **one distinctive name word replaced** (`Imperial Care→Imperial Cole`, `Private→Public`), and records of **another S1 with the same name** (often with an empty address) or of a different business at the same address.

**Quantified signals.**
- Competition decides a lot. `comp_margin` alone has AUC 0.989 on India, and P(match | candidate is not its top claimant) is 0.29 % (E010).
- Unmatched alphanumeric numbers are strong negative evidence. With 0 / 1 / 2 extra numbers in the candidate, 50.7 % / 5.7 % / 0.7 % of pairs are positive (E033).
- 93 % of residual misses have a retrieved true sibling, and for 33 % that sibling is at least 15 similarity points closer to the missed record than the S1 text is (E019). This motivated the collective stage.
- Empty candidate addresses make up 34 % (India) and 42 % (US) of misses, against about 4 % of positives (E010).

### 2.2 Solution Strategy

**Approach Type:** Hybrid. Multi-channel blocking feeds cascaded gradient-boosted classifiers, followed by a collective (anchor/sibling) stage, a residual level-2 model and a Bayes-optimal decision layer.  
**Core Innovation:**
1. **Predicted-anchor collective ER.** Confident matches are cross-fitted during training. They re-query the target pools and supply sibling evidence: +0.0055 over pairwise-only (E021 + E023B).
2. **Native-script dictionary** learned from the training ground truth: +0.0028 (E032).
3. **Generator-aware features** for the dominant distractor patterns (number consensus, name substitution, number distance): +0.0047 (E033–E035).
4. **Residual level-2** model, initialised at logit(p3) and trained on cross-fitted out-of-fold probabilities for 702k entities: +0.00123 on the India + US holdout (E044). On the LB the India part transferred (+0.00073) and the US part did not (−0.00062), see §5.

**Final submission `{{FINAL_SUB_ID}}`:** {{FINAL_RULE}}. Per country: India = {{FINAL_INDIA}}, US = {{FINAL_US}}, France = {{FINAL_FRANCE}}. Holdout F0.5 {{FINAL_HOLDOUT}}; public LB {{FINAL_LB}}.

```
raw TSV → normalise (unidecode, case/punct, NULL tokens, consonant skeletons, canonical digits)
        → India only: native-script dictionary (1,316 tokens, train GT only)  → data version trainT / testT
per (country, target source S2|S3):
  forward IDF top-30 ∪ reverse top-3 ∪ exact capped keys           test: 149,337,340 pairs
  → stage 1  LightGBM, 20 cheap features, keep p1 ≥ 1e-3                  36,366,773
  → stage 2  LightGBM, 73 pairwise/context features → p2
  → anchors  p2 ≥ 0.9 re-query both pools, top-3 each                    +6,692,150 new
                                        = candidate_pairs.tsv            43,058,923 (24.85 / S1)
  → stage 3  LightGBM, 114 features (+ p2, anchor, sibling, number/name/empty-address) → p3
  → residual LightGBM, 204 features, init_score = logit(p3), rows p3 ≥ 1e-3 → p;
    applied per country as in {{FINAL_SUB_ID}} (full / down-only min(p3, p) / raw p3)
  → p ≥ 0.01 → one-owner exclusivity → expected-F0.5 prefix per S1 → matching_results.tsv
```

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** Blocking is partitioned by country (lossless), with S2 and S3 blocked separately. The channels are unioned.

| Channel | Definition | Test rows |
|---|---|---|
| Forward | Rare-term IDF cosine over namespaced name and address terms. These are name and address uni/bigrams, house-number terms, consonant-skeleton keys (`properties`/`praaprttiis` → `prprts`) and a 10-character concatenated-name prefix. Terms are hashed to 2^23 features with an absolute DF cap of 2,000. Keeps the top-30 per (S1, source). | 103,951,900 |
| Reverse | Every S2/S3 record retrieves its top-3 S1 (no depth limit per S1) | 29,906,150 |
| Exact keys | Name-token skeleton × one of the first two canonical address numbers, plus adjacent skeleton bigrams. A key is used only if it occurs in ≤ 10 S1 and ≤ 60 target records. | 48,017,895 |
| Anchors | Each predicted match (p2 ≥ 0.9) queries the S2 and S3 pools and keeps its top-3 neighbours | 17,598,114 (6,692,150 new) |

- **Candidate pairs generated:** The union holds 149,337,340 pairs (86.2 per S1). Stage 1 uses 20 cheap features: channel scores, ranks and membership, plus competition features computed on the full forward table. It was trained with 25 % negative sampling (weighted) and cross-fitted, and it keeps 36,366,773 pairs (24.4 %). On the training data it keeps 22.9 % of pairs and 99.85 % of positives (E039_stage1), at a cost of −0.00004 F0.5 when the same pruning rule was introduced (E012, v1 stage 1). Adding the anchor pairs gives **43,058,923 pairs in `candidate_pairs.tsv`, 24.85 per S1 entity** (France 26.68 · India 25.22 · US 23.69). That is 6.4 × 10⁻⁶ of all within-country pairs, a reduction ratio of 0.9999936. This is exactly the set the stage-3 and residual models score.
- **How we ensured true matches were not lost:** Most blocking misses were *easy* pairs pushed out of the top-30 by look-alikes with generic names (E025). We added channels without a per-S1 depth limit and then used the entity's own predictions to retrieve its siblings.

| Step | Recall evidence |
|---|---|
| Forward + skeleton keys | R@30 0.938 → 0.945 (E002/E002c; 5k India S2 queries) |
| + reverse top-3 | 0.9464 → 0.9583 (E017b, India S2 slice) |
| + exact keys | +0.45 pt on India S2 (E026) |
| v3 union + stage 1 (60k holdout) | pair recall 0.9717, oracle F0.5 0.9901 (E030s2) |
| + predicted anchors | 0.9717 → 0.9785, oracle 0.9918 (E030s3) |
| + native-script dictionary | 0.9785 → 0.9821, oracle 0.9939; native-script misses 886 → 106 (E032) |
| **Final E039, 180k holdout** | before anchors 0.97802 (18.71 candidates/S1); **after anchors 0.98199, oracle F0.5 0.99409** (23.49 candidates/S1) |

---

## 4. Matching Model

All stages are LightGBM binary classifiers. Stages 2 and 3 use lr 0.05, 127 leaves, min_data_in_leaf 200, feature and bagging fraction 0.8 and λ₂ = 1. They were trained on the 720k non-holdout entities of a 900k-entity training sample (E039).

**Leakage control:**
- Every probability used as a feature is **cross-fitted by entity**, so no pair's feature comes from a model that saw its entity: 5 folds for p1 and p2, and 4 folds within the training entities for the p3 fed to the residual.
- The holdout entities' own p1/p2 come from the fold-0 models, which never saw the holdout. Stages 2 and 3 and the residual are trained and early-stopped on training entities only. (The fold-1…4 cross-fit models include holdout entities in their training folds, and so does the deployed stage-1 model, which scores only test data. This can affect holdout scores only through the features of training rows.)

| Level | Features | Output |
|---|---|---|
| Stage 1 | 20 channel and competition features | p1; prune at 1e-3 |
| Stage 2 | 73 pairwise and context features | p2 (0.97741 alone, 180k holdout) |
| Stage 3 | 114: stage-2 features + p2 and its rank + 4 anchor + 7 sibling + 11 number + 8 name/shared-address + 9 distance/empty-address/duplication features | p3 (0.98421) |
| Residual (E044) | 204: p3 + 61 entity-context + 8 edit-type + 21 empty-address/tie features + the other 113 stage-3 features | p = σ(logit p3 + f(x)), trained on India/US; 600 rounds, 63 leaves; applied as in {{FINAL_SUB_ID}} |

**Features used:**
- **Name features:**
  - RapidFuzz ratio, partial, token-sort and token-set scores, and Jaro-Winkler.
  - The same scores on the *core* name, which drops tokens appearing in more than 0.3 % of names (legal and filler words, learned from the data).
  - No-space partial similarity and Levenshtein (e.g. `bangalore south infracon` vs `bangaloresouthinfracon`).
  - Consonant-skeleton similarity and IDF-weighted token overlap.
  - Name-ambiguity counts (how many S1 share the name or core name).
  - **Name substitution**: extra or missing tokens weighted by IDF and by a label-free filler-word noise score (`nm_c_extra_minnoise`).
  - Edit type of each changed token (OCR, anagram, suffix append/drop, one substitution).
- **Address features:**
  - Token-set, sort and partial similarity, skeleton token-set, IDF overlap and an empty-address flag.
  - Digit Jaccard and first-number equality.
  - **Canonical digits** (leading zeros stripped, with 1-digit-truncation and 1-edit tolerance).
  - **Alphanumeric house/unit tokens** (`14B`, `A201`).
  - House-number distance, parity and first differing position.
  - How many S1 share the candidate's address.
- **Other features:**
  - Retrieval evidence: forward score and rank, reverse rank, key hits and frequency, anchor score, rank and hits.
  - **Competition**: the best other S1's score for the same record, the margin and a top-claimant flag.
  - Within-entity ranks and gaps of the similarities; p2 and its rank.
  - **Sibling** similarity to the entity's anchors (name, address and number equality; number of anchors).
  - **Number consensus**: the anchor-weighted support for the candidate's numbers among its siblings.
  - Empty-address context.
  - Level-2 features: the within-entity distribution of p3 (rank, gap to top, share, counts ≥ 0.1/0.5/0.9, expected count − rank), overall and within groups of candidates with identical name, address, digits or similarity profile.
  - Confident matches per source and the remaining source capacity (≤ 5 / ≤ 6).
  - Raw-name equality against competing S1s that have the same name (empty-address ties).
- **Top features by gain:**
  - Stage 2: `comp_margin`, `rev_rank`, `digz_jaccard`, `dig_n_c`, `digz_first_sim`.
  - Stage 3: `p2`, `is_anchor`, `p2_rank`, `nsup_extra_p`, `nm_c_extra_minnoise`, `sib_all`.

**Model type:** LightGBM 4.5.0 gradient-boosted trees (**MIT licence**, CPU), four boosters with 113 / 2,999 / 1,552 / 600 trees. They hold about 0.62 M leaves in total, many orders of magnitude below the 8 B-parameter limit. The final system uses no neural, pre-trained or LLM model and no external data. There are no geography tables either: region tokens are simply down-weighted by IDF. (An abandoned experimental dense retrieval channel built on the pre-trained `intfloat/multilingual-e5-small`, MIT licence, 118M parameters, was never used by any submitted model and is not in the package.)

**Threshold selection method:** There is **no global threshold.** For each S1:
1. Discard candidates with p < 0.01.
2. Apply **one-owner exclusivity**: each S2/S3 record is kept only for its highest-probability S1. This is valid because the ground truth is strictly many-to-one.
3. Choose the prefix *k* of the probability-sorted candidates that maximises the **expected F0.5**:

   E[F_k] = Σ_{a,b} P(A=a) P(B=b) · 1.25a / (0.25(a+b) + k), and E[F₀] = Π(1−pᵢ).

   Here A and B are the Poisson-binomial counts of true matches inside and outside the prefix. The optimum is always a prefix, so scanning *k* is exact.

We chose this over a threshold because the metric is macro-averaged over entities and each singleton is worth a full 1.0. A fixed threshold mishandles entities with several medium-probability candidates, and entities with a single weak one.

Evidence on the E039 holdout (180k):
- threshold 0.5 → 0.98296; best threshold (0.7) → 0.98402; expected F0.5 → 0.98421.
- On the public LB, expected F0.5 with exclusivity scored 0.953 against 0.952 for threshold 0.7 (day1_s2 vs day1_s1).
- The layer is already near Bayes-optimal: a GFM decision gave −0.00015, recalibration ±0.00002 and a miss-aware oracle +0.00012 (E042r).
- Adding a constant "missed matches" term hurts: 0.908 (E010) and 0.9306 (E039).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):**
  - Holdout (180,090 entities): **0.985458** for E044 on India + US (SE 0.00007), +0.00123 over E039 on the same entities (0.98423; E039's own report gives 0.98421 on 180,091). The 600-round checkpoint was chosen from 200/400/600/800 evaluated on this holdout (0.985327–0.985504). 600 was preferred over 800 because the mean logit shift kept growing with rounds.
  - By country: India 0.98422, US 0.98629.
  - By number of true matches: 0 → 0.98837, 1 → 0.9472, 2–3 → 0.98552, 4+ → 0.98934.
  - Public LB: best confirmed {{BEST_LB}} ({{BEST_LB_SUB}}); day3_l2b_india 0.976505, day3_l2b 0.975885, day3_e039 0.975775, day3_e032 0.97385. The final file is **`{{FINAL_SUB_ID}}`** (holdout {{FINAL_HOLDOUT}}), with LB **{{FINAL_LB}}**.

| Holdout progression | Change | F0.5 |
|---|---|---|
| E010 | LightGBM, 56 features, forward top-30 | 0.9578 |
| E011 / E013 | + canonical digits, name-ambiguity counts / + stage-1 pruning (4× fewer pairs) | 0.9614 / 0.96142 |
| E021 / E023B | + stage 3 sibling features / + predicted-anchor candidates | 0.96332 / 0.9669 |
| E030s2 / E030s3 | v3 union (forward ∪ reverse ∪ keys), 300k entities / + stage 3 | 0.9722 / 0.97522 |
| E033 / E034 / E035 | + number consensus / + name substitution, shared address / + number distance, empty-address context | 0.97711 / 0.97942 / 0.97989 |
| E032 | native-script dictionary | 0.98265 |
| E039 | 3× training entities (900k) | 0.98402 (same 60k) / 0.98421 (180k) |
| E042c / **E044** | residual level-2: holdout 5-fold CV / trained on 702k training entities | 0.98506 / **0.985458** |

The holdout had 30k entities for E010–E023B, 60k for E030–E032 and 180k from E039 on. Each gain was measured on identical entities.

| Submission | Model | Decision | Holdout | Public LB |
|---|---|---|---|---|
| day1_s1 / day1_s2 | E013 | threshold 0.7 / expF + exclusivity | 0.9614 / 0.9613 | 0.952 / 0.953 |
| day2_s1 | E030 stage 2 | expF + exclusivity | 0.9718 | 0.9634 |
| day2_s2 | E035 stage 3 | expF + exclusivity | 0.97989 | 0.9705 |
| day3_e032 | E032 | expF + exclusivity | 0.98265 | 0.97385 |
| day3_e039 | E039 | expF + exclusivity | 0.98421 | 0.975775 |
| day3_l2 | E039 + E042 residual (holdout-trained) | expF + exclusivity | 0.98508 | not scored |
| day3_l2b | E039 + E044 residual on India + US | expF + exclusivity | 0.985458 | 0.975885 |
| day3_l2b_india | E039 + E044 residual on India only | expF + exclusivity | 0.98482 (India +0.00147 per India entity) | **0.976505** |
| day3_usdown | E039 + E044: India full, US down-only min(p3, p), France raw | expF + exclusivity | 0.98518 (+0.00095) | {{LB_day3_usdown}} |
| day3_frdown | day3_usdown + E044 down-only on France | expF + exclusivity | 0.98518 (France unvalidatable) | {{LB_day3_frdown}} |

The file in this package is **`{{FINAL_SUB_ID}}`**, public LB {{FINAL_LB}}.

- **Common false positives (wrong merges):** E039 stage 3 made 1,639 false-positive pairs against 598,447 true positives (micro precision 0.99727). On the E032 stage-3 loss budget (60k holdout, E032lb) the false-positive bucket costs 0.0032 of F0.5. A sample of 30 E030 stage-2 holdout false positives, taken before the number and name features were added, showed:
  - About half are generator near-copies with a **neighbouring or replaced house/unit number** (`1002→1004`, `72→93C`, `224→226A`).
  - About a third are records of **another S1 with the same name** (often with an empty address) or in the same building.
  - Another recurring pattern is a **one-word name swap** (`Imperial Care→Imperial Cole`, `Logistics→Global`).

  Neighbouring numbers appear in 6.2 % of false positives vs 1.2 % of true positives, and an extra rare name token in 25.6 % vs 14.7 % (E040, E032 holdout).
- **Common false negatives (missed matches):** E039 missed 24,925 pairs: 11,227 were never retrieved and 13,698 were retrieved but rejected. The most recent full loss budget is for E032 stage 3 on the 60k holdout (E032lb, total loss 0.0173):
  - **Empty-address records** cost 0.0073 (42 %): 0.0024 from retrieval and 0.0049 from the matcher. Of the matcher part, 0.0036 (2,270 pairs) comes from names shared by several S1s. On the E039 holdout, 67 % of the empty-address bucket (worth 0.0046) is **provably tied**, i.e. two S1s have identical evidence (E042r).
  - **Other retrieval misses** cost 0.0039: aliases (`X formerly Y`), changed numbers and heavy corruption. These records are not lexically close to their S1 (E037).
  - **Other matcher misses** cost 0.0030: corrupted numbers and heavy name noise.
  - **Native-script misses** cost only 0.0005.

  The worst group is entities with exactly one true match (0.9441 for E039, 0.9472 after E044).

**What did not work** (holdout F0.5 or recall deltas):

| Idea | Result |
|---|---|
| Relative-DF TF-IDF blocking (E001) / DF cap 10,000 (E002b) | OOM / +0.001 recall at 5× cost |
| Char n-gram channel (E018) / name- and address-IDF, char 3-gram TF-IDF, BM25 and RRF over the union (E037) | +0.0033 recall for +20 candidates/S1 / +0.0025 recall for +114 candidates/S1 |
| Dropping density-dependent features (E016) / anchor features without new candidates (E023A) | no gain / 0.9631 vs 0.9633 |
| XGBoost + LightGBM ensemble (E024) | prediction correlation 0.9989, +0.00013 |
| Stage 4, a second collective round (E036T) / hard anti-match flags (E041) | −0.0005 / +0.00014 (noise) |
| Global assignment instead of greedy exclusivity (E040) | 1 contested holdout record; bound ≲ +0.001 on test and unvalidatable |
| Seed averaging / second anchor round / two-hop retrieval (E042r, E042d) | ≈ +0.0001 / ≤ +0.00008 / +0.00021 but fragile, not deployed |
| Self-training for the unseen country (US → India proxy, E043) | −0.0011 / −0.0016, despite 98.9–99.4 % pseudo-label accuracy |
| Full E044 residual on US test entities (day3_l2b) | holdout +0.00107 per US entity, LB −0.00062 overall (vs day3_l2b_india); upward moves admitted test distractors |

---

## 6. Conclusion

A recall-first, multi-channel candidate set feeds a cascade whose third stage reasons over each entity's own predicted matches. This took macro F0.5 from 0.9578 to 0.985458 on the holdout, while keeping only 24.85 candidates per S1 on test (holdout oracle ceiling 0.994). Every accepted gain came from sampling errors and pricing them in a counterfactual loss budget. Structural changes paid off (sibling anchors, generator-aware number and name features, the native-script dictionary). Model-family ensembles and more similarity channels did not. The remaining loss is dominated by empty-address records whose evidence is provably tied, and by France, the one country with no labels, where `{{FINAL_SUB_ID}}` uses {{FINAL_FRANCE}}. The final file `{{FINAL_SUB_ID}}` ({{FINAL_RULE}}) holds {{FINAL_HOLDOUT}} on the holdout and scores {{FINAL_LB}} on the public LB. The one lesson the leaderboard added: a correction validated on the labelled countries can still fail under test distribution shift (the US residual), so after that feedback the US correction was restricted (the India-only file, and down-only variants that can only remove matches).

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/README.md` gives the exact commands, the memory caps and the expected outputs. `requirements.txt` pins the environment (Python 3.12, LightGBM 4.5.0, RapidFuzz 3.10.1, Unidecode 1.3.8, numba 0.60.0, pandas 2.2.3, pyarrow 18.1.0). The final submission id is a parameter (`--sub-id`).

| Phase | Entry points (`python -m src.<module>` unless stated) | Key files |
|---|---|---|
| Data | `audit_gt`, `audit_sources`, `prep`; `translit build`, then `translit apply --split train` and `--split test` → `trainT`/`testT` | `normalize.py`, `prep.py`, `translit.py` |
| Blocking | `candidates --topk 30`; `candidates --topk 3 --reverse`; `key_channel --max-s1 10 --max-t 60` (per country and source) | `blocking.py`, `candidates.py`, `key_channel.py` |
| Training (E039) | `v3 s1data` → `stage1 --v3 --neg-frac 0.25` → `v3 s2data` → `train` (stage 2) → `collective oof --model lgb` / `build` → `anchor_pass retrieve` / `build` → `patch_extra --sets num,name,ctx` → `train` (stage 3); see `scripts/aws/run_e039.sh` | `v3.py`, `stage1.py`, `features.py`, `train.py`, `collective.py`, `anchor_pass.py`, `extra_features{,2,3}.py` |
| Residual (E044) | `python -m src.l2_train crossfit` → `features` → `holdout` → `train` (600 rounds) → `finalize` → `experiments/L2_E044/` (repo port of the EC2 research scripts crossfit.py / feat_train.py / train_e044.py / finalize_e044.py; folds, seeds, parameters and column order unchanged) | `l2_train.py`, `level2.py` |
| Test → output | `infer_v3 pass1` / `anchors` / `pass2` (`scripts/run_e039_test.sh`) → `scripts/apply_l2.py --model-dir experiments/L2_E044` → per-country rule: day3_l2b = `scores_l2b`; day3_l2b_india = `scores_l2b_in`; day3_usdown / day3_frdown = `scripts/combine_scores.py --rule India=full US=down France=raw\|down` → `infer_v3 write --scores {{FINAL_SCORES}} --sub-id {{FINAL_SUB_ID}}` (README step 7 table) | `infer_v3.py`, `decision.py`, `inference.py`, `submission.py`, `scripts/apply_l2.py`, `scripts/combine_scores.py` |

The write step checks both files with our own streamed verifier (headers, one row per S1, S2/S3 ids only, no duplicates, matches ⊆ candidates) and runs the organisers' validator on `matching_results.tsv`: PASS for every candidate final (day3_l2b, day3_l2b_india, day3_usdown, day3_frdown). The organisers' candidate-file check is skipped because it loads all 43 M ids into Python sets, which its own docstring warns about. The package builder repeats the subset check on the packaged files.

**Compute:**
- Training ran on an 8-vCPU / 61 GB CPU machine: stage 2 in 1,793 s and stage 3 in 1,539 s (E039 report.json).
- Test inference ran on a 16 GB laptop, with each step memory-capped: pass 1 3.6 h, pass 2 2.0 h, the residual 567 s (India) + 362 s (US), and the write step with the validator 186 s.

### B. Additional Results

| E039 stage 3, 180k holdout | India | US | \|T\|=0 | 1 | 2–3 | 4–6 | 7+ |
|---|---|---|---|---|---|---|---|
| macro F0.5 | 0.98273 | 0.98519 | 0.98439 | 0.94405 | 0.98438 | 0.98842 | 0.98967 |

| Test output | Matches | per S1 | Empty rows | India / US changed per 1,000 (holdout ref.) |
|---|---|---|---|---|
| day3_e039 | 5,806,218 | 3.351 | 100,249 (5.79 %) | 0 / 0 |
| day3_l2b | 5,822,762 | 3.361 | 100,438 (5.80 %) | 26.0 / 38.7 (17.4 / 14.1) |
| day3_l2b_india | 5,808,116 | 3.352 | 100,381 (5.79 %) | 26.0 / 0 |
| day3_usdown | 5,800,697 | 3.348 | 100,678 (5.81 %) | 26.0 / down-only |
| day3_frdown | 5,791,925 | 3.343 | 101,053 (5.83 %) | 26.0 / down-only; France 60.6 if fully corrected |

The residual changes 2.7× as many US entities on test as on the holdout (22,151 added and 7,505 removed pairs). The LB confirmed the concern: the US correction lost 0.00062 (day3_l2b vs day3_l2b_india), so the final file `{{FINAL_SUB_ID}}` uses US = {{FINAL_US}}. Both files share the same 43,058,923-pair candidate set: France 6,921,188, India 20,425,805, US 15,711,930. Three India S1 records have no candidates.
