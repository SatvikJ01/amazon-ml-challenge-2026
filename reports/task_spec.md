# Task Specification — Amazon ML Challenge 2026

> Derived **solely** from `6ab5628d5a817_amazon_ml_challenge_problem_statement.pdf` and
> `6ab56657b4f1a_guidelines_and_key_instructions_amazon_ml_challenge_2026.pdf`.
> Anything not stated there is marked **UNKNOWN**. No 2025 assumptions are carried over.

Last updated: 2026-09-25

---

## 1. Canonical spec block

| Field | Value |
|---|---|
| **TASK TYPE** | Entity Resolution / record linkage. Formally: **set-valued prediction per query record** (multi-label retrieval over a cross-source candidate universe). Not classification, not regression, not ranking-only. |
| **TARGET** | For each Source-1 record, the **set** of Source-2 / Source-3 `entity_id`s referring to the same real-world business. The set may be **empty** (singleton), size 1, or many. |
| **INPUT MODALITIES** | Text only (3 short fields): `business_name`, `business_address`, `country`. No images, no numerics, no timestamps. |
| **PRIMARY METRIC** | **F₀.₅ (β = 0.5)**, computed **per Source-1 entity** then **macro-averaged** over all Source-1 entities in the evaluation set. |
| **METRIC FORMULA** | `F_0.5 = (1.25 × P × R) / (0.25 × P + R)` where `P, R` are per-entity precision/recall. Algebraically equivalent closed form used in code: `F_0.5 = 1.25·TP / (0.25·|T| + |P|)`. Conventions: `|P|=0, |T|=0 → 1.0`; `|P|=0, |T|>0 → 0.0`; `|P|>0, |T|=0 → 0.0`. |
| **OPTIMIZATION DIRECTION** | Maximize (∈ [0, 1]). |
| **OUTPUT FORMAT** | Two TSVs in `output/`. **`matching_results.tsv`** (scored): columns `source1_entity_id`, `matched_entity_ids` — the latter a comma-separated ID list, no quoting, empty for singletons. **`candidate_pairs.tsv`** (not scored, audited): columns `source1_entity_id`, `candidate_entity_ids` — the *final* candidate set fed to the matcher. |
| **DATA SIZE** | **UNKNOWN** — dataset not yet present in the workspace (portal-gated). Pipeline is written size-agnostic with sparse/blocked processing; will be filled in `reports/data_diagnostics.md` on arrival. |
| **SPECIAL CONSTRAINTS** | See §3. Key ones: open-set `country` with **France present only in test**; every test S1 entity must appear exactly once; matches must be S2/S3 IDs existing in the test set; no duplicates within a list; matches ⊆ candidates. |
| **MODEL/LICENSE CONSTRAINTS** | Final model must be **MIT or Apache-2.0 licensed** and **≤ 8 B parameters**. |
| **EXTERNAL-DATA RULES** | **Strictly prohibited**: external databases/APIs/services for business identity lookup, government registration lookup, **geocoding APIs for address normalization**, any internet data augmentation. Enforcement = code review, penalty = disqualification. Pretrained *models* are explicitly contemplated (the license clause presumes them) and are permitted; external *entity/address data* is not. |
| **COMPUTE CONSTRAINTS** | Local only (see §4). No stated platform compute limit. |
| **DEADLINE/SUBMISSION RULES** | Window **25 Sep 2026 00:00 IST → 27 Sep 2026 23:59 IST**. **Max 5 leaderboard submissions per day**, 3 days ⇒ 15 total. Public LB = subset of test; **Private LB = remainder and decides final rank**. Full-test predictions submitted in both cases. Plus one final zip package (code + outputs + methodology doc). |

---

## 2. Data contract

All files are **tab-separated**. Must be read with an explicit `sep="\t"` (commas occur inside
addresses *and* inside the ID-list column, so comma-parsing silently collapses to one column).

### Source files — `{train,test}_source{1,2,3}.tsv`

| Column | Notes |
|---|---|
| `entity_id` | Unique per record. Prefix encodes source: `S1-`, `S2-`, `S3-`. |
| `business_name` | Abbreviations, legal suffixes, DBA/trade names, punctuation variants, word-order transpositions, typos, transliterations. |
| `business_address` | Partial addresses, format variations, missing components, landmark references, municipal numbering. |
| `country` | **Open set of string labels.** Train = {US, India}. Test = {US, India, **France**}. Must not be hard-coded, filtered, or one-hot-encoded to the training set. |

There is **no** `source` column — source is implied by the ID prefix and by the file.

### `train_ground_truth.tsv`

| Column | Notes |
|---|---|
| `source1_entity_id` | An S1 record. |
| `matched_entity_ids` | Comma-separated S2/S3 IDs; **empty when no matches**. |

**UNKNOWN pending data:** whether ground truth lists *every* S1 train record (including
singletons) or only matched ones. The loader handles both and reports which.

---

## 3. Hard constraints checklist (submission is rejected on violation)

1. Exact column names, tab-separated.
2. `matched_entity_ids` contains **only** S2/S3 IDs. Self-matches to S1 → rejected.
3. Every ID must **exist in the test set**.
4. **Every** test S1 entity present, **exactly one row** each (France included).
5. No duplicate IDs within a list; no duplicate `source1_entity_id` rows.
6. Empty (not `NaN`, not `""` quoted) for singletons.
7. Final matches should be a **subset** of `candidate_pairs.tsv` (validator warns otherwise).
8. Final model: MIT/Apache-2.0, ≤ 8 B params.

An official `utils/validate_submission.py` ships with the dataset. A stdlib-only equivalent is
implemented at `utils/validate_submission.py` here so we are not blocked; the official one takes
precedence once the data lands and both will be run.

---

## 4. Compute inventory (this machine)

| Resource | Value | Implication |
|---|---|---|
| CPU | Intel i5-1240P, 12 cores / 16 threads | Fine for TF-IDF, sparse similarity, GBDT. |
| RAM | 15 GB total, **~9 GB realistically usable** | Hard constraint. All pairwise work must be sparse + chunked. Never materialize a dense N×M similarity matrix. |
| GPU | RTX 2050, **4 GB VRAM** | Enough for small/base encoders at fp16 (MiniLM, e5-small/base, gte-base). **Not** enough for a 7–8 B model — the license ceiling is a ceiling, not a target. |
| Disk | 78 GB free | Ample for caches. |

Design consequence: the backbone is **sparse char-n-gram TF-IDF blocking + engineered string
similarity features + GBDT**, with a small multilingual encoder as an *optional, measured* add-on
— not a large LLM.

---

## 5. Metric analysis — why this shapes the whole solution

### 5.1 Closed form

`F_0.5 = 1.25·TP / (0.25·|T| + |P|)` (derivation in `src/metrics.py`, verified against the PDF's
worked example = 0.714).

### 5.2 The metric is **per-entity and macro-averaged**

Consequences that most teams will miss:

- **A single global decision threshold is provably suboptimal.** Because the score decomposes
  per S1 entity, the right object is a *per-entity subset selection* problem: given calibrated
  match probabilities `p_i` for that entity's candidates, choose the subset maximizing
  **expected** F₀.₅. `src/decision.py` solves this exactly with a Poisson-binomial DP over
  (in-set successes, out-of-set successes).
- **Entities with few candidates carry the same weight as entities with many.** Macro-averaging
  means a rare/hard entity is worth as much as an easy one. Error analysis must be stratified by
  true-set size.
- **Singletons are free points and free landmines.** Correct empty prediction = 1.0; *any* false
  positive on a singleton = 0.0. If singletons are a large fraction of the data, a conservative
  abstain policy dominates. The expected-F₀.₅ rule handles this automatically (k=0 wins when all
  `p_i` are low) rather than via a hand-tuned threshold.
- **Precision is weighted 2× recall**, but *not* infinitely. With `|T|=1`, predicting the correct
  single ID scores 1.0; adding one wrong ID drops it to `1.25/(0.25+2) = 0.556`. Predicting
  nothing scores 0.0. So the break-even probability for adding a first candidate when `|T|` is
  likely 1 is well below 0.5 — a naive `p > 0.5` threshold **under-predicts** and loses points.
  This is quantified in `src/decision.py`.

### 5.3 Blocking recall is a hard ceiling

Any true match never generated as a candidate is unrecoverable. But because F₀.₅ is
precision-heavy, blocking should aim for **high recall at moderate candidate count**, and let the
expected-F₀.₅ decision layer — not the blocker — do the precision work.

---

## 6. Distribution shift: France

Explicitly stated, and the single largest generalization risk.

- Train covers **US + India**; test adds **France** with **zero** training examples.
- Therefore: no country one-hot; no country-specific hard rules that fail open; no
  `{US, India}` allowlists anywhere in the pipeline.
- Features must be **country-agnostic by construction** (character-level, token-set, positional)
  and any country-conditioned normalization must degrade gracefully to identity for unseen labels.
- **Validation must measure this.** Primary CV is entity-grouped K-fold; the mandatory
  **stress test is leave-one-country-out** (train US → validate India, and train India →
  validate US). The LOCO gap is our only estimator of France performance, and final model
  selection weighs it explicitly against the in-distribution CV.

---

## 7. Structural hypotheses to verify against training data

Each is a testable claim, not an assumption. Verification lands in `reports/data_diagnostics.md`.

| # | Hypothesis | Why it matters | How to test |
|---|---|---|---|
| H1 | Source 1 is deduplicated ⇒ each S2/S3 record matches **at most one** S1 entity. | Enables a global **mutual-exclusivity** post-process: when two S1 entities claim the same S2 record, award it to the higher-scoring one. Directly buys precision. | Check whether `matched_entity_ids` sets are pairwise disjoint across all ground-truth rows. |
| H2 | Country is consistent within a true match group. | If yes, cross-country pairs can be pruned from blocking at near-zero recall cost — a large candidate reduction. | Join ground truth to source country labels; measure cross-country match rate. |
| H3 | Ground truth enumerates all S1 records incl. singletons. | Determines the true singleton rate, which sets the abstain prior and the achievable ceiling. | Compare GT row count to `train_source1.tsv` row count. |
| H4 | S2 and S3 have systematically different noise profiles. | Justifies **source-specific** models/thresholds rather than one pooled matcher. | Compare per-source match rates, field lengths, missingness, similarity distributions. |
| H5 | S2/S3 contain near-duplicate records of each other. | Affects the exclusivity constraint and candidate dedup. | Intra-source similarity scan. |
| H6 | Train/test come from the same generator (beyond the France addition). | Validates that CV transfers to LB. | Adversarial validation on record-level features. |

---

## 8. Solution architecture (staged, each stage independently measurable)

```
 raw TSVs
    │
    ├─ [1] normalization ......... country-agnostic name/address canonicalization
    │                              (legal suffixes, abbreviations, unicode folding,
    │                               token sorting, numeric extraction)
    │
    ├─ [2] blocking .............. union of recall-complementary channels:
    │                              char-ngram TF-IDF top-k (name), token inverted index,
    │                              address-number key, optional dense ANN
    │        └─► candidate_pairs.tsv        ← measured by RECALL CEILING + REDUCTION RATIO
    │
    ├─ [3] pairwise features ..... string similarity battery on name/address,
    │                              token-set stats, numeric/PIN agreement, IDF-weighted
    │                              overlap, plus per-entity CONTEXT features
    │                              (rank, margin-to-next, candidate count)
    │
    ├─ [4] pairwise model ........ GBDT ensemble (LightGBM / CatBoost / XGBoost)
    │                              → calibrated p(match)
    │
    ├─ [5] decision layer ........ per-entity expected-F₀.₅ subset selection (exact DP)
    │                              + global mutual-exclusivity resolution (if H1 holds)
    │        └─► matching_results.tsv
    │
    └─ [6] validation ............ grouped CV (by S1 entity) + leave-one-country-out
                                   stress test + adversarial validation
```

Stage 3's *context features* are important: whether a candidate is the best match for an entity
is relative, not absolute. A raw similarity of 0.8 means something different when the runner-up
is at 0.79 versus 0.3.

---

## 9. Submission budget plan (15 total, 5/day)

| Day | Planned use |
|---|---|
| Day 1 | 1 format-safety submission (cheap, guarantees a SCORED status), 1–2 baseline, 1–2 first modelled. |
| Day 2 | Feature/model iterations; probe the CV↔LB relationship; test decision-layer aggressiveness. |
| Day 3 | Ensemble + final; reserve ≥2 for safety; **stop experimenting early**, spend the tail on validation, packaging, reproducibility. |

Never spend two submissions on near-identical solutions. Every submission must answer a
pre-registered question, logged in `EXPERIMENT_LOG.md` **before** upload.

---

## 10. Open UNKNOWNs (blockers marked ⛔)

- ⛔ **Dataset not present.** `dataset/train/*`, `dataset/test/*` must be downloaded from the
  Unstop portal and placed in `data/raw/`. Everything downstream of §7 is blocked on this.
- Row counts, singleton rate, matches-per-entity distribution, country balance — UNKNOWN.
- Whether the public/private split is random or stratified (e.g. by country) — **UNKNOWN**, not
  stated. We assume nothing and optimize for robust full-test performance.
- Exact contents of the official `utils/validate_submission.py` — ships with the data.
- Whether `Documentation_template.md` imposes fixed sections — ships with the data.
