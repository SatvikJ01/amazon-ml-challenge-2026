# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary

A three-stage pipeline: (1) country-partitioned **rare-term IDF blocking** with
transliteration-robust consonant-skeleton keys, run for *every* Source-1 entity; (2) a
**two-stage LightGBM cascade** whose strongest signals are *competition features* that encode
the fact that each Source-2/3 record belongs to at most one Source-1 entity; (3) a per-entity
decision layer tuned for macro F0.5. Holdout macro F0.5 (30k entities, full ground truth,
blocking losses included): **0.9614**.

---

## 2. Methodology

### 2.1 Problem Analysis

Audits of the full training data (2.2 M S1, 10.3 M S2/S3 records) established three exact
structural facts, which drive the design:

| Fact | Evidence | Used for |
|---|---|---|
| Every S2/S3 record matches **at most one** S1 entity | 7,638,365 GT pairs, 7,638,365 distinct matched ids | competition features, exclusivity |
| Matches never cross countries | 0 of 7.6 M pairs | lossless blocking partition |
| GT lists every S1 incl. singletons | 2,206,821 rows = S1 rows | exact local metric |

Other findings: singleton rate 5.6 %, mean 3.46 matches per entity (max 11); 26 % of S2/S3
records are unmatched distractors; S1 is clean, S2/S3 carry the noise. Observed noise: legal
suffix changes, bracket wrapping (`[LLC]`), domain-style names (`bhgfoundation.com`),
**phonetic transliteration of Indic-script names** (`shiv praaprttiis praiveett limittedd`),
address component reordering, state/region ↔ code swaps, `NULL` inserts, digit corruption
(`2818→02818`, `324→32`), truncated or empty addresses. Test adds **France** (15 % of test
entities), unseen in training, with the same noise generator.

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage classifier + metric-aware decision layer.
**Core Innovation:** competition features derived from the many-to-one structure (a candidate's
best score from *any other* S1 entity), made possible by blocking every S1 entity rather than a
sample. Single-feature AUC 0.989, above every string similarity.

Country-agnostic by construction: no geography tables, no country feature, no external data.
All normalisation is orthographic or learned from document frequencies in the provided data.

---

## 3. Candidate Generation (Blocking)

- **Partition:** country (lossless per GT audit).
- **Representation:** each record → bag of namespaced terms: name unigrams/bigrams, address
  alphabetic unigrams/bigrams, address numbers, a 10-char concatenated-name prefix (catches
  `bhgfoundation.com`), and **consonant-skeleton** terms (merge digraphs and voiced/unvoiced
  pairs, drop vowels, collapse repeats: `praaprttiis`, `properties` → `prprts`).
- **Scoring:** IDF-weighted cosine over *rare* terms (document frequency ≤ 2,000), hashed into
  2^23 features, sparse matrix product in chunks sized to a fixed non-zero budget. Top-30 per S1
  entity per target source.
- **Stage-1 pruning** (the final candidate set): a LightGBM on retrieval and competition features
  keeps pairs with p ≥ 0.001 → ~15 candidates per entity.
- **Candidate pairs:** ~26 M in `candidate_pairs.tsv` (stage-1 survivors: exactly what the final
  model scores).
- **Recall safeguards:** skeleton keys were added after miss analysis showed 76 % of misses were
  transliterated names (R@30 .938 → .945 on India S2); oracle F0.5 ceiling of the final
  candidate set on holdout: 0.9806.

---

## 4. Matching Model

**Features (63):**
- *Name:* rapidfuzz ratio / partial / token-sort / token-set / Jaro-Winkler; same on a "core" name
  (frequent tokens removed, learned per country); no-space variants; skeleton variants; IDF-weighted
  token overlap; token counts.
- *Address:* token-set / token-sort / partial-token-set / ratio, skeleton token-set, IDF overlap,
  numeric agreement on raw and canonical (leading-zero-stripped) numbers with a similarity that
  tolerates a dropped end digit or a single edit.
- *Competition:* best retrieval score of the candidate from any other S1 entity, margin, "is best
  claimant", number of claimants.
- *Context:* per-entity ranks and gaps of retrieval score and key similarities, candidate count.
- *Other:* target source, empty-address flag, transliterated-name flag, S1 name-ambiguity counts.

**Model type:** LightGBM binary classifier (MIT licence), two-stage cascade.
**Threshold selection method:** per-entity macro F0.5 on an entity-grouped holdout, comparing a
global probability threshold with an exact expected-F0.5 prefix selector (Poisson-binomial DP).
The model is well calibrated, and the two tie within noise; threshold 0.7 is used.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro, holdout):** 0.9614 (US 0.966, India 0.955; singletons 0.969).
- **Loss budget (E010):** false positives 0.007, matcher misses 0.016, blocking misses 0.019.
- **Common false positives:** generic names (`… private limited`) whose addresses share a city
  and a truncated house number with the true owner's lookalikes.
- **Common false negatives:** candidates with empty addresses and generic names (inherently
  ambiguous), and digit-corrupted addresses (partly fixed by canonical-digit features, +0.0036).

---

## 6. Conclusion

Exact structural facts found in the data mattered more than model complexity. Blocking every
entity turned the one-owner constraint into the strongest feature, and a cheap cascade made
100 M-pair inference feasible on a 15 GB laptop. Remaining headroom is mostly recall: blocking
misses and empty-address candidates.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/src/`: `prep.py` (normalise) → `candidates.py` (blocking) →
`build_features.py` / `stage1.py` / `make_stage2.py` (features, cascade) → `train.py` →
`inference.py` (scores test, writes and validates both output files). Exact commands in `README.md`.

### B. Additional Results
See `EXPERIMENT_LOG.md` for every experiment with its question, setup and result.
