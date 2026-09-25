# Data Diagnostics

Source of every number below: `reports/gt_audit.json`, `reports/source_audit.json`
(`src/audit_gt.py`, `src/audit_sources.py`), and blocking benchmarks in
`reports/bench_blocking_*.json`. Last updated 2026-09-25.

## 1. Sizes

| File | Rows | Countries |
|---|---:|---|
| train_source1 | 2,206,821 | US 1,323,633 · India 883,188 |
| train_source2 | 5,034,616 | US 3,016,817 · India 2,017,799 |
| train_source3 | 5,285,603 | US 3,170,056 · India 2,115,547 |
| train_ground_truth | 2,206,821 | one row per S1 record |
| test_source1 | **1,732,544** | India 809,986 · US 663,106 · **France 259,452** |
| test_source2 | 4,887,273 | India 2,312,565 · US 1,871,330 · France 703,378 |
| test_source3 | 5,082,316 | India 2,405,000 · US 1,945,701 · France 731,615 |

Submission therefore has **1,732,544 rows**.

## 2. Structural hypotheses — verdicts

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| H1 | Each S2/S3 record matches ≤ 1 S1 entity | **TRUE, exactly** | 7,638,365 GT pairs, 7,638,365 distinct matched ids, 0 shared |
| H2 | Matches never cross countries | **TRUE, exactly** | 0 of 7,638,365 pairs cross-country |
| H3 | GT lists every S1 incl. singletons | **TRUE** | 2,206,821 GT rows = 2,206,821 S1 rows, 0 dup |
| — | Train/test id overlap | none | 0 shared ids for S1, S2, S3 |
| — | Id-order leakage | none | mean abs id diff true 3.332e8 vs random 3.330e8 (ratio 1.0006) |

Consequences: country is a **lossless blocking partition**; the true solution is a strict
**many-to-one assignment** (usable as a feature and as a post-processing constraint);
singletons are enumerated so the local metric is exact.

## 3. Target structure

| Matches per S1 | Entities | Share |
|---:|---:|---:|
| 0 (singleton) | 123,247 | 5.6 % |
| 1 | 119,157 | 5.4 % |
| 2 | 375,212 | 17.0 % |
| 3 | 530,841 | 24.1 % |
| 4 | 484,115 | 21.9 % |
| 5 | 321,957 | 14.6 % |
| 6 | 164,868 | 7.5 % |
| 7–11 | 87,424 | 4.0 % |

Mean 3.46 matches/entity; per-source maxima are 5 (S2) and 6 (S3).
S2/S3 split of GT pairs: 3,693,619 / 3,944,746.
**Distractors** (S2/S3 records matched by nobody): 26.0 % of train S2/S3.

## 4. Field quality

| | S1 (train/test) | S2 / S3 |
|---|---|---|
| Empty address | 0 % | 2.6–3.4 % |
| Non-Latin names (50k sample) | 0 % | S2 ≈ 9–11 %, S3 ≈ 5–6 % (Devanagari, Telugu, Kannada, Tamil, Gujarati, Bengali, …) |
| Leading punctuation junk (`--`, `<<`) | ~0 % | ≈ 1 % |
| Duplicate (name, address) | 0 % | 0.4–0.7 % |
| Duplicate names | 30 % of S1 | — (generic business names repeat a lot) |

S1 is clean, Latin-only, fully populated: consistent with a curated reference source.

## 5. Noise model (from inspecting true pairs)

Names: legal suffix add/drop/swap (`Limited → Ltd → ∅`, `Group → LLC`), bracket wrapping
(`[LLC]`, `[Tnrust]`), typos, domain form (`bhgfoundation.com`, `Eyegroup.Com`), prefixed
honorifics (`Sri`), word moves (`LLC Moncada …`), **phonetic transliteration of Indic-script
names** (`shiv praaprttiis praiveett limittedd`), and occasional complete replacement by an
unrelated string (`Dovasynxylo`) where only the address links the records.

Addresses: component reordering (`OR, 415 SUTTLE STREET, PORTLAND`), case change,
abbreviation (`Drive→DR`, `Rue→R.`), literal `NULL` inserts, state/region ↔ code/department
swaps (`Utah↔UT`, `Andhra Pradesh↔AP`, `Hauts-de-France↔Nord`), state in native script,
house-number prefixes (`H.NO`, `#`, `Door No`), digit truncation/corruption (`20-52/2 → 20-52/`),
truncation to a short form, fully empty addresses.

**France** (test only) shows the same generator: `SARL/SAS/EURL/SCI` suffixes, `[EURL]`,
`Rue→R.`, region↔department swaps, accent damage (`Àmicale`).

## 6. Distribution shift train → test

- **New country**: France = 15.0 % of test S1 entities, 0 % of train.
- **Country mix**: train S1 is 60 % US / 40 % India; test is 47 % India / 38 % US / 15 % France.
  Local scores are reported per country and re-weighted to the test mix.
- **Pool density**: S2+S3 records per S1 entity is 4.68 in train vs 5.75 in test
  (US 4.67→5.76, India 4.68→5.82, France 5.53). Either test entities have more matches or test has more
  distractors per entity. If the latter, test precision will be harder than local CV suggests,
  which argues for a conservative decision rule. Tracked as an open risk.

## 7. Blocking (India S2, 5k queries vs full 2.02 M pool)

| Variant | R@5 | R@10 | R@30 | R@50 | Peak RSS |
|---|---:|---:|---:|---:|---:|
| token+bigram+nc, max_df 2000 | .897 | .919 | .938 | — | 3.8 GB |
| same, max_df 10000 | .901 | .922 | .939 | — | 4.1 GB |
| + consonant skeleton keys | .895 | .922 | **.945** | **.956** | 4.6 GB |

Miss analysis (`reports/blocking_misses_*.tsv`): before skeleton keys, 76 % of missed pairs had
zero name-token overlap, almost all phonetically transliterated names with truncated addresses.
First version (relative max_df = 2 %) was OOM-killed: see CHANGELOG.
