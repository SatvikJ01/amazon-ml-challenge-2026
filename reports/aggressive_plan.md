# Aggressive plan — remaining window (2026-09-26 15:30 → 09-27 23:59 IST, ~32 h)

## 0. Where the loss is (E035, 60k holdout, F0.5 0.9799, loss 0.0201)

Micro precision **0.9965**, micro recall **0.9543** — the remaining loss is overwhelmingly *recall*.

| Bucket | Pairs | F0.5 if fixed | Share |
|---|---|---|---|
| Retrieval miss — other | 2,115 | +0.0042 | |
| Retrieval miss — native script (India) | 886 | +0.0027 | |
| Retrieval miss — empty address | 1,446 | +0.0024 | |
| **Retrieval total** | **4,447** | **≈ +0.0093** | **46 %** |
| Matcher miss — empty address, shared S1 name | 2,269 | +0.0037 | |
| Matcher miss — empty address, unique S1 name | 865 | +0.0013 | |
| Matcher miss — other | 1,731 | +0.0031 | |
| Matcher miss — native script | 159 | +0.0003 | |
| **Matcher total** | **5,024** | **≈ +0.0084** | **42 %** |
| False positives | 701 | +0.0036 | 18 % |

By true-set size: |T|=1 **0.926** (3,194 ent.), singletons 0.976, 2–3 0.980, 4–6 0.986, 7+ 0.988.
Candidate oracle 0.9918 (recall .9785). Record IDs carry no cluster information (corr 0.0001).
LB = holdout − 0.0084 on both v1 and v3 submissions (gains transfer 1:1).

Leader 0.9906 LB ≈ 0.999 holdout-equivalent: requires near-complete retrieval **and** near-perfect
matching. Honest expectation with everything below: holdout 0.984–0.988 → LB 0.976–0.980.

## 1. Compute

Measured on the laptop (i5-1240P 12C/16T, 15.7 GB, RTX 2050 4 GB): test pass 1 = 11 parts × 7–17 min
(~2.3 h), train stage-2/3 LightGBM 10–16 min, GPU XGB 5-fold OOF ~20 min but **peaks ~9 GB**, anchor
retrieval 10–25 min/country, test pass 2 ~1–1.5 h. A candidate-retrieval change = full rebuild
≈ 9–10 h sequential here. RAM, not CPU, forces serialisation.

| Option | RAM / CPU / GPU | Fits workload? | Setup | Full retrieval rebuild (train+test) | Parallel workers |
|---|---|---|---|---|---|
| Laptop | 15.7 GB (~9 usable) / 16 thr / 4 GB | stage-3/4 iterations only | 0 | 9–10 h | 1 heavy |
| **AWS r7i.8xlarge** (recommended) | 256 GB / 32 vCPU / — | everything CPU; GPU XGB → CPU hist (fine at 32 cores) | ~1.5 h (launch, upload ~0.8 GB compressed raw + code, venv, prep) | **~2.5–3.5 h** | 4–6 (country × source × part) |
| AWS r7iz.4xlarge | 128 GB / 16 vCPU | yes | ~1.5 h | ~4–5 h | 3–4 |
| AWS g5.4xlarge (A10G 24 GB) | 64 GB / 16 vCPU | only if we train a neural cross-encoder | ~1.5 h | ~5 h | 2–3 |
| Kaggle | 30 GB / 4 CPU / 2×T4, 12 h | secondary GPU worker (XGB OOF, encoder); CPU too weak for string features | ~1 h (private dataset upload) | n/a | 1 |
| Colab free | ~12 GB / 2 CPU / T4 (variable) | experiments only; unreliable | ~0.5 h | n/a | — |

Rough on-demand prices (check the console): r7i.8xlarge ≈ $2/h, r7iz.4xlarge ≈ $1.5/h, g5.4xlarge ≈ $1.6/h.
~20 h of r7i.8xlarge ≈ $40–45. Data stays in the team's private instance (rules forbid external data,
not cloud compute). **I cannot create the AWS account/instance or handle its credentials** — the user
launches it (Ubuntu 22.04/24.04, 200 GB gp3) and adds an SSH host alias; I do everything after that.

## 2. Parallel DAG

```
raw → prep (per country) ─┬─ forward/reverse/key/NEW channels (country × source: 6 train + 6 test blocks, independent)
                          │
train: s1data (per country) → stage 1 (joint) → s2data (per country) → stage 2 (joint) ─┬─ OOF (5 folds, independent)
                                                                                         └─ anchors retrieve/build (per country)
                                                                                            → stage 3 (joint) → OOF → stage 4
test:  pass1 (11 parts, independent) → rescore → anchors (per country) → pass2 (11 parts, independent) → write
```
Independent at once on 256 GB: all 12 blocking blocks (≈6–8 GB each, CPU-bound → run 6 at a time),
test pass-1 parts (≈6 GB each → 5–6 at a time, 4–6 threads each), OOF folds, and **train chain ‖ test
pass 1 of the previous model**. Laptop: one heavy job at a time (a second one caused today's global OOM).

## 3. Memory remediation (in order of payoff)

1. `collective.oof_stage2`: full float32 X + XGB quantisation peak ~9 GB → batch iterator (done) +
   int8/category `country`, free per-country columns; on cloud, split folds across processes.
2. `anchor_pass.build` / `infer_v3.pass2`: pandas read of the whole survivor table + merge + concat +
   groupby copies → Arrow append-column pattern (as `patch_extra`), smaller entity parts (100k).
3. `infer_v3.rescore`: whole-part materialisation → streamed batches (done, 1.8 GB).
4. `_texts` / `TokenStats` / `NameStats` / `PoolNames` rebuilt from Parquet in every process →
   cache once per (split, country) as pickled dicts / dictionary-encoded Arrow.
5. String columns → Arrow dictionary/large_string; ids int64; features float32 everywhere (mostly done).
6. Candidate union (`v3.union_source`) builds pandas frames of the full forward table for competition
   aggregates → compute aggregates in Arrow/numpy only (already mostly numpy).

## 4. Top levers (ranked by expected LB gain per hour)

| # | Lever | Targets | Exp. F0.5 gain | Recall gain | Wall clock (laptop / cloud) | Difficulty | FP risk |
|---|---|---|---|---|---|---|---|
| 1 | **Retrieval v4**: name-only channel (empty-address + noisy address), structured-address channel (number+street+locality; alias names like `ZEPHHALO`), native-script dictionary (E032), union/RRF → existing stage-1 filter | retrieval 0.0093 | **+0.003 – 0.005** | +0.5–0.8 pt | 9–10 h / **3–4 h** | medium (channels exist as code patterns) | low (stage-1/2 filter) |
| 2 | **All training entities** (2.1 M S1 instead of 300k) for stages 1–3 | matcher 0.0084 + FP | +0.001 – 0.003 | — | infeasible / +1–2 h on cloud | low | low |
| 3 | **Stage 4** (collective features from cross-fitted p3), E036 | matcher, FP | +0.001 – 0.002 | — | 1 h + 1.5 h test | low (built) | low |
| 4 | **Empty-address ownership model**: cross-S1 competition for name-only records (all S1 with that name, their p3, whether they already hold an empty-address record, count caps per source ≤5/6) | 0.0050 bucket | +0.001 – 0.003 | — | needs #2's all-entity scoring for unbiased features; 3–4 h cloud | medium-high | medium |
| 5 | **Neural cross-encoder feature** (multilingual MiniLM/e5-small, MIT) on uncertain pairs only (p ∈ [0.02, 0.98], ~3–5 M test pairs) | typos, aliases, native script | +0.001 – 0.004 (uncertain) | — | Kaggle T4×2 or g5: ~4–6 h incl. training | high | medium |

Considered and deprioritised: BM25 / char n-gram TF-IDF (E-series char channel: +0.1 pt recall only),
ANN embeddings for retrieval (dictionary beats it on native script; ANN over 2–3 M records/country
is feasible but ~1 day of work), Fellegi–Sunter (GBDT on the same comparisons dominates),
LGB+XGB ensembles (corr 0.999, +0.0001), country-specific models (France unseen; LOCO shows transfer
loss, so shared model + comparison features is safer), self-training for France (LB gap has stayed
exactly 0.0084 across two very different models → France is not diverging; keep as a later check).

## 5. Recommended order

| When | Laptop | Cloud (once launched) |
|---|---|---|
| now → 17:30 | day2_s2 (E035) test pass → **submit** (exp. LB ≈ 0.971) | user launches r7i.8xlarge; I set it up (~1.5 h) |
| 17:30 → 19:00 | retrieval-v4 **pilot** on the 60k holdout: recall of new channels on the 4,447 misses; then E036 stage 4 | prep + blocking v4 (all channels, 12 blocks in parallel) |
| 19:00 → 02:00 | E036 → day3 candidate; stage-3 error-driven features | train chain v4 on **all entities** (+ E033–E035 features, stage 4); test pass 1 in parallel |
| 02:00 → 09:00 | idle / checks | v4 holdout → test pass 2 → **day3_s1** |
| 09:00 → 18:00 | cross-encoder or empty-address model only if v4 lands early | v4 + empty-address ownership / cross-encoder → day3_s2–s3 |
| 18:00 → 23:59 | final package (zip, docs, reproducibility) | final best → day3_s4; keep 1 submission spare |

Validation for every major method: overall, US, India, singletons, |T|=1, noise buckets (native,
empty address, shared name), plus US→India / India→US LOCO for unseen-country robustness and the
label-free France diagnostics (anchor rate, predicted set sizes, empty rate).

Rollback preserved: E013 (day1), day2_s1 (v3 stage 2), day2_s2 (E035) artefacts are never modified.

## 6. Cloud runbook (prepared 2026-09-26 16:15)

1. User: launch Ubuntu 24.04 on **r7i.8xlarge** (32 vCPU / 256 GB; fallback r7iz.4xlarge 16 / 128), 200 GB gp3,
   SSH key; add `Host ml` (HostName, User ubuntu, IdentityFile) to `~/.ssh/config` on the laptop.
2. Laptop: `scripts/aws/push.sh ml` — git-tracked code + official validator + raw dataset as a zstd stream
   (~0.7 GB on the wire).
3. Instance: `bash scripts/aws/bootstrap.sh` (≈5 min; `EMBED=1` adds CPU torch + sentence-transformers + faiss).
4. Instance: `scripts/aws/run_v4.sh "--channels <pilot winners> --entities 1000000 --stage4"` — 114-job graph
   (`src/v4_dag.py`, runner `src/dag.py`): per-job memory caps and pinned cores, resumable (`.done` markers),
   one retry, dependants skipped on failure. Logs: `logs/dag/v4/_dag.log`, one log per job.
5. Laptop: `scripts/aws/pull.sh ml` — reports, holdout predictions, DAG logs, finished submissions.

Estimated critical path on r7i.8xlarge (1 M training entities, stage 4): setup 0.4 h → blocking 1–1.5 h
(≈60 blocks, 6–8 at a time) → s1data/stage 1 0.7 h → s2data/stage 2 1 h → OOF (5 parallel folds) 0.5 h →
anchors/extra 0.5 h → stage 3 0.7 h → stage 4 1.2 h; test pass 1 overlaps training; pass 2 + write 0.7 h.
≈ 6–7 h to a stage-3 submission file, ≈ 7.5 h with stage 4 (laptop: not feasible at this scale).
Dense e5 channel: GPU worker recommended (≈12 M records to encode; hours on CPU).
