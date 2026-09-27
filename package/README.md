# Business entity resolution: how to reproduce

This folder is `code/business_entity_resolution/` of the submission zip. It has the same layout as
our repository root. Every Python module works out its paths as `Path(__file__).parents[1]`, and
every shell script first changes to the folder root. So all commands below are run **from this
folder**, and the Python modules are run as `python -m src.<module>`.

```
code/business_entity_resolution/
├── src/                       all pipeline code (every module used below)
├── scripts/                   drivers (run_*.sh, aws/run_e039.sh) and the Python programs apply_l2.py,
│                              combine_scores.py, make_package.py
├── experiments/               the shipped final models (table "Final artefacts")
│   ├── E039_stage1/  E039_stage2/  E039_stage3/  L2_E044/
├── data/processed/translit_dict.json     native-script dictionary (learned from train ground truth)
├── student_resource/utils/validate_submission.py   the organisers' validator (the write step runs it)
├── tests/                     unit tests (metric, decision rule)
├── docs/EXPERIMENT_LOG.md     every experiment with its result
├── requirements.txt           pinned environment
└── README.md                  this file
```

Two more files sit in this folder: `SHA256SUMS` (sha256 of every file in the zip, check with
`sha256sum -c code/business_entity_resolution/SHA256SUMS` from the zip root) and `PACKAGE_INFO.json`
(build record: submission id, git commit, per-file sources).

`scripts/` holds the shell drivers and three Python programs. Two of them, `apply_l2.py` (applies
the E044 residual) and `combine_scores.py` (per-country down-only rules), are part of the final
path. They sit next to `src/` rather than inside it because this folder mirrors the repository root,
and they import `src/`.

**Which submission this package holds.** `output/` at the zip root is a copy of
`submissions/{{FINAL_SUB_ID}}/`, submission id **`{{FINAL_SUB_ID}}`** (also recorded as
`submission_id` in `PACKAGE_INFO.json`): {{FINAL_RULE}}. Its holdout F0.5 is {{FINAL_HOLDOUT}} and
its public-LB score {{FINAL_LB}}. To reproduce it, run step 8 with the scores directory
`{{FINAL_SCORES}}` of its row in the step-7 table. The write step writes `./output/` in *this*
folder (not at the zip root) and archives a copy under `submissions/<new id>/`.

## Final system in one paragraph

1. **Normalisation.** Records are normalised: unidecode, punctuation, NULL tokens, consonant
   skeletons and canonical digits. India names in native scripts are then translated with a
   token dictionary learned from the training ground truth. This gives the data versions
   `trainT` / `testT`.
2. **Candidates.** Candidates are the union of three channels: forward rare-term IDF top-30,
   reverse top-3 and exact capped keys. A cheap LightGBM (stage 1) keeps pairs with p1 ≥ 1e-3.
3. **Stage 2.** A LightGBM on 73 string/address/context features gives p2.
4. **Anchor retrieval.** Predicted matches with p2 ≥ 0.9 are used as "anchors". Each anchor
   retrieves its top-3 new candidates.
5. **Stage 3.** A LightGBM on 114 features (stage-2 features, sibling/anchor, number-consensus,
   name-substitution and context features) gives prob3. This is model E039, trained on 900k S1
   entities.
6. **Residual E044.** A residual LightGBM with 204 features and `init_score = logit(prob3)`
   corrects prob3. It was trained on 702k training entities using cross-fitted out-of-fold prob3.
7. **Decision.** Each entity gets the expected-F0.5-optimal prefix of its candidates, with
   one-owner exclusivity.

All models are LightGBM (MIT licence), trained from scratch on the provided training data. No
pre-trained model and no neural network is used by the final system, and it uses no external data
and makes no API or network calls (see section 5).

## Final artefacts (shipped in `experiments/`)

| Directory | What it is | Features | Trees | Produced by |
|---|---|---|---|---|
| `E039_stage1/` | stage 1 LightGBM (candidate pruning, keep p1 ≥ 1e-3) | 20 | 113 | `src.stage1 --tag v5s1 --out E039_stage1 --v3 --neg-frac 0.25` |
| `E039_stage2/` | stage 2 pairwise LightGBM → p2 | 73 | 2,999 | `src.train --tag v5c --exp E039_stage2` |
| `E039_stage3/` | stage 3 collective LightGBM → prob3 (holdout 180,091 entities: 0.98421) | 114 | 1,552 | `src.train --tag v5c_ancz --exp E039_stage3` |
| `L2_E044/` | residual level-2 LightGBM, variant ALL_nc, 600 rounds (holdout 0.985458 = +0.00123 over E039) | 204 | 600 | `src.l2_train finalize --tag f0123 --iteration 600` |

Each directory has `model.txt` (LightGBM text model) and `features.json` (the column order).
Stage 1 to 3 also have `report.json`. `L2_E044/` also has `meta.json` (training description and
holdout evaluation) and `ref_X5000.npy` / `ref_raw5000.npy`. These last two are 5,000 reference
rows and their raw scores. `scripts/apply_l2.py` uses them to check that the model reproduces its
scores on the current machine before applying it, and it stops if the maximum difference is
above 1e-6.

The shipped `data/processed/translit_dict.json` is identical (md5 `5b5ce09f…`) to the one that
`src.translit build` regenerated on the training machine.

## 1. Hardware and environment

| Machine | Used for | Specification |
|---|---|---|
| Laptop | test candidates, test inference, residual application, write | Intel i5-1240P, 12 cores / 16 threads, 15.7 GB RAM (about 9 GB usable), no GPU used by the final path |
| AWS EC2 r7i.2xlarge | all training (E039 chain, E044 residual) | 8 vCPU, 61 GB RAM, no GPU |

- On the laptop every heavy step ran under a memory cap:
  `systemd-run --user --scope -q -p MemoryMax=<cap> -p MemorySwapMax=0 <command>`. The caps we
  used are given with each command. The cap is optional on a larger machine.
- On EC2 nothing was capped. The run used `scripts/aws/run_e039.sh`, which runs India and US
  side by side.
- Disk: about 17 GB for `data/processed`, 1.2 GB for `data/interim`, 13 GB for the test run
  directory `experiments/E039_test` and about 1 GB for `experiments/E044`. Plan for 50 GB or
  more of free space.

Set up Python 3.12 (we used 3.12.3 on the laptop and 3.12.14 on EC2). The venv must be named
`.venv` in this folder, because the shell scripts call `.venv/bin/python`.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Leave the environment variable `V4_CHANNELS` unset. It switches on experimental retrieval
channels that the final system does not use. Leave `PASS2_SAVE_FEATS` unset as well (default
`1`), because pass 2 must save the stage-3 inputs that the residual needs.

## 2. Dataset location

The code reads the raw TSVs from `data/raw/dataset/` (`src/audit_gt.py`, `src/audit_sources.py`,
`src/submission.py`):

```
data/raw/dataset/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
data/raw/dataset/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```

The package has no `data/raw/` folder (it ships only `data/processed/translit_dict.json`), so
create it first. A symlink is fine:

```bash
mkdir -p data/raw && ln -s <absolute path>/student_resource/dataset data/raw/dataset
```

The dataset folder must contain `train/{train_source1,train_source2,train_source3,train_ground_truth}.tsv`
and `test/{test_source1,test_source2,test_source3}.tsv`. The write step runs the validator at
`student_resource/utils/validate_submission.py` with `--test-dir data/raw/dataset/test`.

## Two ways to reproduce

- **Route A: inference with the shipped models.** Run steps 1, 2 (apply only), 3, 6, 7 and 8.
  This takes about 8 h on the laptop, most of it in step 6.
- **Route B: full retraining.** Run all steps, 1 to 8. Steps 4 and 5 need a machine with 32 GB or
  more of RAM; we used EC2.

  Retraining **overwrites** `experiments/E039_stage{1,2,3}`, because the training modules write to
  `experiments/<exp>/`. Copy the shipped models first:

  ```bash
  cp -a experiments experiments_shipped
  ```

  Route B also rewrites the shipped `data/processed/translit_dict.json` (step 2 `translit build`,
  and line 17 of `scripts/aws/run_e039.sh`). The regenerated dictionary is identical (md5
  `5b5ce09fd857fd19bea8ef626c5e6e8b`); to keep the shipped file anyway, also run
  `cp -a data/processed/translit_dict.json translit_dict_shipped.json`. Training also creates
  `experiments/E039_collective/` and `experiments/E039_anchor/`.

  E044 writes to `experiments/E044/`, and the final model ends up in
  `experiments/E044/final_E044/`.

## 3. Step-by-step commands

### Step 1: audit and normalisation (both routes)

```bash
python -m src.audit_gt          # GT pairs -> data/interim/gt_pair_{s1,m}.npy, reports/gt_audit.json
python -m src.audit_sources     # raw TSV -> data/interim/{train,test}_s{1,2,3}.parquet, reports/source_audit.json
python -m src.prep              # -> data/processed/{train,test}_s{1,2,3}_{country}.parquet (normalised, per country)
```

On EC2 these took 9 s / 1.3 GB, 149 s / 5.5 GB and 228 s / 1.3 GB (wall time / peak RSS). The
`src.prep` docstring recommends a 5 GB cap on the laptop.

### Step 2: native-script dictionary

```bash
python -m src.translit build                 # route B only (route A ships the dictionary); needs train GT
python -m src.translit apply --split train   # route B only: -> data/processed/trainT_* (India S2/S3 rewritten, rest hard links)
python -m src.translit apply --split test    # both routes:  -> data/processed/testT_*
```

Each command takes seconds and peaks below 2.5 GB.

### Step 3: test candidates (both routes, run on the laptop)

- US and France were blocked on the plain `test` data version (tag `testall`). India was blocked
  again on `testT` (tag `tstT`).
- The US/France `tstT` files are **hard links** of the `testall` files. The US/France data are
  identical in `test` and `testT`, because `translit apply` hard-links them.
- These commands are the ones in `scripts/run_blocking.sh`, `scripts/run_reverse.sh`,
  `scripts/run_keys.sh` and `scripts/run_e032b.sh`.
- Laptop caps: 5.5 GB (forward), 5 GB (reverse), 6 GB (keys), 6.5 GB (India on `testT`).

```bash
for c in US France; do for s in 2 3; do
  python -m src.candidates --split test --tag testall --topk 30 --countries $c --sources $s --skip-existing
  python -m src.candidates --split test --tag testall --topk 3  --countries $c --sources $s --reverse --skip-existing
  python -m src.key_channel --split test --tag testall --country $c --source $s --max-s1 10 --max-t 60
done; done
for s in 2 3; do
  python -m src.candidates --split testT --tag tstT --topk 30 --countries India --sources $s --skip-existing
  python -m src.candidates --split testT --tag tstT --topk 3  --countries India --sources $s --reverse --skip-existing
  python -m src.key_channel --split testT --tag tstT --country India --source $s --max-s1 10 --max-t 60
done
P=data/processed
for s in 2 3; do for sfx in "" "_rev" "_key"; do for c in US France; do
  ln -f $P/cands_testall_${c}_s${s}${sfx}.parquet $P/cands_tstT_${c}_s${s}${sfx}.parquet
done; done; done
```

Laptop times per (country, source): forward 50–320 s, reverse 64–462 s, keys 8–63 s. The whole
step takes about 1 h.

### Step 4: E039 training (route B, EC2)

**4a. Training entity samples.** These files are not shipped, because the package excludes data
tables.

- `entities_v3.npy`: 300k S1, the E032 training sample.
- `entities_v5.npy`: 900k S1, that sample plus 600k new entities.

They are rebuilt deterministically from the processed train S1 files. We checked that the
snippet below reproduces both original files byte for byte (md5 `2b45985c…` for v3 and
`4eecf5da…` for v5). The v5 part is the same code as the heredoc in `scripts/aws/run_e039.sh`.

```bash
python - <<'PY'
import numpy as np
from src.candidates import sample_entities
trn = sample_entities("train", 150_000, 0)                     # original 150k sample (entities_trn.npy)
pool = sample_entities("train", 400_000, 1)
v3 = np.unique(np.concatenate([trn, pool[~np.isin(pool, trn)][:150_000]]))
np.save("data/processed/entities_v3.npy", v3)
pool = sample_entities("train", 1_400_000, 7)
v5 = np.unique(np.concatenate([v3, pool[~np.isin(pool, v3)][:600_000]]))
np.save("data/processed/entities_v5.npy", v5); print(len(v3), len(v5))   # 300000 900000
PY
```

**4b. The rest of the training chain.** Once `data/processed/entities_v3.npy` exists, the original
driver `bash scripts/aws/run_e039.sh` runs the whole chain. It needs GNU time, because every step
runs under `/usr/bin/time` (`sudo apt-get install time`; the packaged `scripts/aws/bootstrap.sh`
installs it). It does the following:

1. Re-runs step 1 and step 2 (train) and blocks `trainT` for India and US.
2. Rebuilds `entities_v5.npy` with the same code as 4a.
3. Runs the training steps below, with India and US side by side.

The same chain, one step at a time:

```bash
P=data/processed
# training candidates on trainT (both countries), tag trnT
for c in India US; do for s in 2 3; do
  python -m src.candidates --split trainT --tag trnT --topk 30 --countries $c --sources $s --skip-existing
  python -m src.candidates --split trainT --tag trnT --topk 3  --countries $c --sources $s --reverse --skip-existing
  python -m src.key_channel --split trainT --tag trnT --country $c --source $s --max-s1 10 --max-t 60
done; done
# stage 1
for c in India US; do python -m src.v3 s1data --split trainT --tag trnT --prefix v5 --country $c --entities $P/entities_v5.npy; done
cp $P/entities_v5.npy $P/entities_v5s1.npy
python -m src.stage1 --tag v5s1 --out E039_stage1 --v3 --neg-frac 0.25
# stage 2
for c in India US; do python -m src.v3 s2data --split trainT --tag trnT --prefix v5 --country $c --stage1 E039_stage1 --entities $P/entities_v5.npy; done
python -m src.train --tag v5c --exp E039_stage2
# cross-fitted p2 -> sibling tables -> anchor retrieval -> anchor tables -> extra features
python -m src.collective oof --tag v5c --exp E039_stage2 --out E039_collective --model lgb
for c in India US; do python -m src.collective build --tag v5c --out E039_collective --country $c --split trainT; done
for c in India US; do python -m src.anchor_pass retrieve --split trainT --country $c --oof-dir E039_collective --hits-dir E039_anchor; done
for c in India US; do python -m src.anchor_pass build --country $c --old-tag v5c --out-tag v5c_anc --entities entities_v5c.npy --hits-dir E039_anchor --train-split trainT --train-tag trnT; done
for c in India US; do python -m src.patch_extra --country $c --tag v5c_anc --out-tag v5c_ancz --sets num,name,ctx --split trainT; done
# stage 3
python -m src.train --tag v5c_ancz --exp E039_stage3
```

How the holdout is defined:

- The holdout is `entity_fold(s1, 5) == 0` of the 900k entities, which gives 180,091 S1.
  `src.train` scores it with the official macro F0.5 against the full ground truth.
- `experiments/E039_stage3/holdout_pred.parquet` holds the holdout's stage-3 probabilities. Step 5
  needs this file.

### Step 5: E044 residual (route B, EC2, after step 4)

The sub-commands of `src/l2_train.py`, the repository port of the EC2 research scripts. The port
was checked against the original run on EC2:

- `evaluate` of the shipped model gives F0.5 = 0.985458, which equals `L2_E044/meta.json`.
- `finalize` applied to the research `model_f0123` directory writes files byte-identical to
  `experiments/L2_E044` (all 5 files). On a fresh retrain, `meta.json` differs only in the
  informational `vs_cv` / `se_vs_cv` fields: they are null without `--cv-ref`, whose input comes
  from an unported holdout-CV research script. `model.txt` depends on the thread count, as described
  below.
- The holdout level-2 features, fold-0 features and cross-fit design are bit-identical.
- `features` and `holdout` must use the same `--work` as `crossfit`. `--wait` only matters when
  `features` runs at the same time as `crossfit`.

```bash
python -m src.l2_train crossfit            # 4-fold cross-fit of stage 3 inside the 720k training entities -> OOF prob3
python -m src.l2_train features --wait     # level-2 features of the training rows (rows with OOF prob >= 1e-3)
python -m src.l2_train holdout             # E039 holdout table + its level-2 features (needs E039_stage3/holdout_pred.parquet)
L2_THREADS=8 python -m src.l2_train train --tag f0123 --num-leaves 63 --rounds 800 \
    --checkpoints 200,300,400,600,800 --lr 0.05 --folds 0,1,2,3
python -m src.l2_train finalize --tag f0123 --iteration 600     # -> experiments/E044/final_E044/
python -m src.l2_train evaluate --model-dir experiments/E044/final_E044 --checkpoints 600   # optional re-check
```

- **Training set.** 3,509,054 rows from 702,007 entities, with a positive rate of 0.6970.
- **Holdout F0.5.** Raw E039 scores 0.984231. After the residual:

  | Trees | 200 | 400 | 600 (shipped) | 800 |
  |---|---|---|---|---|
  | Holdout F0.5 | 0.985327 | 0.985385 | **0.985458** | 0.985504 |

  At 600 trees, India is 0.984219 and US 0.986287. The 600-tree checkpoint was chosen on this
  holdout from the four above. 600 was preferred over 800 because the mean logit shift kept
  growing with the number of trees.
- **Using a retrained model.** Pass `--model-dir experiments/E044/final_E044` in step 7, or copy
  that directory over `experiments/L2_E044`.
- **Byte-identical retraining.** LightGBM results depend slightly on the thread count, because
  `deterministic` is not set. Byte-identical `model.txt` files need the thread counts we used:
  - `crossfit` and `train`: 8 threads, the default here, with `L2_THREADS=8` for `train`.
  - Stage 1, 2 and 3: `num_threads=12` from `src.train.LGB_PARAMS` and `src/stage1.py`, which is
    oversubscribed on 8 vCPU.

  With other thread counts, scores should agree within noise.

### Step 6: test inference with E039 (both routes, run on the laptop)

This is the logic of `scripts/run_e039_test.sh` without its EC2 wait and rsync lines. Each part
holds at most 200k S1, which gives 11 parts: France 2, India 5 and US 4. Every part is written as
a checkpoint, and re-running a command skips parts that already exist.

```bash
T="--run E039_test --split testT --tag tstT"
for c in France India US; do python -m src.infer_v3 pass1   --country $c $T --stage1 E039_stage1 --stage2 E039_stage2; done   # cap 6G
for c in France India US; do python -m src.infer_v3 anchors --country $c $T; done                                            # cap 6G
for c in France India US; do python -m src.infer_v3 pass2   --country $c $T --stage3 E039_stage3 --scores scores; done         # cap 7G
```

Pass 1 applies the union of the candidate channels, then stage 1 (p1 ≥ 1e-3), the string and
context features and stage 2. It writes `pass1_feats/` and `pass1_anchors/`.

Pass 2 adds the anchor pairs and the sibling/anchor features, then applies stage 3. It writes:

- `experiments/E039_test/scores/{C}_p{i}.parquet`: s1, cand, src, prob3.
- `experiments/E039_test/scores_feats/{C}_p{i}_c{k}.parquet`: the 114 stage-3 inputs plus prob3,
  in 4 chunks per part. The residual reads these.

Before step 7, check that every part exists (the `guard` of `scripts/run_e039_test.sh`):

```bash
python - <<'PY'
import math, pathlib, pyarrow.parquet as pq
for sub in ("pass1_feats", "scores"):
    d = pathlib.Path("experiments/E039_test") / sub
    for c in ("France", "India", "US"):
        n = pq.ParquetFile(f"data/processed/testT_s1_{c}.parquet").metadata.num_rows
        miss = [i for i in range(math.ceil(n / 200_000)) if not (d / f"{c}_p{i}.parquet").exists()]
        print(sub, c, "missing", miss); assert not miss
PY
```

### Step 7: apply the E044 residual and build the scores directory of the chosen submission

`scripts/apply_l2.py` computes the level-2 features on the full `testT` split. It then sets
`prob = sigmoid(logit(prob3) + residual)` for rows with prob3 ≥ 1e-3 in the countries listed in
`--countries`. Every other country keeps prob3. On the laptop we ran one capped process per
country:

```bash
for c in India US France; do
  OMP_NUM_THREADS=4 python scripts/apply_l2.py --run E039_test --countries India US --process $c \
      --model-dir experiments/L2_E044 --out-dir scores_l2b --threads 4                    # cap 5G, peak 2.5 GB
done
python scripts/apply_l2.py --run E039_test --countries India US --process India US France \
    --model-dir experiments/L2_E044 --out-dir scores_l2b --no-gates | tail -2             # must print "ALL PARTS COMPLETE"
```

This gives `scores_l2b`: India and US corrected, France at prob3. (`scripts/run_day3_l2b.sh` is
the driver we actually ran for this step and the two writes after it. In this package its
completeness check writes to a `mktemp` file; the original wrote to a session-local path.)

The other leaderboard variants are built from the same models. Every one of them has the same
candidate set, because only the probabilities differ.

| Submission id | India | US | France | Scores dir | Holdout F0.5 | Public LB |
|---|---|---|---|---|---|---|
| `day3_e039` | prob3 | prob3 | prob3 | `scores` | 0.98421 | 0.975775 |
| `day3_l2b` | corrected | corrected | prob3 | `scores_l2b` | 0.985458 | 0.975885 |
| `day3_l2b_india` | corrected | prob3 | prob3 | `scores_l2b_in` | 0.98482 (India +0.00147 per India entity) | 0.976505 |
| `day3_usdown` | corrected | min(prob3, corrected) | prob3 | `scores_usdown` | 0.98518 (+0.00095 vs E039) | {{LB_day3_usdown}} |
| `day3_frdown` | corrected | min(prob3, corrected) | min(prob3, corrected) | `scores_frdown` | 0.98518 (France has no labels) | {{LB_day3_frdown}} |

Holdout values are on the same 180,090 holdout entities (raw E039 0.98423 there; 0.98421 in E039's
own report on 180,091). The `day3_l2b_india` and `day3_usdown` values add the per-country gains
(India +0.00147 per India entity, US down-only +0.00061 per US entity) to the raw score. **This
package holds `{{FINAL_SUB_ID}}`.**

The commands for the variant directories:

```bash
# day3_l2b_india: India corrected, US/France prob3 (India ~10 min; US/France are copies of prob3)
python scripts/apply_l2.py --run E039_test --countries India --model-dir experiments/L2_E044 --out-dir scores_l2b_in --threads 4
# day3_usdown: needs scores_l2b and scores_l2b_in
python scripts/combine_scores.py --run E039_test --raw scores_l2b_in --corrected scores_l2b \
    --rule India=full US=down France=raw --out scores_usdown
# day3_frdown: France-only correction, then France down-only on top of scores_usdown
python scripts/apply_l2.py --run E039_test --countries France --process France --model-dir experiments/L2_E044 --out-dir scores_l2fr --threads 4
python scripts/combine_scores.py --run E039_test --raw scores_usdown --corrected scores_l2fr --rule France=down --out scores_frdown
```

Notes on these commands:

- **How `scores_l2b_in` was made originally.** We built it by copying `scores_l2b/India_*` and
  taking the prob3 copies of US and France from an earlier India-only run (see
  `scripts/run_day3_l2b.sh`). We checked that the `apply_l2.py --countries India` output above is
  identical: the India parts equal `scores_l2b`, and the US/France parts equal prob3.
- **Completeness message for `scores_l2fr`.** The completeness check reports India and US as
  missing for `scores_l2fr`. This is expected: `combine_scores.py` reads only the France files
  from it.
- **Checks on the combined directories.** We recomputed `scores_usdown` and `scores_frdown` from
  their inputs with the rules above and they matched exactly.

### Step 8: write, check and archive the submission

Set `SCORES` and `SUB` to the chosen row of the table:

```bash
SCORES=scores_l2b_in; SUB=repro_day3_l2b_india       # example
python -m src.infer_v3 write --run E039_test --split testT --tag tstT --scores $SCORES --sub-id $SUB --note "reproduction"   # cap 6G, ~3 min
```

The write step does the following:

1. Reads every pair of `experiments/E039_test/$SCORES/`. Pairs with prob ≥ 0.01 go through
   one-owner exclusivity (`src.inference.enforce_exclusivity`) and the expected-F0.5 prefix
   selection (`src.decision.select_sets`).
2. Writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`. The candidate file lists
   every pair stage 3 scored, which is 43,058,923 pairs (24.85 per S1).
3. Re-reads both files and checks every format rule, then runs the official validator on the
   matching file.
4. Copies both files and a `meta.json` to `submissions/$SUB/`.

Two things to watch:

- `submissions/<id>` is never overwritten, so use a new id. Also, `output/` holds the files of
  the **last** write only.
- The validator run needs only the matching file and `test_source1.tsv`. Our own streamed checks
  cover both files (step 3 of the list above), including the rule that every matched id is also a
  candidate id.

To run the validator by hand, from this folder (the same call the write step makes):

```bash
python student_resource/utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/__skip_candidate_check__.tsv --test-dir data/raw/dataset/test
```

This takes 8 s and 1.4 GB, and should end with "PASS — no blocking issues found". The
`--candidate` argument names a file that does not exist on purpose: without `--candidate` the
validator loads `output/candidate_pairs.tsv` by default (43 M ids into Python sets, several GB;
under a 5 GB cap it was killed). Run `--candidate output/candidate_pairs.tsv` and `--check-ids`
only on a machine with enough RAM.

**Package build.** Run `python scripts/make_package.py --team <TEAM> --members "<names>" --sub <SUB>`
from the committed repository. It copies `submissions/<SUB>/` into the zip's `output/`, checks both
files streamed (same S1 order, matched ids ⊆ candidate ids, no duplicates, S2-/S3- ids only), adds
this folder, fills the team and submission fields of this README and of the methodology document,
and then checks the zip: file list, CRC, the paths the spec requires, and the sha256 of both output
files. It refuses to build from uncommitted code, with unfilled document fields, or with a
machine-local path in any packaged file.

## 4. Runtimes and peak memory

All values were measured. EC2 values come from `/usr/bin/time` in `logs/E039.log` and the E044
logs. Laptop values come from the logs and file timestamps of the E039 test run.

| Step | Machine | Wall time | Peak RSS |
|---|---|---|---|
| audit_gt / audit_sources / prep | EC2 | 9 s / 149 s / 228 s | 1.3 / 5.5 / 1.3 GB |
| translit build / apply | EC2 | 7 s / 11 s | 1.8 / 2.4 GB |
| train blocking, per (country, source): forward / reverse / keys | EC2 | 177–322 s / 236–370 s / 32–48 s | 3.0–3.9 / 2.1–2.3 / 2.6–4.7 GB |
| v3 s1data India / US | EC2 | 100 s / 163 s | 7.8 / 12.2 GB |
| stage1 | EC2 | 586 s | 13.6 GB |
| v3 s2data India / US | EC2 | 223 s / 294 s | 10.5 / 11.9 GB |
| train E039_stage2 | EC2 | 30 min | 8.8 GB |
| collective oof (5-fold LightGBM p2) | EC2 | **2 h 38 min** | 11.9 GB |
| collective build India / US | EC2 | 73 s / 98 s | 10.1 / 14.7 GB |
| anchor_pass retrieve India / US | EC2 | 350 s / 442 s | 3.9 / 4.9 GB |
| anchor_pass build India / US | EC2 | 167 s / 244 s | 14.4 / **21.8 GB** |
| patch_extra India / US | EC2 | 229 s / 312 s | 8.6 / 12.5 GB |
| train E039_stage3 | EC2 | 26 min | **15.7 GB** |
| **E039 chain total** (run_e039.sh, India/US in parallel) | EC2 | **4 h 40 min** | — |
| l2_train crossfit (4 folds × 16–21 min) | EC2 | 76 min | about 15.5 GB (7.7 GB feature matrix) |
| l2_train features (4 folds) | EC2 | about 1 min per fold and country (ran alongside crossfit) | 5.0 GB |
| l2_train holdout | EC2 | about 3 min | 7.7 GB |
| l2_train train (800 rounds, 8 threads) | EC2 | 233 s | 8.9 GB |
| l2_train finalize / evaluate | EC2 | 1 s / 30–105 s | 0.3 / 2.1 GB |
| test blocking, all countries | laptop | about 1 h | ≤ 6.5 GB cap |
| infer_v3 pass1 (11 parts) | laptop | about 3.8 h (2.3 h in an earlier run of the same code) | 6 GB cap |
| infer_v3 anchors | laptop | ≤ 40 min | 6 GB cap |
| infer_v3 pass2 (11 parts) | laptop | about 2 h | 7 GB cap |
| apply_l2 India / US / France (prob3 copy) | laptop | about 10 min / 6 min / 3 s | 2.5 GB |
| infer_v3 write + validator | laptop | 186 s | 6 GB cap |

**Memory-heavy steps.** These need 12 GB or more: anchor_pass build US (21.8 GB), stage 3
training (15.7 GB), l2_train crossfit (about 15.5 GB), collective build US (14.7 GB),
anchor_pass build India (14.4 GB), stage1 (13.6 GB), patch_extra US (12.5 GB) and v3 s1data US
(12.2 GB).

`run_e039.sh` runs India and US side by side, which peaks at about 25 GB (collective build).
Run one step at a time on a 32 GB machine. None of the training steps fit the laptop, but every
test-side step does.

## 5. Fair play and licences

- **Data.** Only the provided train/test files are used. The native-script dictionary is learned
  from the training ground truth (`src/translit.py build`). There is no geocoding, no external
  list and no network access: the final system downloads nothing.
- **Experimental code left out.** Our repository also holds an abandoned experimental retrieval
  study ("v4": `src/v4_dag.py`, `src/v4_entities.py`, `src/exp_retrieval_v4.py` and a dense
  channel `src/embed_channel.py` that would use the pre-trained `intfloat/multilingual-e5-small`,
  MIT licence, 118M parameters). No submitted model or file used it. The final stage-1 features
  (`experiments/E039_stage1/features.json`) contain no v4 channel feature, and `V4_CHANNELS` was
  never set. It is **not part of this package**, and neither are its drivers or the historical E030
  driver `run_test_v3c.sh`. The list is in `PACKAGE_INFO.json` (`excluded_repo_files`). The sparse
  optional channels in `src/channels.py` are still packaged, because `src/v3.py` imports that
  module lazily; they stay off while `V4_CHANNELS` is unset. `scripts/aws/bootstrap.sh` still has an
  `EMBED=1` option for that stack: leave it unset.
- **Models.** Every model is LightGBM 4.5.0 (MIT licence), trained from scratch. The largest,
  stage 2, is a 42 MB text model with about 3k trees, far below the 8B-parameter limit.
- **XGBoost.** It appears only in optional code paths of earlier experiments and is not used here.
