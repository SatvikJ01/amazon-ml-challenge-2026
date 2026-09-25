# Amazon ML Challenge 2026 — Business Entity Resolution

For every Source-1 business record, find all Source-2 / Source-3 records that refer to the same
real-world business. Scored by per-entity F0.5, macro-averaged (singletons count).
Full spec: [reports/task_spec.md](reports/task_spec.md). Data facts: [reports/data_diagnostics.md](reports/data_diagnostics.md).

## Pipeline

```
raw TSV ──► audit ──► prep (normalise, split by country)
                          │
                          ▼
             candidates (rare-term IDF blocking, per country × source, top-30)
                          │
                          ▼
             features (pairwise + context + competition)  ──►  LightGBM matcher
                          │                                          │
                          ▼                                          ▼
             candidate_pairs.tsv                     expected-F0.5 set selection
                                                                 │
                                                                 ▼
                                                     matching_results.tsv
```

| Stage | Module | Output |
|---|---|---|
| Forensics | `src/audit_gt.py`, `src/audit_sources.py` | `reports/*.json`, `data/interim/*.parquet` |
| Normalisation | `src/prep.py` (`src/normalize.py`) | `data/processed/{split}_s{1,2,3}_{country}.parquet` |
| Blocking | `src/candidates.py` (`src/blocking.py`) | `data/processed/cands_{tag}_{country}.parquet` |
| Features | `src/build_features.py` (`src/features.py`) | `data/processed/feats_{tag}_{country}.parquet` |
| Training + validation | `src/train.py` | `experiments/<EXP>/{model.txt,report.json,...}` |
| Decision layer | `src/decision.py` | — |
| Inference + submission | `src/inference.py`, `src/submission.py` | `output/*.tsv`, `submissions/<id>/` |
| Metric | `src/metrics.py` (tests in `tests/`) | — |

## Reproduce

Hardware used: 12-core laptop CPU, 15 GB RAM. **Every heavy step runs under a cgroup memory cap**
so an overrun kills only that step:

```bash
CAP="systemd-run --user --scope -q -p MemoryMax=5500M -p MemorySwapMax=0"
PY=.venv/bin/python
```

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
$PY -m src.audit_gt && $CAP $PY -m src.audit_sources
$CAP $PY -m src.prep
for c in India US; do $CAP $PY -m src.candidates --split train --tag trnall --topk 30 --countries $c; done
for c in France India US; do $CAP $PY -m src.candidates --split test --tag testall --topk 30 --countries $c; done
$PY -c "import numpy as np; from src.candidates import sample_entities, PROCESSED; np.save(PROCESSED/'entities_trn.npy', sample_entities('train', 150000, 0))"
for c in India US; do $CAP $PY -m src.build_features --split train --tag trnall --depth 30 --entities data/processed/entities_trn.npy --out-tag trn --countries $c; done
$CAP $PY -m src.train --tag trn --exp <EXP>
$CAP $PY -m src.inference --exp <EXP> --sub-id <SUB_ID> --depth 30
```

Tests: `$PY -m pytest -q tests/`

## Rules compliance

- No external data, APIs, geocoding, or lookup tables of places or companies. All normalisation is
  orthographic or learned from the provided data (IDF, document-frequency stop tokens).
- Country is never a model feature (France is absent from training); it is used only as a
  blocking partition, which the ground truth shows is lossless.
- Models: LightGBM (MIT). No pretrained neural model is used so far.
