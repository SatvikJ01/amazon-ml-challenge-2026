# Changelog

## 2026-09-25
- Workspace: removed `__MACOSX/` and `.DS_Store` from the dataset; moved it to
  `student_resource/`, made `dataset/` read-only, symlinked as `data/raw/dataset`.
  Original zip kept untouched.
- `src/metrics.py`: exact macro F0.5 with the problem statement's empty-set conventions;
  tests in `tests/test_metrics.py` (PDF example 0.714, 5k random cross-checks).
- `src/ids.py`: int64 codec for entity ids (source * 1e10 + number).
- `src/audit_gt.py`, `src/audit_sources.py`: forensics -> `reports/*.json`.
- `src/normalize.py`: unicode folding, punctuation stripping, consonant skeleton keys.
- `src/blocking.py`: **rewritten** after OOM (E001) -> hashing, absolute DF cap, nnz-budgeted
  chunks, skeleton terms.
- `src/prep.py`: streamed per-country normalisation (first version OOM'd under the 5 GB cap).
- `src/candidates.py`: streamed full blocking per country/source.
- `src/features.py`, `src/build_features.py`: pairwise, context and competition features.
- `src/decision.py`: exact expected-F0.5 prefix selection (numba), brute-force tested.
- `src/train.py`: LightGBM with entity-grouped folds, inner early stopping, full-truth scoring.
- `src/submission.py`: strict writer + official validator + versioned archive.
- `src/stage1.py`, `src/make_stage2.py`: cross-fitted cascade; `src/inference.py`: per-part checkpoints, per-country processes.
- `src/submission.py`: streamed writer + streamed format verifier; official validator on matching file.
- `src/candidates.py`: reverse channel (`--reverse`, target -> top-k S1).
- Experiments: `src/exp_cascade.py`, `src/exp_reverse_blocking.py`, `src/exp_char_channel.py`, `src/evaluate_full.py`.
- Reports: `reports/current_state_audit.md`, `reports/improvement_plan.md`.
