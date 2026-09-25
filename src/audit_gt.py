"""Stage-1 forensics: the ground-truth file.

Answers the structural questions from ``reports/task_spec.md`` section 7 that
determine the whole solution design, in a single streaming pass so it runs inside
our RAM budget:

* H3 -- does ground truth enumerate every S1 record (singletons included)?
* singleton rate and the full match-count distribution  -> the abstain prior and
  the achievable score ceiling
* H1 -- is each S2/S3 record claimed by **at most one** S1 entity?  If so we can
  apply a global mutual-exclusivity constraint, which is pure precision for free.
* S2 vs S3 composition of the match sets  -> whether to model the sources apart
* coverage -- what fraction of S2/S3 records participate in any match at all

Run:  python -m src.audit_gt
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .ids import OFFSET

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "raw" / "dataset" / "train"
REPORTS = ROOT / "reports"
INTERIM = ROOT / "data" / "interim"

CHUNK = 250_000


def main() -> dict:
    t0 = time.time()
    gt_path = TRAIN / "train_ground_truth.tsv"

    n_rows = 0
    match_counts: Counter[int] = Counter()
    n_s2_per_row: Counter[int] = Counter()
    n_s3_per_row: Counter[int] = Counter()

    # Flat arrays of every (s1_code, matched_code) pair, accumulated per chunk.
    s1_parts: list[np.ndarray] = []
    m_parts: list[np.ndarray] = []
    s1_all: list[np.ndarray] = []

    reader = pd.read_csv(
        gt_path,
        sep="\t",
        dtype={"source1_entity_id": "string", "matched_entity_ids": "string"},
        keep_default_na=False,
        na_values=[],
        chunksize=CHUNK,
    )

    for chunk in reader:
        n_rows += len(chunk)

        s1 = chunk["source1_entity_id"].str.slice(3).astype("int64").to_numpy() + OFFSET
        s1_all.append(s1.astype(np.int64))

        lists = chunk["matched_entity_ids"].fillna("")
        # Empty string -> zero matches. Everything else splits on ",".
        counts = np.where(
            lists.str.len().to_numpy() == 0,
            0,
            lists.str.count(",").to_numpy(na_value=0) + 1,
        ).astype(np.int32)
        match_counts.update(counts.tolist())

        nonempty = counts > 0
        if nonempty.any():
            sub_ids = lists[nonempty]
            sub_s1 = s1[nonempty]
            sub_counts = counts[nonempty]

            # Explode without pandas' object overhead: one big split, then repeat
            # the S1 code to line up.
            flat = sub_ids.str.cat(sep=",").split(",")
            flat_arr = pd.Series(flat, dtype="string")
            tag = flat_arr.str.slice(1, 2).astype("int64").to_numpy()
            num = flat_arr.str.slice(3).astype("int64").to_numpy()
            codes = num + tag * OFFSET

            s1_parts.append(np.repeat(sub_s1, sub_counts))
            m_parts.append(codes)

            # Per-row S2/S3 split.
            bounds = np.concatenate([[0], np.cumsum(sub_counts)])
            is_s2 = (tag == 2).astype(np.int32)
            cs = np.concatenate([[0], np.cumsum(is_s2)])
            n2 = cs[bounds[1:]] - cs[bounds[:-1]]
            n_s2_per_row.update(n2.tolist())
            n_s3_per_row.update((sub_counts - n2).tolist())

    s1_codes = np.concatenate(s1_all)
    pair_s1 = np.concatenate(s1_parts) if s1_parts else np.empty(0, np.int64)
    pair_m = np.concatenate(m_parts) if m_parts else np.empty(0, np.int64)

    del s1_all, s1_parts, m_parts

    # ---- H1: is each matched S2/S3 record claimed by at most one S1 entity? ----
    uniq_m, inv, cnt = np.unique(pair_m, return_inverse=True, return_counts=True)
    multi_claimed = int((cnt > 1).sum())

    # For the ones claimed more than once, how many distinct S1 entities claim them?
    # (A record could legitimately appear twice only if GT had a duplicate row.)
    order = np.argsort(inv, kind="stable")
    shared_examples = []
    if multi_claimed:
        dup_codes = uniq_m[cnt > 1][:5]
        for dc in dup_codes:
            owners = np.unique(pair_s1[pair_m == dc])
            shared_examples.append(
                {"matched": int(dc), "n_owners": int(owners.size)}
            )
    del order, inv

    # ---- Composition ----
    tag_m = (pair_m // OFFSET).astype(np.int8)
    n_pairs = int(pair_m.size)
    n_s2_pairs = int((tag_m == 2).sum())
    n_s3_pairs = n_pairs - n_s2_pairs

    dist = {int(k): int(v) for k, v in sorted(match_counts.items())}
    n_singleton = dist.get(0, 0)

    src1_rows = sum(1 for _ in open(TRAIN / "train_source1.tsv", encoding="utf-8")) - 1

    res = {
        "gt_rows": n_rows,
        "train_source1_rows": src1_rows,
        "H3_gt_covers_all_s1": n_rows == src1_rows,
        "gt_duplicate_s1_rows": int(n_rows - np.unique(s1_codes).size),
        "singleton_count": n_singleton,
        "singleton_rate": n_singleton / n_rows,
        "match_count_distribution": dist,
        "mean_matches_per_entity": n_pairs / n_rows,
        "mean_matches_given_nonsingleton": (
            n_pairs / (n_rows - n_singleton) if n_rows > n_singleton else 0.0
        ),
        "max_matches": max(dist) if dist else 0,
        "total_gt_pairs": n_pairs,
        "pairs_to_s2": n_s2_pairs,
        "pairs_to_s3": n_s3_pairs,
        "distinct_matched_records": int(uniq_m.size),
        "H1_records_claimed_by_multiple_s1": multi_claimed,
        "H1_holds_exclusive": multi_claimed == 0,
        "H1_shared_examples": shared_examples,
        "n_s2_per_row_distribution": {
            int(k): int(v) for k, v in sorted(n_s2_per_row.items())
        },
        "n_s3_per_row_distribution": {
            int(k): int(v) for k, v in sorted(n_s3_per_row.items())
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }

    INTERIM.mkdir(parents=True, exist_ok=True)
    np.save(INTERIM / "gt_pair_s1.npy", pair_s1)
    np.save(INTERIM / "gt_pair_m.npy", pair_m)
    np.save(INTERIM / "gt_s1_codes.npy", s1_codes)

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "gt_audit.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2)[:4000])
    return res


if __name__ == "__main__":
    main()
