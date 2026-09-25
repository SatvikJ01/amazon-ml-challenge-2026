"""Stage-2 forensics: the six source files.

Covers the dataset-audit checklist -- counts, dtypes, missingness, duplicates,
constant/near-constant fields, country balance, field-length distributions,
character-script mix -- plus the cross-cutting checks that actually change the
design:

* **H2** country consistency inside a true match group (can we block per country?)
* train/test entity-id overlap (is any record reused across splits?)
* numeric-id leakage: is a matched record's id predictable from the S1 id?
  Synthetic datasets often leak through id assignment order; if this fires it
  dominates everything else, so it is checked explicitly rather than assumed away.
* distractor rate: S2/S3 records that match nothing

Run:  python -m src.audit_sources
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .ids import OFFSET

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "dataset"
REPORTS = ROOT / "reports"
INTERIM = ROOT / "data" / "interim"

FILES = {
    ("train", 1): RAW / "train" / "train_source1.tsv",
    ("train", 2): RAW / "train" / "train_source2.tsv",
    ("train", 3): RAW / "train" / "train_source3.tsv",
    ("test", 1): RAW / "test" / "test_source1.tsv",
    ("test", 2): RAW / "test" / "test_source2.tsv",
    ("test", 3): RAW / "test" / "test_source3.tsv",
}

SAMPLE_N = 200_000  # rows sampled for the expensive string-level statistics


def _script_of(text: str) -> str:
    """Coarse writing-system label for a string -- 'latin', 'devanagari', ..."""
    for ch in text:
        if ch.isspace() or not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        for script in ("LATIN", "DEVANAGARI", "BENGALI", "TAMIL", "TELUGU",
                       "KANNADA", "MALAYALAM", "GUJARATI", "GURMUKHI", "ORIYA",
                       "ARABIC", "CYRILLIC"):
            if name.startswith(script):
                return script.lower()
        return "other"
    return "none"


def _load(path: Path, source: int, usecols=None) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        keep_default_na=False,
        na_values=[],
        usecols=usecols,
    )
    df["code"] = df["entity_id"].str.slice(3).astype("int64") + source * OFFSET
    return df


def audit_one(split: str, source: int, path: Path) -> dict:
    t0 = time.time()
    df = _load(path, source)

    n = len(df)
    name = df["business_name"]
    addr = df["business_address"]
    country = df["country"]

    name_empty = int((name.str.strip().str.len() == 0).sum())
    addr_empty = int((addr.str.strip().str.len() == 0).sum())

    codes = df["code"].to_numpy()
    n_unique_ids = int(np.unique(codes).size)

    # Duplicate *content* (same name + address) -- matters because S1 is claimed
    # to be deduplicated, and because intra-source dupes affect exclusivity.
    key = name.str.strip().str.lower() + "\x00" + addr.str.strip().str.lower()
    n_dup_content = int(n - key.nunique())
    n_dup_name = int(n - name.str.strip().str.lower().nunique())

    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=min(SAMPLE_N, n), replace=False)
    name_s = name.iloc[idx]
    addr_s = addr.iloc[idx]

    name_len = name_s.str.len().to_numpy()
    addr_len = addr_s.str.len().to_numpy()
    name_tok = name_s.str.split().str.len().to_numpy()
    addr_tok = addr_s.str.split().str.len().to_numpy()

    scripts = Counter(_script_of(s) for s in name_s.head(50_000))
    addr_scripts = Counter(_script_of(s) for s in addr_s.head(50_000))

    # Leading/trailing junk noise markers seen in the head sample.
    junk_lead = int(name_s.str.match(r"^\s*[^\w\s]{2,}").sum())
    has_digit = int(name_s.str.contains(r"\d", regex=True).sum())

    res = {
        "split": split,
        "source": source,
        "rows": n,
        "unique_entity_ids": n_unique_ids,
        "duplicate_entity_ids": n - n_unique_ids,
        "id_min": int(codes.min() % OFFSET),
        "id_max": int(codes.max() % OFFSET),
        "country_counts": {k: int(v) for k, v in country.value_counts().items()},
        "name_empty": name_empty,
        "name_empty_pct": round(100 * name_empty / n, 4),
        "address_empty": addr_empty,
        "address_empty_pct": round(100 * addr_empty / n, 4),
        "duplicate_name_address_pairs": n_dup_content,
        "duplicate_name_address_pct": round(100 * n_dup_content / n, 4),
        "duplicate_names": n_dup_name,
        "name_len": {
            "mean": round(float(name_len.mean()), 2),
            "p50": int(np.percentile(name_len, 50)),
            "p95": int(np.percentile(name_len, 95)),
            "max": int(name_len.max()),
        },
        "addr_len": {
            "mean": round(float(addr_len.mean()), 2),
            "p50": int(np.percentile(addr_len, 50)),
            "p95": int(np.percentile(addr_len, 95)),
            "max": int(addr_len.max()),
        },
        "name_tokens_mean": round(float(np.nanmean(name_tok.astype(float))), 2),
        "addr_tokens_mean": round(float(np.nanmean(addr_tok.astype(float))), 2),
        "name_scripts_50k": {k: int(v) for k, v in scripts.most_common()},
        "addr_scripts_50k": {k: int(v) for k, v in addr_scripts.most_common()},
        "name_leading_punct_junk_pct": round(100 * junk_lead / len(name_s), 3),
        "name_contains_digit_pct": round(100 * has_digit / len(name_s), 3),
        "elapsed_sec": round(time.time() - t0, 1),
    }

    # Persist the compact columnar form for reuse by the rest of the pipeline.
    INTERIM.mkdir(parents=True, exist_ok=True)
    out = INTERIM / f"{split}_s{source}.parquet"
    df[["code", "business_name", "business_address", "country"]].to_parquet(
        out, index=False, compression="zstd"
    )
    res["cached_parquet"] = str(out.relative_to(ROOT))
    print(f"  [{split} S{source}] {n:,} rows -> {out.name} ({res['elapsed_sec']}s)")
    return res


def cross_checks(per_file: dict) -> dict:
    """Checks that span files: id overlap, H2, leakage, distractor rate."""
    out: dict = {}

    # ---- train/test entity-id overlap ----
    overlaps = {}
    for source in (1, 2, 3):
        tr = pd.read_parquet(INTERIM / f"train_s{source}.parquet", columns=["code"])["code"].to_numpy()
        te = pd.read_parquet(INTERIM / f"test_s{source}.parquet", columns=["code"])["code"].to_numpy()
        inter = np.intersect1d(tr, te, assume_unique=False)
        overlaps[f"S{source}"] = {
            "train": int(tr.size),
            "test": int(te.size),
            "shared_ids": int(inter.size),
        }
    out["train_test_id_overlap"] = overlaps

    # ---- H2: country consistency within a true match group ----
    pair_s1 = np.load(INTERIM / "gt_pair_s1.npy")
    pair_m = np.load(INTERIM / "gt_pair_m.npy")

    s1 = pd.read_parquet(INTERIM / "train_s1.parquet", columns=["code", "country"])
    s2 = pd.read_parquet(INTERIM / "train_s2.parquet", columns=["code", "country"])
    s3 = pd.read_parquet(INTERIM / "train_s3.parquet", columns=["code", "country"])
    others = pd.concat([s2, s3], ignore_index=True)

    c1 = pd.Series(s1["country"].to_numpy(), index=s1["code"].to_numpy())
    cm = pd.Series(others["country"].to_numpy(), index=others["code"].to_numpy())

    left = c1.reindex(pair_s1).to_numpy()
    right = cm.reindex(pair_m).to_numpy()
    same = left == right
    out["H2_country_consistency"] = {
        "pairs_checked": int(same.size),
        "same_country": int(same.sum()),
        "cross_country": int((~same).sum()),
        "cross_country_rate": float((~same).mean()),
        "matched_id_not_found_in_sources": int(pd.isna(right).sum()),
    }

    # ---- distractor rate: S2/S3 records matched by nobody ----
    matched = np.unique(pair_m)
    other_codes = others["code"].to_numpy()
    is_matched = np.isin(other_codes, matched, assume_unique=False)
    tag = (other_codes // OFFSET).astype(np.int8)
    out["distractors"] = {
        "s2_total": int((tag == 2).sum()),
        "s2_unmatched": int(((tag == 2) & ~is_matched).sum()),
        "s3_total": int((tag == 3).sum()),
        "s3_unmatched": int(((tag == 3) & ~is_matched).sum()),
        "overall_unmatched_rate": float((~is_matched).mean()),
    }

    # ---- numeric-id leakage probe ----
    # If ids were assigned in generation order, |id(S1) - id(match)| would be far
    # smaller for true pairs than for random pairs. Compare the two directly.
    rng = np.random.default_rng(0)
    k = min(400_000, pair_s1.size)
    sel = rng.choice(pair_s1.size, size=k, replace=False)
    true_d = np.abs((pair_s1[sel] % OFFSET).astype(np.int64)
                    - (pair_m[sel] % OFFSET).astype(np.int64))
    rand_m = pair_m[rng.choice(pair_m.size, size=k, replace=False)]
    rand_d = np.abs((pair_s1[sel] % OFFSET).astype(np.int64)
                    - (rand_m % OFFSET).astype(np.int64))
    out["id_leakage_probe"] = {
        "true_pair_mean_abs_id_diff": float(true_d.mean()),
        "random_pair_mean_abs_id_diff": float(rand_d.mean()),
        "true_pair_median": float(np.median(true_d)),
        "random_pair_median": float(np.median(rand_d)),
        "ratio": float(true_d.mean() / rand_d.mean()) if rand_d.mean() else None,
        "verdict": (
            "LEAK SUSPECTED - investigate"
            if true_d.mean() < 0.5 * rand_d.mean()
            else "no id-order leakage detected"
        ),
    }

    # ---- country balance across splits ----
    out["country_by_split_source"] = {
        f"{sp}_S{src}": per_file[f"{sp}_S{src}"]["country_counts"]
        for sp in ("train", "test")
        for src in (1, 2, 3)
    }
    return out


def main() -> dict:
    t0 = time.time()
    per_file = {}
    for (split, source), path in FILES.items():
        per_file[f"{split}_S{source}"] = audit_one(split, source, path)

    res = {"per_file": per_file, "cross": cross_checks(per_file)}
    res["total_elapsed_sec"] = round(time.time() - t0, 1)

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "source_audit.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res["cross"], indent=2, default=str))
    return res


if __name__ == "__main__":
    main()
