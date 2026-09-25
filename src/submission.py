"""Write, verify and archive a submission (streamed, bounded memory).

``write_submission`` is the only code path that produces the two output files.
It writes them in entity chunks straight from sorted pair arrays, so the ~100 M
candidate ids never exist as Python strings at once.

Rules enforced (problem statement, "Output Format" and "Constraints"):
* exact headers, tab-separated, no quoting, UTF-8, one row per test S1 entity in
  the raw file's order, no extra rows
* S2-/S3- ids only, no duplicates within a list, only ids present in the test
  S2/S3 files
* final matches are a subset of the same entity's candidates

Verification is two-layered:
1. ``verify_file`` re-reads each written file line by line and checks every rule
   above (streamed; used for both files).
2. The organisers' ``utils/validate_submission.py`` is run on
   ``matching_results.tsv`` -- the scored file.  It is *not* run on the
   candidate file, because it loads all candidate ids into Python sets
   (~8-10 GB for ~100 M ids), which its own docstring warns about.

Every submission is copied to ``submissions/<sub_id>/`` with a ``meta.json``;
existing folders are never overwritten.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .ids import OFFSET, decode_ids

ROOT = Path(__file__).resolve().parents[1]
RAW_TEST = ROOT / "data" / "raw" / "dataset" / "test"
OUTPUT = ROOT / "output"
SUBMISSIONS = ROOT / "submissions"
VALIDATOR = ROOT / "student_resource" / "utils" / "validate_submission.py"

MATCH_HEADER = ("source1_entity_id", "matched_entity_ids")
CAND_HEADER = ("source1_entity_id", "candidate_entity_ids")


def test_s1_ids() -> np.ndarray:
    """Test S1 ids as strings, in the raw file's order."""
    return pd.read_csv(RAW_TEST / "test_source1.tsv", sep="\t", usecols=["entity_id"],
                       dtype="string", keep_default_na=False)["entity_id"].to_numpy()


def test_target_codes() -> np.ndarray:
    """Sorted unique encoded ids of every test S2/S3 record."""
    parts = []
    for s in (2, 3):
        ids = pd.read_csv(RAW_TEST / f"test_source{s}.tsv", sep="\t", usecols=["entity_id"],
                          dtype="string", keep_default_na=False)["entity_id"]
        parts.append(ids.str.slice(3).astype("int64").to_numpy() + s * OFFSET)
    return np.unique(np.concatenate(parts))


def _sorted_pairs(s1: np.ndarray, cand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort pairs by (s1, cand) and drop exact duplicates."""
    order = np.lexsort((cand, s1))
    s1, cand = s1[order], cand[order]
    keep = np.r_[True, (s1[1:] != s1[:-1]) | (cand[1:] != cand[:-1])]
    return s1[keep], cand[keep]


def _write_lists(path: Path, header: tuple[str, str], s1_ids: np.ndarray, s1_codes: np.ndarray,
                 ps1: np.ndarray, pc: np.ndarray, chunk: int = 100_000) -> None:
    """Stream one results file in the raw S1 order.  ``ps1``/``pc`` sorted by ``ps1``."""
    start = np.searchsorted(ps1, s1_codes, side="left")
    stop = np.searchsorted(ps1, s1_codes, side="right")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{header[0]}\t{header[1]}\n")
        for c0 in range(0, s1_ids.size, chunk):
            c1 = min(c0 + chunk, s1_ids.size)
            a, b = start[c0:c1], stop[c0:c1]
            lens = b - a
            offs = np.concatenate([[0], np.cumsum(lens)])
            total = int(offs[-1])
            # Gather every id of this chunk's entities and decode them in one call.
            idx = np.repeat(a - offs[:-1], lens) + np.arange(total)
            strs = decode_ids(pc[idx]) if total else np.empty(0, dtype=object)
            f.writelines(
                f"{sid}\t{','.join(strs[offs[i]:offs[i + 1]])}\n"
                for i, sid in enumerate(s1_ids[c0:c1])
            )


def write_submission(match_s1: np.ndarray, match_cand: np.ndarray,
                     cand_s1: np.ndarray, cand_cand: np.ndarray,
                     *, out_dir: Path = OUTPUT) -> dict:
    """Write both files to ``out_dir`` after strict checks. Returns stats."""
    s1_ids = test_s1_ids()
    s1_codes = pd.Series(s1_ids).str.slice(3).astype("int64").to_numpy() + OFFSET
    assert np.unique(s1_codes).size == s1_codes.size, "duplicate S1 ids in test file"

    ms1, mc = _sorted_pairs(np.asarray(match_s1, np.int64), np.asarray(match_cand, np.int64))
    cs1, cc = _sorted_pairs(np.asarray(cand_s1, np.int64), np.asarray(cand_cand, np.int64))

    valid = test_target_codes()
    for name, s1a, arr in (("matches", ms1, mc), ("candidates", cs1, cc)):
        assert np.all(np.isin(s1a, s1_codes)), f"{name}: S1 ids not in the test S1 file"
        src = arr // OFFSET
        assert np.all((src == 2) | (src == 3)), f"{name}: non S2/S3 ids"
        pos = np.minimum(np.searchsorted(valid, arr), valid.size - 1)
        assert np.all(valid[pos] == arr), f"{name}: ids absent from the test S2/S3 files"

    # matches ⊆ candidates, pair-wise.  Encode a pair as (rank of s1) << 36 | cand:
    # cand codes are < 4e10 < 2**36 and there are < 2**21 S1 entities, so the key
    # is a unique, order-preserving int64.
    uniq = np.unique(cs1)
    ckey = (np.searchsorted(uniq, cs1).astype(np.int64) << 36) | cc
    rm = np.minimum(np.searchsorted(uniq, ms1), uniq.size - 1)
    mkey = (rm.astype(np.int64) << 36) | mc
    pos = np.minimum(np.searchsorted(ckey, mkey), ckey.size - 1)
    ok = (uniq[rm] == ms1) & (ckey[pos] == mkey)
    assert ok.all(), f"{int((~ok).sum())} matched pairs are not candidates"
    del uniq, ckey, mkey, pos, ok

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_lists(out_dir / "matching_results.tsv", MATCH_HEADER, s1_ids, s1_codes, ms1, mc)
    _write_lists(out_dir / "candidate_pairs.tsv", CAND_HEADER, s1_ids, s1_codes, cs1, cc)

    n_nonempty = int(np.unique(ms1).size)
    return {
        "rows": int(s1_codes.size),
        "nonempty_rows": n_nonempty,
        "empty_rows": int(s1_codes.size - n_nonempty),
        "total_matches": int(mc.size),
        "mean_matches_per_entity": float(mc.size / s1_codes.size),
        "total_candidates": int(cc.size),
        "mean_candidates_per_entity": float(cc.size / s1_codes.size),
    }


def verify_file(path: Path, header: tuple[str, str], required: set[str]) -> list[str]:
    """Streamed re-check of one written file against every format rule."""
    errors: list[str] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        head = f.readline().rstrip("\n").split("\t")
        if tuple(head) != header:
            return [f"{path.name}: bad header {head}"]
        for ln, line in enumerate(f, start=2):
            s1, tab, rest = line.rstrip("\n").partition("\t")
            if not tab:
                errors.append(f"{path.name}:{ln} no tab"); continue
            if s1 in seen:
                errors.append(f"{path.name}:{ln} duplicate row {s1}")
            seen.add(s1)
            if rest:
                ids = rest.split(",")
                if len(ids) != len(set(ids)):
                    errors.append(f"{path.name}:{ln} duplicate id in list")
                if any(not (x.startswith("S2-") or x.startswith("S3-")) for x in ids):
                    errors.append(f"{path.name}:{ln} non S2/S3 id")
            if len(errors) > 20:
                break
    if seen != required:
        errors.append(f"{path.name}: {len(required - seen)} missing, {len(seen - required)} extra S1 rows")
    return errors


def run_official_validator(out_dir: Path = OUTPUT) -> tuple[int, str]:
    """Official validator on the scored file (candidate file skipped, see module doc)."""
    cmd = [sys.executable, str(VALIDATOR),
           "--matching", str(out_dir / "matching_results.tsv"),
           "--candidate", str(out_dir / "__skip_candidate_check__.tsv"),
           "--test-dir", str(RAW_TEST)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def archive(sub_id: str, meta: dict, out_dir: Path = OUTPUT) -> Path:
    dest = SUBMISSIONS / sub_id
    if dest.exists():
        raise FileExistsError(f"{dest} exists -- submissions are never overwritten")
    dest.mkdir(parents=True)
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        shutil.copy2(out_dir / name, dest / name)
    meta = dict(meta, sub_id=sub_id, created=dt.datetime.now().isoformat(timespec="seconds"))
    (dest / "meta.json").write_text(json.dumps(meta, indent=2, default=float))
    return dest
