#!/usr/bin/env python3
"""Build the official final-submission package ``<team>_submission.zip``.

Usage (from anywhere; paths are resolved against the repo root)::

    python scripts/make_package.py --team <TEAM_NAME> --sub <submission id, e.g. day3_l2b> [--out-dir DIR]

The package follows ``student_resource/README.md`` ("Final Submission Package")::

    <team>_submission.zip
    ├── output/
    │   ├── matching_results.tsv        <- submissions/<sub>/matching_results.tsv  (the leaderboard file)
    │   └── candidate_pairs.tsv         <- submissions/<sub>/candidate_pairs.tsv   (set the final matcher scored)
    ├── code/business_entity_resolution/
    │   ├── src/                        <- repo src/*.py (every module, no __pycache__)
    │   ├── scripts/                    <- scripts/*.sh, scripts/aws/*.sh, scripts/*.py (apply_l2.py, this file, ...)
    │   ├── experiments/                <- final model artefacts (E039 stage 1/2/3 + L2_E044 residual)
    │   ├── data/processed/translit_dict.json   <- native-script dictionary learned from train GT
    │   ├── student_resource/utils/validate_submission.py   <- organisers' validator (src/submission.py runs it)
    │   ├── tests/                      <- repo tests/*.py
    │   ├── docs/EXPERIMENT_LOG.md      <- full experiment log
    │   ├── README.md                   <- package/README.md
    │   ├── requirements.txt            <- package/requirements.txt (else the repo requirements.txt)
    │   ├── SHA256SUMS                  <- sha256 of every other file in the zip (``sha256sum -c`` from zip root)
    │   └── PACKAGE_INFO.json           <- submission id + meta.json, git commit, build time, sizes
    └── Documentation_template.md       <- package/Documentation_template.md (filled-in methodology)

Layout choice: ``code/business_entity_resolution/`` mirrors the repo root.  Every module computes its
root as ``Path(__file__).parents[1]`` and every shell script does ``cd "$(dirname "$0")/.."``
(``scripts/aws/*`` ``/../..``), so keeping ``scripts/`` next to ``src/`` (and ``experiments/``,
``data/`` below the same folder) means all commands run unchanged from
``code/business_entity_resolution/``.  The raw dataset goes to
``code/business_entity_resolution/data/raw/dataset/{train,test}`` (see the package README).

Never packaged: data/raw, data/interim, any parquet / npy data table, logs/, .venv, __pycache__,
.git, the student_resource dataset, keys / secrets.  The builder refuses (exit 2) when a required
input is missing, when a path matches the forbidden list, or when a text file contains something
that looks like a credential.  ``--draft`` replaces a missing README / methodology document /
E044 training module with a clearly marked placeholder (for dry runs while those are being
written); the output files and the model artefacts are required even in draft mode.

After zipping, the archive is re-opened and checked: file list and sizes equal the staging tree,
CRC of every member (``testzip``), the spec's mandatory paths exist, and the two output files'
sha256 in the zip equal those of ``submissions/<sub>/``.  Memory use is small (everything is streamed).

Further guards (a non-draft build refuses, exit 2):
  * the two output files are checked streamed row by row: same S1 order in both files, every matched
    id also listed as a candidate, no duplicate ids, only S2-/S3- ids (the organisers' validator is run
    by the write step on matching_results.tsv only, see src/submission.py);
  * the documents are templates: ``{{TOKEN}}`` fields (team, submission id, its per-country rule,
    holdout and leaderboard score) are filled at staging time from ``--team`` / ``--members`` /
    ``--sub``, the VARIANTS table below and ``submissions/SUBMISSIONS.md``; any unfilled field or
    template placeholder (``[Team Name]``, ``[pending]``, ...) left in a packaged document is refused;
  * no packaged code / document may contain a machine-local path (a session scratchpad under /tmp or
    the build laptop's home directory, LOCAL_PATH_PATTERNS);
  * the packaged code must be committed (``git status`` clean for every packaged source), so that
    PACKAGE_INFO.json's commit identifies it; ``--allow-dirty`` overrides this for dry runs.
Experimental code that no submitted model uses (the abandoned v4 retrieval DAG and its dense
channel, and their drivers) is left out (EXCLUDE below); a few historical shell drivers get
plumbing-only rewrites at staging time (REWRITES below), recorded per file in PACKAGE_INFO.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = "code/business_entity_resolution"          # code folder inside the zip

# Modules reached from the final pipeline's entry points (audit_gt, audit_sources, prep, translit,
# candidates, key_channel, v3, stage1, train, collective, anchor_pass, patch_extra, infer_v3,
# eval_subset, level2, l2_train) through their imports.  All other src/*.py files are packaged too,
# but only these make the build fail when missing.
REQUIRED_SRC = [
    "__init__", "anchor_pass", "audit_gt", "audit_sources", "blocking", "build_features", "candidates",
    "channels", "collective", "decision", "eval_subset", "extra_features", "extra_features2",
    "extra_features3", "extra_features4", "features", "ids", "infer_v3", "inference", "key_channel",
    "level2", "make_stage2", "metrics", "normalize", "patch_extra", "prep", "stage1", "stage4",
    "submission", "train", "translit", "v3",
]
L2_TRAIN_SRC = "l2_train"                        # repo port of the EC2 E044 training scripts
SCRIPT_FILES = ["scripts/apply_l2.py", "scripts/make_package.py"]          # required
SCRIPT_GLOBS = ["scripts/*.sh", "scripts/aws/*.sh", "scripts/*.py"]      # every driver / helper (e.g. combine_scores.py)
MODEL_FILES = {                                  # experiments/<dir>/<file>, all required
    "E039_stage1": ["model.txt", "features.json", "report.json"],
    "E039_stage2": ["model.txt", "features.json", "report.json"],
    "E039_stage3": ["model.txt", "features.json", "report.json"],
    "L2_E044": ["model.txt", "features.json", "meta.json", "ref_X5000.npy", "ref_raw5000.npy"],
}
SMALL_DATA = ["data/processed/translit_dict.json"]
VALIDATOR = "student_resource/utils/validate_submission.py"
OUTPUT_FILES = ["matching_results.tsv", "candidate_pairs.tsv"]
OUTPUT_HEADERS = {"matching_results.tsv": b"source1_entity_id\tmatched_entity_ids\n",
                  "candidate_pairs.tsv": b"source1_entity_id\tcandidate_entity_ids\n"}

# Mandatory paths of the official spec (checked inside the finished zip).
SPEC_PATHS = ["output/matching_results.tsv", "output/candidate_pairs.tsv", f"{PKG}/README.md",
              f"{PKG}/requirements.txt", "Documentation_template.md"]
SPEC_DIRS = [f"{PKG}/src/"]

# Anything whose repo-relative source path or archive name matches one of these is refused.
FORBIDDEN = [
    (re.compile(r"(^|/)data/raw(/|$)"), "raw dataset"),
    (re.compile(r"(^|/)data/interim(/|$)"), "interim data"),
    (re.compile(r"(^|/)student_resource/dataset(/|$)"), "student_resource dataset"),
    (re.compile(r"(^|/)(dataset)/(train|test)(/|$)"), "dataset folder"),
    (re.compile(r"(^|/)logs(/|$)"), "logs"),
    (re.compile(r"(^|/)\.venv(/|$)|(^|/)venv(/|$)"), "virtualenv"),
    (re.compile(r"(^|/)__pycache__(/|$)|\.pyc$"), "bytecode"),
    (re.compile(r"(^|/)\.git(/|$)"), "git metadata"),
    (re.compile(r"\.(parquet|feather|arrow|pkl|joblib|npz|partial)$"), "data table"),
    (re.compile(r"\.(pem|key|p12|pfx|ppk)$|(^|/)id_(rsa|ed25519|ecdsa)|(^|/)\.env$|(^|/)\.ssh(/|$)"), "secret"),
    (re.compile(r"(^|/)\.aws(/|$)|credentials$|(^|/)\.netrc$"), "secret"),
]
# The only .npy files allowed are the tiny reference matrices of the residual model.
NPY_ALLOWED = {f"{PKG}/experiments/L2_E044/ref_X5000.npy", f"{PKG}/experiments/L2_E044/ref_raw5000.npy"}
SECRET_PATTERNS = re.compile(
    rb"-----BEGIN [A-Z ]*PRIVATE KEY-----|AKIA[0-9A-Z]{16}|aws_secret_access_key\s*[=:]|ghp_[A-Za-z0-9]{30,}"
    rb"|xox[bap]-[A-Za-z0-9-]{10,}|sk-[A-Za-z0-9]{32,}|hf_[A-Za-z0-9]{30,}")
MAX_CODE_FILE = 80 * 2**20                       # warn above this (largest model.txt is ~42 MB)
# Machine-local paths that must not appear in any packaged code / document (after REWRITES).
# (Written split so that this file does not match its own check.)
LOCAL_PATH_PATTERNS = re.compile(rb"/tmp/clau" rb"de-|/home/alpha" rb"tron")

# Repo files deliberately NOT packaged: experimental code that no submitted model or file uses.
# (The v4 retrieval DAG; src/embed_channel.py is its dense channel, which would download the
# pre-trained intfloat/multilingual-e5-small from the HF hub and needs torch / faiss, not pinned.)
EXCLUDE = {
    "src/embed_channel.py": "experimental v4 dense retrieval channel (never used by a submitted model)",
    "src/v4_dag.py": "experimental v4 retrieval DAG driver (never used by a submitted model)",
    "src/v4_entities.py": "experimental v4 entity sampler (never used by a submitted model)",
    "src/exp_retrieval_v4.py": "experimental v4 retrieval study (never used by a submitted model)",
    "scripts/aws/run_v4.sh": "driver of src/v4_dag.py (not packaged)",
    "scripts/run_queue_evening.sh": "driver of src/exp_retrieval_v4.py (not packaged)",
    "scripts/run_queue_evening2.sh": "driver of src/exp_retrieval_v4.py (not packaged)",
    "scripts/run_test_v3c.sh": "historical E030 (day2_s1) one-off driver; waited on a session-local pid file",
}
EXCLUDED_MODULES = {Path(p).stem for p in EXCLUDE if p.startswith("src/")}
EXCLUDED_REF = re.compile(
    rb"(-m\s+src\.|from\s+src\.|import\s+src\.|from\s+\.)(" + "|".join(sorted(EXCLUDED_MODULES)).encode() + rb")\b")

# Plumbing-only rewrites of shell drivers at staging time (no pipeline logic changes):
# (file, regex, replacement, line inserted before the first rewritten line or None, description).
# Each rule is applied only when its pattern is present, so once the same edit is made in the repo the
# rule becomes a no-op.
_SCRATCH = rb"/tmp/clau" rb"de-[^\s\"'|;]*?/(?:scratchpad/)?[A-Za-z0-9_]+\.txt"
REWRITES = [
    ("scripts/run_day3_l2b.sh", _SCRATCH, rb'"$CHK"', b'CHK=$(mktemp)   # completeness-check output of apply_l2 --no-gates',
     "session scratchpad check file -> mktemp"),
    ("scripts/run_day3_l2.sh", _SCRATCH, rb'"$CHK"', b'CHK=$(mktemp)   # completeness-check output of apply_l2 --no-gates',
     "session scratchpad check file -> mktemp"),
    ("scripts/aws/bootstrap.sh", rb"build-essential zstd tmux htop >/dev/null",
     rb"build-essential zstd tmux htop time >/dev/null", None,
     "install GNU time (scripts/aws/run_e039.sh calls /usr/bin/time)"),
    ("scripts/aws/bootstrap.sh",
     rb"# EMBED=1 also installs the dense-channel stack \(CPU torch \+ sentence-transformers \+ faiss\)\.\n",
     rb"# EMBED=1 also installs the dense-channel stack (CPU torch + sentence-transformers + faiss).\n"
     rb"# That stack is only for the experimental v4 dense channel, which no submitted model uses and which\n"
     rb"# is not part of the submission package: leave EMBED unset.\n", None,
     "comment: EMBED is only for the unpackaged experimental v4 channel"),
]

# The final-candidate submissions: per-country use of the E044 residual, holdout F0.5 on the
# 180,090-entity E039 holdout (raw E039 0.98423 on the same entities), scores dir under
# experiments/E039_test/.  Numbers from EXPERIMENT_LOG.md (E039, E044) and submissions/SUBMISSIONS.md.
VARIANTS = {
    "day3_e039": dict(india="stage-3 prob3", us="stage-3 prob3", france="stage-3 prob3",
                      rule="no residual correction (stage-3 prob3 in every country)",
                      holdout="0.98421", scores="scores"),
    "day3_l2b": dict(india="full E044 correction", us="full E044 correction", france="stage-3 prob3",
                     rule="E044 residual applied in full on India and US; France keeps the stage-3 prob3",
                     holdout="0.985458", scores="scores_l2b"),
    "day3_l2b_india": dict(india="full E044 correction", us="stage-3 prob3", france="stage-3 prob3",
                           rule="E044 residual applied in full on India only; US and France keep the stage-3 prob3",
                           holdout="0.98482", scores="scores_l2b_in"),
    "day3_usdown": dict(india="full E044 correction", us="down-only E044 correction min(prob3, p)", france="stage-3 prob3",
                        rule="E044 residual in full on India, down-only min(prob3, p) on US; France keeps the stage-3 prob3",
                        holdout="0.98518", scores="scores_usdown"),
    "day3_frdown": dict(india="full E044 correction", us="down-only E044 correction min(prob3, p)",
                        france="down-only E044 correction min(prob3, p)",
                        rule="E044 residual in full on India, down-only min(prob3, p) on US and on France",
                        holdout="0.98518 (France unlabelled, its part is not validatable)", scores="scores_frdown"),
}
# Left-over template placeholders that a non-draft build refuses in a packaged document.
DOC_PLACEHOLDERS = re.compile(r"\{\{[A-Za-z0-9_]+\}\}|\[BEST_LB|\[FINAL_[A-Z]+\]|\[Team Name\]|\[Team Members\]|\[TEAM|\[FINAL_SUB_ID\]|\[FINAL_LB\]"
                              r"|\[pending\]|DRAFT PLACEHOLDER")
TEMPLATED_DOCS = {"package/README.md", "package/Documentation_template.md"}

PLACEHOLDER = """# DRAFT PLACEHOLDER -- {what}

`{src}` did not exist when this package was built with `--draft`.
This file must be replaced before the final package is submitted:
re-run `python scripts/make_package.py --team <TEAM> --sub <SUB>` (without `--draft`).
"""


@dataclass
class Entry:
    """One file of the package: archive path, source file (or generated text) and its role."""
    arc: str
    src: Path | None = None
    text: str | bytes | None = None               # generated / rewritten content (then src is provenance)
    kind: str = "code"                           # output | code | model | doc | meta
    executable: bool = False
    note: str | None = None                      # staging-time rewrite / template fill applied to src


def rel(p: Path) -> str:
    """Repo-relative POSIX path (absolute path if outside the repo)."""
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.resolve().as_posix()


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    """Streaming sha256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def count_lines(path: Path, chunk: int = 8 << 20) -> int:
    """Number of newline characters (streamed)."""
    n = 0
    with open(path, "rb") as f:
        while b := f.read(chunk):
            n += b.count(b"\n")
    return n


def forbidden_reason(src_rel: str | None, arc: str) -> str | None:
    """Why this file must not be packaged, or None if it is allowed."""
    for name in filter(None, (src_rel, arc)):
        for pat, why in FORBIDDEN:
            if pat.search(name):
                return why
    if arc.endswith(".npy") and arc not in NPY_ALLOWED:
        return "data table (.npy)"
    if arc.startswith(f"{PKG}/data/") and arc not in {f"{PKG}/{d}" for d in SMALL_DATA}:
        return "data file outside the allow-list"
    return None


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return str(n)


def git_info(packaged: set[str] | None = None) -> dict:
    """HEAD commit and uncommitted changes (git status --porcelain) of the packaged code: every line
    under src / scripts / tests / package / requirements.txt, restricted to ``packaged`` (repo-relative
    source paths) plus anything under package/ when that set is given (best effort)."""
    def run(*a):
        r = subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
        return r.stdout.rstrip() if r.returncode == 0 else None
    head = run("rev-parse", "HEAD")
    dirty = run("status", "--porcelain", "--untracked-files=all", "--",
                "src", "scripts", "tests", "package", "requirements.txt")
    lines = [ln for ln in (dirty.splitlines() if dirty else []) if ln.strip()]
    if packaged is not None:
        lines = [ln for ln in lines if ln[3:].strip('"') in packaged or ln[3:].startswith("package/")]
    return {"commit": head, "uncommitted_code_changes": lines}


# ----------------------------------------------------------------------------------------------
# collection
# ----------------------------------------------------------------------------------------------
def collect(args) -> tuple[list[Entry], list[str], list[str]]:
    """All package entries; returns (entries, missing required inputs, warnings)."""
    entries: list[Entry] = []
    missing: list[str] = []
    warns: list[str] = []

    def need(path: Path, arc: str, kind: str, required: bool = True, executable: bool = False) -> None:
        if path.is_file():
            entries.append(Entry(arc, src=path, kind=kind, executable=executable))
        elif required:
            missing.append(rel(path))
        else:
            warns.append(f"optional input missing, not packaged: {rel(path)}")

    def need_doc(path: Path, arc: str, what: str) -> None:
        """README / methodology document: required, or a placeholder in --draft mode."""
        if path.is_file():
            entries.append(Entry(arc, src=path, kind="doc"))
        elif args.draft:
            warns.append(f"DRAFT: {rel(path)} missing -> placeholder at {arc}")
            entries.append(Entry(arc, text=PLACEHOLDER.format(what=what, src=rel(path)), kind="doc"))
        else:
            missing.append(rel(path))

    # --- output/: the two files of the chosen submission
    sub = ROOT / "submissions" / args.sub
    for name in OUTPUT_FILES:
        need(sub / name, f"output/{name}", "output")
    meta = sub / "meta.json"
    if not meta.is_file():
        missing.append(rel(meta))

    # --- src/: every module (required ones must exist)
    src = ROOT / "src"
    have = {p.stem for p in src.glob("*.py")}
    for m in REQUIRED_SRC:
        if m not in have:
            missing.append(f"src/{m}.py")
    if L2_TRAIN_SRC not in have:
        (warns if args.draft else missing).append(
            f"src/{L2_TRAIN_SRC}.py (E044 residual training code)" + (" -- DRAFT: not packaged" if args.draft else ""))
    for p in sorted(src.rglob("*.py")):
        if "__pycache__" in p.parts or rel(p) in EXCLUDE:
            continue
        entries.append(Entry(f"{PKG}/{rel(p)}", src=p, kind="code"))

    # --- scripts/: orchestration actually used (all shell drivers + apply_l2.py + this builder)
    for s in SCRIPT_FILES:
        need(ROOT / s, f"{PKG}/{s}", "code", executable=True)
    for g in SCRIPT_GLOBS:
        found = sorted(ROOT.glob(g))
        if not found:
            missing.append(g)
        for p in found:
            if rel(p) not in EXCLUDE:
                entries.append(Entry(f"{PKG}/{rel(p)}", src=p, kind="code", executable=True))
    for s in ("scripts/run_e039_test.sh", "scripts/run_day3_l2b.sh", "scripts/run_e032b.sh",
              "scripts/aws/run_e039.sh"):
        if not (ROOT / s).is_file():
            missing.append(s)

    # --- tests/ (optional)
    for p in sorted((ROOT / "tests").glob("*.py")):
        entries.append(Entry(f"{PKG}/{rel(p)}", src=p, kind="code"))

    # --- final model artefacts + small learned data
    for d, files in MODEL_FILES.items():
        for f in files:
            need(ROOT / "experiments" / d / f, f"{PKG}/experiments/{d}/{f}", "model")
    for d in SMALL_DATA:
        need(ROOT / d, f"{PKG}/{d}", "model")

    # --- organisers' validator (src/submission.py calls it from <root>/student_resource/utils/)
    if not args.no_validator:
        need(ROOT / VALIDATOR, f"{PKG}/{VALIDATOR}", "code", required=False)

    # --- documents
    pkg = ROOT / "package"
    need_doc(pkg / "README.md", f"{PKG}/README.md", "code/business_entity_resolution/README.md")
    req = pkg / "requirements.txt" if (pkg / "requirements.txt").is_file() else ROOT / "requirements.txt"
    need(req, f"{PKG}/requirements.txt", "doc")
    need_doc(pkg / "Documentation_template.md", "Documentation_template.md", "methodology document")
    need(ROOT / "EXPERIMENT_LOG.md", f"{PKG}/docs/EXPERIMENT_LOG.md", "doc", required=False)

    # de-duplicate archive names (a file matched twice keeps its first entry)
    seen, uniq = set(), []
    for e in entries:
        if e.arc not in seen:
            seen.add(e.arc)
            uniq.append(e)
    return uniq, missing, warns


def leaderboard_score(sub: str) -> str | None:
    """Public-LB score of ``sub`` from submissions/SUBMISSIONS.md (column "LB score"), or None when the
    row is missing or the score is still pending."""
    reg = ROOT / "submissions" / "SUBMISSIONS.md"
    if not reg.is_file():
        return None
    for line in reg.read_text().splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 7 and cells[1] == sub:
            lb = cells[6]
            if lb.lower().startswith(("pending", "not submitted", "—", "-")):
                return None
            m = re.search(r"0\.\d{3,6}", lb)
            return m.group(0) if m else None
    return None


def best_leaderboard() -> tuple[str, str]:
    """(best public-LB score, its submission id) over every scored row of submissions/SUBMISSIONS.md."""
    reg = ROOT / "submissions" / "SUBMISSIONS.md"
    best = ("[BEST_LB]", "[BEST_LB_SUB]")
    for line in (reg.read_text().splitlines() if reg.is_file() else []):
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 7 and re.fullmatch(r"[A-Za-z0-9._-]+", cells[1]) and cells[1] != "ID":
            lb = leaderboard_score(cells[1])
            if lb and (best[0].startswith("[") or float(lb) > float(best[0])):
                best = (lb, cells[1])
    return best


def template_fields(args) -> dict[str, str]:
    """Values of the ``{{TOKEN}}`` fields of the packaged README / methodology document."""
    v = VARIANTS.get(args.sub, {})
    lb = leaderboard_score(args.sub)
    return {
        "TEAM_NAME": args.team_display or args.team,
        "TEAM_MEMBERS": args.members or "[Team Members]",
        "FINAL_SUB_ID": args.sub,
        "FINAL_RULE": v.get("rule", "[FINAL_RULE]"),
        "FINAL_INDIA": v.get("india", "[FINAL_INDIA]"),
        "FINAL_US": v.get("us", "[FINAL_US]"),
        "FINAL_FRANCE": v.get("france", "[FINAL_FRANCE]"),
        "FINAL_HOLDOUT": v.get("holdout", "[FINAL_HOLDOUT]"),
        "FINAL_SCORES": v.get("scores", "[FINAL_SCORES]"),
        "FINAL_LB": lb if lb else "not yet scored when this package was built (see submissions/SUBMISSIONS.md)",
        **{f"LB_{k}": leaderboard_score(k) or "not scored when packaged" for k in VARIANTS},
        **dict(zip(("BEST_LB", "BEST_LB_SUB"), best_leaderboard())),
    }


def transform(entries: list[Entry], args, warns: list[str]) -> None:
    """Staging-time content changes (in place): REWRITES of shell drivers and the ``{{TOKEN}}`` fields
    of the templated documents.  The source files in the repo are never modified."""
    by_src = {rel(e.src): e for e in entries if e.src is not None}
    for f, pat, repl, insert, why in REWRITES:
        e = by_src.get(f)
        if e is None:
            continue
        data = e.text if isinstance(e.text, bytes) else e.src.read_bytes()
        m = re.search(pat, data)
        if not m:
            continue
        if insert is not None:
            ls = data.rfind(b"\n", 0, m.start()) + 1
            data = data[:ls] + insert + b"\n" + data[ls:]
        data, n = re.subn(pat, repl, data)
        e.text = data
        e.note = ((e.note + "; ") if e.note else "") + f"staging rewrite ({n}x): {why}"
        warns.append(f"rewrote {f} at staging time ({n}x): {why}")
    fields = template_fields(args)
    for f in TEMPLATED_DOCS:
        e = by_src.get(f)
        if e is None:
            continue
        txt = e.src.read_text()
        used = sorted(set(re.findall(r"\{\{([A-Za-z0-9_]+)\}\}", txt)))
        for k in used:
            if k in fields:
                txt = txt.replace("{{" + k + "}}", fields[k])
        e.text = txt.encode()
        e.note = "template filled: " + ", ".join(used) if used else None


def entry_bytes(e: Entry) -> bytes:
    """Content that will be staged for entry ``e``."""
    if e.text is not None:
        return e.text if isinstance(e.text, bytes) else e.text.encode()
    return e.src.read_bytes()


def audit(entries: list[Entry], draft: bool = False) -> list[str]:
    """Forbidden paths, credential-looking content, machine-local paths, references to excluded
    modules and (non-draft) unfilled document placeholders; returns a list of problems."""
    bad = []
    for e in entries:
        why = forbidden_reason(rel(e.src) if e.src else None, e.arc)
        if why:
            bad.append(f"forbidden ({why}): {e.arc} <- {rel(e.src) if e.src else 'generated'}")
            continue
        if e.kind == "output" or (e.src is None and e.text is None):
            continue
        if e.arc.endswith(".npy"):
            continue
        data = entry_bytes(e)
        m = SECRET_PATTERNS.search(data)
        if m:
            bad.append(f"credential-like content in {e.arc}: {m.group(0)[:12]!r}...")
        m = LOCAL_PATH_PATTERNS.search(data)
        if m:
            ln = data[:m.start()].count(b"\n") + 1
            bad.append(f"machine-local path in {e.arc}:{ln}: {m.group(0)!r}...")
        if e.arc.endswith((".py", ".sh")):
            m = EXCLUDED_REF.search(data)
            if m:
                bad.append(f"{e.arc} references an excluded module: {m.group(0)!r}")
        if e.kind == "doc" and not draft and e.arc.endswith(".md") and not e.arc.endswith("EXPERIMENT_LOG.md"):
            hits = sorted(set(DOC_PLACEHOLDERS.findall(data.decode("utf-8", "replace"))))
            if hits:
                bad.append(f"unfilled placeholder(s) in {e.arc}: {', '.join(hits)}")
    return bad


def subset_check(match: Path, cand: Path, max_report: int = 5) -> tuple[list[str], dict]:
    """Streamed spec check of the two output files, row by row (both are written in the same S1
    order by src/submission.py): same S1 id on every row, matched ids a subset of that row's
    candidate ids, no duplicate id within a row, every S1 id prefixed S1- and every other id S2- or
    S3-.  About 40 s and < 300 MB for the 1.73 M-row / 43 M-id files."""
    errs: list[str] = []
    st = {"rows": 0, "matched_ids": 0, "candidate_ids": 0, "not_in_candidates": 0,
          "duplicate_ids": 0, "bad_prefix": 0, "s1_mismatch": 0}
    ok_pre = ("S2-", "S3-")

    def ids(field: str) -> list[str]:
        return [x for x in field.split(",") if x] if field else []

    def err(msg: str) -> None:
        if len(errs) < max_report:
            errs.append(msg)

    with open(match, encoding="utf-8") as fm, open(cand, encoding="utf-8") as fc:
        fm.readline(); fc.readline()                       # headers are checked by the caller
        for ln, (a, b) in enumerate(zip(fm, fc), start=2):
            s1a, _, ma = a.rstrip("\n").partition("\t")
            s1b, _, cb = b.rstrip("\n").partition("\t")
            st["rows"] += 1
            if s1a != s1b:
                st["s1_mismatch"] += 1; err(f"line {ln}: S1 {s1a!r} in matching vs {s1b!r} in candidates")
                continue
            if not s1a.startswith("S1-"):
                st["bad_prefix"] += 1; err(f"line {ln}: S1 id {s1a!r}")
            m, c = ids(ma), ids(cb)
            st["matched_ids"] += len(m); st["candidate_ids"] += len(c)
            sm, sc = set(m), set(c)
            if len(sm) != len(m) or len(sc) != len(c):
                st["duplicate_ids"] += 1; err(f"line {ln} ({s1a}): duplicate ids")
            if any(not x.startswith(ok_pre) for x in c) or any(not x.startswith(ok_pre) for x in m):
                st["bad_prefix"] += 1; err(f"line {ln} ({s1a}): id without S2-/S3- prefix")
            miss = sm - sc
            if miss:
                st["not_in_candidates"] += len(miss); err(f"line {ln} ({s1a}): matched but not a candidate: {sorted(miss)[:3]}")
        if fm.readline() or fc.readline():
            st["s1_mismatch"] += 1; err("files have different row counts")
    bad = {k: v for k, v in st.items() if k not in ("rows", "matched_ids", "candidate_ids") and v}
    if bad and not errs:
        errs.append(f"subset check failed: {bad}")
    elif bad:
        errs.insert(0, f"subset check failed: {bad}")
    return errs, st


def check_outputs(args) -> tuple[list[str], dict]:
    """Headers, row counts (both files, and test_source1 when present), meta.json validator status."""
    errs, info = [], {}
    sub = ROOT / "submissions" / args.sub
    rows = {}
    for name in OUTPUT_FILES:
        p = sub / name
        with open(p, "rb") as f:
            head = f.readline()
        if head != OUTPUT_HEADERS[name]:
            errs.append(f"{rel(p)}: unexpected header {head[:80]!r}")
        rows[name] = count_lines(p) - 1
    info["rows"] = rows
    if rows["matching_results.tsv"] != rows["candidate_pairs.tsv"]:
        errs.append(f"row counts differ: {rows}")
    s1 = ROOT / "data" / "raw" / "dataset" / "test" / "test_source1.tsv"
    if s1.is_file():
        n = count_lines(s1) - 1
        info["test_source1_rows"] = n
        if n != rows["matching_results.tsv"]:
            errs.append(f"matching_results rows {rows['matching_results.tsv']:,} != test_source1 rows {n:,}")
    if not errs:
        sub_errs, info["subset"] = subset_check(sub / "matching_results.tsv", sub / "candidate_pairs.tsv")
        errs += sub_errs
    meta = json.loads((sub / "meta.json").read_text())
    info["meta"] = meta
    if meta.get("validator") != "PASS":
        errs.append(f"submissions/{args.sub}/meta.json: validator = {meta.get('validator')!r}, expected 'PASS'")
    return errs, info


# ----------------------------------------------------------------------------------------------
# staging + zip + verification
# ----------------------------------------------------------------------------------------------
def stage(entries: list[Entry], staging: Path) -> None:
    """Copy (never link) every entry into the staging tree.  PACKAGE_INFO.json is written first as a
    marker, so a later --force may recognise (and only then delete) a staging dir of this script."""
    (staging / PKG).mkdir(parents=True, exist_ok=True)
    (staging / PKG / "PACKAGE_INFO.json").write_text("{}\n")
    for e in entries:
        dst = staging / e.arc
        dst.parent.mkdir(parents=True, exist_ok=True)
        if e.text is not None:
            dst.write_bytes(e.text if isinstance(e.text, bytes) else e.text.encode())
        else:
            shutil.copy2(e.src, dst)
        os.chmod(dst, 0o755 if e.executable else 0o644)


def write_meta_files(entries: list[Entry], staging: Path, args, out_info: dict, warns: list[str],
                     git: dict) -> dict:
    """SHA256SUMS (every file, sha256sum format, paths relative to the zip root) and PACKAGE_INFO.json."""
    sums, files = [], []
    for e in sorted(entries, key=lambda x: x.arc):
        p = staging / e.arc
        h = sha256(p)
        sums.append(f"{h}  {e.arc}")
        rec = {"path": e.arc, "bytes": p.stat().st_size, "sha256": h, "kind": e.kind,
               "source": rel(e.src) if e.src else "generated placeholder"}
        if e.note:
            rec["staging_change"] = e.note
        files.append(rec)
    (staging / PKG / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    info = {
        "team": args.team, "submission_id": args.sub, "draft": bool(args.draft),
        "built": dt.datetime.now().isoformat(timespec="seconds"), "git": git,
        "submission_variant": VARIANTS.get(args.sub), "public_lb": leaderboard_score(args.sub),
        "submission_meta": out_info.get("meta"), "output_rows": out_info.get("rows"),
        "output_subset_check": out_info.get("subset"),
        "excluded_repo_files": EXCLUDE, "warnings": warns, "files": files,
    }
    (staging / PKG / "PACKAGE_INFO.json").write_text(json.dumps(info, indent=1) + "\n")
    return info


def make_zip(staging: Path, zip_path: Path, level: int) -> list[str]:
    """Deflate every staged file (sorted) into zip_path; returns the member names."""
    names = sorted(p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_file())
    tmp = zip_path.with_name(zip_path.name + ".partial")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=level, allowZip64=True) as zf:
        for n in names:
            zf.write(staging / n, n)
    os.replace(tmp, zip_path)
    return names


def verify_zip(zip_path: Path, staging: Path, names: list[str], out_sha: dict) -> list[str]:
    """Re-open the zip: members == staging tree, sizes, CRCs, spec paths, output sha256."""
    errs = []
    with zipfile.ZipFile(zip_path) as zf:
        infos = {i.filename: i for i in zf.infolist()}
        if sorted(infos) != names:
            errs.append(f"zip members differ from staging: {sorted(set(infos) ^ set(names))[:10]}")
        for n in names:
            if n in infos and infos[n].file_size != (staging / n).stat().st_size:
                errs.append(f"size mismatch in zip: {n}")
        bad = zf.testzip()
        if bad:
            errs.append(f"CRC error in zip member {bad}")
        for p in SPEC_PATHS:
            if p not in infos:
                errs.append(f"spec path missing in zip: {p}")
        for d in SPEC_DIRS:
            if not any(n.startswith(d) and n.endswith(".py") for n in infos):
                errs.append(f"spec folder empty in zip: {d}")
        tops = {n.split("/", 1)[0] for n in infos}
        if tops != {"output", "code", "Documentation_template.md"}:
            errs.append(f"unexpected top-level entries: {sorted(tops)}")
        if {n for n in infos if n.startswith("output/")} != {f"output/{f}" for f in OUTPUT_FILES}:
            errs.append("output/ must hold exactly matching_results.tsv and candidate_pairs.tsv")
        for n in infos:
            why = forbidden_reason(None, n)
            if why:
                errs.append(f"forbidden member in zip ({why}): {n}")
        for f in OUTPUT_FILES:
            h = hashlib.sha256()
            with zf.open(f"output/{f}") as fh:
                while b := fh.read(8 << 20):
                    h.update(b)
            if h.hexdigest() != out_sha[f]:
                errs.append(f"sha256 of output/{f} in zip != submissions copy")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--team", required=True, help="team name -> <team>_submission.zip")
    ap.add_argument("--team-display", default=None, help="team name as written in the documents (default --team)")
    ap.add_argument("--members", default=None,
                    help="team members as written in the methodology document (required unless --draft)")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="build although packaged code has uncommitted changes (dry runs only)")
    ap.add_argument("--sub", required=True, help="submission id under submissions/ (e.g. day3_l2b)")
    ap.add_argument("--out-dir", default=str(ROOT / "dist"), help="where the staging dir and the zip go (default dist/)")
    ap.add_argument("--draft", action="store_true",
                    help="placeholders for a missing README / methodology doc / src/l2_train.py (dry runs only)")
    ap.add_argument("--no-validator", action="store_true", help="do not package student_resource/utils/validate_submission.py")
    ap.add_argument("--no-zip", action="store_true", help="stage only")
    ap.add_argument("--rm-staging", action="store_true", help="delete the staging dir after a verified zip")
    ap.add_argument("--force", action="store_true", help="replace an existing staging dir / zip built by this script")
    ap.add_argument("--level", type=int, default=6, help="deflate level (default 6)")
    args = ap.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.team):
        print(f"ERROR: team name {args.team!r} must be letters, digits, '.', '_' or '-'"); return 2
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.sub) or not (ROOT / "submissions" / args.sub).is_dir():
        print(f"ERROR: submissions/{args.sub}/ not found"); return 2
    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir == ROOT:
        print("ERROR: --out-dir must not be the repo root"); return 2
    for protected in ("submissions", "experiments", "data", "src", "scripts", "student_resource", "package", ".git"):
        pd_ = (ROOT / protected).resolve()
        if out_dir == pd_ or pd_ in out_dir.parents:
            print(f"ERROR: --out-dir must not be inside {protected}/"); return 2

    if not args.draft:
        if not args.members:
            print("ERROR: --members is required for a non-draft build (Team Members of the methodology document)"); return 2
        if args.sub not in VARIANTS:
            print(f"ERROR: --sub {args.sub} is not one of the final candidates {sorted(VARIANTS)} "
                  "(add it to VARIANTS with its per-country rule, or use --draft)"); return 2

    print(f"== make_package: team={args.team} sub={args.sub} draft={args.draft} repo={ROOT}")
    entries, missing, warns = collect(args)
    transform(entries, args, warns)
    git = git_info({rel(e.src) for e in entries if e.src is not None})
    dirty = git["uncommitted_code_changes"]
    if dirty:
        msg = "uncommitted changes in packaged code: " + "; ".join(dirty)
        if not (args.draft or args.allow_dirty):
            print(f"ERROR: {msg}\n   commit first (PACKAGE_INFO.json records the commit), or pass --allow-dirty for a dry run")
            return 2
        warns.append(msg)
    if missing:
        print("ERROR: required inputs missing -- refusing to build:")
        for m in missing:
            print(f"   - {m}")
        return 2
    problems = audit(entries, draft=args.draft)
    if problems:
        print("ERROR: compliance audit failed -- refusing to build:")
        for p in problems:
            print(f"   - {p}")
        return 2
    out_errs, out_info = check_outputs(args)
    if out_errs:
        print("ERROR: output files failed the pre-checks:")
        for e in out_errs:
            print(f"   - {e}")
        return 2
    for e in entries:
        if e.src is not None and e.kind != "output" and e.src.stat().st_size > MAX_CODE_FILE:
            warns.append(f"large code-side file {e.arc}: {human(e.src.stat().st_size)}")
    print(f"   rows: {out_info['rows']}  test_source1: {out_info.get('test_source1_rows', 'n/a')}  "
          f"meta validator: {out_info['meta'].get('validator')}")
    print(f"   subset check: {out_info.get('subset')}")
    print(f"   variant: {VARIANTS.get(args.sub, {}).get('rule', 'not a listed final candidate')}  "
          f"public LB: {leaderboard_score(args.sub) or 'not scored yet'}")
    print(f"   submission {args.sub}: run={out_info['meta'].get('run')}  created={out_info['meta'].get('created')}\n"
          f"   note: {out_info['meta'].get('note')}")

    # --- staging dir (only ever deletes a previous build of this script: it holds our PACKAGE_INFO.json)
    out_dir.mkdir(parents=True, exist_ok=True)
    if ROOT in out_dir.parents:                   # keep build output out of git status
        gi = out_dir / ".gitignore"
        if not gi.exists():
            gi.write_text("# build output of scripts/make_package.py -- never commit\n*\n")
    staging = out_dir / f"{args.team}_submission"
    zip_path = out_dir / f"{args.team}_submission.zip"
    for p in (staging, zip_path):
        if p.exists():
            ours = (p / PKG / "PACKAGE_INFO.json").is_file() if p.is_dir() else zipfile.is_zipfile(p)
            if not args.force:
                print(f"ERROR: {p} exists (use --force to rebuild)"); return 2
            if not ours:
                print(f"ERROR: {p} exists and was not built by this script -- not touching it"); return 2
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    stage(entries, staging)
    info = write_meta_files(entries, staging, args, out_info, warns, git)
    out_sha = {f["path"].split("/", 1)[1]: f["sha256"] for f in info["files"] if f["kind"] == "output"}

    # --- manifest
    print(f"\n== manifest ({len(info['files'])} files + SHA256SUMS + PACKAGE_INFO.json)")
    total = 0
    for f in info["files"]:
        total += f["bytes"]
        tail = f"  sha256={f['sha256']}" if f["kind"] == "output" else ""
        print(f"   {f['bytes']:>12,}  {f['path']}{tail}")
    by_kind = {}
    for f in info["files"]:
        by_kind[f["kind"]] = by_kind.get(f["kind"], 0) + f["bytes"]
    print(f"   total {human(total)}  by kind: " + ", ".join(f"{k} {human(v)}" for k, v in sorted(by_kind.items())))
    for n in OUTPUT_FILES:
        print(f"   sha256 output/{n} = {out_sha[n]}")
    for w in warns:
        print(f"   WARNING: {w}")

    if args.no_zip:
        print(f"\nstaged (no zip): {staging}")
        return 0
    print(f"\n== zipping -> {zip_path}")
    names = make_zip(staging, zip_path, args.level)
    errs = verify_zip(zip_path, staging, names, out_sha)
    if errs:
        print("ERROR: zip verification failed:")
        for e in errs:
            print(f"   - {e}")
        return 1
    print(f"   verified: {len(names)} members, CRC ok, spec paths present, output sha256 match")
    print(f"   zip size {zip_path.stat().st_size:,} bytes ({human(zip_path.stat().st_size)})")
    if args.rm_staging:
        shutil.rmtree(staging)
        print(f"   staging removed")
    else:
        print(f"   staging kept: {staging}")
    if dirty:
        print(f"WARNING: built from UNCOMMITTED code ({len(dirty)} paths; git commit {git['commit']} does not "
              "contain them) -- rebuild after committing for the final package")
    print("DRAFT BUILD -- not for submission" if args.draft else
          ("PACKAGE OK (dry run: uncommitted code)" if dirty else "PACKAGE OK"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
