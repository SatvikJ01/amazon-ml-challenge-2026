"""Native-script name dictionary learned from training ground truth.

18.2 % of India S2/S3 names arrive in an Indic script (Devanagari, Telugu,
Kannada, Tamil, ...).  Generic ``unidecode`` turns them into phonetic garble
(``praaivett limittedd``) that defeats retrieval and matching.  But the
generator transliterates each English word consistently: aligning native-script
names with their S1 names in the training ground truth gives a 96.6 %
deterministic token mapping over only ~1.3 k native tokens, covering 96.4 % of
test token occurrences (E027).

``build``  learns native token -> English token from positionally aligned pairs
           (same token count), keeping mappings seen >= MIN_COUNT times with a
           modal share >= MIN_SHARE.  Uses train data only.
``apply``  writes a parallel data version ``{split}T`` in data/processed/: records
           whose raw name contains Indic script get ``name_norm`` rebuilt from the
           dictionary (unknown tokens fall back to the usual normalisation); every
           other record and every S1 file is hard-linked unchanged.  The original
           ``train``/``test`` files are never modified (rollback baseline).

Usage:
    python -m src.translit build
    python -m src.translit apply --split train
    python -m src.translit apply --split test
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .normalize import normalize_text

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
DICT_PATH = ROOT / "data" / "processed" / "translit_dict.json"
INDIC = re.compile(r"[ऀ-෿]")
PUN = re.compile(r"[^\w\s]")
MIN_COUNT, MIN_SHARE = 3, 0.8


def _toks(x: str) -> list[str]:
    return PUN.sub(" ", x.lower()).split()


def build() -> dict:
    ps1, pm = np.load(INTERIM / "gt_pair_s1.npy"), np.load(INTERIM / "gt_pair_m.npy")
    owner = pd.Series(ps1, index=pm)
    s1 = pd.read_parquet(INTERIM / "train_s1.parquet", columns=["code", "business_name"]).set_index("code")["business_name"]
    mp: dict[str, Counter] = defaultdict(Counter)
    n_rec = 0
    for s in (2, 3):
        t = pd.read_parquet(INTERIM / f"train_s{s}.parquet", columns=["code", "business_name"])
        t = t[t["business_name"].str.contains(INDIC)]
        t = t.assign(s1=owner.reindex(t["code"]).to_numpy()).dropna(subset=["s1"])
        en = s1.reindex(t["s1"].astype(np.int64)).to_numpy()
        for raw, e in zip(t["business_name"], en):
            a, b = raw.split(), _toks(e)
            if len(a) != len(b):
                continue
            n_rec += 1
            for x, y in zip(a, b):
                if INDIC.search(x):
                    mp[x][y] += 1
    d = {}
    for tok, cnt in mp.items():
        (best, c), tot = cnt.most_common(1)[0], sum(cnt.values())
        if tot >= MIN_COUNT and c / tot >= MIN_SHARE:
            d[tok] = best
    DICT_PATH.write_text(json.dumps(d, ensure_ascii=False))
    print(f"aligned records {n_rec:,}; native tokens seen {len(mp):,}; kept {len(d):,} "
          f"(count>={MIN_COUNT}, share>={MIN_SHARE}) -> {DICT_PATH.name}")
    return d


def translate(raw: str, d: dict) -> str:
    return " ".join(d.get(t, t) for t in raw.split())


def apply(split: str) -> None:
    d = json.loads(DICT_PATH.read_text())
    out_split = f"{split}T"
    for p in sorted(PROCESSED.glob(f"{split}_s[123]_*.parquet")):
        dst = PROCESSED / p.name.replace(f"{split}_", f"{out_split}_", 1)
        if dst.exists():
            dst.unlink()
        country = p.stem.split("_", 2)[2]
        src_id = p.stem.split("_")[1]
        if src_id == "s1" or country != "India":
            os.link(p, dst)                      # unchanged: hard link, no copy
            continue
        df = pd.read_parquet(p)
        raw = pd.read_parquet(INTERIM / f"{split}_{src_id}.parquet", columns=["code", "business_name"],
                              filters=[("country", "==", "India")]).set_index("code")["business_name"]
        raw = raw.reindex(df["code"].to_numpy())
        mask = raw.str.contains(INDIC).fillna(False).to_numpy(dtype=bool)
        tr = pd.Series([translate(x, d) for x in raw[mask]], dtype="string")
        new = normalize_text(tr).to_numpy()
        before = df.loc[mask, "name_norm"].to_numpy()
        df.loc[mask, "name_norm"] = new
        df.to_parquet(dst, index=False, compression="zstd")
        changed = int((before != new).sum())
        print(f"  {dst.name}: {int(mask.sum()):,} native-script names, {changed:,} rewritten "
              f"(e.g. {before[0]!r} -> {new[0]!r})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "apply"])
    ap.add_argument("--split", default="train")
    args = ap.parse_args()
    build() if args.step == "build" else apply(args.split)


if __name__ == "__main__":
    main()
