"""Label-free train/test shift check on blocking-score profiles.

Test has ~23 % more S2/S3 records per S1 entity than train (5.75 vs 4.68).  If
the extra records are distractors, test entities face more near-duplicates and
precision will be harder than local CV suggests.  This compares, per (split,
country, source), the distribution of retrieval scores using candidate files
only -- no labels -- so train, test and the unseen France block are comparable.

Usage: python -m src.shift_check
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .candidates import cand_path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def profile(tag: str, country: str, source: int) -> dict:
    t = pq.read_table(cand_path(tag, country, source), columns=["blk_score", "blk_rank"])
    sc, rk = t.column("blk_score").to_numpy(), t.column("blk_rank").to_numpy()
    top1, top2, top5 = sc[rk == 0], sc[rk == 1], sc[rk == 4]
    n = top1.size
    strong = np.bincount(np.cumsum(rk == 0)[sc > 0.5] - 1, minlength=n) if n else np.array([])
    return {
        "entities": int(n),
        "top1_mean": float(top1.mean()),
        "top1_p10": float(np.percentile(top1, 10)),
        "top2_mean": float(top2.mean()),
        "top5_mean": float(top5.mean()),
        "gap12_mean": float((top1[: top2.size] - top2).mean()),
        "n_score_gt_0.5_mean": float(strong.mean()),
        "n_score_gt_0.5_p90": float(np.percentile(strong, 90)),
    }


def main() -> None:
    rows = []
    for tag, countries in (("trnall", ["India", "US"]), ("testall", ["France", "India", "US"])):
        for c in countries:
            for s in (2, 3):
                if cand_path(tag, c, s).exists():
                    r = profile(tag, c, s)
                    r.update(split=tag, country=c, source=s)
                    rows.append(r)
    df = pd.DataFrame(rows).set_index(["split", "country", "source"])
    pd.set_option("display.width", 200)
    print(df.round(3).to_string())
    (REPORTS / "shift_check.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
