"""Training-entity sample for v4: ``n`` S1 records drawn uniformly across countries
(all of them when ``n`` exceeds the pool).  Same seed as earlier samples."""
from __future__ import annotations

import argparse

import numpy as np

from .candidates import countries_for, sample_entities
from .v3 import PROCESSED


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    a = ap.parse_args()
    import pyarrow.parquet as pq
    total = sum(pq.ParquetFile(PROCESSED / f"{a.split}_s1_{c}.parquet").metadata.num_rows for c in countries_for(a.split))
    e = sample_entities(a.split, min(a.n, total), 0)
    np.save(a.out, e)
    print(f"{len(e):,} of {total:,} training S1 entities -> {a.out}")


if __name__ == "__main__":
    main()
