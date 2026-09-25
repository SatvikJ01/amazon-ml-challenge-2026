"""One-off preprocessing: normalise every source file once, split by country.

Output: ``data/processed/{split}_s{source}_{country}.parquet`` with columns

    code        encoded entity id (int64, see ids.py)
    name_norm   normalised business name
    addr_norm   normalised business address
    nonascii    1 if the *raw* name contained non-ASCII characters (i.e. it was
                transliterated) -- a useful signal that the name is noisy

Country values are discovered from the data, never enumerated in code, so the
France files for the test split appear automatically.

Run (under a memory cap):
    systemd-run --user --scope -p MemoryMax=5G -p MemorySwapMax=0 \
        .venv/bin/python -m src.prep
"""

from __future__ import annotations

import gc
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .normalize import normalize_text

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
_NON_ASCII = re.compile(r"[^\x00-\x7F]")


def safe_name(country: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", country).strip("_") or "UNK"


def process(split: str, source: int, batch_rows: int = 400_000) -> list[str]:
    """Stream one source file in row batches and append to per-country outputs.

    Normalising a 5 M-row file in one go exceeded a 5 GB cap (every pandas
    string op materialises a full copy), so rows are processed in batches and
    written through one ParquetWriter per country.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    t0 = time.time()
    src = pq.ParquetFile(INTERIM / f"{split}_s{source}.parquet")
    writers: dict[str, pq.ParquetWriter] = {}
    counts: dict[str, int] = {}
    try:
        for batch in src.iter_batches(batch_size=batch_rows):
            df = batch.to_pandas()
            out = pd.DataFrame(
                {
                    "code": df["code"].to_numpy(),
                    "name_norm": normalize_text(df["business_name"]).to_numpy(),
                    "addr_norm": normalize_text(df["business_address"]).to_numpy(),
                    "nonascii": df["business_name"].str.contains(_NON_ASCII, regex=True)
                    .to_numpy().astype(np.int8),
                }
            )
            country = df["country"].to_numpy()
            del df
            for c in np.unique(country):
                part = out[country == c]
                table = pa.Table.from_pandas(part, preserve_index=False)
                if c not in writers:
                    path = PROCESSED / f"{split}_s{source}_{safe_name(c)}.parquet"
                    writers[c] = pq.ParquetWriter(path, table.schema, compression="zstd")
                    counts[c] = 0
                writers[c].write_table(table)
                counts[c] += len(part)
            del out
            gc.collect()
    finally:
        for w in writers.values():
            w.close()
    written = [f"{split}_s{source}_{safe_name(c)} ({n:,})" for c, n in sorted(counts.items())]
    print(f"[{split} S{source}] {time.time() - t0:.0f}s -> {', '.join(written)}", flush=True)
    return written


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        for source in (1, 2, 3):
            process(split, source)
            gc.collect()


if __name__ == "__main__":
    main()
