"""Compact integer codec for entity ids.

Entity ids look like ``S1-925783039``: a source tag plus a decimal integer.  At
this dataset's scale (~26 M records) keeping them as Python strings is the
difference between a pipeline that fits in RAM and one that does not --
``sys.getsizeof("S2-166376419") == 61`` bytes plus 8 for the pointer, versus 8
bytes for an ``int64``.

Encoding
--------
``code = source_tag * OFFSET + numeric_part`` with ``source_tag in {1, 2, 3}``.

``OFFSET`` is 10**10, comfortably above the largest observed numeric part, so the
source tag is recoverable by integer division and the mapping is a bijection.
The result fits in int64 with ~9 orders of magnitude to spare.

This gives us, for free:
* O(1) source identification (``code // OFFSET``) with no string work,
* sortable / hashable keys usable directly as numpy array values,
* `np.isin`, `np.searchsorted` and pandas joins at C speed.
"""

from __future__ import annotations

import numpy as np

__all__ = ["OFFSET", "encode_ids", "decode_ids", "source_of", "encode_one", "decode_one"]

OFFSET: int = 10**10

_PREFIX = {1: "S1-", 2: "S2-", 3: "S3-"}


def encode_ids(raw: np.ndarray | list[str], source: int | None = None) -> np.ndarray:
    """Vectorised ``["S2-123", ...] -> int64 array``.

    Parameters
    ----------
    raw:
        Array/list of id strings.  Passed through pandas' string accessor when it
        is an object array, which is the fast path for the ~5 M-row source files.
    source:
        If given, the source tag is taken as a constant and the prefix is only
        *stripped*, not parsed -- a useful speedup when loading a file whose
        source is known.  When ``None`` the tag is read from each string.
    """
    import pandas as pd

    s = pd.Series(raw, dtype="string")
    num = s.str.slice(3).astype("int64").to_numpy()

    if source is not None:
        return num + source * OFFSET

    tag = s.str.slice(1, 2).astype("int64").to_numpy()
    return num + tag * OFFSET


def decode_ids(codes: np.ndarray) -> np.ndarray:
    """Vectorised ``int64 array -> object array of id strings``.

    Only used when writing submissions, so it is allowed to be the slow direction.
    """
    import pandas as pd

    codes = np.asarray(codes, dtype=np.int64)
    tag = (codes // OFFSET).astype(np.int64)
    num = (codes % OFFSET).astype(np.int64)

    prefix = pd.Series(tag).map(_PREFIX)
    if prefix.isna().any():
        bad = int(tag[prefix.isna().to_numpy()][0])
        raise ValueError(f"unknown source tag {bad} in encoded ids")

    return (prefix + pd.Series(num).astype("string")).to_numpy()


def source_of(codes: np.ndarray) -> np.ndarray:
    """Return the source tag (1/2/3) for each encoded id."""
    return (np.asarray(codes, dtype=np.int64) // OFFSET).astype(np.int8)


def encode_one(raw: str) -> int:
    """Scalar encode -- for tests and debugging, not hot paths."""
    return int(raw[1]) * OFFSET + int(raw[3:])


def decode_one(code: int) -> str:
    """Scalar decode -- for tests and debugging, not hot paths."""
    return f"{_PREFIX[int(code) // OFFSET]}{int(code) % OFFSET}"
