"""Score a pair-text parquet with a saved E046 cross-encoder (local GPU):
    python kaggle/e046_ce/local_score.py <model_dir> <pairs.parquet> <out.parquet>"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ce_kernel as K  # noqa: E402


def main():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    md, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    dev = torch.device("cuda:0")
    tok = AutoTokenizer.from_pretrained(md)
    model = AutoModelForSequenceClassification.from_pretrained(md).float().to(dev)
    te = pd.read_parquet(src)
    parts, t0, CH = [], time.time(), 400000
    for i in range(0, len(te), CH):
        c = te.iloc[i:i + CH]
        a, b = K.texts(c)
        f_, o_, l_ = K.tokenize(tok, a, b)
        parts.append(K.predict(torch, model, f_, o_, l_, tok.pad_token_id, dev, bs=256))
        print(f"{min(i + CH, len(te)):,}/{len(te):,} scored, {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame({"s1": te.s1.to_numpy(), "cand": te.cand.to_numpy(), "logit": np.concatenate(parts)}).to_parquet(dst, index=False)
    print("->", dst, flush=True)


if __name__ == "__main__":
    main()
