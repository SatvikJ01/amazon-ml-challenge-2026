"""Run the E046 kernel worker on the local GPU (RTX 2050, 4 GB) with a small backbone:
    python kaggle/e046_ce/local_run.py <in_dir> <out_dir> <backbone> <batch> <lr> [freeze]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ce_kernel as K  # noqa: E402

K.IN, K.OUT = sys.argv[1], sys.argv[2]
K.BACKBONES = {0: [sys.argv[3]]}
K.BS, K.LR = int(sys.argv[4]), float(sys.argv[5])
K.WARMUP, K.EVAL_EVERY, K.INFER_BS = 500, 3000, 256
K.FREEZE_EMB = len(sys.argv) > 6 and sys.argv[6] == "freeze"
K.worker(0)
