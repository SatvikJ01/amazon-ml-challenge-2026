"""E046 epoch-2 cross-encoder kernel (continues mDeBERTa / LaBSE from e046-ce-mdl).
E046 cross-encoder kernel (Kaggle, 2x T4 16 GB, fp16).

A pretrained multilingual transformer reads both records of a pair as one sequence
    <s> S1 name | S1 address [| transliteration] </s></s> candidate name | candidate address [| translit] </s>
and is fine-tuned (all layers, binary cross-entropy on the match label) to say whether they are the same
business.  One worker per GPU, each with its own backbone, so the two models can be compared and averaged:
    GPU 0: FacebookAI/xlm-roberta-base        (MIT; masked LM on 2.5 TB of CommonCrawl, 100 languages)
    GPU 1: microsoft/mdeberta-v3-base         (MIT; ELECTRA-style, 100 languages), falls back to
           intfloat/multilingual-e5-base      (MIT) if it cannot be loaded.

Inputs (Kaggle dataset, built by src/ce_data.py): train_text.parquet (E039 training entities, OOF stage-3
band sample, labels), hold_text.parquet (E039 holdout band, labels, only used for the monitoring AUC here),
test_text.parquet (test band).  Outputs in /kaggle/working: ce{k}_hold.parquet, ce{k}_test.parquet
(s1, cand, logit), model{k}/ (fp16 weights + tokenizer), log{k}.json.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

SMOKE = False                    # True: tiny subsets, end-to-end check of the kernel in a few minutes
IN = "/kaggle/input/e046-ce-pairs"
if not os.path.exists(f"{IN}/train_text.parquet"):          # the dataset mount path differs between sessions
    import glob
    _hits = glob.glob("/kaggle/input/**/train_text.parquet", recursive=True)
    if _hits:
        IN = os.path.dirname(_hits[0])
OUT = "/kaggle/working"
import glob as _g                # epoch 2: continue from the e046-ce-mdl checkpoints (kernel output mounted as input)
def _ck(k):
    h = sorted(_g.glob(f"/kaggle/input/**/model{k}/config.json", recursive=True))
    return [os.path.dirname(h[0])] if h else []
BACKBONES = {0: _ck(0), 1: _ck(1)}
MAX_LEN = 128
BS = 64
LR = 1e-5
WD = 0.01
WARMUP = 500
EPOCHS = 1
EVAL_EVERY = 4000
INFER_BS = 512
MAX_TRAIN_SEC = 6.0 * 3600       # stop training early if needed so that inference still fits in 12 h
SEED = 47
HALVES = False                   # True: worker k trains only on the entities with hash % 2 == k (bagging)
FREEZE_EMB = {1: True}                # True: freeze the word-embedding matrix (large models: memory, transfer)


def log(k, *a):
    print(f"[gpu{k} {time.strftime('%H:%M:%S')}]", *a, flush=True)


def texts(df):
    """Side A / side B strings of every pair."""
    def side(name, addr, tr):
        s = np.where(addr.to_numpy() != "", name.to_numpy() + " | " + addr.to_numpy(), name.to_numpy())
        s = np.where(tr.to_numpy() != "", s + " | " + tr.to_numpy(), s)
        return s.tolist()
    return side(df.a_name, df.a_addr, df.a_tr), side(df.b_name, df.b_addr, df.b_tr)


def tokenize(tok, a, b, chunk=50000):
    """Pair token ids as one flat int32 array + offsets (memory: 4 bytes per token)."""
    ids, lens = [], []
    for i in range(0, len(a), chunk):
        enc = tok(a[i:i + chunk], b[i:i + chunk], truncation=True, max_length=MAX_LEN,
                  return_attention_mask=False, return_token_type_ids=False)["input_ids"]
        lens.append(np.fromiter((len(x) for x in enc), np.int32, len(enc)))
        ids.append(np.fromiter((t for x in enc for t in x), np.int32, int(lens[-1].sum())))
    lens = np.concatenate(lens)
    off = np.zeros(len(lens) + 1, np.int64)
    np.cumsum(lens, out=off[1:])
    return np.concatenate(ids), off, lens


def batch_tensors(torch, flat, off, lens, rows, pad_id, dev):
    L = int(lens[rows].max())
    x = np.full((len(rows), L), pad_id, np.int64)
    m = np.zeros((len(rows), L), np.int64)
    for j, r in enumerate(rows):
        n = lens[r]
        x[j, :n] = flat[off[r]:off[r] + n]
        m[j, :n] = 1
    return (torch.from_numpy(x).pin_memory().to(dev, non_blocking=True),
            torch.from_numpy(m).pin_memory().to(dev, non_blocking=True))


def auc(y, s):
    y = np.asarray(y).astype(bool)
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def predict(torch, model, flat, off, lens, pad_id, dev, bs=INFER_BS):
    """Logits for every row, batched by length (sorted) for speed."""
    order = np.argsort(lens, kind="stable")
    out = np.empty(len(lens), np.float32)
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(order), bs):
            rows = order[i:i + bs]
            x, m = batch_tensors(torch, flat, off, lens, rows, pad_id, dev)
            out[rows] = model(input_ids=x, attention_mask=m).logits.float().squeeze(-1).cpu().numpy()
    model.train()
    return out


def worker(k):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(SEED + k)
    rng = np.random.default_rng(SEED + k)
    dev = torch.device("cuda:0")
    info = {"gpu": k, "device": torch.cuda.get_device_name(0), "torch": torch.__version__}
    tok = model = None
    for name in BACKBONES[k]:
        try:
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1).float()
            info["backbone"] = name
            break
        except Exception as e:  # noqa: BLE001 - try the next backbone
            log(k, f"cannot load {name}: {type(e).__name__}: {e}")
    if model is None:
        raise SystemExit(f"gpu{k}: no backbone could be loaded")
    if (FREEZE_EMB.get(k, False) if isinstance(FREEZE_EMB, dict) else FREEZE_EMB):
        for n_, p_ in model.named_parameters():
            if "word_embeddings" in n_:
                p_.requires_grad = False
    model.to(dev)
    log(k, f"backbone {info['backbone']} ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params), "
           f"fast tokenizer {tok.is_fast}, {info['device']}")
    pad_id = tok.pad_token_id

    # ---------------------------------------------------------------- data
    tr = pd.read_parquet(f"{IN}/train_text.parquet")
    if HALVES:
        tr = tr[tr.s1.to_numpy() % 2 == k].reset_index(drop=True)
    ho = pd.read_parquet(f"{IN}/hold_text.parquet")
    if SMOKE:
        tr = tr.sample(3000, random_state=0).reset_index(drop=True)
        ho = ho.sample(2000, random_state=0).reset_index(drop=True)
    t0 = time.time()
    a, b = texts(tr)
    flat, off, lens = tokenize(tok, a, b)
    y = tr.label.to_numpy().astype(np.float32)
    del a, b
    log(k, f"train pairs {len(tr):,} tokenized in {time.time() - t0:.0f}s; mean len {lens.mean():.1f}, "
           f"p99 {np.percentile(lens, 99):.0f}, truncated {(lens >= MAX_LEN).mean():.4f}")
    info.update(train_pairs=len(tr), mean_len=float(lens.mean()))
    a, b = texts(ho)
    hflat, hoff, hlens = tokenize(tok, a, b)
    del a, b
    mon = np.sort(rng.choice(len(ho), min(30000, len(ho)), replace=False))   # monitoring subset
    a, b = texts(ho.iloc[mon])
    mflat, moff, mlens = tokenize(tok, a, b)
    del a, b

    # ---------------------------------------------------------------- optimisation
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    decay = [p for n, p in params if not any(s in n for s in ("bias", "LayerNorm", "layernorm"))]
    nodecay = [p for n, p in params if any(s in n for s in ("bias", "LayerNorm", "layernorm"))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": WD}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=LR, betas=(0.9, 0.98), eps=1e-6)
    n = len(tr)
    steps_per_epoch = (n // (BS * 100)) * 100 + (n % (BS * 100)) // BS   # full batches of the blocks below
    total = steps_per_epoch * EPOCHS
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / WARMUP) * max(0.0, 1 - s / max(total, 1)))
    scaler = torch.cuda.amp.GradScaler()
    lossf = torch.nn.BCEWithLogitsLoss()
    info["history"] = []
    step, t_start, run_loss, seen = 0, time.time(), 0.0, 0
    stop = False
    for ep in range(EPOCHS):
        perm = rng.permutation(n)
        # length-grouped batches: sort inside blocks of 100 batches, then shuffle the batches
        blocks = [perm[i:i + BS * 100] for i in range(0, n, BS * 100)]
        batches = []
        for blk in blocks:
            blk = blk[np.argsort(lens[blk], kind="stable")]
            batches += [blk[i:i + BS] for i in range(0, len(blk) - BS + 1, BS)]
        rng.shuffle(batches)
        for rows in batches:
            x, m = batch_tensors(torch, flat, off, lens, rows, pad_id, dev)
            yt = torch.from_numpy(y[rows]).to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logit = model(input_ids=x, attention_mask=m).logits.squeeze(-1)
            loss = lossf(logit.float(), yt)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            run_loss += float(loss)
            seen += len(rows)
            if step % 200 == 0:
                el = time.time() - t_start
                log(k, f"step {step}/{total} loss {run_loss / 200:.4f} {seen / el:.0f} pairs/s "
                       f"lr {sched.get_last_lr()[0]:.2e} eta {(total - step) * el / step / 60:.0f} min")
                run_loss = 0.0
            if step % EVAL_EVERY == 0 or step == total:
                sub = mon
                sl = predict(torch, model, mflat, moff, mlens, pad_id, dev)
                yh = ho.label.to_numpy()[sub]
                pr = 1 / (1 + np.exp(-sl.astype(np.float64)))
                ll = float(-np.mean(yh * np.log(np.clip(pr, 1e-7, 1)) + (1 - yh) * np.log(np.clip(1 - pr, 1e-7, 1))))
                cty = ho.country.to_numpy()[sub]
                h = dict(step=step, auc=auc(yh, sl), logloss=ll,
                         auc_US=auc(yh[cty == "US"], sl[cty == "US"]), auc_India=auc(yh[cty == "India"], sl[cty == "India"]),
                         auc_p39=auc(yh, ho.p39.to_numpy()[sub]), auc_pf=auc(yh, ho.pf.to_numpy()[sub]),
                         elapsed_min=round((time.time() - t_start) / 60, 1))
                info["history"].append(h)
                log(k, "EVAL", json.dumps(h))
                json.dump(info, open(f"{OUT}/log{k}.json", "w"), indent=1)
            if time.time() - t_start > MAX_TRAIN_SEC:
                log(k, f"time budget reached at step {step}; stopping training")
                info["stopped_early_at"] = step
                stop = True
                break
        if stop:
            break
    info["train_min"] = round((time.time() - t_start) / 60, 1)
    del flat, off, lens, y, tr
    model.half().save_pretrained(f"{OUT}/model{k}")
    tok.save_pretrained(f"{OUT}/model{k}")
    model.float()

    # ---------------------------------------------------------------- scoring
    t0 = time.time()
    s = predict(torch, model, hflat, hoff, hlens, pad_id, dev)
    pd.DataFrame({"s1": ho.s1.to_numpy(), "cand": ho.cand.to_numpy(), "logit": s}).to_parquet(f"{OUT}/ce{k}_hold.parquet", index=False)
    info["hold_auc"] = auc(ho.label.to_numpy(), s)
    info["hold_auc_pf"] = auc(ho.label.to_numpy(), ho.pf.to_numpy())
    log(k, f"holdout scored ({len(ho):,} pairs, {time.time() - t0:.0f}s): AUC {info['hold_auc']:.5f} (pf {info['hold_auc_pf']:.5f})")
    del hflat, hoff, hlens
    te = pd.read_parquet(f"{IN}/test_text.parquet")
    if SMOKE:
        te = te.sample(2000, random_state=0).reset_index(drop=True)
    parts, t0 = [], time.time()
    CH = 400000
    for i in range(0, len(te), CH):
        c = te.iloc[i:i + CH]
        a, b = texts(c)
        f_, o_, l_ = tokenize(tok, a, b)
        parts.append(predict(torch, model, f_, o_, l_, pad_id, dev))
        log(k, f"test {min(i + CH, len(te)):,}/{len(te):,} scored, {time.time() - t0:.0f}s")
    pd.DataFrame({"s1": te.s1.to_numpy(), "cand": te.cand.to_numpy(), "logit": np.concatenate(parts)}).to_parquet(
        f"{OUT}/ce{k}_test.parquet", index=False)
    info["test_min"] = round((time.time() - t0) / 60, 1)
    json.dump(info, open(f"{OUT}/log{k}.json", "w"), indent=1)
    log(k, "done", json.dumps({x: info[x] for x in ("backbone", "train_min", "test_min", "hold_auc", "hold_auc_pf")}))


def _run(k):
    # the parent never touches CUDA: each forked child pins its GPU before importing torch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(k)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    worker(k)


def main():
    import multiprocessing as mp
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout, flush=True)
    gpus = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip().splitlines()
    print("gpus", len(gpus), "| inputs", os.listdir(IN), flush=True)
    ctx = mp.get_context("fork")
    procs = [ctx.Process(target=_run, args=(k,)) for k in range(min(2, len(gpus)))]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    codes = [p.exitcode for p in procs]
    print("worker exit codes", codes, flush=True)
    for k in range(len(procs)):
        f = f"{OUT}/log{k}.json"
        if os.path.exists(f):
            print(open(f).read(), flush=True)


if __name__ == "__main__":
    main()
