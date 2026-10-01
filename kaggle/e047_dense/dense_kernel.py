"""E047 dense-retrieval rescue kernel (Kaggle, 2x T4 16 GB, fp16).

Phase 1 (GPU 0): fine-tune intfloat/multilingual-e5-small (MIT) as a bi-encoder on the ground-truth pairs of the
non-holdout training entities: mean-pooled, L2-normalised embeddings, symmetric InfoNCE with in-batch negatives
from the same country (temperature 0.05), word embeddings frozen.
Phase 2 (one worker per GPU, by country): embed every record, exact kNN by cosine in both directions within the
country (each S1 -> its top-K_F records; each record -> its top-K_R S1 among ALL S1 of the split), drop the pairs
already in our candidate set (Bloom filter of the 47M existing pairs), keep the best new pairs and score them with
the E046 LaBSE cross-encoder (checkpoint = output `model1` of the e046-ce-mdl kernel).
Holdout split = E039 holdout S1 (queries) against the full training pools; test split = all test S1.
Outputs: dense_{hold,test}_{country}.parquet (s1, cand, cos, rank_f, rank_r, ce), biencoder/, log_*.json.
"""
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

SMOKE = os.environ.get("E047_SMOKE") == "1"
OUT = os.environ.get("E047_OUT", "/kaggle/working")
BASE = "intfloat/multilingual-e5-small"
MAX_LEN = 48
BS_TRAIN = int(os.environ.get("E047_BS", "512"))
LR = 5e-5
TEMP = 0.05
N_TRAIN = 4_000_000
MAX_TRAIN_SEC = 55 * 60
K_F, K_R = 10, 2            # retrieved per S1 / per record
KEEP_F, KEEP_R = 5, 2       # new pairs kept for cross-encoder scoring (forward rank <= KEEP_F or reverse rank <= KEEP_R)
ENC_BS = 1024
CE_BS = 512
CTY = {0: "US", 1: "India", 2: "France"}
BLOOM_BITS, BLOOM_K = 1 << 29, 7


def log(tag, *a):
    print(f"[{tag} {time.strftime('%H:%M:%S')}]", *a, flush=True)


def find(name):
    h = glob.glob(f"{os.environ.get('E047_INPUT', '/kaggle/input')}/**/{name}", recursive=True)
    if not h:
        raise SystemExit(f"input not found: {name}")
    return h[0]


IN = os.path.dirname(find("rec_train_s1.parquet"))


def load_records(split, sources=(1, 2, 3)):
    tr = pd.read_parquet(f"{IN}/translit.parquet").set_index("code").tr
    parts = []
    for s in sources:
        t = pd.read_parquet(f"{IN}/rec_{split}_s{s}.parquet")
        if SMOKE:
            t = t.groupby("cty", group_keys=False).head(20000)
        x = tr.reindex(t.code.to_numpy())
        has = x.notna().to_numpy()
        txt = t.text.to_numpy().astype(object)
        txt[has] = txt[has] + " | " + x.to_numpy()[has]
        t["text"] = txt
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


# ------------------------------------------------------------------------------------------- Bloom filter
def pair_hash(s1, cand):
    with np.errstate(over="ignore"):
        x = (s1.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) ^ cand.astype(np.uint64)
        x ^= x >> np.uint64(33)
        x *= np.uint64(0xFF51AFD7ED558CCD)
        x ^= x >> np.uint64(33)
        x *= np.uint64(0xC4CEB9FE1A85EC53)
        x ^= x >> np.uint64(33)
    return x


def in_bloom(flags, s1, cand):
    h = pair_hash(s1, cand)
    ok = np.ones(len(h), bool)
    with np.errstate(over="ignore"):
        h1 = h & np.uint64(BLOOM_BITS - 1)
        h2 = (h >> np.uint64(32)) | np.uint64(1)
        for i in range(BLOOM_K):
            ok &= flags[((h1 + np.uint64(i) * h2) & np.uint64(BLOOM_BITS - 1)).astype(np.int64)]
    return ok


# ------------------------------------------------------------------------------------------- encoder utils
def tok_batch(tok, texts, prefix="query: "):
    return tok([prefix + t for t in texts], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")


def mean_pool(out, mask):
    m = mask.unsqueeze(-1).to(out.dtype)
    return (out * m).sum(1) / m.sum(1).clamp(min=1)


def encode(torch, tok, model, texts, dev, tag):
    """L2-normalised fp16 embeddings (on the GPU) of a list of texts, length-sorted batches."""
    lens = np.fromiter((len(t) for t in texts), np.int32, len(texts))
    order = np.argsort(lens, kind="stable")
    E = torch.empty((len(texts), model.config.hidden_size), dtype=torch.float16, device=dev)
    model.eval()
    t0 = time.time()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(order), ENC_BS):
            idx = order[i:i + ENC_BS]
            b = tok_batch(tok, [texts[j] for j in idx]).to(dev)
            e = mean_pool(model(**b).last_hidden_state, b["attention_mask"])
            E[torch.from_numpy(idx).to(dev)] = torch.nn.functional.normalize(e.float(), dim=-1).half()
    log(tag, f"encoded {len(texts):,} texts in {time.time() - t0:.0f}s")
    return E


def topk(torch, Q, P, k, qbs=2048, pbs=1_000_000):
    """Exact top-k by inner product: returns (idx int64 [nq,k], sim float32 [nq,k]) on the CPU."""
    nq = Q.shape[0]
    oi = np.empty((nq, k), np.int64)
    os_ = np.empty((nq, k), np.float32)
    k_ = min(k, P.shape[0])
    for qs in range(0, nq, qbs):
        q = Q[qs:qs + qbs]
        bv = bi = None
        for ps in range(0, P.shape[0], pbs):
            v, i = torch.topk(q @ P[ps:ps + pbs].T, min(k_, P[ps:ps + pbs].shape[0]), dim=1)
            i = i + ps
            if bv is None:
                bv, bi = v, i
            else:
                v2, j = torch.topk(torch.cat([bv, v], 1), k_, dim=1)
                bi = torch.gather(torch.cat([bi, i], 1), 1, j)
                bv = v2
        n = bv.shape[0]
        oi[qs:qs + n, :k_] = bi.cpu().numpy()
        os_[qs:qs + n, :k_] = bv.float().cpu().numpy()
        if k_ < k:
            oi[qs:qs + n, k_:] = -1
            os_[qs:qs + n, k_:] = -1
    return oi, os_


# ------------------------------------------------------------------------------------------- phase 1
def train_biencoder():
    import torch
    from transformers import AutoModel, AutoTokenizer

    tag = "train"
    dev = torch.device("cuda:0")
    torch.manual_seed(47)
    rng = np.random.default_rng(47)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModel.from_pretrained(BASE).float().to(dev)
    for n_, p_ in model.named_parameters():
        if "word_embeddings" in n_:
            p_.requires_grad = False
    R = load_records("train")
    text = pd.Series(R.text.to_numpy(), index=R.code.to_numpy())
    cty = pd.Series(R.cty.to_numpy(), index=R.code.to_numpy())
    G = pd.read_parquet(f"{IN}/gt_train.parquet")
    G = G[G.s1.isin(text.index) & G.m.isin(text.index)]
    if len(G) > N_TRAIN:
        G = G.sample(N_TRAIN, random_state=47)
    if SMOKE:
        G = G.head(20000)
    G = G.reset_index(drop=True)
    G["c"] = cty.reindex(G.s1.to_numpy()).to_numpy()
    batches = []
    for c, g in G.groupby("c"):
        idx = rng.permutation(g.index.to_numpy())
        batches += [idx[i:i + BS_TRAIN] for i in range(0, len(idx) - BS_TRAIN + 1, BS_TRAIN)]
    rng.shuffle(batches)
    qa = text.reindex(G.s1.to_numpy()).to_numpy()
    pa = text.reindex(G.m.to_numpy()).to_numpy()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
    total = len(batches)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 300) * max(0.0, 1 - s / max(total, 1)))
    scaler = torch.cuda.amp.GradScaler()
    log(tag, f"{len(G):,} positive pairs, {total} batches of {BS_TRAIN}")
    t0, run = time.time(), 0.0
    model.train()
    for step, b in enumerate(batches, 1):
        rows = b
        q = tok_batch(tok, list(qa[rows])).to(dev)
        p = tok_batch(tok, list(pa[rows])).to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            eq = torch.nn.functional.normalize(mean_pool(model(**q).last_hidden_state, q["attention_mask"]).float(), dim=-1)
            ep = torch.nn.functional.normalize(mean_pool(model(**p).last_hidden_state, p["attention_mask"]).float(), dim=-1)
        logits = eq @ ep.T / TEMP
        lab = torch.arange(len(rows), device=dev)
        loss = (torch.nn.functional.cross_entropy(logits, lab) + torch.nn.functional.cross_entropy(logits.T, lab)) / 2
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        run += float(loss)
        if step % 200 == 0:
            el = time.time() - t0
            log(tag, f"step {step}/{total} loss {run / 200:.4f} {step * BS_TRAIN / el:.0f} pairs/s eta {(total - step) * el / step / 60:.0f} min")
            run = 0.0
        if time.time() - t0 > MAX_TRAIN_SEC:
            log(tag, f"time budget reached at step {step}")
            break
    model.save_pretrained(f"{OUT}/biencoder")
    tok.save_pretrained(f"{OUT}/biencoder")
    json.dump({"pairs": len(G), "steps": step, "train_min": round((time.time() - t0) / 60, 1)}, open(f"{OUT}/log_train.json", "w"))
    log(tag, "saved biencoder")


# ------------------------------------------------------------------------------------------- phase 2
def ce_score(torch, tok, model, a, b, dev, chunk=200_000):
    if len(a) > chunk:
        return np.concatenate([ce_score(torch, tok, model, a[i:i + chunk], b[i:i + chunk], dev, chunk)
                               for i in range(0, len(a), chunk)])
    enc = tok(a, b, truncation=True, max_length=128, return_attention_mask=False, return_token_type_ids=False)["input_ids"]
    lens = np.fromiter((len(x) for x in enc), np.int32, len(enc))
    order = np.argsort(lens, kind="stable")
    out = np.empty(len(enc), np.float32)
    pad = tok.pad_token_id
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(order), CE_BS):
            rows = order[i:i + CE_BS]
            L = int(lens[rows].max())
            x = np.full((len(rows), L), pad, np.int64)
            m = np.zeros((len(rows), L), np.int64)
            for j, r in enumerate(rows):
                x[j, :lens[r]] = enc[r]
                m[j, :lens[r]] = 1
            out[rows] = model(input_ids=torch.from_numpy(x).to(dev), attention_mask=torch.from_numpy(m).to(dev)).logits.float().squeeze(-1).cpu().numpy()
    return out


def retrieve(torch, tok, model, Rs1, Rpool, queries, dev, tag):
    """New (s1, cand) pairs for the query S1 codes: forward top-K_F and reverse top-K_R (among all S1 in Rs1)."""
    Es1 = encode(torch, tok, model, list(Rs1.text.to_numpy()), dev, tag + " S1")
    Ep = encode(torch, tok, model, list(Rpool.text.to_numpy()), dev, tag + " pool")
    s1c, pc_ = Rs1.code.to_numpy(), Rpool.code.to_numpy()
    qmask = np.isin(s1c, queries)
    qi = np.flatnonzero(qmask)
    t0 = time.time()
    fi, fs = topk(torch, Es1[torch.from_numpy(qi).to(dev)], Ep, K_F)
    ri, rs = topk(torch, Ep, Es1, K_R)
    log(tag, f"kNN fwd {len(qi):,} x {len(pc_):,}, rev {len(pc_):,} x {len(s1c):,}: {time.time() - t0:.0f}s")
    del Es1, Ep
    torch.cuda.empty_cache()
    f = pd.DataFrame({"s1": np.repeat(s1c[qi], K_F), "cand": pc_[np.clip(fi.ravel(), 0, None)],
                      "cos": fs.ravel(), "rank_f": np.tile(np.arange(K_F), len(qi))})
    f = f[fi.ravel() >= 0]
    r = pd.DataFrame({"s1": s1c[np.clip(ri.ravel(), 0, None)], "cand": np.repeat(pc_, K_R),
                      "cos": rs.ravel(), "rank_r": np.tile(np.arange(K_R), len(pc_))})
    r = r[(ri.ravel() >= 0) & np.isin(s1c[np.clip(ri.ravel(), 0, None)], queries)]
    D = f.merge(r, on=["s1", "cand"], how="outer", suffixes=("", "_r"))
    D["cos"] = D.cos.fillna(D.cos_r).astype(np.float32)
    D = D.drop(columns=["cos_r"])
    D["rank_f"] = D.rank_f.fillna(99).astype(np.int16)
    D["rank_r"] = D.rank_r.fillna(99).astype(np.int16)
    return D


def phase2(k, countries):
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    tag = f"gpu{k}"
    dev = torch.device("cuda:0")
    flags = np.unpackbits(np.load(f"{IN}/bloom.npy"), bitorder="little").astype(bool)
    hold = np.load(f"{IN}/holdout_s1.npy")
    tok = AutoTokenizer.from_pretrained(f"{OUT}/biencoder")
    model = AutoModel.from_pretrained(f"{OUT}/biencoder").to(dev)
    ce_dir = os.environ.get("E047_CE") or os.path.dirname(find("model1/config.json"))
    info = {}
    for split in ("train", "test"):
        R = load_records(split)
        for c in countries:
            if split == "train" and c == "France":
                continue
            cid = {v: kk for kk, v in CTY.items()}[c]
            Rc = R[R.cty == cid]
            Rs1 = Rc[Rc.code // 10**10 == 1].reset_index(drop=True)
            Rpool = Rc[Rc.code // 10**10 != 1].reset_index(drop=True)
            queries = hold if split == "train" else Rs1.code.to_numpy()
            name = "hold" if split == "train" else "test"
            D = retrieve(torch, tok, model, Rs1, Rpool, queries, dev, f"{tag} {name} {c}")
            n_all = len(D)
            D = D[~in_bloom(flags, D.s1.to_numpy(), D.cand.to_numpy())]
            n_new = len(D)
            D = D[(D.rank_f < KEEP_F) | (D.rank_r < KEEP_R)].reset_index(drop=True)
            log(tag, f"{name} {c}: dense pairs {n_all:,}, new {n_new:,}, kept for CE {len(D):,}")
            D.to_parquet(f"{OUT}/dense_{name}_{c}.parquet", index=False)     # checkpoint before CE scoring
            info[f"{name}_{c}"] = dict(dense=n_all, new=n_new, kept=len(D))
            # cross-encoder (LaBSE, E046) on the kept new pairs
            del model
            torch.cuda.empty_cache()
            ctok = AutoTokenizer.from_pretrained(ce_dir)
            cem = AutoModelForSequenceClassification.from_pretrained(ce_dir).float().to(dev)
            txt = pd.Series(Rc.text.to_numpy(), index=Rc.code.to_numpy())
            t0 = time.time()
            D["ce"] = ce_score(torch, ctok, cem, list(txt.reindex(D.s1.to_numpy()).to_numpy()),
                               list(txt.reindex(D.cand.to_numpy()).to_numpy()), dev) if len(D) else np.zeros(0, np.float32)
            log(tag, f"{name} {c}: CE-scored {len(D):,} pairs in {time.time() - t0:.0f}s")
            D.to_parquet(f"{OUT}/dense_{name}_{c}.parquet", index=False)
            del cem
            torch.cuda.empty_cache()
            model = AutoModel.from_pretrained(f"{OUT}/biencoder").to(dev)
            json.dump(info, open(f"{OUT}/log_gpu{k}.json", "w"), indent=1)
        del R
    log(tag, "done", json.dumps(info))


def _run(fn, k, *args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(k)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    fn(*args)


def main():
    import multiprocessing as mp
    print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout, "inputs:", IN, flush=True)
    ctx = mp.get_context("fork")
    if not os.path.exists(f"{OUT}/biencoder/config.json"):
        p = ctx.Process(target=_run, args=(train_biencoder, 0))
        p.start()
        p.join()
        if p.exitcode != 0:
            raise SystemExit(f"bi-encoder training failed ({p.exitcode})")
    if os.environ.get("E047_ONE_GPU") == "1":
        ws = [ctx.Process(target=_run, args=(phase2, 0, 0, ["India", "US", "France"]))]
    else:
        ws = [ctx.Process(target=_run, args=(phase2, 0, 0, ["India"])),
              ctx.Process(target=_run, args=(phase2, 1, 1, ["US", "France"]))]
    for w in ws:
        w.start()
    for w in ws:
        w.join()
    print("worker exit codes", [w.exitcode for w in ws], flush=True)


if __name__ == "__main__":
    main()
