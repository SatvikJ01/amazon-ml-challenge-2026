"""v4 job graph for the high-RAM machine (runner: ``src/dag.py``).

Same modules as the laptop pipeline (v3 + stage 3 E033-E035 features + stage 4),
plus the extra retrieval channels chosen from the E037 pilot and a larger training
sample.  Independent work runs concurrently:

  setup      audit -> prep (-> native-script dictionary, --translit)
  blocking   every (split, country, source) x (forward, reverse, key, extra channels)
  train      s1data per country -> stage 1 -> s2data per country -> stage 2 (LightGBM
             and XGBoost in parallel) -> 5 OOF folds in parallel -> sibling tables,
             anchor retrieval, anchor tables, extra features per country -> stage 3
             -> 5 OOF folds -> stage-4 tables -> stage 4
  test       pass 1 per entity part (parallel, as soon as stage 1+2 exist) -> rescore
             -> anchors per country -> pass 2 per part -> write

Usage:
    python -m src.v4_dag --run v4 --channels name,caddr --entities 1000000 --mem 230 --cpus 32
    python -m src.v4_dag ... --dry-run          # print the graph only
"""
from __future__ import annotations

import argparse
import math
import sys

from .dag import Job, run

PY = [sys.executable, "-u", "-m"]
TRAIN_C, TEST_C = ["India", "US"], ["France", "India", "US"]
TEST_S1 = {"France": 259_452, "India": 809_986, "US": 663_106}   # test S1 counts (fixed by the data)


def build(a) -> list[Job]:
    J: list[Job] = []
    E = a.exp
    sp_tr, sp_te = ("trainT", "testT") if a.translit else ("train", "test")
    tg_tr, tg_te = ("trnT", "tstT") if a.translit else ("trnall", "testall")
    chans = [c for c in a.channels.split(",") if c]
    env = {"V4_CHANNELS": ",".join(chans), "XGB_DEVICE": a.xgb_device}

    def job(name, mod, *args, deps=(), mem=8, th=4):
        J.append(Job(name, PY + [mod, *map(str, args)], list(deps), mem, th, env))
        return name

    # ---- setup ---------------------------------------------------------------------
    setup = [job("audit_gt", "src.audit_gt", mem=16, th=2)]
    setup.append(job("audit_sources", "src.audit_sources", deps=setup, mem=24, th=2))
    setup.append(job("prep", "src.prep", deps=setup[-1:], mem=24, th=4))
    if a.translit:
        setup.append(job("translit_build", "src.translit", "build", deps=["prep"], mem=16, th=2))
        for sp in ("train", "test"):
            setup.append(job(f"translit_{sp}", "src.translit", "apply", "--split", sp, deps=["translit_build"], mem=16, th=2))

    # ---- blocking ------------------------------------------------------------------
    blk: dict[tuple[str, str], list[str]] = {}
    for sp, tg, countries in ((sp_tr, tg_tr, TRAIN_C), (sp_te, tg_te, TEST_C)):
        for c in countries:
            names = []
            for s in (2, 3):
                b = f"{sp}_{c}_s{s}"
                names.append(job(f"fwd_{b}", "src.candidates", "--split", sp, "--tag", tg, "--topk", 30,
                                 "--countries", c, "--sources", s, deps=setup, mem=14, th=4))
                names.append(job(f"rev_{b}", "src.candidates", "--split", sp, "--tag", tg, "--topk", 3,
                                 "--countries", c, "--sources", s, "--reverse", deps=setup, mem=14, th=4))
                names.append(job(f"key_{b}", "src.key_channel", "--split", sp, "--tag", tg, "--country", c,
                                 "--source", s, "--max-s1", 10, "--max-t", 60, deps=setup, mem=8, th=2))
                for ch in chans:
                    if ch == "embed":
                        names.append(job(f"ch_embed_{b}", "src.embed_channel", "--split", sp, "--tag", tg, "--country", c,
                                         "--source", s, "--topk", a.topk, deps=setup, mem=24, th=16))
                    else:
                        names.append(job(f"ch_{ch}_{b}", "src.channels", "--split", sp, "--tag", tg, "--country", c,
                                         "--source", s, "--channel", ch, "--topk", a.topk, deps=setup, mem=14, th=4))
            blk[(sp, c)] = names

    # ---- training chain ------------------------------------------------------------
    P = "data/processed"
    ents = job("entities", "src.v4_entities", "--n", a.entities, "--out", f"{P}/entities_v4.npy", deps=setup, mem=8, th=1)
    s1d = [job(f"s1data_{c}", "src.v3", "s1data", "--split", sp_tr, "--tag", tg_tr, "--prefix", "v4", "--country", c,
               "--entities", f"{P}/entities_v4.npy", deps=blk[(sp_tr, c)] + [ents], mem=48, th=4) for c in TRAIN_C]
    st1 = job("stage1", "src.stage1", "--tag", "v4s1", "--out", f"{E}_stage1", "--v3", "--neg-frac", a.neg_frac,
              deps=s1d, mem=96, th=16)
    s2d = [job(f"s2data_{c}", "src.v3", "s2data", "--split", sp_tr, "--tag", tg_tr, "--prefix", "v4", "--country", c,
               "--stage1", f"{E}_stage1", "--entities", f"{P}/entities_v4.npy", deps=[st1], mem=48, th=8) for c in TRAIN_C]
    st2 = job("stage2", "src.train", "--tag", "v4c", "--exp", f"{E}_stage2", deps=s2d, mem=96, th=16)
    st2x = job("stage2x", "src.train", "--tag", "v4c", "--exp", f"{E}_stage2x", "--model", "xgb", deps=s2d, mem=96, th=16)
    f2 = [job(f"oof2_f{k}", "src.collective", "oof", "--tag", "v4c", "--exp", f"{E}_stage2", "--out", f"{E}_collective",
              "--model", "xgb", "--fold", k, deps=[st2], mem=64, th=6) for k in range(5)]
    m2 = job("oof2_merge", "src.collective", "oof-merge", "--out", f"{E}_collective", deps=f2, mem=16, th=1)
    patched = []
    for c in TRAIN_C:
        sib = job(f"sib_{c}", "src.collective", "build", "--tag", "v4c", "--out", f"{E}_collective", "--country", c,
                  "--split", sp_tr, deps=[m2], mem=40, th=4)
        ar = job(f"ancret_{c}", "src.anchor_pass", "retrieve", "--split", sp_tr, "--country", c, "--oof-dir",
                 f"{E}_collective", "--hits-dir", f"{E}_anchor", deps=[m2], mem=32, th=4)
        ab = job(f"ancbuild_{c}", "src.anchor_pass", "build", "--country", c, "--old-tag", "v4c", "--out-tag", "v4c_anc",
                 "--entities", "entities_v4c.npy", "--hits-dir", f"{E}_anchor", "--train-split", sp_tr,
                 "--train-tag", tg_tr, deps=[sib, ar], mem=64, th=4)
        patched.append(job(f"extra_{c}", "src.patch_extra", "--country", c, "--tag", "v4c_anc", "--out-tag", "v4c_ancz",
                           "--sets", "num,name,ctx", "--split", sp_tr, deps=[ab], mem=48, th=2))
    st3 = job("stage3", "src.train", "--tag", "v4c_ancz", "--exp", f"{E}_stage3", deps=patched, mem=96, th=16)
    final = st3
    if a.stage4:
        f3 = [job(f"oof3_f{k}", "src.collective", "oof", "--tag", "v4c_ancz", "--exp", f"{E}_stage3", "--out", f"{E}_oof3",
                  "--model", "xgb", "--fold", k, deps=[st3], mem=64, th=6) for k in range(5)]
        m3 = job("oof3_merge", "src.collective", "oof-merge", "--out", f"{E}_oof3", deps=f3, mem=16, th=1)
        b4 = [job(f"s4build_{c}", "src.stage4", "build", "--country", c, "--tag", "v4c_ancz", "--out-tag", "v4c_s4",
                  "--oof", f"{E}_oof3", "--split", sp_tr, deps=[m3], mem=48, th=4) for c in TRAIN_C]
        final = job("stage4", "src.train", "--tag", "v4c_s4", "--exp", f"{E}_stage4", deps=b4, mem=96, th=16)

    # ---- test inference --------------------------------------------------------------
    run_dir, common = f"{E}_test", ["--run", f"{E}_test", "--split", sp_te, "--tag", tg_te]
    p2_all = []
    for c in TEST_C:
        parts = math.ceil(TEST_S1[c] / 200_000)
        p1 = [job(f"pass1_{c}_p{i}", "src.infer_v3", "pass1", "--country", c, "--part", i, *common,
                  "--stage1", f"{E}_stage1", "--stage2", f"{E}_stage2", deps=blk[(sp_te, c)] + [st1, st2], mem=16, th=4)
              for i in range(parts)]
        rs = job(f"rescore_{c}", "src.infer_v3", "rescore", "--country", c, *common, "--stage2x", f"{E}_stage2x",
                 deps=p1 + [st2x], mem=12, th=8)
        an = job(f"anchors_{c}", "src.infer_v3", "anchors", "--country", c, *common, deps=[rs], mem=32, th=4)
        extra = ["--stage4", f"{E}_stage4"] if a.stage4 else []
        p2_all += [job(f"pass2_{c}_p{i}", "src.infer_v3", "pass2", "--country", c, "--part", i, *common,
                       "--stage3", f"{E}_stage3", *extra, "--scores", "scores", deps=[an, final], mem=20, th=4)
                   for i in range(parts)]
    job("write", "src.infer_v3", "write", *common, "--scores", "scores", "--sub-id", a.sub_id,
        "--note", f"{E} v4 ({','.join(chans)})", deps=p2_all, mem=24, th=4)
    return J


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v4")
    ap.add_argument("--exp", default="E040")
    ap.add_argument("--channels", default="name,caddr")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--entities", type=int, default=1_000_000)
    ap.add_argument("--neg-frac", type=float, default=0.1)
    ap.add_argument("--translit", action="store_true")
    ap.add_argument("--stage4", action="store_true")
    ap.add_argument("--xgb-device", default="cpu")
    ap.add_argument("--sub-id", default="cloud_v4")
    ap.add_argument("--mem", type=float, default=230)
    ap.add_argument("--cpus", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    jobs = build(a)
    if a.dry_run:
        for j in jobs:
            print(f"{j.name:28s} mem {j.mem_gb:4g} th {j.threads:2d}  deps {len(j.deps):2d}  {' '.join(j.cmd[3:])[:110]}")
        print(f"{len(jobs)} jobs")
        return
    sys.exit(0 if run(jobs, a.run, a.mem, a.cpus) else 1)


if __name__ == "__main__":
    main()
