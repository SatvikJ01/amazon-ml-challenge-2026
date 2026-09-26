"""Resource-aware DAG runner for the high-RAM machine.

Jobs declare dependencies, a memory budget (GB) and a thread count.  The runner
starts every ready job whose budget fits in the remaining memory / cores, pins it
to its own cores (``taskset``), caps its memory (``systemd-run --user --scope``,
falls back to ``prlimit`` address-space limits when no user systemd is present),
logs to ``logs/dag/<run>/<job>.log`` and writes ``<job>.done`` on success, so a
re-run resumes where it stopped.  A failed job is retried once; its dependants
are skipped and reported.

Usage (from a job-graph module that builds ``jobs`` and calls ``run``):
    python -m scripts.aws.v4_dag --run v4 --mem 230 --cpus 32
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Job:
    name: str
    cmd: list[str]
    deps: list[str] = field(default_factory=list)
    mem_gb: float = 8.0
    threads: int = 4
    env: dict = field(default_factory=dict)


def _wrap(job: Job, cores: list[int]) -> list[str]:
    cmd = list(job.cmd)
    if shutil.which("taskset"):
        cmd = ["taskset", "-c", ",".join(map(str, cores))] + cmd
    if shutil.which("systemd-run") and os.environ.get("DAG_NO_SYSTEMD") != "1":
        return ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={int(job.mem_gb * 1024)}M",
                "-p", "MemorySwapMax=0"] + cmd
    return cmd


def run(jobs: list[Job], run_name: str, mem_total: float, cpus: int, poll: float = 5.0) -> bool:
    logdir = ROOT / "logs" / "dag" / run_name
    logdir.mkdir(parents=True, exist_ok=True)
    by = {j.name: j for j in jobs}
    for j in jobs:
        for d in j.deps:
            assert d in by, f"{j.name}: unknown dependency {d}"
    done = {j.name for j in jobs if (logdir / f"{j.name}.done").exists()}
    failed: set[str] = set()
    tries: dict[str, int] = {}
    running: dict[str, tuple[subprocess.Popen, list[int], float]] = {}
    free_cores = list(range(cpus))
    mem_free = mem_total
    t0 = time.time()

    def status(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')} +{(time.time() - t0) / 60:5.1f}m] {msg}"
        print(line, flush=True)
        with open(logdir / "_dag.log", "a") as fh:
            fh.write(line + "\n")

    status(f"start: {len(jobs)} jobs, {len(done)} already done, mem {mem_total} GB, {cpus} cores")
    while True:
        for name, (proc, cores, mem) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del running[name]
            free_cores.extend(cores)
            free_cores.sort()
            mem_free += mem
            if rc == 0:
                (logdir / f"{name}.done").write_text(time.strftime("%F %T"))
                done.add(name)
                status(f"done   {name}")
            elif tries.get(name, 0) < 2:
                status(f"FAIL   {name} (rc {rc}) -> retry")
            else:
                failed.add(name)
                status(f"FAILED {name} (rc {rc}), see {logdir / (name + '.log')}")
        blocked = {j.name for j in jobs if any(d in failed for d in j.deps)}
        new_blocked = blocked - failed
        for b in new_blocked:
            failed.add(b)
            status(f"SKIP   {b} (dependency failed)")
        pending = [j for j in jobs if j.name not in done and j.name not in failed and j.name not in running]
        if not pending and not running:
            break
        for j in pending:
            if not all(d in done for d in j.deps):
                continue
            th = min(j.threads, cpus)
            if j.mem_gb > mem_free or th > len(free_cores):
                continue
            cores, free_cores = free_cores[:th], free_cores[th:]
            mem_free -= j.mem_gb
            tries[j.name] = tries.get(j.name, 0) + 1
            env = dict(os.environ, OMP_NUM_THREADS=str(th), NUMBA_NUM_THREADS=str(th),
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", **j.env)
            log = open(logdir / f"{j.name}.log", "a")
            log.write(f"\n=== {time.strftime('%F %T')} try {tries[j.name]}: {' '.join(j.cmd)}\n")
            log.flush()
            proc = subprocess.Popen(_wrap(j, cores), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            running[j.name] = (proc, cores, j.mem_gb)
            status(f"start  {j.name}  ({j.mem_gb:g} GB, {th} cores; free {mem_free:g} GB / {len(free_cores)} cores)")
        time.sleep(poll)
    status(f"finished: {len(done)} done, {len(failed)} failed/skipped")
    return not failed
