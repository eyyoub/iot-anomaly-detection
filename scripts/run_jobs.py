"""Run a stage of the experiments, N jobs in parallel on one GPU.

Stages (run in this order):
  smoke   2-epoch versions of every job type, separate results dir
  grid    27-combo HP search for M1 (dense) and M2 (lstm), seed 0, val loss only
  main    B1, B2, B3(+M3), M1, M2, M4 with 5 seeds (needs grid)
  sweeps  M2b (LSTM low dropout) and M5 (dense-dropout) with 3 seeds (needs grid)
  window  window-size sensitivity: B3(+M3) and M1 at other W, 3 seeds (needs grid)

Usage:
  python -m scripts.run_jobs --stage smoke --workers 4
  python -m scripts.run_jobs --stage grid  --workers 8
Finished jobs are skipped, so a stage can be restarted at any time.
"""
import argparse
import itertools
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import experiments as E  # noqa: E402
from src.data import DEFAULT_W, get_data  # noqa: E402

DATASETS = ("SKAB", "NAB")
SEEDS_MAIN = range(5)
SEEDS_SWEEP = range(3)
GRID = {"p": [0.05, 0.10, 0.20], "latent": [16, 32, 64], "lr": [5e-4, 1e-3, 2e-3]}
OTHER_W = {"SKAB": [20, 50, 100], "NAB": [20, 30, 100]}

B3 = dict(kind="ae", arch="dense", p=0.0, lr=1e-3, latent=32, epochs=50, patience=5)
MCD_BUDGET = dict(epochs=100, patience=10)

# REV_FAST=1: tiny version of every stage, to test the notebooks end to end
FAST = os.environ.get("REV_FAST") == "1"
if FAST:
    SEEDS_MAIN = SEEDS_SWEEP = range(2)
    GRID = {"p": [0.05, 0.1], "latent": [32], "lr": [1e-3]}
    OTHER_W = {"SKAB": [20], "NAB": [30]}
    B3 = {**B3, "epochs": 1, "patience": 1}
    MCD_BUDGET = dict(epochs=1, patience=1)


def grid_jobs():
    return [dict(kind="ae", dataset=ds, W=DEFAULT_W[ds], arch=arch, p=p, lr=lr,
                 latent=lat, seed=0, mode="grid", **MCD_BUDGET)
            for ds in DATASETS for arch in ("dense", "lstm")
            for p, lat, lr in itertools.product(GRID["p"], GRID["latent"], GRID["lr"])]


def grid_winners():
    """Lowest val loss per (dataset, arch), same criterion as nb 04/05."""
    E.RESULTS = os.environ.get("REV_RESULTS", E.RESULTS)
    best = {}
    for job in grid_jobs():
        path = os.path.join(E.RESULTS, "records", f"{E.tag_of(job)}.json")
        if not os.path.exists(path):
            sys.exit(f"grid not finished: missing {path}. Run --stage grid first.")
        with open(path) as f:
            r = json.load(f)
        k = (job["dataset"], job["arch"])
        if k not in best or r["best_val"] < best[k]["best_val"]:
            best[k] = r
    return {k: {"p": v["p"], "lr": v["lr"], "latent": v["latent"]} for k, v in best.items()}


def main_jobs(win):
    jobs = []
    for ds in DATASETS:
        W = DEFAULT_W[ds]
        jobs.append(dict(kind="b1", dataset=ds, W=W, seed=0))
        for s in SEEDS_MAIN:
            jobs.append(dict(kind="b2", dataset=ds, W=W, seed=s))
            jobs.append(dict(dataset=ds, W=W, seed=s, **B3))                       # B3 + M3
            jobs.append(dict(kind="ae", dataset=ds, W=W, arch="dense", seed=s,
                             **win[(ds, "dense")], **MCD_BUDGET))                 # M1
            jobs.append(dict(kind="ae", dataset=ds, W=W, arch="lstm", seed=s,
                             **win[(ds, "lstm")], **MCD_BUDGET))                  # M2
            jobs.append(dict(kind="ae", dataset=ds, W=W, arch="lstm", seed=s,
                             **{**win[(ds, "lstm")], "p": 0.0}, **MCD_BUDGET))    # M4
    return jobs


def sweep_jobs(win):
    jobs = []
    for ds in DATASETS:
        W = DEFAULT_W[ds]
        lw = win[(ds, "lstm")]
        for p, s in itertools.product([0.001, 0.005, 0.01, 0.02, 0.05], SEEDS_SWEEP):
            jobs.append(dict(kind="ae", dataset=ds, W=W, arch="lstm", p=p, lr=lw["lr"],
                             latent=lw["latent"], seed=s, **MCD_BUDGET))          # M2b
        for p, s in itertools.product([0.05, 0.1, 0.2, 0.3, 0.5], SEEDS_SWEEP):
            jobs.append(dict(kind="ae", dataset=ds, W=W, arch="densedrop", p=p,
                             lr=1e-3, latent=64, bs=512, seed=s, **MCD_BUDGET))   # M5
    return jobs


def window_jobs(win):
    jobs = []
    for ds in DATASETS:
        for W, s in itertools.product(OTHER_W[ds], SEEDS_SWEEP):
            jobs.append(dict(dataset=ds, W=W, seed=s, **B3))
            jobs.append(dict(kind="ae", dataset=ds, W=W, arch="dense", seed=s,
                             **win[(ds, "dense")], **MCD_BUDGET))
    return jobs


def smoke_jobs():
    short = dict(epochs=2, patience=1)
    jobs = []
    for ds in DATASETS:
        W = DEFAULT_W[ds]
        jobs += [dict(kind="b1", dataset=ds, W=W, seed=0),
                 dict(kind="b2", dataset=ds, W=W, seed=0),
                 dict(kind="ae", dataset=ds, W=W, arch="dense", p=0.0, lr=1e-3, latent=32, seed=0, **short),
                 dict(kind="ae", dataset=ds, W=W, arch="dense", p=0.05, lr=1e-3, latent=32, seed=0, **short),
                 dict(kind="ae", dataset=ds, W=W, arch="lstm", p=0.05, lr=1e-3, latent=32, seed=0, **short),
                 dict(kind="ae", dataset=ds, W=W, arch="densedrop", p=0.5, lr=1e-3, latent=64, bs=512, seed=0, **short)]
    return jobs


def stage_tags(stage):
    """Tags of every job of a stage (used by the notebooks to pick their rows)."""
    return {E.tag_of(j) for j in list_jobs(stage, quiet=True)}


def _init(results, t_mc):
    E.RESULTS, E.T_MC = results, t_mc
    import torch
    torch.set_num_threads(2)          # 8 workers x 2 threads on a 16-vCPU pod
    torch.backends.cudnn.benchmark = True


def _work(job, device):
    tag = E.tag_of(job)
    try:
        t0 = time.time()
        E.run_job(job, device=device)
        return tag, None, time.time() - t0
    except Exception:
        err_dir = os.path.join(E.RESULTS, "errors")
        os.makedirs(err_dir, exist_ok=True)
        tb = traceback.format_exc()
        with open(os.path.join(err_dir, f"{tag}.txt"), "w") as f:
            f.write(tb)
        return tag, tb.strip().splitlines()[-1], 0.0


def list_jobs(stage, quiet=False):
    if stage == "smoke":
        return smoke_jobs()
    if stage == "grid":
        return grid_jobs()
    win = grid_winners()
    if not quiet:
        print("grid winners:", {f"{k[0]}/{k[1]}": v for k, v in win.items()})
    return {"main": main_jobs, "sweeps": sweep_jobs, "window": window_jobs}[stage](win)


def run_stage(stage, workers=4, device=None, results=None, dry=False):
    """Run every unfinished job of a stage. Returns the list of failed tags."""
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    E.RESULTS = results or ("results_smoke" if stage == "smoke" else "results")
    t_mc = 5 if (stage == "smoke" or FAST) else E.T_MC
    jobs = list_jobs(stage)

    # dedupe (e.g. M2b p=0.05 == M2 when the grid picks p=0.05) and skip done
    uniq = {E.tag_of(j): j for j in jobs}
    todo = [j for j in uniq.values() if not E.is_done(j)]
    # longest first: LSTM on NAB dominates the wall-clock time
    todo.sort(key=lambda j: (j.get("arch") != "lstm", j["dataset"] != "NAB"))
    print(f"stage={stage}  jobs={len(uniq)}  todo={len(todo)}  "
          f"workers={workers}  device={device}  results={E.RESULTS}", flush=True)
    if dry or not todo:
        for j in todo:
            print(" ", E.tag_of(j))
        return []

    for ds, W in sorted({(j["dataset"], j.get("W")) for j in todo}):
        get_data(ds, W)          # build caches once, before workers start

    t0, failed = time.time(), []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init,
                             initargs=(E.RESULTS, t_mc)) as ex:
        futs = [ex.submit(_work, j, device) for j in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            tag, err, dt = fut.result()
            status = f"FAILED: {err}" if err else f"{dt:6.0f}s"
            if err:
                failed.append(tag)
            print(f"[{i}/{len(todo)}  {(time.time() - t0) / 60:5.1f} min] {tag}  {status}",
                  flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min, {len(failed)} failed", flush=True)
    if failed:
        print("tracebacks in", os.path.join(E.RESULTS, "errors"))
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["smoke", "grid", "main", "sweeps", "window"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--results", default=None)
    ap.add_argument("--dry", action="store_true", help="list jobs and exit")
    a = ap.parse_args()
    run_stage(a.stage, a.workers, a.device, a.results, a.dry)


if __name__ == "__main__":
    main()
