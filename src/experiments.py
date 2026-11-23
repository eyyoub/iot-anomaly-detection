"""One job = one trained model (or one classical baseline) on one dataset.

Each job writes
  results/records/<tag>.json   all scalar metrics (one file per job, so
                               parallel workers never write the same file)
  results/arrays/<tag>.npz     per-window test scores / uncertainties /
                               p-values / labels / file ids (for plots and
                               file-level bootstrap CIs)

How the paper's methods map onto jobs:
  B1  kind=b1       z-score of the raw input (what nb 03 actually computes)
  B2  kind=b2       Isolation Forest on flattened windows
  B3  kind=ae, arch=dense, p=0          -> "det_val/*" metrics
  M3  same job as B3                     -> "det_cp/*"  (conformal on B3's
                                            own scores, no retraining)
  M1  kind=ae, arch=dense, p>0           -> "mc_val/*" (+ "mc_cp/*")
  M2  kind=ae, arch=lstm,  p>0
  M2b kind=ae, arch=lstm,  p in sweep
  M4  kind=ae, arch=lstm,  p=0           -> "det_cp/*"
  M5  kind=ae, arch=densedrop, p in sweep -> det_val (dropout off) + mc_val
"""
import json
import os
import time

import numpy as np

from . import metrics as M
from .data import get_data

ALPHA = 0.05
THR_PCT = 95
T_MC = 50
RESULTS = "results"


def tag_of(job):
    keys = ["kind", "dataset", "W", "arch", "p", "lr", "latent", "epochs",
            "patience", "bs", "seed", "mode"]
    return "_".join(f"{k}{job[k]}" for k in keys if k in job and job[k] is not None)


def _paths(tag):
    return (os.path.join(RESULTS, "records", f"{tag}.json"),
            os.path.join(RESULTS, "arrays", f"{tag}.npz"))


def is_done(job):
    return os.path.exists(_paths(tag_of(job))[0])


def _save(job, record, arrays):
    tag = tag_of(job)
    rec_path, arr_path = _paths(tag)
    os.makedirs(os.path.dirname(rec_path), exist_ok=True)
    os.makedirs(os.path.dirname(arr_path), exist_ok=True)
    if arrays:
        np.savez_compressed(arr_path, **arrays)
    record = {"tag": tag, **job, **record}
    tmp = rec_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(record, f, default=float)
    os.replace(tmp, rec_path)          # atomic: a record exists only if complete
    return record


def _evaluate(prefix, y, test_s, thr, signals, rec, curves, seed):
    """Decision y_hat = 1[test_s >= thr], then detection + uncertainty metrics."""
    y_hat = (test_s >= thr).astype(int)
    for k, v in M.detection(y, test_s, y_hat).items():
        rec[f"{prefix}/{k}"] = v
    rec[f"{prefix}/threshold"] = float(thr)
    sig = {**signals, "margin": M.margin(test_s, thr)}
    r, c = M.uncertainty_eval(y, y_hat, sig, seed=seed)
    rec.update({f"{prefix}/{k}": v for k, v in r.items()})
    curves.update({f"rc_{prefix}/{k}": v for k, v in c.items()})
    return y_hat


def _conformal(prefix, y, test_s, cal_s, signals, rec, curves, seed, arrays):
    """Conformal decision y_hat = 1[p <= alpha] with p from the cal split."""
    p = M.conformal_pvalues(test_s, cal_s)
    rec.update({f"{prefix}/{k}": v for k, v in M.conformal_validity(p, y).items()})
    rec[f"{prefix}/n_cal"] = int(len(cal_s))
    y_hat = (p <= ALPHA).astype(int)
    for k, v in M.detection(y, test_s, y_hat).items():
        rec[f"{prefix}/{k}"] = v
    sig = {**signals,
           "p_margin": -np.abs(np.log(p) - np.log(ALPHA))}
    r, c = M.uncertainty_eval(y, y_hat, sig, seed=seed)
    rec.update({f"{prefix}/{k}": v for k, v in r.items()})
    curves.update({f"rc_{prefix}/{k}": v for k, v in c.items()})
    arrays[f"{prefix}_p"] = p.astype(np.float32)
    arrays[f"{prefix}_yhat"] = y_hat.astype(np.int8)


def run_b1(job, data):
    flat = data["X_train"].reshape(-1, data["X_train"].shape[-1])
    mu, sd = flat.mean(0), flat.std(0) + 1e-8
    score = lambda X: np.abs((X - mu) / sd).mean(axis=(1, 2))
    return score(data["X_val"]), score(data["X_cal"]), score(data["X_test"]), {}


def run_b2(job, data):
    from sklearn.ensemble import IsolationForest
    f = lambda X: X.reshape(len(X), -1)
    iso = IsolationForest(n_estimators=100, contamination="auto",
                          random_state=job["seed"], n_jobs=4).fit(f(data["X_train"]))
    s = lambda X: -iso.score_samples(f(X))
    return s(data["X_val"]), s(data["X_cal"]), s(data["X_test"]), {}


def run_job(job, device="cuda"):
    t0 = time.time()
    data = get_data(job["dataset"], job.get("W"))
    y = data["y_test"]
    rec, curves = {}, {}
    arrays = {"y": y, "file_id": data["file_test"]}
    seed = job.get("seed", 0)
    np.random.seed(seed)

    if job["kind"] in ("b1", "b2"):
        v, c, t, _ = (run_b1 if job["kind"] == "b1" else run_b2)(job, data)
        _evaluate("det_val", y, t, M.val_threshold(v, THR_PCT), {}, rec, curves, seed)
        _conformal("det_cp", y, t, c, {}, rec, curves, seed, arrays)
        arrays["det_score"] = t.astype(np.float32)
        return _save(job, {**rec, "time_s": time.time() - t0}, {**arrays, **curves})

    import torch
    from .models import build, det_scores, mc_scores, train
    torch.manual_seed(seed)
    model = build(job["arch"], data["window_size"], data["n_features"],
                  latent_dim=job.get("latent", 32), dropout_p=job["p"])
    model, hist = train(model, data["X_train"], data["X_val"], lr=job["lr"],
                        epochs=job["epochs"], patience=job["patience"],
                        batch_size=job.get("bs", 256), seed=seed, device=device)
    rec.update({"best_val": hist["best_val"], "n_epochs": hist["epochs"]})

    if job.get("mode") == "grid":        # HP search: val loss is all we need
        return _save(job, {**rec, "time_s": time.time() - t0}, None)

    # Deterministic pass (dropout off): B3 / M3 / M4 / M5-DET
    dv, dc, dt = (det_scores(model, data[f"X_{s}"], device) for s in ("val", "cal", "test"))
    _evaluate("det_val", y, dt, M.val_threshold(dv, THR_PCT), {}, rec, curves, seed)
    _conformal("det_cp", y, dt, dc, {}, rec, curves, seed, arrays)
    arrays["det_score"] = dt.astype(np.float32)

    # MC Dropout: M1 / M2 / M2b / M5-MCD
    if job["p"] > 0:
        mv, mc, mt = (mc_scores(model, data[f"X_{s}"], T_MC, device, seed=seed + i)
                      for i, s in enumerate(("val", "cal", "test")))
        mean_v, mean_c, mean_t = mv.mean(1), mc.mean(1), mt.mean(1)
        std_t = mt.std(1)
        thr = M.val_threshold(mean_v, THR_PCT)
        signals = {"std": std_t,
                   "cv": std_t / (mean_t + 1e-12),
                   "vote": M.vote_disagreement(mt, thr)}
        _evaluate("mc_val", y, mean_t, thr, signals, rec, curves, seed)
        _conformal("mc_cp", y, mean_t, mean_c, signals, rec, curves, seed, arrays)
        # does the uncertainty just track the score?
        from scipy.stats import spearmanr
        rec["mc/spearman_score_std"] = float(spearmanr(mean_t, std_t)[0])
        arrays.update({"mc_score": mean_t.astype(np.float32),
                       "mc_std": std_t.astype(np.float32),
                       "mc_vote": signals["vote"].astype(np.float32)})

    rec["time_s"] = time.time() - t0
    return _save(job, rec, {**arrays, **curves})
