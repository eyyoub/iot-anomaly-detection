"""Metrics for the revision.

Definitions (these go into Section 4.6 of the manuscript):

  AUC-PR      sklearn average_precision_score (step-wise average precision,
              not trapezoidal), computed on all test windows pooled across
              files (micro). Chance level = test prevalence.
  threshold   tau = 95th percentile of the *val* scores (normal-only);
              never fitted on test. y_hat = 1[score >= tau].
  conformal   p(s) = (1 + #{i : s_i >= s}) / (n_cal + 1), s_i = cal scores;
              y_hat = 1[p <= alpha].
  error       E = 1[y_hat != y].
  label-AUROC AUROC(u vs y): the paper's old "AUROC(unc)".
  error-AUROC AUROC(u vs E): does uncertainty flag wrong decisions?
  AURC        area under the risk-coverage curve: sort by u ascending
              (most confident first), risk(k) = error rate of the k most
              confident windows; AURC = mean risk over k. Lower is better.
"""
import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


def auroc(y, s):
    y = np.asarray(y).astype(int)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


def detection(y, scores, y_hat):
    return {
        "auc_pr": float(average_precision_score(y, scores)),
        "auc_pr_chance": float(np.mean(y)),
        "auroc": auroc(y, scores),
        "f1": float(f1_score(y, y_hat, zero_division=0)),
        "precision": float(precision_score(y, y_hat, zero_division=0)),
        "recall": float(recall_score(y, y_hat, zero_division=0)),
        "error_rate": float(np.mean(np.asarray(y) != np.asarray(y_hat))),
        "alarm_rate": float(np.mean(y_hat)),
    }


def val_threshold(val_scores, pct=95):
    return float(np.percentile(val_scores, pct))


def conformal_pvalues(test_scores, cal_scores):
    cal = np.sort(np.asarray(cal_scores))
    n_ge = len(cal) - np.searchsorted(cal, test_scores, side="left")
    return (1.0 + n_ge) / (len(cal) + 1.0)


def conformal_validity(p, y, alphas=(0.01, 0.05, 0.1)):
    """Empirical FPR / coverage on *normal* test windows at each alpha."""
    pn = p[np.asarray(y) == 0]
    out = {}
    for a in alphas:
        fpr = float(np.mean(pn <= a))
        out[f"fpr@{a}"] = fpr
        out[f"coverage@{a}"] = 1.0 - fpr
    return out


def risk_coverage(err, u, n_points=101):
    """Risk-coverage curve, rejecting the most uncertain first."""
    err = np.asarray(err, float)
    order = np.argsort(u, kind="stable")
    risks = np.cumsum(err[order]) / np.arange(1, len(err) + 1)
    grid = np.linspace(0.01, 1.0, n_points)
    idx = np.clip((grid * len(err)).astype(int) - 1, 0, len(err) - 1)
    return float(risks.mean()), grid, risks[idx]


def uncertainty_eval(y, y_hat, signals, seed=0):
    """Evaluate each uncertainty signal (dict name -> array, larger = less sure).

    Adds 'random' (reject at random) and 'oracle' (reject errors first) so
    every AURC can be read against its floor and ceiling.
    """
    y, y_hat = np.asarray(y).astype(int), np.asarray(y_hat).astype(int)
    err = (y != y_hat).astype(int)
    rng = np.random.default_rng(seed)
    sig = dict(signals)
    sig["random"] = rng.random(len(y))
    sig["oracle"] = err + 1e-6 * rng.random(len(y))
    res, curves = {}, {}
    for name, u in sig.items():
        u = np.asarray(u, float)
        aurc, grid, risk = risk_coverage(err, u)
        res[f"{name}/label_auroc"] = auroc(y, u)
        res[f"{name}/error_auroc"] = auroc(err, u)
        res[f"{name}/aurc"] = aurc
        for c in (0.8, 0.9):
            res[f"{name}/risk@{int(c * 100)}"] = float(risk[np.argmin(np.abs(grid - c))])
        # among alarms: are false alarms more uncertain than true alarms?
        m = y_hat == 1
        res[f"{name}/error_auroc_alarms"] = auroc(err[m], u[m]) if m.sum() > 1 else float("nan")
        m = y_hat == 0
        res[f"{name}/error_auroc_silent"] = auroc(err[m], u[m]) if m.sum() > 1 else float("nan")
        curves[name] = risk
    curves["coverage_grid"] = grid
    return res, curves


def margin(scores, thr):
    """Free baseline every detector has: closeness of the score to tau.
    Returned as an uncertainty (larger = closer to the threshold)."""
    return -np.abs(np.log(np.asarray(scores) + 1e-12) - np.log(thr + 1e-12))


def vote_disagreement(pass_scores, thr):
    """MCD decision disagreement: fraction f of the T passes that flag the
    window; uncertainty = f(1 - f), 0 when all passes agree."""
    f = (pass_scores >= thr).mean(axis=1)
    return f * (1 - f)


def file_bootstrap(fn, files, *arrays, n_boot=1000, seed=0, level=0.95):
    """CI by resampling whole files (windows inside a file are dependent)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(files)
    idx_by_file = {f: np.flatnonzero(files == f) for f in uniq}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([idx_by_file[f] for f in pick])
        v = fn(*(a[idx] for a in arrays))
        if not np.isnan(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def _self_test():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    y_hat = np.where(rng.random(4000) < 0.2, 1 - y, y)
    err = (y != y_hat).astype(int)
    r, _ = uncertainty_eval(y, y_hat, {"perfect": err + 0.01 * rng.random(4000),
                                       "noise": rng.random(4000)})
    assert r["perfect/error_auroc"] > 0.99
    assert abs(r["noise/error_auroc"] - 0.5) < 0.05
    assert r["oracle/aurc"] < r["perfect/aurc"] + 1e-3 < r["random/aurc"]
    p = conformal_pvalues(rng.random(20000), rng.random(5000))
    assert abs(conformal_validity(p, np.zeros(20000))["fpr@0.05"] - 0.05) < 0.01
    print("metrics self-test passed")


if __name__ == "__main__":
    _self_test()
