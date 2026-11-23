"""Turn results/records/*.json into the tables and figures of the revision.

All new or changed figures are saved as figures_rev/REV_*.png so they stand
out from the figures of the submitted version.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

FIG_DIR = "figures_rev"


def load(results="results"):
    rows = [json.load(open(p)) for p in glob.glob(os.path.join(results, "records", "*.json"))]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["p"] = df.get("p", np.nan)
    return df


def method_of(r):
    """Name of the paper method a record belongs to (decision views differ)."""
    if r["kind"] == "b1":
        return "B1 Z-score"
    if r["kind"] == "b2":
        return "B2 IsoForest"
    if r.get("mode") == "grid":
        return "grid"
    if r["arch"] == "dense":
        return "B3 Dense AE" if r["p"] == 0 else "M1 Dense+MCD"
    if r["arch"] == "lstm":
        return "LSTM AE (det.)" if r["p"] == 0 else "M2 LSTM+MCD"
    if r["arch"] == "densedrop":
        return "M5 DenseDropout"
    return "?"


def views(df):
    """One row per (method view, record): which metric prefix answers which
    paper method. E.g. a B3 record yields 'B3' via det_val and 'M3' via det_cp."""
    out = []
    for _, r in df.iterrows():
        m = method_of(r)
        if m == "grid":
            continue
        base = {"dataset": r["dataset"], "W": r["W"], "seed": r["seed"],
                "p": r.get("p"), "tag": r["tag"]}
        pairs = []
        if m in ("B1 Z-score", "B2 IsoForest"):
            pairs = [(m, "det_val"), (m + " + CP", "det_cp")]
        elif m == "B3 Dense AE":
            pairs = [("B3 Dense AE", "det_val"), ("M3 Dense+CP", "det_cp")]
        elif m == "LSTM AE (det.)":
            pairs = [("LSTM AE (det.)", "det_val"), ("M4 LSTM+CP", "det_cp")]
        elif m in ("M1 Dense+MCD", "M2 LSTM+MCD", "M5 DenseDropout"):
            pairs = [(m, "mc_val"), (m + " + CP", "mc_cp")]
            if m == "M5 DenseDropout":
                pairs.append(("M5 DenseDropout (DET)", "det_val"))
        for name, pre in pairs:
            row = {**base, "method": name, "view": pre}
            for k, v in r.items():
                if isinstance(k, str) and k.startswith(pre + "/"):
                    row[k[len(pre) + 1:]] = v
            out.append(row)
    return pd.DataFrame(out)


def mean_std(df, cols, by=("dataset", "method"), digits=3):
    """'mean ± std (n)' per group, the format used in all revised tables."""
    g = df.groupby(list(by))
    m, s, n = g[cols].mean(), g[cols].std(ddof=1), g[cols].count()
    fmt = lambda a, b, c: (f"{a:.{digits}f}" if c < 2 or np.isnan(b)
                           else f"{a:.{digits}f} ± {b:.{digits}f}")
    out = m.copy().astype(object)
    for c in cols:
        out[c] = [fmt(a, b, k) for a, b, k in zip(m[c], s[c], n[c])]
    out["n_seeds"] = n[cols[0]]
    return out


def save_fig(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    if not name.startswith("REV_"):
        name = "REV_" + name
    path = os.path.join(FIG_DIR, name if name.endswith(".png") else name + ".png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path


def risk_coverage_curves(results, tag, view, signals):
    z = np.load(os.path.join(results, "arrays", f"{tag}.npz"))
    grid = z[f"rc_{view}/coverage_grid"]
    return grid, {s: z[f"rc_{view}/{s}"] for s in signals if f"rc_{view}/{s}" in z}
