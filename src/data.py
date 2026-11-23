"""Revised data protocol (review-fix): leak-free per-file chronological splits
into train / val / cal / test.

What changed vs notebooks 01/02 (the protocol used in the submitted paper):
  * Files are never concatenated before splitting; every file is cut by time.
  * Anomalies are never moved. A window keeps the period it occurs in.
    Train/val/cal windows that contain an anomaly are dropped (models train
    on normal data only); test keeps every window of its period.
  * A separate calibration split (cal) is used only for conformal
    prediction. Early stopping and model selection use val only.
  * The scaler is fitted per file on that file's train rows.
  * Windows never cross split or file boundaries.
  * Every window keeps its source file id (for file-level bootstrap CIs).

SKAB: each experiment file is ~570 normal rows, then one fault (~400 rows),
then a normal tail. A naive "last 20% of the file" cut would leave most of
the fault in the training period. So the pre-fault normal prefix is split
train/val/cal/test-head by `skab_ratios` (default 60/10/10/20), and the
test split is the test-head plus everything from the fault onward.
anomaly-free.csv is split chronologically by the same ratios.

NAB: each file is split chronologically by `nab_ratios` (default 40/10/10/40,
see README_REVIEW_FIX.md for why not 60/10/10/20).
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

SKAB_ROOT = "SKAB/data"
NAB_ROOT = "NAB"
SKAB_FEATURES = ["Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure",
                 "Temperature", "Thermocouple", "Voltage", "Volume Flow RateRMS"]
SPLITS = ("train", "val", "cal", "test")


def sliding_window(X, y, window_size, step=1):
    """(n, f) -> (n_win, W, f); window label = 1 if any step is anomalous."""
    n = len(X)
    if n < window_size:
        return (np.empty((0, window_size, X.shape[1]), np.float32),
                np.empty((0,), np.int8))
    idx = np.arange(0, n - window_size + 1, step)[:, None] + np.arange(window_size)
    return X[idx].astype(np.float32), y[idx].any(axis=1).astype(np.int8)


def _cuts(n, ratios):
    """Row indices where train|val|cal|test begin/end for a prefix of length n."""
    c = np.cumsum(ratios)[:3]
    return [0, *[int(round(r * n)) for r in c]]


def split_file(X, y, cuts, window_size):
    """Split one file at row cuts [0, a, b, c] -> train [0,a), val [a,b),
    cal [b,c), test [c, end). Returns {split: (X_win, y_win)}."""
    a, b, c = cuts[1], cuts[2], cuts[3]
    fit_rows = X[:a][y[:a] == 0]
    sc = MinMaxScaler().fit(fit_rows if len(fit_rows) else X[:a])
    Xs = sc.transform(X)
    bounds = {"train": (0, a), "val": (a, b), "cal": (b, c), "test": (c, len(X))}
    out = {}
    for name, (lo, hi) in bounds.items():
        Xw, yw = sliding_window(Xs[lo:hi], y[lo:hi], window_size)
        if name != "test":
            keep = yw == 0
            Xw, yw = Xw[keep], yw[keep]
        out[name] = (Xw, yw)
    return out


def _read_skab(path):
    df = pd.read_csv(path, sep=";", parse_dates=["datetime"]).sort_values("datetime")
    X = df[SKAB_FEATURES].to_numpy(float)
    y = df["anomaly"].to_numpy(int) if "anomaly" in df else np.zeros(len(df), int)
    return X, y


def load_skab(window_size=30, ratios=(0.6, 0.1, 0.1, 0.2), root=SKAB_ROOT):
    per_file = {}
    for path in sorted(glob.glob(os.path.join(root, "*", "*.csv"))):
        group = os.path.basename(os.path.dirname(path))
        X, y = _read_skab(path)
        if y.any():
            prefix = int(np.flatnonzero(y)[0])      # rows before the fault
            cuts = _cuts(prefix, ratios)
        else:
            cuts = _cuts(len(X), ratios)
        key = f"{group}/{os.path.basename(path)}"
        per_file[key] = {**split_file(X, y, cuts, window_size), "group": group}
    return _assemble(per_file, "SKAB", window_size, {"ratios": list(ratios)})


def _read_nab(root, key, wins):
    df = pd.read_csv(os.path.join(root, "data", key), parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    y = np.zeros(len(df), int)
    for s, e in wins:
        y[(df.timestamp >= pd.Timestamp(s)) & (df.timestamp <= pd.Timestamp(e))] = 1
    return df[["value"]].to_numpy(float), y


def load_nab(window_size=50, ratios=(0.4, 0.1, 0.1, 0.4), root=NAB_ROOT):
    with open(os.path.join(root, "labels", "combined_windows.json")) as f:
        windows = json.load(f)
    per_file, skipped = {}, []
    for key, wins in sorted(windows.items()):
        X, y = _read_nab(root, key, wins)
        cuts = _cuts(len(X), ratios)
        if (y[:cuts[1]] == 0).sum() < window_size:
            skipped.append(key)
            continue
        per_file[key] = {**split_file(X, y, cuts, window_size),
                         "group": key.split("/")[0]}
    return _assemble(per_file, "NAB", window_size,
                     {"ratios": list(ratios), "skipped_files": skipped})


def _assemble(per_file, name, window_size, meta):
    data = {"name": name, "window_size": window_size, **meta}
    for s in SPLITS:
        Xs, ys, fs = [], [], []
        for f, d in per_file.items():
            Xw, yw = d[s]
            if len(Xw):
                Xs.append(Xw); ys.append(yw); fs.append(np.full(len(yw), f))
        data[f"X_{s}"] = np.concatenate(Xs)
        data[f"y_{s}"] = np.concatenate(ys)
        data[f"file_{s}"] = np.concatenate(fs)
    groups = {f: d["group"] for f, d in per_file.items()}
    data["group_test"] = np.array([groups[f] for f in data["file_test"]])
    data["n_features"] = data["X_train"].shape[2]
    check(data)
    return data


def check(data):
    for s in ("train", "val", "cal"):
        assert data[f"y_{s}"].sum() == 0, f"anomalous window in {s}"
    for s in SPLITS:
        n = len(data[f"X_{s}"])
        assert n == len(data[f"y_{s}"]) == len(data[f"file_{s}"]) and n > 0, s
    assert data["y_test"].sum() > 0, "no anomalies in test"


def summary(data):
    rows = []
    for s in SPLITS:
        y = data[f"y_{s}"]
        rows.append({"split": s, "windows": len(y), "anomalous": int(y.sum()),
                     "prevalence": round(float(y.mean()), 4),
                     "files": len(set(data[f"file_{s}"]))})
    return pd.DataFrame(rows)


# ── Cache: build once, load fast in every worker ─────────────────────────────
LOADERS = {"SKAB": load_skab, "NAB": load_nab}
DEFAULT_W = {"SKAB": 30, "NAB": 50}


def cache_path(name, window_size, cache_dir="cache"):
    return os.path.join(cache_dir, f"{name}_W{window_size}.npz")


def get_data(name, window_size=None, cache_dir="cache", rebuild=False):
    window_size = window_size or DEFAULT_W[name]
    path = cache_path(name, window_size, cache_dir)
    if os.path.exists(path) and not rebuild:
        z = np.load(path, allow_pickle=True)
        data = {k: z[k] for k in z.files}
        for k in ("name",):
            data[k] = str(data[k])
        for k in ("window_size", "n_features"):
            data[k] = int(data[k])
        return data
    data = LOADERS[name](window_size=window_size)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(path, **{k: (np.asarray(v, dtype=object)
                                     if isinstance(v, list) else v)
                                 for k, v in data.items()})
    return data
