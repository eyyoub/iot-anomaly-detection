"""Generate notebooks R01-R06.  python -m scripts.make_notebooks

Edit this file, not the notebooks. Heavy code lives in src/.
"""
import nbformat as nbf

SETUP = '''# ── Settings ──────────────────────────────────────────────────────────────
import os
WORKERS = int(os.environ.get("REV_WORKERS", 8))   # training jobs side by side on the GPU
DEVICE  = None     # None = use the GPU if there is one
RESULTS = os.environ.get("REV_RESULTS", "results")
RUN     = os.environ.get("REV_RUN", "1") == "1"   # False = skip training, rebuild tables/figures from results/

import warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display, Markdown
from src import report, plots
from scripts.run_jobs import run_stage, stage_tags
warnings.filterwarnings("ignore")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
os.makedirs("tables_rev", exist_ok=True)

def export(df, name):
    """Save a table as tables_rev/REV_<name>.csv and show it."""
    path = f"tables_rev/REV_{name}.csv"; df.to_csv(path); display(df); print("saved", path)'''

LOAD = '''df = report.load(RESULTS)
V = report.views(df)
print(len(df), "records")'''


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


def save(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, name)
    print("wrote", name)


# ─────────────────────────────────────────────────────────────────────────────
R01 = [
    md("""
# R01 · Data splits

Each file is split **chronologically on its own** into train / val / cal / test:

* Anomalies stay in the period where they occur. Windows with an anomaly are dropped from train, val and cal,
  so the models train on normal data only.
* The **cal** split is used only to calibrate conformal prediction; early stopping uses val.
* The MinMax scaler is fitted per file on that file's train rows.
* Windows never cross a split or file boundary.

**SKAB:** every experiment file is ≈570 normal rows → one fault (≈400 rows) → a normal tail. The *normal
prefix* before the fault is split 60/10/10/20 (train/val/cal/test-head); the test split is the test-head
plus everything from the fault onward. `anomaly-free.csv` is split 60/10/10/20 by time.

**NAB:** each file split 40/10/10/40 by time (the table below shows why not 60/10/10/20).
"""),
    code(SETUP),
    code('''from src.data import get_data, summary, load_nab
data = {ds: get_data(ds, rebuild=True) for ds in ("SKAB", "NAB")}
summ = {ds: summary(d) for ds, d in data.items()}
for ds, s in summ.items():
    display(Markdown(f"**{ds}** (W={data[ds]['window_size']})")); display(s)'''),
    md("""
### Why NAB uses 40/10/10/40 and not 60/10/10/20
NAB anomalies are spread over the whole series. With a 60/10/10/20 cut, most labelled anomalies fall in the
training period (where their windows are dropped) and many files have no anomaly in test at all.
"""),
    code('''opts = []
for r in [(0.6,0.1,0.1,0.2), (0.5,0.1,0.1,0.3), (0.4,0.1,0.1,0.4)]:
    d = load_nab(ratios=r)
    ft = pd.Series(d["y_test"]).groupby(d["file_test"]).max()
    opts.append({"ratios": "/".join(str(int(x*100)) for x in r), "test windows": len(d["y_test"]),
                 "test prevalence": round(d["y_test"].mean(), 3),
                 "files with a test anomaly": f"{int(ft.sum())}/{len(ft)}"})
export(pd.DataFrame(opts).set_index("ratios"), "nab_split_options")'''),
    md("### Per-file view (SKAB)\nEvery file contributes train/val/cal windows from its own normal prefix, and its own fault to test."),
    code('''d = data["SKAB"]
pf = pd.DataFrame({s: pd.Series(d[f"file_{s}"]).value_counts() for s in ("train","val","cal","test")}).fillna(0).astype(int)
pf["test anomalous"] = pd.Series(d["y_test"]).groupby(d["file_test"]).sum()
display(pf.sort_index())'''),
    md("### Split composition (Figure 1)"),
    code('''fig, path = plots.split_composition(summ); plt.show(); print("saved", path)'''),
    md("### Checks"),
    code('''from src.data import check
for ds, d in data.items():
    check(d)                                   # no anomalous window in train/val/cal, nothing empty
    assert set(d["file_cal"]) <= set(d["file_train"]) | set(d["file_val"]) | set(d["file_test"])
print("all split checks passed")'''),
]

# ─────────────────────────────────────────────────────────────────────────────
R02 = [
    md("""
# R02 · Hyperparameter grid search (M1 Dense+MCD, M2 LSTM+MCD)

Grid searched for the two MC Dropout models:

* learning rate ∈ {5e-4, 1e-3, 2e-3}, latent dimension ∈ {16, 32, 64}, dropout p ∈ {0.05, 0.10, 0.20}
* 27 combinations × 2 architectures × 2 datasets = 108 runs, seed 0, up to 100 epochs, patience 10.
* **Selection criterion: lowest validation reconstruction loss** . It uses
  no labels and never touches the calibration or test split.

Why grid search: the space is small (27 points), so an exhaustive search is cheap, fully reproducible, and
gives every method the same tuning budget, unlike random or Bayesian search, which add their own randomness.
"""),
    code(SETUP),
    code('''if RUN:
    failed = run_stage("grid", workers=WORKERS, device=DEVICE, results=RESULTS)'''),
    code('''df = report.load(RESULTS)
g = df[df["mode"] == "grid"][["dataset","arch","p","latent","lr","best_val","n_epochs","time_s"]]
g = g.sort_values(["dataset","arch","best_val"])
for (ds, arch), t in g.groupby(["dataset","arch"]):
    display(Markdown(f"**{ds} · {arch}** (best first)")); display(t.head(10).reset_index(drop=True))
export(g.reset_index(drop=True), "grid_search_all")'''),
    md("### Selected configurations (used by R03–R05)"),
    code('''from scripts.run_jobs import grid_winners
win = pd.DataFrame(grid_winners()).T
win.index.names = ["dataset","arch"]
export(win, "grid_winners")'''),
    md("""
**Note:** selecting by validation loss favours the smallest dropout rate, because dropout adds noise to the
reconstruction. This is why p = 0.05 is selected every time.
"""),
]

# ─────────────────────────────────────────────────────────────────────────────
R03 = [
    md("""
# R03 · Main experiments: detection and uncertainty against **errors**, 5 seeds

Uncertainty is evaluated both against the anomaly labels and against the **errors** of the detector, to see
whether it identifies unreliable predictions. Each model is trained with 5 seeds (0–4). Each B3 run also gives M3 (conformal on **the same** B3 scores, no
retraining). Each deterministic LSTM run gives M4.

### How to read the uncertainty columns
* **label-AUROC**: does uncertainty separate anomalies from normal windows?
* **error-AUROC**: does uncertainty separate **wrong** decisions from **right** ones? 0.5 = useless.
* **AURC**: average error rate when you keep only the most confident windows, over all coverage levels.
  Lower is better. Compare with **random** (reject at random) and **oracle** (reject exactly the errors).
* **margin**: a free baseline every detector already has: how close the score is to the threshold.
  An uncertainty method is only useful if it beats this.
* **vote**: MCD decision disagreement: share of the 50 dropout passes that flag the window, f·(1−f).
"""),
    code(SETUP),
    code('''if RUN:
    failed = run_stage("main", workers=WORKERS, device=DEVICE, results=RESULTS)'''),
    code(LOAD),
    md("## Table A · Detection performance (mean ± std over seeds)"),
    code('''MAIN = ["B1 Z-score","B2 IsoForest","B3 Dense AE","M1 Dense+MCD","M2 LSTM+MCD",
        "LSTM AE (det.)","M3 Dense+CP","M4 LSTM+CP"]
V = V[V.tag.isin(stage_tags("main"))]          # only the runs of this stage
Vm = V[V.method.isin(MAIN)]
ta = report.mean_std(Vm, ["auc_pr","auroc","f1","precision","recall"])
ta["AUC-PR chance"] = Vm.groupby(["dataset","method"])["auc_pr_chance"].first().round(3)
export(ta, "tableA_detection")'''),
    md("## Table B · Uncertainty quality: label-AUROC vs error-AUROC"),
    code('''U = V[V.method.isin(["M1 Dense+MCD","M2 LSTM+MCD"])]
cols = [c for c in ["std/label_auroc","std/error_auroc","vote/error_auroc","cv/error_auroc",
                    "margin/error_auroc"] if c in U]
export(report.mean_std(U, cols), "tableB_uncertainty_auroc")'''),
    md("## Table C · Selective prediction: AURC (lower = better)"),
    code('''cols = ["std/aurc","vote/aurc","margin/aurc","random/aurc","oracle/aurc"]
export(report.mean_std(U, cols, digits=4), "tableC_aurc")'''),
    md("## Table D · Conformal prediction: false-alarm rate on **normal** test windows (target α = 0.05)"),
    code('''CPM = ["M3 Dense+CP","M4 LSTM+CP","M1 Dense+MCD + CP","M2 LSTM+MCD + CP"]
C_ = V[V.method.isin(CPM)]
export(report.mean_std(C_, ["fpr@0.01","fpr@0.05","fpr@0.1","coverage@0.05","n_cal"]), "tableD_conformal")'''),
    md("## Figures"),
    code('''def first_tag(method, ds, seed=0):
    t = V[(V.method == method) & (V.dataset == ds) & (V.seed == seed)]
    return t.iloc[0]["tag"]
for ds in ("SKAB", "NAB"):
    items = [(f"{ds} · M1 Dense+MCD", first_tag("M1 Dense+MCD", ds), "mc_val", ["std","vote","margin"]),
             (f"{ds} · M2 LSTM+MCD",  first_tag("M2 LSTM+MCD", ds),  "mc_val", ["std","vote","margin"])]
    fig, path = plots.risk_coverage(RESULTS, items, f"REV_risk_coverage_{ds}"); plt.show(); print("saved", path)'''),
    code('''for ds in ("SKAB", "NAB"):
    t = U[U.dataset == ds].groupby("method")[["std/label_auroc","std/error_auroc","margin/error_auroc"]].agg(["mean","std"])
    t.columns = [f"{a}_{b}" for a, b in t.columns]; t = t.reset_index()
    fig, path = plots.dot_compare(t, ["std/label_auroc","std/error_auroc","margin/error_auroc"],
                                  ["MCD std vs anomaly labels","MCD std vs errors",
                                   "Score margin vs errors"],
                                  f"REV_label_vs_error_auroc_{ds}")
    plt.show(); print("saved", path)'''),
    code('''t = C_.groupby(["dataset","method"])["fpr@0.05"].agg(["mean","std"]).reset_index()
t.columns = ["dataset","method","fpr_mean","fpr_std"]
fig, path = plots.conformal_fpr(t); plt.show(); print("saved", path)'''),
]

# ─────────────────────────────────────────────────────────────────────────────
R04 = [
    md("""
# R04 · Dropout-rate sweeps (M2b LSTM, M5 Dense-Dropout), 3 seeds each

* **M2b:** LSTM+MCD with p ∈ {0.001, 0.005, 0.01, 0.02, 0.05}, using the M2 learning rate / latent size from R02.
* **M5:** single-dropout dense model, p ∈ {0.05, 0.1, 0.2, 0.3, 0.5}, dropout OFF (DET) and MCD inference.

Same metrics as R03. Each point is the mean over 3 seeds; the band shows the spread.
"""),
    code(SETUP),
    code('''if RUN:
    failed = run_stage("sweeps", workers=WORKERS, device=DEVICE, results=RESULTS)'''),
    code(LOAD),
    code('''V = V[V.tag.isin(stage_tags("sweeps"))]        # only the runs of this stage
lstm = V[V.method == "M2 LSTM+MCD"]
cols = ["auc_pr","std/label_auroc","std/error_auroc","margin/error_auroc","std/aurc","margin/aurc"]
export(report.mean_std(lstm, cols, by=("dataset","p")), "tableE_M2b_sweep")
fig, path = plots.sweep_lines(lstm, "p", ["auc_pr","std/label_auroc","std/error_auroc"],
                              ["AUC-PR","MCD std vs anomaly labels","MCD std vs errors"],
                              "REV_M2b_dropout_sweep", logx=True, ref=0.5)
plt.show(); print("saved", path)'''),
    code('''m5 = V[V.method == "M5 DenseDropout"]
m5d = V[V.method == "M5 DenseDropout (DET)"]
t = report.mean_std(m5, cols, by=("dataset","p"))
t["DET auc_pr"] = report.mean_std(m5d, ["auc_pr"], by=("dataset","p"))["auc_pr"]
export(t, "tableF_M5_sweep")
fig, path = plots.sweep_lines(m5, "p", ["auc_pr","std/label_auroc","std/error_auroc"],
                              ["AUC-PR","MCD std vs anomaly labels","MCD std vs errors"],
                              "REV_M5_dropout_sweep", ref=0.5)
plt.show(); print("saved", path)'''),
]

# ─────────────────────────────────────────────────────────────────────────────
R05 = [
    md("""
# R05 · Window-size sensitivity

The default window sizes are W = 30 for SKAB and W = 50 for NAB: SKAB is sampled once per second and a fault lasts several minutes, so
30 s windows see a local stretch of the fault; NAB series are mostly 5-minute samples, so 50 steps ≈ 4 hours.
This notebook checks that the **conclusions** do not depend on that choice: B3 (+M3) and M1 are rerun with
other window sizes, 3 seeds each.
Window size is **not** selected on test data: the table is a robustness check, not a tuning step.
"""),
    code(SETUP),
    code('''if RUN:
    failed = run_stage("window", workers=WORKERS, device=DEVICE, results=RESULTS)'''),
    code(LOAD),
    code('''tags = stage_tags("window") | stage_tags("main")   # main runs give the default-W point
W = V[V.tag.isin(tags) & V.method.isin(["B3 Dense AE","M3 Dense+CP","M1 Dense+MCD"])
      & V.seed.isin(range(3))]
cols = ["auc_pr","f1","std/error_auroc","margin/error_auroc","fpr@0.05"]
cols = [c for c in cols if c in W]
export(report.mean_std(W, cols, by=("dataset","method","W")), "tableG_window_size")
fig, path = plots.sweep_lines(W[W.method == "B3 Dense AE"], "W", ["auc_pr"], ["B3 AUC-PR"], "REV_window_size_B3")
plt.show(); print("saved", path)
fig, path = plots.sweep_lines(W[W.method == "M1 Dense+MCD"], "W", ["std/error_auroc"],
                              ["M1 MCD std vs errors"], "REV_window_size_M1", ref=0.5)
plt.show(); print("saved", path)'''),
]

# ─────────────────────────────────────────────────────────────────────────────
R06 = [
    md("""
# R06 · Confidence intervals

Windows from the same file are strongly dependent (stride 1), so CIs are
computed by **resampling whole files** (file-level bootstrap, 1000 resamples), not single windows.
Computed for seed 0 of each main model; the seed-to-seed spread is in R03.
"""),
    code(SETUP),
    code(LOAD),
    code('''from src import metrics as M
rows = []
V = V[V.tag.isin(stage_tags("main"))]
for method, view in [("B3 Dense AE","det_val"), ("M1 Dense+MCD","mc_val"), ("M2 LSTM+MCD","mc_val"),
                     ("M3 Dense+CP","det_cp"), ("M4 LSTM+CP","det_cp")]:
    for ds in ("SKAB","NAB"):
        t = V[(V.method == method) & (V.dataset == ds) & (V.seed == 0)]
        if t.empty: continue
        z = np.load(f"{RESULTS}/arrays/{t.iloc[0].tag}.npz")
        y, f = z["y"], z["file_id"]
        s = z["mc_score"] if view.startswith("mc") else z["det_score"]
        from sklearn.metrics import average_precision_score
        lo, hi = M.file_bootstrap(lambda a, b: average_precision_score(a, b), f, y, s)
        row = {"dataset": ds, "method": method, "AUC-PR": round(average_precision_score(y, s), 3),
               "AUC-PR 95% CI": f"[{lo:.3f}, {hi:.3f}]"}
        if view == "mc_val":
            thr = t.iloc[0]["threshold"]; e = ((s >= thr).astype(int) != y).astype(int)
            lo, hi = M.file_bootstrap(M.auroc, f, e, z["mc_std"])
            row.update({"error-AUROC (std)": round(M.auroc(e, z["mc_std"]), 3),
                        "error-AUROC 95% CI": f"[{lo:.3f}, {hi:.3f}]"})
        if view == "det_cp":
            normal = y == 0
            lo, hi = M.file_bootstrap(lambda p: float(np.mean(p <= 0.05)), f[normal], z["det_cp_p"][normal])
            row.update({"FPR@0.05": round(float(np.mean(z["det_cp_p"][normal] <= 0.05)), 3),
                        "FPR 95% CI": f"[{lo:.3f}, {hi:.3f}]"})
        rows.append(row)
export(pd.DataFrame(rows).set_index(["dataset","method"]), "tableH_bootstrap_ci")'''),
    md("### Output files"),
    code('''for d in ("figures_rev", "tables_rev"):
    print(d); [print("  ", f) for f in sorted(os.listdir(d))] if os.path.isdir(d) else print("   (none)")'''),
]

if __name__ == "__main__":
    for name, cells in [("R01_Data_Splits.ipynb", R01), ("R02_Grid_Search.ipynb", R02),
                        ("R03_Main_Experiments.ipynb", R03), ("R04_Dropout_Sweeps.ipynb", R04),
                        ("R05_Window_Size.ipynb", R05), ("R06_Confidence_Intervals.ipynb", R06)]:
        save(name, cells)
