"""Figures for the revised manuscript. Every file is saved as
figures_rev/REV_<name>.png (see report.save_fig)."""
import matplotlib.pyplot as plt
import numpy as np

from .report import risk_coverage_curves, save_fig

# fixed categorical order (never cycled); gray = reference lines
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
GRAY, INK, MUTED = "#8a8a85", "#2b2b29", "#6b6b66"

plt.rcParams.update({
    "figure.dpi": 110, "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "lines.linewidth": 2,
    "legend.frameon": False,
})


def split_composition(summaries, name="REV_fig1_split_composition"):
    """summaries: {dataset: DataFrame from data.summary}."""
    parts = [("train", "Train (normal)", C[0]), ("val", "Validation (normal)", C[2]),
             ("cal", "Calibration (normal)", C[3]), ("test_n", "Test – normal", C[4]),
             ("test_a", "Test – anomalous", C[1])]
    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.7 * len(summaries)))
    for i, (ds, s) in enumerate(summaries.items()):
        s = s.set_index("split")
        v = {"train": s.loc["train", "windows"], "val": s.loc["val", "windows"],
             "cal": s.loc["cal", "windows"],
             "test_n": s.loc["test", "windows"] - s.loc["test", "anomalous"],
             "test_a": s.loc["test", "anomalous"]}
        tot, left = sum(v.values()), 0
        for key, lab, col in parts:
            w = v[key] / tot
            ax.barh(i, w, left=left, color=col, edgecolor="white", linewidth=2,
                    label=lab if i == 0 else None)
            if w > 0.035:
                ax.text(left + w / 2, i, f"{v[key]:,}", ha="center", va="center",
                        color="white", fontsize=8 if w > 0.07 else 6.5,
                        fontweight="bold", rotation=0 if w > 0.07 else 90)
            left += w
    ax.set_yticks(range(len(summaries)), list(summaries))
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of windows")
    ax.grid(False)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.35), fontsize=8)
    return fig, save_fig(fig, name)


def risk_coverage(results, items, name):
    """items: list of (title, tag, view, signals). One panel per item."""
    labels = {"std": "MCD std", "cv": "MCD std / mean", "vote": "MCD vote disagreement",
              "margin": "Score margin (baseline)", "p_margin": "p-value margin",
              "random": "Random rejection", "oracle": "Oracle"}
    colors = {"std": C[0], "cv": C[2], "vote": C[1], "margin": C[3], "p_margin": C[4]}
    fig, axes = plt.subplots(1, len(items), figsize=(4.2 * len(items), 3.3), squeeze=False)
    for ax, (title, tag, view, signals) in zip(axes[0], items):
        grid, curves = risk_coverage_curves(results, tag, view, signals + ["random", "oracle"])
        for s, r in curves.items():
            style = dict(color=GRAY, lw=1.2, ls="--" if s == "random" else ":") \
                if s in ("random", "oracle") else dict(color=colors.get(s, C[5]))
            ax.plot(grid, r, label=labels.get(s, s), **style)
        ax.set_title(title, color=INK, fontsize=10)
        ax.set_xlabel("Coverage (share of windows kept)")
        ax.set_ylabel("Error rate of kept windows")
    axes[0][-1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    return fig, save_fig(fig, name)


def dot_compare(table, metrics, labels, name, ref=0.5, xlabel="AUROC"):
    """Dot plot: one row per method, one dot colour per metric, error bar = std
    over seeds. `table` has columns method, <m>_mean, <m>_std."""
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(table) + 1.2))
    y = np.arange(len(table))
    off = np.linspace(-0.15, 0.15, len(metrics))
    for j, (m, lab) in enumerate(zip(metrics, labels)):
        ax.errorbar(table[f"{m}_mean"], y + off[j], xerr=table[f"{m}_std"], fmt="o",
                    ms=7, color=C[j], ecolor=C[j], elinewidth=1.5, capsize=0,
                    markeredgecolor="white", markeredgewidth=1.5, label=lab)
    ax.axvline(ref, color=GRAY, ls="--", lw=1)
    ax.set_yticks(y, table["method"])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(metrics), fontsize=8)
    fig.tight_layout()
    return fig, save_fig(fig, name)


def sweep_lines(df, x, metrics, labels, name, logx=False, ref=None, title=None):
    """One panel per metric (never two y-scales). df: rows per seed."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.1), squeeze=False)
    for ax, m, lab in zip(axes[0], metrics, labels):
        for i, (ds, d) in enumerate(df.groupby("dataset")):
            g = d.groupby(x)[m].agg(["mean", "std"]).reset_index()
            ax.plot(g[x], g["mean"], "-o", ms=5, color=C[i], label=ds,
                    markeredgecolor="white", markeredgewidth=1.2)
            ax.fill_between(g[x], g["mean"] - g["std"].fillna(0),
                            g["mean"] + g["std"].fillna(0), color=C[i], alpha=0.15, lw=0)
        if ref is not None and "auroc" in m:
            ax.axhline(ref, color=GRAY, ls="--", lw=1)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(x)
        ax.set_title(lab, color=INK, fontsize=10)
    axes[0][0].legend(fontsize=8)
    if title:
        fig.suptitle(title, color=INK)
    fig.tight_layout()
    return fig, save_fig(fig, name)


def conformal_fpr(table, name="REV_conformal_false_alarm_rate", alpha=0.05):
    """table: method, dataset, fpr_mean, fpr_std."""
    dss = list(table["dataset"].unique())
    fig, axes = plt.subplots(1, len(dss), figsize=(4.5 * len(dss), 0.4 * table["method"].nunique() + 1.4),
                             squeeze=False, sharey=True)
    for ax, ds in zip(axes[0], dss):
        t = table[table.dataset == ds]
        y = np.arange(len(t))
        ax.barh(y, t["fpr_mean"], xerr=t["fpr_std"], color=C[0], height=0.6,
                error_kw=dict(ecolor=INK, elinewidth=1))
        ax.axvline(alpha, color=C[1], ls="--", lw=1.5, label=f"Target α = {alpha}")
        ax.set_yticks(y, t["method"])
        ax.set_title(ds, color=INK)
        ax.set_xlabel("False-alarm rate on normal test windows")
    axes[0][0].invert_yaxis()
    axes[0][-1].legend(fontsize=8)
    fig.tight_layout()
    return fig, save_fig(fig, name)
