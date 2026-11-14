# Quantum Cognitive Machine Learning Project
# QCML111326_3: thesis figures for the results chapter
# What it does and how: see the notes at the end of this file.


import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Repository set-up: run from this folder
import os
import sys
_FOLDER = os.path.dirname(os.path.abspath(__file__))
os.chdir(_FOLDER)                                                 # data and outputs live in this folder


PLOT_DIR = os.path.join("QCML Plots", "qcml111326_3_plots")
os.makedirs(PLOT_DIR, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                     "axes.formatter.use_mathtext": True, "font.size": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                     "legend.frameon": False, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.grid": True, "axes.axisbelow": True,
                     "grid.color": "#e1e0d9", "grid.linewidth": 0.5})
QCML_COLOR, DM_COLOR, KRR_COLOR = "#eb6834", "#1baf7a", "#2a78d6"
GRAY, DARK_GRAY, INK, MUTED_INK = "#898781", "#52514e", "#0b0b0b", "#52514e"
WIDE = (6.3, 2.75)            # full text width of the thesis, in inches


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(PLOT_DIR, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"   {os.path.join(PLOT_DIR, name)}.pdf")


def read_csv(path):
    with open(path) as file:
        return list(csv.DictReader(file))


def decimal_ticks(ax, values):
    """Log-scale y ticks written as plain decimals (no powers of ten, no scaling)."""
    ax.set_yticks(values, [f"{v:g}" for v in values])
    ax.minorticks_off()


def time_label(seconds):
    if seconds < 1:
        return f"{seconds * 1000:g} ms"
    if seconds < 60:
        return f"{seconds:g} s"
    if seconds < 3600:
        return f"{seconds / 60:.2g} min"
    if seconds < 86400:
        return f"{seconds / 3600:.2g} h"
    return f"{seconds / 86400:.2g} days"


def nice_ticks(low, high):
    """Plain-decimal ticks (1, 2, 5 per decade) inside [low, high]."""
    out = []
    for exponent in range(-5, 1):
        for mantissa in (1, 2, 5):
            v = mantissa * 10.0 ** exponent
            if low <= v <= high:
                out.append(round(v, 8))
    return out


print(f"figures saved to {PLOT_DIR}:")

# 1. Test NRMSE against the number of weak runs, every readout (QCML111326 Part A)
weak = read_csv("qcml111326_weak.csv")
WEAK_STYLE = [  # readout in the CSV, label, color, marker, filled
    ("Ridge regression", "ridge regression", GRAY, "s", True),
    ("Diffusion map", "diffusion map", DM_COLOR, "s", True),
    ("Supervised QCML", "supervised QCML", DARK_GRAY, "^", False),
    ("Kernel ridge", "kernel ridge regression", KRR_COLOR, "o", True),
    ("QCML+", "QCML+", QCML_COLOR, "^", True)]
for delay in ("1", "20"):
    rows = [r for r in weak if r["delay"] == delay]
    runs = sorted({float(r["runs"]) for r in rows})
    fig, ax = plt.subplots(figsize=WIDE, layout="constrained")
    lows, highs, ends = [], [], []
    for name, label, color, marker, filled in WEAK_STYLE:
        cells = sorted([r for r in rows if r["readout"] == name], key=lambda r: float(r["runs"]))
        means = np.array([float(r["NRMSE mean"]) for r in cells])
        stds = np.array([float(r["NRMSE std"]) for r in cells])
        lows.append(means.min())
        highs.append(means.max())
        ax.errorbar(runs, means, yerr=stds, color=color, marker=marker, ms=5, mfc=color if filled else "white",
                    mec=color if not filled else "white", mew=0.8 if not filled else 0.5, lw=1.2,
                    elinewidth=0.7, capsize=1.5, label=label, zorder=3 if name == "QCML+" else 2)
        ends.append((np.log10(means[-1]), label))
    # direct labels at the right end, pushed apart where two lines end close together
    ends.sort()
    placed = []
    for height, label in ends:
        if placed and height - placed[-1][0] < 0.13:
            height = placed[-1][0] + 0.13
        placed.append((height, label))
    for height, label in placed:
        ax.annotate(label, (runs[-1] * 1.35, 10 ** height), va="center", fontsize=7.5, color=MUTED_INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(runs, [f"$10^{{{round(np.log10(r))}}}$\n{time_label(r * 2000 * 100e-9)}" for r in runs])
    ax.set_xlim(runs[0] / 1.6, runs[-1] * 9)
    decimal_ticks(ax, nice_ticks(min(lows) * 0.8, max(highs) * 1.25))
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("runs of the input sequence, with the experiment time they take")
    ax.set_ylabel("test NRMSE")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, handletextpad=0.4, columnspacing=1.2)
    save(fig, f"results_weak_runs_delay_{delay}")

# 2. Test NRMSE against experiment time (QCML111326 Part B, and the published readout from
#    QCML102426_2)
projective = read_csv("qcml111326_projective.csv")
published = read_csv("qcml102426_2_cost.csv")
TIME_TICKS = [(1e-2, "10 ms"), (1, "1 s"), (60, "1 min"), (3600, "1 h"), (86400, "1 day"),
              (2.63e6, "1 month"), (3.156e7, "1 year")]


def curve(rows, readout):
    cells = sorted([r for r in rows if r["readout"] == readout], key=lambda r: float(r["experiment time (s)"]))
    return (np.array([float(r["experiment time (s)"]) for r in cells]),
            np.array([float(r["NRMSE mean"]) for r in cells]))


def time_to_reach(times, errors, level):
    """First experiment time at which the curve reaches `level` (log-log interpolation)."""
    lt, le, target = np.log(times), np.log(errors), np.log(level)
    if le[0] <= target:
        return times[0]
    for i in range(1, len(lt)):
        if le[i] <= target:
            share = (target - le[i - 1]) / (le[i] - le[i - 1])
            return float(np.exp(lt[i - 1] + share * (lt[i] - lt[i - 1])))
    return None


gaps = {}
for delay in ("1", "20"):
    fig, ax = plt.subplots(figsize=WIDE, layout="constrained")
    weak_rows = [r for r in weak if r["delay"] == delay]
    proj_rows = [r for r in projective if r["delay"] == delay]
    old_rows = [r for r in published if r["delay"] == delay and r["protocol"].startswith("theirs")]
    old_t = np.array([float(r["experiment time (s)"]) for r in old_rows])
    old_e = np.array([float(r["Linear readout (theirs) mean"]) for r in old_rows])
    series = [  # (times, errors, label, color, marker, filled)
        (old_t, old_e, "published linear readout, projective", GRAY, "o", True),
        (*curve(proj_rows, "Kernel ridge"), "kernel ridge regression, projective", KRR_COLOR, "o", False),
        (*curve(proj_rows, "QCML+"), "QCML+, projective", QCML_COLOR, "^", False),
        (*curve(weak_rows, "Kernel ridge"), "kernel ridge regression, weak", KRR_COLOR, "o", True),
        (*curve(weak_rows, "QCML+"), "QCML+, weak", QCML_COLOR, "^", True)]
    for times, errors, label, color, marker, filled in series:
        ax.plot(times, errors, "-", color=color, marker=marker, ms=5, lw=1.2,
                mfc=color if filled else "white", mec="white" if filled else color, mew=0.5 if filled else 0.9,
                label=label, zorder=3 if filled and color == QCML_COLOR else 2)
    # the gap: how long kernel ridge on the projective record needs to match QCML+ on 2 s of weak record
    t_weak = 1e4 * 2000 * 100e-9
    level = float([r for r in weak_rows if r["readout"] == "QCML+" and float(r["runs"]) == 1e4][0]["NRMSE mean"])
    t_krr = time_to_reach(*curve(proj_rows, "Kernel ridge"), level)
    t_qp = time_to_reach(*curve(proj_rows, "QCML+"), level)
    gaps[delay] = (level, t_krr, t_qp)
    if t_krr is not None:
        ax.annotate("", xy=(t_krr, level), xytext=(t_weak, level),
                    arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9, shrinkA=3, shrinkB=3), zorder=4)
        ax.annotate(f"{t_krr / t_weak:,.0f}$\\times$ longer", (np.sqrt(t_weak * t_krr), level),
                    textcoords="offset points", xytext=(0, 4), ha="center", va="bottom", fontsize=8, color=INK)
    ax.annotate("$10^4$ runs, 2 s", (t_weak, level), textcoords="offset points", xytext=(-4, -9), ha="right",
                va="top", fontsize=7, color=MUTED_INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([t for t, _ in TIME_TICKS], [name for _, name in TIME_TICKS])
    ax.set_xlim(1e-2 / 2, 3.156e7 * 1.5)
    all_errors = np.concatenate([s[1] for s in series])
    decimal_ticks(ax, nice_ticks(all_errors.min() * 0.8, all_errors.max() * 1.25))
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("experiment time")
    ax.set_ylabel("test NRMSE")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.4, columnspacing=1.5)
    save(fig, f"results_time_delay_{delay}")
for delay, (level, t_krr, t_qp) in gaps.items():
    print(f"   delay {delay}: QCML+ weak 1e4 runs {level:.5f}; kernel ridge, projective, reaches it at "
          + ("never" if t_krr is None else f"{time_label(t_krr)} ({t_krr / 2:,.0f} x 2 s)")
          + "; QCML+, projective, at " + ("never" if t_qp is None else f"{time_label(t_qp)} ({t_qp / 2:,.0f} x)"))

# 3. The final comparison (QCML110226), redrawn as in QCML110426 but with plain decimal ticks
final = read_csv("qcml110226_readouts.csv")
SETTINGS = ["vacuum only, exact", "40 numbers, exact", "vacuum only, 9^5 shots", "40 numbers, 9^5 shots",
            "all 160, 9^5 shots", "weak record, 1e4 runs", "weak record, 1e6 runs"]   # the table's order
TICKS = ["vacuum,\nnoise-free", "40 values,\nnoise-free", "vacuum,\n$9^5$ shots", "40 values,\n$9^5$ shots",
         "all 160,\n$9^5$ shots", "weak,\n$10^4$ runs", "weak,\n$10^6$ runs"]
READOUTS = [("Diffusion map", "diffusion map", DM_COLOR, "s"), ("Kernel ridge", "kernel ridge regression", KRR_COLOR, "o"),
            ("QCML+", "QCML+", QCML_COLOR, "^")]
for delay in ("1", "20"):
    cells = {row["setting"]: row for row in final if row["delay"] == delay}
    fig, ax = plt.subplots(figsize=(6.3, 2.7), layout="constrained")
    xs = np.arange(len(SETTINGS))
    values = []
    for k, (name, label, color, marker) in enumerate(READOUTS):
        means = [float(cells[s][f"{name} mean"]) for s in SETTINGS]
        stds = [float(cells[s][f"{name} std"]) for s in SETTINGS]
        values += means
        ax.errorbar(xs + (k - 1) * 0.2, means, yerr=stds, color=color, marker=marker, ms=5.5,
                    mec="white", mew=0.6, ls="none", capsize=1.5, elinewidth=0.8, label=label)
    ax.set_yscale("log")
    decimal_ticks(ax, nice_ticks(min(values) * 0.8, max(values) * 1.25))
    ax.set_xticks(xs, TICKS)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("test NRMSE")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    save(fig, f"results_final_delay_{delay}")

# 4. Breast cancer with features removed (QCML111326_2)
removal = read_csv("qcml111326_2_removal.csv")
CLASSIFIERS = [("Supervised QCML", "supervised QCML"), ("QCML", "unsupervised QCML"),
               ("Diffusion map", "diffusion map"), ("Logistic regression", "logistic regression"),
               ("PCA", "PCA"), ("Nearest centroid", "nearest centroid")]
full = [r for r in removal if r["removed"] == "none"][0]
top = [r for r in removal if r["removed"].startswith("Features 29")][0]
singles = [r for r in removal if r["feature number"]]
fig, ax = plt.subplots(figsize=(6.3, 2.9), layout="constrained")
ys = np.arange(len(CLASSIFIERS))[::-1]
rng = np.random.default_rng(0)
for y, (name, _) in zip(ys, CLASSIFIERS):
    values = np.array([float(r[f"{name} accuracy mean"]) for r in singles])
    ax.scatter(values, y + rng.uniform(-0.16, 0.16, len(values)), s=9, color=GRAY, alpha=0.75, lw=0,
               label="one feature removed (each of the 30)" if y == ys[0] else None, zorder=2)
    ax.scatter(float(full[f"{name} accuracy mean"]), y, s=46, facecolor="none", edgecolor=INK, lw=1.0,
               label="all 30 features" if y == ys[0] else None, zorder=5)
    ax.scatter(float(top[f"{name} accuracy mean"]), y, s=46, marker="D", color=QCML_COLOR, edgecolor="white",
               lw=0.5, label="Features 29, 14 and 13 removed together" if y == ys[0] else None, zorder=4)
ax.set_yticks(ys, [label for _, label in CLASSIFIERS])
ax.grid(axis="y", visible=False)
ax.set_xlabel("validation accuracy, mean over the ten splits")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handletextpad=0.3, columnspacing=1.2)
save(fig, "results_wdbc_removal")


# ========================================================================================
# EXPLANATION
# Notes on the code above, in the order of its sections. A tag [note k] in the code
# marks the section that note k explains.
# ========================================================================================
#
# [note 0] About this script
#
# Draws, from saved CSVs only (nothing is recomputed):
#   results_weak_runs_delay_{1,20}  test NRMSE against the number of weak runs, every readout
#                                   (QCML111326, Part A)
#   results_time_delay_{1,20}       test NRMSE against experiment time: QCML+ on the weak and the
#                                   projective record, kernel ridge on the projective record, and
#                                   the published linear readout (QCML111326 Part B; QCML102426_2
#                                   for the published readout, all 160 probabilities, no delays)
#   results_wdbc_removal            breast cancer accuracy with features removed (QCML111326_2)
# In Computer Modern like the other thesis figures (QCML110426), no titles, PDF + PNG, in
# "QCML Plots/qcml111326_3_plots". Copy the PDFs to Overleaf_Methods/figures.
#
# Colors: the three validated thesis hues (kernel ridge blue, QCML+ orange, diffusion map aqua).
# No fourth hue passes the color-vision checks against them, so supervised QCML and ridge
# regression are drawn in grays and labeled directly.
#
# Run with python3.13 QCML111326_3.py. Takes a few seconds.
