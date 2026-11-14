# Quantum Cognitive Machine Learning Project
# QCML102826: thesis versions of the methods-chapter figures, built on QCML102726
# Runs in about 30 seconds.
# What it does and how: see the notes at the end of this file.


import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D

import QCML102726 as base     # reruns the 10/27 computations and keeps their results

# Repository set-up: run from this folder
import os
import sys
_FOLDER = os.path.dirname(os.path.abspath(__file__))
os.chdir(_FOLDER)                                                 # data and outputs live in this folder


PLOT_DIR = os.path.join("QCML Plots", "qcml102826_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                     "axes.formatter.use_mathtext": True, "font.size": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                     "legend.frameon": False, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.grid": True, "axes.axisbelow": True,
                     "grid.color": "#e1e0d9", "grid.linewidth": 0.5})
QCML_COLOR, DM_COLOR = "#eb6834", "#1baf7a"
GRAY, INK, HELD_BACK = "#898781", "#0b0b0b", "#f4a3a3"
SEQUENTIAL = LinearSegmentedColormap.from_list("blues", ["#b7d3f6", "#104281"])


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(PLOT_DIR, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"   {os.path.join(PLOT_DIR, name)}.pdf")


def egg_axes(ax):
    ax.set_aspect("equal")
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")


def egg_panel(name, key, color, run):
    """One method on one test: the learned points (gray), the held-back points (pink), where
    the method puts each (its color, black), the true egg (dashed) and the test's cut."""
    keep = base.RUNS[run]
    recon = base.reconstructions[(key, run)]
    fig, ax = plt.subplots(figsize=(3.1, 3.1), layout="constrained")
    ax.scatter(*base.data[keep].T, s=2, alpha=0.35, color=GRAY, linewidths=0, rasterized=True)
    ax.scatter(*base.data[~keep].T, s=2, alpha=0.6, color=HELD_BACK, linewidths=0, rasterized=True)
    ax.scatter(*recon[keep].T, s=2, alpha=0.7, color=color, linewidths=0, rasterized=True)
    ax.scatter(*recon[~keep].T, s=2.5, alpha=0.9, color=INK, linewidths=0, rasterized=True)
    ax.plot(*base.egg_line.T, "--", color=INK, lw=0.8)
    if run == "cut-off":
        ax.axvline(base.CUTOFF, color=INK, lw=0.7, ls=":")
    if run == "chunks":
        for center in base.chunk_centers:
            ax.add_patch(plt.Circle(center, base.CHUNK_RADIUS, fill=False, ls=":", lw=0.7, color=INK))
    egg_axes(ax)
    save(fig, name)


print(f"\nthesis figures saved to {PLOT_DIR}:")
dm_key = f"DM m={base.DM_M}"
egg_panel("egg_diffusion_map_cut_off", dm_key, DM_COLOR, "cut-off")
egg_panel("egg_qcml_cut_off", "QCML", QCML_COLOR, "cut-off")
egg_panel("egg_diffusion_map_chunks", dm_key, DM_COLOR, "chunks")
egg_panel("egg_qcml_chunks", "QCML", QCML_COLOR, "chunks")

# One step of the diffusion map's random walk from the top of the egg
fig, ax = plt.subplots(figsize=(3.1, 2.9), layout="constrained")
order = np.argsort(base.one_step)
shown = base.one_step[order] > 1e-6
ax.scatter(*base.data[order][~shown].T, s=2, color="#e1e0d9", linewidths=0, rasterized=True)
dots = ax.scatter(*base.data[order][shown].T, c=base.one_step[order][shown], s=3, cmap=SEQUENTIAL,
                  norm=LogNorm(vmin=1e-6, vmax=base.one_step.max()), linewidths=0, rasterized=True)
ax.scatter(*base.data[base.start_point], s=40, marker="*", color=INK, zorder=3)
fig.colorbar(dots, ax=ax, shrink=0.8, label="step probability")
egg_axes(ax)
save(fig, "diffusion_map_one_step")

# The first two diffusion coordinates, colored by angle around the egg
fig, ax = plt.subplots(figsize=(3.1, 2.9), layout="constrained")
dots = ax.scatter(base.dm_all.right[:, 1], base.dm_all.right[:, 2], c=base.theta, s=2,
                  cmap="twilight", linewidths=0, rasterized=True)
fig.colorbar(dots, ax=ax, shrink=0.8, label=r"angle $\theta$ around the egg")
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(r"diffusion coordinate $\psi_1$")
ax.set_ylabel(r"diffusion coordinate $\psi_2$")
save(fig, "diffusion_coordinates")

# Feature 1's four eigenvalues during training, against the benign patients' range
fig, ax = plt.subplots(figsize=(6.0, 3.0), layout="constrained")
low, high = base.benign[:, base.FEATURE].min(), base.benign[:, base.FEATURE].max()
band = ax.axhspan(low, high, color="#e1e0d9", alpha=0.6, lw=0,
                  label="mean radius of the 250 benign training patients")
epochs = np.arange(base.WDBC_EPOCHS + 1)
shades = ["#f6c3a8", "#f0956a", "#eb6834", "#a8441a"]
handles = [band]
for k in range(base.CLS_N):
    handles += ax.plot(epochs, base.spectra[:, k], color=shades[k], lw=1.4, label=f"eigenvalue {k + 1}")
    ax.scatter([base.WDBC_EPOCHS + 12], [base.spectrum_end[k]], color=shades[k], s=14, zorder=3)
handles.append(Line2D([], [], ls="", marker="o", ms=4, color="#52514e", label="after the L-BFGS polish"))
ax.set_xticks(list(range(0, base.WDBC_EPOCHS + 1, 25)) + [base.WDBC_EPOCHS + 12],
              [str(t) for t in range(0, base.WDBC_EPOCHS + 1, 25)] + ["L-BFGS"])
ax.set_xlabel("Adam epoch (0 is the random start)")
ax.set_ylabel("eigenvalue (standard deviations)")
fig.legend(handles=handles, loc="outside lower center", ncol=3)
save(fig, "wdbc_feature1_eigenvalues")


# ========================================================================================
# EXPLANATION
# Notes on the code above, in the order of its sections. A tag [note k] in the code
# marks the section that note k explains.
# ========================================================================================
#
# [note 0] About this script
#
# Nothing here changes a result. It imports QCML102726, which reruns the egg and the
# breast-cancer Feature 1 model exactly as before (and saves its own outputs again), then
# redraws the seven plots the methods chapter uses in the chapter's style:
#   - Computer Modern, the LaTeX font, so the plots match the text
#   - no titles or legends on the egg panels (the LaTeX captions describe them)
#   - sized for the page: egg panels for a 2 x 2 grid, the eigenvalue plot full width
# Each plot is saved as a PDF (for the LaTeX chapter) and a PNG (for quick viewing) in
# "QCML Plots/qcml102826_plots". Run with python3.13 QCML102826.py.
# Runs in about 30 seconds.
