# Quantum Cognitive Machine Learning Project
# QCML110426_2: figures for the introduction, from the egg of QCML102726
# What it does and how: see the notes at the end of this file.


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


PLOT_DIR = os.path.join("QCML Plots", "qcml110426_2_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# The egg of QCML102726: the same 2,500 noisy points, chunks and cut-off
L, B, w = 3.0, 2.0, 0.2        # length, breadth and the shift of its widest point
N = 2500
SIGMA = np.sqrt(0.01)          # noise standard deviation
SEED = 0
CUTOFF = 1.0
CHUNK_SEED, N_CHUNKS, CHUNK_RADIUS = 1, 3, 0.35


def egg(theta):
    """Points on the true egg at angles theta."""
    return np.column_stack([
        (L / 2) * np.cos(theta),
        (B / 2) * L * np.sin(theta) / np.sqrt(L**2 + 4 * w * L * np.cos(theta) + 4 * w**2)])


rng = np.random.default_rng(SEED)
theta = rng.uniform(0.0, 2.0 * np.pi, size=N)
data = egg(theta) + rng.normal(loc=0.0, scale=SIGMA, size=(N, 2))
egg_line = egg(np.linspace(0.0, 2.0 * np.pi, 500))
egg_curve = egg(np.linspace(0.0, 2.0 * np.pi, 2000, endpoint=False))
chunk_rng = np.random.default_rng(CHUNK_SEED)
chunk_centers = egg(chunk_rng.uniform(0.0, 2.0 * np.pi, size=N_CHUNKS))
in_chunk = (np.sqrt(((data[:, None, :] - chunk_centers[None, :, :]) ** 2).sum(axis=2))
            < CHUNK_RADIUS).any(axis=1)


def distance_to_egg(points):
    """Distance from each point to the nearest spot on the true egg."""
    gaps = points[:, None, :] - egg_curve[None, :, :]
    return np.sqrt((gaps ** 2).sum(axis=2)).min(axis=1)


# PCA with one component: the straight line through the average along the widest spread
center = data.mean(0)
direction = np.linalg.svd(data - center, full_matrices=False)[2][:1]
pca_recon = center + ((data - center) @ direction.T) @ direction
print(f"noisy points: {distance_to_egg(data).mean():.3f} from the egg;  "
      f"PCA, one component: {distance_to_egg(pca_recon).mean():.3f}")
print(f"chunks keep {(~in_chunk).mean():.0%} of the points; the cut-off keeps {(data[:, 0] < CUTOFF).mean():.0%}")

plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                     "axes.formatter.use_mathtext": True, "font.size": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                     "legend.frameon": False, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.grid": True, "axes.axisbelow": True,
                     "grid.color": "#e1e0d9", "grid.linewidth": 0.5})
PCA_COLOR, GRAY, INK = "#2a78d6", "#898781", "#0b0b0b"


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


print(f"\nfigures saved to {PLOT_DIR}:")

# Missing data: the points that remain (gray), the true egg (dashed) and the missing region (dotted)
for name, keep in [("intro_egg_missing_chunks", ~in_chunk), ("intro_egg_missing_cutoff", data[:, 0] < CUTOFF)]:
    fig, ax = plt.subplots(figsize=(3.1, 3.1), layout="constrained")
    ax.scatter(*data[keep].T, s=2, alpha=0.5, color=GRAY, linewidths=0, rasterized=True)
    ax.plot(*egg_line.T, "--", color=INK, lw=0.8)
    if name.endswith("chunks"):
        for c in chunk_centers:
            ax.add_patch(plt.Circle(c, CHUNK_RADIUS, fill=False, ls=":", lw=0.8, color=INK))
    else:
        ax.axvline(CUTOFF, color=INK, lw=0.8, ls=":")
    egg_axes(ax)
    save(fig, name)

# PCA with one component: every reconstruction (blue) lies on one straight line
fig, ax = plt.subplots(figsize=(3.1, 3.1), layout="constrained")
ax.scatter(*data.T, s=2, alpha=0.35, color=GRAY, linewidths=0, rasterized=True)
ax.scatter(*pca_recon.T, s=2, alpha=0.7, color=PCA_COLOR, linewidths=0, rasterized=True)
ax.plot(*egg_line.T, "--", color=INK, lw=0.8)
egg_axes(ax)
save(fig, "intro_egg_pca")


# ========================================================================================
# EXPLANATION
# Notes on the code above, in the order of its sections. A tag [note k] in the code
# marks the section that note k explains.
# ========================================================================================
#
# [note 0] About this script
#
# Nothing here is new: the egg, its missing regions and PCA with one component are exactly
# those of QCML102726. The plots are drawn in the style of the methods figures (QCML102826):
#   - intro_egg_missing_chunks   three disks removed from inside the data
#   - intro_egg_missing_cutoff   every point beyond x = 1 withheld
#   - intro_egg_pca              PCA with one component flattens the egg onto a line
# Each plot is saved as a PDF (for the LaTeX chapter) and a PNG in "QCML Plots/qcml110426_2_plots".
# Run with python3.13 QCML110426_2.py. Runs in a few seconds.
