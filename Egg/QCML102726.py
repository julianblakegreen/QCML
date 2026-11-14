# Quantum Cognitive Machine Learning Project
# QCML102726: pictures for the methods chapter, built on QCML092926 and QCML102326_2
# Runs in about 3 minutes.
# What it does and how: see the notes at the end of this file.


import csv
import os
import textwrap
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D


# Repository set-up: run from this folder, with QCML from 1_Plain_QCML
import os
import sys
_FOLDER = os.path.dirname(os.path.abspath(__file__))
os.chdir(_FOLDER)                                                 # data and outputs live in this folder
sys.path.insert(0, os.path.join(_FOLDER, "..", "1_Plain_QCML"))   # qcml.py and hermitian.py

from qcml import QCML

PLOT_DIR = os.path.join("QCML Plots", "qcml102726_plots")   # every plot is saved here as its own PNG


# PART A: THE EGG (as in QCML092926)

# The three numbers that define the egg
L = 3.0   # length:    the egg runs from x = -L/2 to x = +L/2
B = 2.0   # breadth:   the height at its widest point
w = 0.2   # shift:     the position of its widest point

N = 2500                   # number of points
VARIANCE = 0.01            # noise variance
SIGMA = np.sqrt(VARIANCE)  # noise standard deviation
SEED = 0                   # noise seed

CUTOFF = 1.0               # run 2 of QCML_92226: only points with x < CUTOFF are learned from
CHUNK_SEED = 1             # run 2 of QCML092926: where the chunks land
N_CHUNKS = 3               # how many pieces are cut out
CHUNK_RADIUS = 0.35        # radius of each piece, in egg units

# QCML settings of the egg scripts
n = 8          # matrix size
C1 = 1.0       # position error weight
C2 = 0.1       # quantum fluctuation weight
EPOCHS = 100
BATCH = 250

# PCA: 1 component is a straight line. 2 components would keep every point exactly where it
# is, since the egg data only have 2 numbers (x, y) per point.
PCA_COMPONENTS = 1

# Diffusion map  [note 1]
DM_NEIGHBORS = 30
DM_M = 10
DM_M_SHOWN = (2, 10, 50)


def egg(theta):
    """Points on the true egg at angles theta."""
    return np.column_stack([
        (L / 2) * np.cos(theta),
        (B / 2) * L * np.sin(theta) / np.sqrt(L**2 + 4 * w * L * np.cos(theta) + 4 * w**2)])


# The same 2,500 noisy points as QCML092926: random angles, then normal noise on x and y
rng = np.random.default_rng(SEED)
theta = rng.uniform(0.0, 2.0 * np.pi, size=N)
clean = egg(theta)
noise = rng.normal(loc=0.0, scale=SIGMA, size=(N, 2))
data = clean + noise
egg_line = egg(np.linspace(0.0, 2.0 * np.pi, 500))

# A dense copy of the true egg, used only to measure how close results are to it
egg_curve = egg(np.linspace(0.0, 2.0 * np.pi, 2000, endpoint=False))


def distance_to_egg(points):
    """Distance from each point to the nearest spot on the true egg."""
    gaps = points[:, None, :] - egg_curve[None, :, :]
    return np.sqrt((gaps ** 2).sum(axis=2)).min(axis=1)


# The three runs. keep = True for the points a method may learn from.
chunk_rng = np.random.default_rng(CHUNK_SEED)
chunk_centers = egg(chunk_rng.uniform(0.0, 2.0 * np.pi, size=N_CHUNKS))
in_chunk = (np.sqrt(((data[:, None, :] - chunk_centers[None, :, :]) ** 2).sum(axis=2))
            < CHUNK_RADIUS).any(axis=1)
RUNS = {"all data": np.ones(N, dtype=bool),
        "cut-off": data[:, 0] < CUTOFF,
        "chunks": ~in_chunk}
for name, keep in RUNS.items():
    print(f"{name}: {keep.sum()} of {N} points learned from ({100 * keep.mean():.0f}%)")


# PCA  [note 2]

def pca_fit(points, k=PCA_COMPONENTS):
    center = points.mean(0)
    directions = np.linalg.svd(points - center, full_matrices=False)[2][:k]
    return lambda new: center + ((new - center) @ directions.T) @ directions


# QCML (as in QCML092926)

def compute_loss(model, points):
    """Loss for a batch of points: C1*L1 + C2*L2. Also returns L1 and L2."""
    psi = model.ground_state(points)                             # ground state for every point
    L1 = ((model.bloch(psi) - points) ** 2).sum(dim=1).mean()    # average squared distance
    L2 = model.uncertainty(psi).mean()                           # average fluctuation
    return C1 * L1 + C2 * L2, L1, L2


def qcml_fit(points):
    """Train a fresh model on `points` exactly as QCML092926 does. Returns the model, its
    normalization numbers and a function that reconstructs new points in egg units."""
    mean = points.mean(axis=0)
    scale = np.sqrt(((points - mean) ** 2).sum(axis=1).mean())
    X = torch.tensor((points - mean) / scale, dtype=torch.float32)
    M = len(X)

    torch.manual_seed(SEED)                       # same starting matrices every run
    model = QCML(D=2, n=n, w=C2 / C1)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(1, EPOCHS + 1):
        order = torch.randperm(M)
        for start in range(0, M, BATCH):
            loss, _, _ = compute_loss(model, X[order[start:start + BATCH]])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=300,
                              line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        loss, _, _ = compute_loss(model, X)
        loss.backward()
        return loss

    lbfgs.step(closure)

    def reconstruct(new):
        Xn = torch.tensor((new - mean) / scale, dtype=torch.float32)
        return model.reconstruct(Xn).numpy() * scale + mean       # <psi|A_k|psi>
    return model, mean, scale, reconstruct


# DIFFUSION MAP (as in QCML102326)  [note 3]

def squared_distances(A, B):
    return np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T, 0)


class DiffusionMap:
    def __init__(self, points, neighbors=DM_NEIGHBORS):
        self.points, self.neighbors = points, neighbors
        d2 = squared_distances(points, points)
        self.width = np.sqrt(np.sort(d2, axis=1)[:, neighbors])     # column 0 is the point itself
        K = np.exp(-d2 / np.outer(self.width, self.width))
        self.degree = K.sum(1)
        S = K / np.sqrt(np.outer(self.degree, self.degree))     # symmetric form of the walk
        eigenvalues, V = np.linalg.eigh(S)
        order = np.argsort(eigenvalues)[::-1]
        self.eigenvalues, V = eigenvalues[order], V[:, order]
        self.right = V / np.sqrt(self.degree)[:, None]           # the walk's eigenvectors
        self.left = V * np.sqrt(self.degree)[:, None]            # their duals: left.T @ right = I

    def steps(self, points):
        """Step probabilities from (new) points to every training point."""
        d2 = squared_distances(points, self.points)
        width = np.sqrt(np.sort(d2, axis=1)[:, self.neighbors - 1])
        K = np.exp(-d2 / np.outer(width, self.width))
        return K / K.sum(1, keepdims=True)

    def rebuild(self, values, points, m):
        """Rebuild `values` (one row per training point) at `points` from the constant
        eigenvector plus the first m others."""
        at_points = self.steps(points) @ self.right[:, :m + 1] / self.eigenvalues[:m + 1]
        return at_points @ (self.left[:, :m + 1].T @ values)


# Train every method on every run, and send all 2,500 points through each trained model
reconstructions, seconds, models = {}, {}, {}
for run, keep in RUNS.items():
    start = time.time()
    reconstructions[("PCA", run)] = pca_fit(data[keep])(data)
    seconds[("PCA", run)] = time.time() - start

    start = time.time()
    dm = DiffusionMap(data[keep])
    for m in DM_M_SHOWN:
        reconstructions[(f"DM m={m}", run)] = dm.rebuild(data[keep], data, m)
    seconds[("DM", run)] = time.time() - start
    models[("DM", run)] = dm

    start = time.time()
    model, mean, scale, reconstruct = qcml_fit(data[keep])
    reconstructions[("QCML", run)] = reconstruct(data)
    seconds[("QCML", run)] = time.time() - start
    models[("QCML", run)] = (model, mean, scale)
    print(f"   {run}: PCA {seconds[('PCA', run)]:.2f} s, diffusion map {seconds[('DM', run)]:.1f} s, "
          f"QCML {seconds[('QCML', run)]:.1f} s")


# Scores: average distance to the true egg, for the points learned from and those held back
ROWS = [("PCA, 1 component", "PCA"), ("PCA, 2 components (the noisy points as they are)", "identity"),
        ("QCML, n = 8", "QCML")] + [(f"Diffusion map, m = {m}", f"DM m={m}") for m in DM_M_SHOWN]
COLUMNS = [("all data", None), ("cut-off", True), ("cut-off", False), ("chunks", True), ("chunks", False)]


def score(key, run, learned):
    keep = RUNS[run]
    points = data if key == "identity" else reconstructions[(key, run)]
    rows = keep if learned in (None, True) else ~keep
    return float(distance_to_egg(points[rows]).mean())


table = [(label, [score(key, run, learned) for run, learned in COLUMNS]) for label, key in ROWS]
header = ["Method", "All data", "Cut-off: learned", "Cut-off: held back", "Chunks: learned",
          "Chunks: held back"]
print("\nAverage distance to the true egg (egg units; the noise puts raw points 0.081 away)")
print("   " + "".join(f"{h:>20}" for h in header))
for label, cells in table:
    print(f"   {label:<48}" + "".join(f"{c:>16.3f}" for c in cells))
with open("qcml102726_egg_table.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    for label, cells in table:
        writer.writerow([label] + [f"{c:.4f}" for c in cells])
print("saved to qcml102726_egg_table.csv")
print("\nseconds to fit on all data: " + "   ".join(
    f"{m} {seconds[(m, 'all data')]:.2f}" for m in ("PCA", "DM", "QCML")))


# Intrinsic dimension of the all-data QCML model.  [note 4]
model, mean, scale = models[("QCML", "all data")]
metric = model.quantum_metric(torch.tensor((data - mean) / scale, dtype=torch.float32))
spectrum = torch.linalg.eigvalsh(metric).numpy()
participation = spectrum.sum(1) ** 2 / (spectrum ** 2).sum(1)
print(f"\nQCML intrinsic dimension on the egg: mean {participation.mean():.2f}, "
      f"median {np.median(participation):.2f} (2 numbers per point, one curve)")


# PART B: HOW THE DIFFUSION MAP SEES THE EGG
dm_all = models[("DM", "all data")]
start_point = int(np.argmin(((data - egg(np.array([np.pi / 2]))) ** 2).sum(1)))   # top of the egg
one_step = dm_all.steps(data[[start_point]])[0]
print(f"\none step of the walk from the top of the egg: the {DM_NEIGHBORS} likeliest points get "
      f"{np.sort(one_step)[::-1][:DM_NEIGHBORS].sum():.0%} of the probability; "
      f"{(one_step > 1e-3).sum()} points get more than 0.1%")
print(f"diffusion map eigenvalues 1-6: {np.round(dm_all.eigenvalues[1:7], 4).tolist()}")


# PART C: WHAT QCML'S NUMBERS LOOK LIKE ON BREAST CANCER (as in QCML102326_2)

WDBC_EPOCHS = 150
WDBC_BATCH = 64
CLS_N, CLS_C2 = 4, 0.1        # one manifold per class, for classifying
TRAIN_FRACTION = 0.7
SPLIT = 0                     # which stratified split (seed)
FEATURE = 0                   # Feature 1, mean radius

rows = np.genfromtxt("wdbc.data", delimiter=",", dtype=str)
labels = (rows[:, 1] == "M").astype(int)          # 1 = malignant, 0 = benign
features = rows[:, 2:].astype(float)              # (569, 30)


def split_data(seed):
    """Stratified random split, exactly as QCML102326_2."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for value in (0, 1):
        rows_of_class = rng.permutation(np.flatnonzero(labels == value))
        cut = round(TRAIN_FRACTION * len(rows_of_class))
        train_idx.append(rows_of_class[:cut])
        val_idx.append(rows_of_class[cut:])
    return np.sort(np.concatenate(train_idx)), np.sort(np.concatenate(val_idx))


def normalize(train_points, points):
    """Z-score `points` using the mean and std of `train_points` (one per feature)."""
    return (points - train_points.mean(0)) / train_points.std(0)


train_idx, val_idx = split_data(SPLIT)
Ztr = normalize(features[train_idx], features[train_idx])
Zva = normalize(features[train_idx], features[val_idx])
benign = Ztr[labels[train_idx] == 0]                       # 250 benign training patients
X = torch.tensor(benign, dtype=torch.float32)

# The benign model, trained as in QCML102326_2's train_qcml, with Feature 1's matrix
# recorded along the way
torch.manual_seed(SPLIT)
wdbc_model = QCML(D=benign.shape[1], n=CLS_N, w=CLS_C2)
raw_start = wdbc_model.observables.raw[FEATURE].detach().clone()
A_start = wdbc_model.observables()[FEATURE].detach().clone()
spectra = [torch.linalg.eigvalsh(A_start).numpy()]


def wdbc_loss(points):
    psi = wdbc_model.ground_state(points)
    L1 = ((wdbc_model.bloch(psi) - points) ** 2).sum(dim=1).mean()    # position error
    L2 = wdbc_model.uncertainty(psi).mean()                           # fluctuation
    return L1 + CLS_C2 * L2


optimizer = torch.optim.Adam(wdbc_model.parameters(), lr=0.01)
for epoch in range(1, WDBC_EPOCHS + 1):
    order = torch.randperm(len(X))
    for start in range(0, len(X), WDBC_BATCH):
        loss = wdbc_loss(X[order[start:start + WDBC_BATCH]])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        spectra.append(torch.linalg.eigvalsh(wdbc_model.observables()[FEATURE]).numpy())

lbfgs = torch.optim.LBFGS(wdbc_model.parameters(), lr=1.0, max_iter=300, line_search_fn="strong_wolfe")


def wdbc_closure():
    lbfgs.zero_grad()
    loss = wdbc_loss(X)
    loss.backward()
    return loss


lbfgs.step(wdbc_closure)
with torch.no_grad():
    raw_end = wdbc_model.observables.raw[FEATURE].detach().clone()
    A_end = wdbc_model.observables()[FEATURE].detach().clone()
    spectrum_end = torch.linalg.eigvalsh(A_end).numpy()
spectra = np.array(spectra)


def show(A):
    return "\n".join("   " + "  ".join(f"{z.real:+.2f}{z.imag:+.2f}i" for z in row) for row in A.numpy())


print(f"\nBreast cancer, benign model of split {SPLIT} (n = {CLS_N}, w = {CLS_C2}), Feature 1 (mean radius)")
print("raw array R at the start:\n" + "\n".join("   " + "  ".join(f"{v:+.2f}" for v in row)
                                                for row in raw_start.numpy()))
print("A_1 at the start:\n" + show(A_start))
print("raw array R after training:\n" + "\n".join("   " + "  ".join(f"{v:+.2f}" for v in row)
                                                  for row in raw_end.numpy()))
print("A_1 after training:\n" + show(A_end))
print(f"eigenvalues at the start   : {np.round(spectra[0].astype(float), 2).tolist()}")
print(f"eigenvalues after Adam     : {np.round(spectra[-1].astype(float), 2).tolist()}")
print(f"eigenvalues after L-BFGS   : {np.round(spectrum_end.astype(float), 2).tolist()}")
print(f"Feature 1 in the benign training patients: {benign[:, FEATURE].min():+.2f} to "
      f"{benign[:, FEATURE].max():+.2f} (z-scores), mean {benign[:, FEATURE].mean():+.2f}")

with open("qcml102726_wdbc_feature1_matrix.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["stage", "row", "column", "raw R", "A real part", "A imaginary part"])
    for stage, raw, A in (("start", raw_start, A_start), ("trained", raw_end, A_end)):
        for i in range(CLS_N):
            for j in range(CLS_N):
                writer.writerow([stage, i + 1, j + 1, f"{raw[i, j].item():.4f}",
                                 f"{A[i, j].real.item():.4f}", f"{A[i, j].imag.item():.4f}"])
with open("qcml102726_wdbc_feature1_eigenvalues.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["epoch"] + [f"eigenvalue {k + 1}" for k in range(CLS_N)])
    for epoch, values in enumerate(spectra):
        writer.writerow([epoch] + [f"{v:.4f}" for v in values])
    writer.writerow(["after L-BFGS"] + [f"{v:.4f}" for v in spectrum_end])

# One patient's read-back of Feature 1: the first benign validation patient. Their ground
# state overlaps each eigenvector of A_1 by p_k, and the read-back is sum_k p_k * lambda_k.
patient = int(np.flatnonzero(labels[val_idx] == 0)[0])
x = torch.tensor(Zva[[patient]], dtype=torch.float32)
with torch.no_grad():
    psi = wdbc_model.ground_state(x)[0]
    values, vectors = torch.linalg.eigh(A_end)
    overlaps = (vectors.conj().T @ psi).abs() ** 2
    read_back = wdbc_model.bloch(psi[None])[0, FEATURE].item()
    full_read_back = wdbc_model.bloch(psi[None])[0].numpy()
print(f"\nread-back example: benign validation patient {val_idx[patient]} (row in wdbc.data)")
print(f"   eigenvalues lambda_k : {np.round(values.numpy().astype(float), 2).tolist()}")
print(f"   overlaps p_k         : {np.round(overlaps.numpy().astype(float), 3).tolist()}  (sum {overlaps.sum():.3f})")
print(f"   sum p_k lambda_k = {(overlaps * values).sum():+.3f}   <psi|A_1|psi> = {read_back:+.3f}   "
      f"patient's own Feature 1 = {Zva[patient, FEATURE]:+.3f}")
print(f"   whole read-back: distance {np.linalg.norm(full_read_back - Zva[patient]):.2f} from the "
      f"patient's 30 values (the average benign training patient is "
      f"{np.linalg.norm(benign.mean(0) - Zva[patient]):.2f} away)")
with open("qcml102726_wdbc_read_back_example.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["k", "eigenvalue lambda_k", "overlap p_k", "p_k * lambda_k"])
    for k in range(CLS_N):
        writer.writerow([k + 1, f"{values[k].item():.4f}", f"{overlaps[k].item():.4f}",
                         f"{(overlaps[k] * values[k]).item():.4f}"])
    writer.writerow(["read-back", "", f"{overlaps.sum().item():.4f}", f"{read_back:.4f}"])
    writer.writerow(["patient's value", "", "", f"{Zva[patient, FEATURE]:.4f}"])
print("saved to qcml102726_wdbc_feature1_matrix.csv, qcml102726_wdbc_feature1_eigenvalues.csv, "
      "qcml102726_wdbc_read_back_example.csv")


# Plots: one PNG per plot, numbered in reading order, saved in PLOT_DIR
os.makedirs(PLOT_DIR, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "figure.titlesize": 13,
                     "figure.titleweight": "bold", "legend.fontsize": 9,
                     "legend.title_fontsize": 9, "legend.frameon": False,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.grid": True, "axes.axisbelow": True,
                     "grid.color": "#e1e0d9", "grid.linewidth": 0.6})
QCML_COLOR, PCA_COLOR, DM_COLOR = "#eb6834", "#2a78d6", "#1baf7a"
GRAY, INK, MUTED_INK, RULE = "#898781", "#0b0b0b", "#52514e", "#c3c2b7"
HELD_BACK = "#f4a3a3"
SEQUENTIAL = LinearSegmentedColormap.from_list("blues", ["#b7d3f6", "#104281"])


def new_plot(title, subtitle, size=(8.5, 7)):
    """A figure holding a single plot, with a bold title and a gray subtitle."""
    fig, ax = plt.subplots(figsize=size, layout="constrained")
    fig.suptitle(title)
    ax.set_title(subtitle, color=MUTED_INK)
    return fig, ax


def save_figure(fig, name):
    path = os.path.join(PLOT_DIR, f"{name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"   {path}")


def dot(color, label, size=6, marker="o", alpha=1.0):
    """A legend entry for scattered points, drawn at a readable size."""
    return Line2D([], [], ls="", marker=marker, ms=size, color=color, alpha=alpha, label=label)


def save_plot(fig, handles, name, ncol=2):
    """Put the legend under the plot, inside the figure's layout, then save and close it."""
    fig.legend(handles=handles, loc="outside lower center", ncol=ncol, alignment="left")
    save_figure(fig, name)


def table_plot(name, title, subtitle, header, rows, note="", label_width=0.44, size=11,
               digits=3):
    """A results table as a PNG. rows: (label, cells); a cell is text or a number."""
    show_cell = lambda c: c if isinstance(c, str) else f"{c:.{digits}f}"
    lines = [max(h.count("\n") + 1 for h in header)] + [label.count("\n") + 1 for label, _ in rows]
    line, pad, top = 0.22, 0.18, 1.0                        # inches
    note = "\n".join(textwrap.wrap(note, int(size * 14)))      # fits the page width
    note_height = 0.3 + 0.16 * note.count("\n") + 0.2 if note else 0.25
    height = top + sum(k * line + pad for k in lines) + note_height
    fig = plt.figure(figsize=(size, height))
    y_of = lambda inches: 1 - inches / height
    fig.text(0.02, y_of(0.35), title, fontsize=13, fontweight="bold", va="center")
    fig.text(0.02, y_of(0.65), subtitle, fontsize=10, color=MUTED_INK, va="center")
    width = (0.98 - label_width) / (len(header) - 1)
    y = top
    for i, (label, cells) in enumerate([(header[0], header[1:])] + rows):
        h = lines[i] * line + pad
        weight = "bold" if i == 0 else "normal"
        fig.text(0.02, y_of(y + h / 2), label, va="center", fontweight=weight, linespacing=1.4)
        for j, cell in enumerate(cells):
            fig.text(label_width + width * (j + 0.5), y_of(y + h / 2), show_cell(cell), ha="center",
                     va="center", fontweight=weight, linespacing=1.4)
        y += h
        fig.add_artist(plt.Line2D([0.02, 0.98], [y_of(y)] * 2, color=INK if i == 0 else RULE,
                                  lw=1 if i == 0 else 0.6))
    if note:
        fig.text(0.02, y_of(y + 0.2), note, fontsize=8.5, color=MUTED_INK, va="top",
                 linespacing=1.4)
    save_figure(fig, name)


def egg_plot(name, method, key, color, run, title):
    """One method on one run: the points it learned from, the points held back, where it
    put both, and the true egg."""
    keep = RUNS[run]
    recon = reconstructions[(key, run)]
    seen = score(key, run, True)
    subtitle = (f"Average distance to the true egg: {seen:.3f} for the learned points, "
                f"{score(key, run, False):.3f} for the held-back ones" if run != "all data" else
                f"Average distance to the true egg: {seen:.3f} (the noisy points sit 0.081 away)")
    fig, ax = new_plot(title, subtitle, size=(7.5, 8.4))
    handles = [dot(GRAY, "noisy points learned from (x, y)", alpha=0.6)]
    ax.scatter(*data[keep].T, s=5, alpha=0.3, color=GRAY, linewidths=0)
    if run != "all data":
        ax.scatter(*data[~keep].T, s=5, alpha=0.5, color=HELD_BACK, linewidths=0)
        handles.append(dot(HELD_BACK, "noisy points held back"))
    ax.scatter(*recon[keep].T, s=5, alpha=0.6, color=color, linewidths=0)
    handles.append(dot(color, f"{method} reconstruction, learned points"))
    if run != "all data":
        ax.scatter(*recon[~keep].T, s=6, alpha=0.8, color=INK, linewidths=0)
        handles.append(dot(INK, f"{method} reconstruction, held-back points"))
    handles += ax.plot(*egg_line.T, "--", color=INK, lw=1.2, label="true egg curve")
    if run == "cut-off":
        handles.append(ax.axvline(CUTOFF, color=INK, lw=1.0, ls=":", label=f"cut-off line, x = {CUTOFF:g}"))
    if run == "chunks":
        for center in chunk_centers:
            ax.add_patch(plt.Circle(center, CHUNK_RADIUS, fill=False, ls=":", lw=1.0, color=INK))
        handles.append(Line2D([], [], ls=":", color=INK, lw=1.0, label=f"removed chunks, radius {CHUNK_RADIUS}"))
    ax.set_aspect("equal")
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    save_plot(fig, handles, name)


print(f"\nplots saved to {PLOT_DIR}:")
pca_name = "PCA"
egg_plot("01_pca_all_data", pca_name, "PCA", PCA_COLOR, "all data",
         "PCA with one component flattens the egg onto a straight line")
egg_plot("02_pca_cut_off", pca_name, "PCA", PCA_COLOR, "cut-off",
         "Trained left of the cut-off line, PCA still draws a straight line")
egg_plot("03_pca_chunks", pca_name, "PCA", PCA_COLOR, "chunks",
         "With three chunks removed, PCA still draws a straight line")
egg_plot("04_qcml_all_data", "QCML", "QCML", QCML_COLOR, "all data",
         "QCML pulls the noisy points onto the egg")
egg_plot("05_qcml_cut_off", "QCML", "QCML", QCML_COLOR, "cut-off",
         "QCML recreates the egg where it had data, not past the cut-off")
egg_plot("06_qcml_chunks", "QCML", "QCML", QCML_COLOR, "chunks",
         "QCML fills the three removed chunks back in")
dm_name = "Diffusion map"
egg_plot("07_diffusion_map_all_data", dm_name, f"DM m={DM_M}", DM_COLOR, "all data",
         "The diffusion map also pulls the noisy points onto the egg")
egg_plot("08_diffusion_map_cut_off", dm_name, f"DM m={DM_M}", DM_COLOR, "cut-off",
         "Past the cut-off line, the diffusion map also leaves the egg")
egg_plot("09_diffusion_map_chunks", dm_name, f"DM m={DM_M}", DM_COLOR, "chunks",
         "The diffusion map also fills the three chunks back in")

table_plot("10_egg_results_table", "How close each method's reconstruction lies to the true egg",
           "Average distance to the true egg in egg units, after every point is sent through the "
           "trained model",
           ["Method", "All data", "Cut-off:\nlearned", "Cut-off:\nheld back", "Chunks:\nlearned",
            "Chunks:\nheld back"], table,
           note=(f"Cut-off: trained only on points with x < {CUTOFF:g} ({RUNS['cut-off'].sum():,} of {N:,}). "
                 f"Chunks: trained without {N_CHUNKS} random chunks of radius {CHUNK_RADIUS} "
                 f"({RUNS['chunks'].sum():,} of {N:,} kept). PCA uses 1 component, QCML n = {n} and "
                 f"w = {C2}, and the diffusion map m = {DM_M} coordinates with {DM_NEIGHBORS} neighbors "
                 "setting its kernel widths. Source: qcml102726_egg_table.csv."),
           label_width=0.40, size=12)

fig, ax = new_plot("One step of the random walk stays near where it starts",
                   f"Probability of stepping from the starred point to each of the {N:,} points\n"
                   f"(each point's kernel width is its distance to its {DM_NEIGHBORS}th nearest neighbor)",
                   size=(7.5, 8.4))
order = np.argsort(one_step)
shown = one_step[order] > 1e-6
ax.scatter(*data[order][~shown].T, s=5, color="#e1e0d9", linewidths=0)
dots = ax.scatter(*data[order][shown].T, c=one_step[order][shown], s=9, cmap=SEQUENTIAL,
                  norm=LogNorm(vmin=1e-6, vmax=one_step.max()), linewidths=0)
ax.scatter(*data[start_point], s=160, marker="*", color=INK, zorder=3)
fig.colorbar(dots, ax=ax, shrink=0.6, label="step probability")
ax.set_aspect("equal")
ax.set_xlim(-2.0, 2.0)
ax.set_ylim(-2.0, 2.0)
ax.set_xlabel("x")
ax.set_ylabel("y")
save_plot(fig, [dot("#e1e0d9", "points it cannot reach in one step (probability below one in a million)"),
                dot("#2a5a9a", "points it can reach (darker = more likely)"),
                dot(INK, "starting point", size=12, marker="*")], "11_diffusion_map_one_step", ncol=1)

fig, ax = new_plot("The diffusion map's first two coordinates turn the egg into a circle",
                   f"Each of the {N:,} points placed by its diffusion coordinates 1 and 2,\n"
                   "colored by its angle around the egg", size=(7.5, 8.0))
dots = ax.scatter(dm_all.right[:, 1], dm_all.right[:, 2], c=theta, s=6, cmap="twilight",
                  linewidths=0)
fig.colorbar(dots, ax=ax, shrink=0.6, label="angle around the egg (radians)")
ax.set_aspect("equal")
ax.set_xlabel("diffusion coordinate 1")
ax.set_ylabel("diffusion coordinate 2")
save_figure(fig, "12_diffusion_coordinates")

fig, ax = new_plot("Training pulls Feature 1's four eigenvalues inside the benign patients' range",
                   f"Eigenvalues of the 4 x 4 matrix for Feature 1 (mean radius) in the benign model of "
                   f"split {SPLIT},\nafter each Adam epoch and after the L-BFGS polish", size=(9, 7))
low, high = benign[:, FEATURE].min(), benign[:, FEATURE].max()
band = ax.axhspan(low, high, color="#e1e0d9", alpha=0.6,
                  label=f"Feature 1 (mean radius) of the {len(benign)} benign training patients, lowest to highest")
epochs = np.arange(WDBC_EPOCHS + 1)
shades = ["#f6c3a8", "#f0956a", "#eb6834", "#a8441a"]
handles = [band]
for k in range(CLS_N):
    handles += ax.plot(epochs, spectra[:, k], color=shades[k], lw=2, label=f"eigenvalue {k + 1}")
    ax.scatter([WDBC_EPOCHS + 12], [spectrum_end[k]], color=shades[k], s=40, zorder=3)
handles.append(dot(MUTED_INK, "after the L-BFGS polish"))
ax.set_xticks(list(range(0, WDBC_EPOCHS + 1, 25)) + [WDBC_EPOCHS + 12],
              [str(t) for t in range(0, WDBC_EPOCHS + 1, 25)] + ["L-BFGS"])
ax.set_xlabel("Adam epoch (0 = the random start)")
ax.set_ylabel("eigenvalue (in standard deviations of Feature 1)")
save_plot(fig, handles, "13_wdbc_feature1_eigenvalues", ncol=2)


# ========================================================================================
# EXPLANATION
# Notes on the code above, in the order of its sections. A tag [note k] in the code
# marks the section that note k explains.
# ========================================================================================
#
# [note 0] About this script
#
# Nothing here changes a result. It reruns the egg and one breast-cancer model to show,
# in pictures, what each method does:
# PART A  The egg (as in QCML092926): 2,500 noisy points around an egg-shaped curve. PCA,
#         QCML and a diffusion map each learn the shape three times: from all the points,
#         from only the points left of a cut-off line (x < 1, the second run of
#         QCML_92226), and with three random chunks cut out (the second run of
#         QCML092926). Every point, seen or not, is then sent through the trained model,
#         and the distance of its reconstruction from the true egg says how well the
#         method recreated the shape. Also QCML's intrinsic dimension on the egg: the egg
#         is one curve drawn in two dimensions, so the answer should be about 1.
# PART B  How the diffusion map sees the egg: where one step of its random walk can go,
#         and its first two coordinates, which turn the egg into a circle.
# PART C  What QCML's numbers look like on the breast cancer data: the 4 x 4 Hermitian
#         matrix of Feature 1 (mean radius) in the benign model of split 0, at the start
#         and after training, how its four eigenvalues move while it trains, and how one
#         patient's read-back of Feature 1 is a weighted average of them.
#
# Plots in "QCML Plots/qcml102726_plots", one PNG each, numbered in reading order; CSVs
# named qcml102726_*.csv. Run with python3.13 QCML102726.py.
# Runs in about 3 minutes.
#
# [note 1]
# Diffusion map: 30 neighbors set each point's kernel width (the middle of the 7, 30, 100
# tried on breast cancer and the reservoir), and m = 10 coordinates rebuild each point. The
# validation block cannot choose m here: a point is rebuilt from its own position, so more
# coordinates always match a held-out point more closely, and by m = 50 the rebuild is the
# noisy point itself. m = 2 and m = 50 are scored too, to show that trade-off.
#
# [note 2] PCA
#
# PCA fits the flat subspace that passes closest to the points: with 1 component, the
# straight line through their average along the direction they spread most. A point's
# reconstruction is its projection onto that line.
#
# [note 3] DIFFUSION MAP (as in QCML102326)
#
# Every pair of points gets a similarity exp(-distance^2 / (s_i s_j)), where s_i is point
# i's distance to its k-th nearest neighbor. Turned into step probabilities (each row
# divided by its sum, its "degree"), the similarities give a random walk that mostly hops
# between neighbors. The walk's slowest-changing patterns (its top eigenvectors) are
# smooth coordinates along the data. A new point steps onto the training points with its
# own probabilities, so its coordinates are averages over its neighbors (the Nystrom
# extension). Anything known on the training points, here their own (x, y), is rebuilt
# from the constant eigenvector plus the first m others (geometric harmonics).
#
# [note 4]
# Intrinsic dimension of the all-data QCML model. The quantum metric g(x) measures how fast
# the ground state turns as each of the 2 inputs is nudged. Its participation ratio,
# (sum of eigenvalues)^2 / (sum of squared eigenvalues), counts the directions that matter:
# 1 when the state only turns for moves along the egg, 2 when every direction counts.
