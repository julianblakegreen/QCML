# Quantum Cognitive Machine Learning Project
# QCML110426: tables and figures for the results chapter
# Runs in about 2 minutes.
# What it does and how: see the notes at the end of this file.


import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


# Repository set-up: run from this folder, with QCML from 1_Plain_QCML
import os
import sys
_FOLDER = os.path.dirname(os.path.abspath(__file__))
os.chdir(_FOLDER)                                                 # data and outputs live in this folder
sys.path.insert(0, os.path.join(_FOLDER, "..", "1_Plain_QCML"))   # qcml.py and hermitian.py

from qcml import QCML

torch.set_num_threads(1)      # the matrices are tiny, so extra threads cost more than they save

PLOT_DIR = os.path.join("QCML Plots", "qcml110426_plots")

# Settings, all as in QCML102326_2 (Part A2) and QCML100626 (the linear baselines)
EPOCHS = 150
BATCH = 64
CLS_N, CLS_C2 = 4, 0.1        # one manifold per diagnosis
SUP_N = 8                     # supervised QCML Hilbert-space dimension
SUP_LAMBDA, SUP_W = 0.1, 0.1  # mixed loss: weight of the reconstruction part and its fluctuation weight
SUP_EPOCHS = 100              # supervised QCML training passes (no L-BFGS: it overfitted)
N_SPLITS = 10
TRAIN_FRACTION = 0.7
COMPONENTS = np.arange(1, 21) # PCA components k and diffusion-map eigenvectors m to try
DM_NEIGHBOURS = 7
LOGISTIC_STEPS, LOGISTIC_RATE, LOGISTIC_PENALTY = 600, 0.05, 1e-3


# PART A: BREAST CANCER

rows = np.genfromtxt("wdbc.data", delimiter=",", dtype=str)
labels = (rows[:, 1] == "M").astype(int)          # 1 = malignant, 0 = benign
features = rows[:, 2:].astype(float)              # (569, 30)
N, D = features.shape


def normalize(train_points, points):
    """Z-score `points` using the mean and std of `train_points` (one per feature)."""
    return (points - train_points.mean(0)) / train_points.std(0)


def split_data(seed):
    """Stratified random split: 70% of the benign and 70% of the malignant row
    numbers train, the rest are held back for validation."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for value in (0, 1):
        rows_of_class = rng.permutation(np.flatnonzero(labels == value))
        cut = round(TRAIN_FRACTION * len(rows_of_class))
        train_idx.append(rows_of_class[:cut])
        val_idx.append(rows_of_class[cut:])
    return np.sort(np.concatenate(train_idx)), np.sort(np.concatenate(val_idx))


SPLITS = [split_data(seed) for seed in range(N_SPLITS)]
N_TRAIN, N_VAL = len(SPLITS[0][0]), len(SPLITS[0][1])


# QCML (as in QCML102326_2)

def compute_loss(model, points, c2):
    psi = model.ground_state(points)
    L1 = ((model.bloch(psi) - points) ** 2).sum(dim=1).mean()    # position error
    L2 = model.uncertainty(psi).mean()                           # fluctuation
    return L1 + c2 * L2, L1, L2


def train_qcml(points, n, c2, seed=0):
    """Train a QCML model on `points`, a numpy table with one row per sample."""
    X = torch.tensor(points, dtype=torch.float32)
    M = len(X)
    torch.manual_seed(seed)
    model = QCML(D=points.shape[1], n=n, w=c2)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(EPOCHS):
        order = torch.randperm(M)
        for start in range(0, M, BATCH):
            loss, _, _ = compute_loss(model, X[order[start:start + BATCH]], c2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=300,
                              line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        loss, _, _ = compute_loss(model, X, c2)
        loss.backward()
        return loss

    lbfgs.step(closure)
    return model


def distance_to_manifold(model, points):
    """Per sample: squared distance from its reconstruction. How well it fits."""
    X = torch.tensor(points, dtype=torch.float32)
    with torch.no_grad():
        psi = model.ground_state(X)
        return ((model.bloch(psi) - X) ** 2).sum(1).numpy()


def ground_state_from(model, values, known):
    """Ground state built from the `known` features only: their matrices alone make the
    error Hamiltonian. Differentiable, so it can be trained through."""
    H = model.observables()[known]
    X = torch.as_tensor(values, dtype=torch.float32)
    I = torch.eye(model.n, dtype=H.dtype)
    E = (torch.einsum("dij,djk->ik", H, H) - 2 * torch.einsum("bd,dij->bij", X.to(H.dtype), H)
         + (X ** 2).sum(1).to(H.dtype)[:, None, None] * I)
    return torch.linalg.eigh(E)[1][:, :, 0]


def expectation(psi, A):
    """<psi|A|psi> for a batch of states: shape (batch,)."""
    return torch.einsum("bi,ij,bj->b", psi.conj(), A, psi).real


def train_supervised(inputs, target, n, loss_fn, lam=0.0, w=0.1, seed=0, epochs=EPOCHS):
    """Supervised QCML (Di Caro et al., 2025): K input matrices plus a target observable B
    (the last matrix), trained so <psi(x)|B|psi(x)> predicts `target`; lam > 0 adds the
    mixed-loss reconstruction term. No L-BFGS polish, as in QCML102326_2's breast-cancer runs."""
    X = torch.tensor(inputs, dtype=torch.float32)
    Y = torch.tensor(target, dtype=torch.float32)
    K = X.shape[1]
    known = list(range(K))
    torch.manual_seed(seed)
    model = QCML(D=K + 1, n=n)

    def loss_on(rows):
        psi = ground_state_from(model, X[rows], known)
        H = model.observables()
        loss = loss_fn(expectation(psi, H[K]), Y[rows])
        if lam:
            A = H[:K]
            position = torch.einsum("bi,dij,bj->bd", psi.conj(), A, psi).real
            fluctuation = (expectation(psi, torch.einsum("dij,djk->ik", A, A))
                           - (position ** 2).sum(1))
            loss = loss + lam * (((position - X[rows]) ** 2).sum(1).mean()
                                 + w * fluctuation.mean())
        return loss

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(epochs):
        order = torch.randperm(len(X))
        for start in range(0, len(X), BATCH):
            loss = loss_on(order[start:start + BATCH])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def supervised_forecast(model, inputs):
    """<psi(x)|B|psi(x)> for every row of `inputs`, B being the last matrix."""
    K = inputs.shape[1]
    with torch.no_grad():
        psi = ground_state_from(model, torch.tensor(inputs, dtype=torch.float32), list(range(K)))
        return expectation(psi, model.observables()[K]).numpy()


# PCA AND THE DIFFUSION MAP (as in QCML102326_2): one model per diagnosis

def pca_distances(train_points, points, ks=COMPONENTS):
    """Squared distance from each sample to the flat PCA subspace of `train_points`,
    for every k in ks at once: shape (samples, len(ks))."""
    centre = train_points.mean(0)
    _, _, directions = np.linalg.svd(train_points - centre, full_matrices=False)
    offset = points - centre
    explained = np.cumsum((offset @ directions[:ks.max()].T) ** 2, axis=1)
    return (offset ** 2).sum(1, keepdims=True) - explained[:, ks - 1]


def squared_distances(A, B):
    return np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T, 0)


class DiffusionMap:
    def __init__(self, points, neighbours=DM_NEIGHBOURS):
        self.points, self.neighbours = points, neighbours
        d2 = squared_distances(points, points)
        self.width = np.sqrt(np.sort(d2, axis=1)[:, neighbours])    # column 0 is the point itself
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
        width = np.sqrt(np.sort(d2, axis=1)[:, self.neighbours - 1])
        K = np.exp(-d2 / np.outer(width, self.width))
        return K / K.sum(1, keepdims=True)

    def rebuild(self, values, points, ms):
        """Rebuild `values` (one row per training point) at `points` from the constant
        eigenvector plus the first m others, for every m in ms:
        shape (len(ms), points, columns of values)."""
        J = ms.max() + 1
        at_points = self.steps(points) @ self.right[:, :J] / self.eigenvalues[:J]
        shares = self.left[:, :J].T @ values
        partial = np.cumsum(at_points[:, :, None] * shares[None, :, :], axis=1)
        return np.moveaxis(partial[:, ms, :], 1, 0)


def dm_distances(train_points, points, ms=COMPONENTS):
    """Squared distance from each sample to its diffusion-map reconstruction, for every m
    in ms: shape (samples, len(ms))."""
    rebuilt = DiffusionMap(train_points).rebuild(train_points, points, ms)
    return ((points[None, :, :] - rebuilt) ** 2).sum(2).T


# THE LINEAR BASELINES (as in QCML100626)

def logistic_scores(Ztr, ytr, Zva):
    """Logistic regression: log-odds w.x + b, trained by Adam on the cross-entropy with a
    small penalty on the weights."""
    W = torch.zeros(D, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    Xt = torch.tensor(Ztr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    opt = torch.optim.Adam([W, b], lr=LOGISTIC_RATE)
    for _ in range(LOGISTIC_STEPS):
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(Xt @ W + b, yt)
                + LOGISTIC_PENALTY * W.square().sum())
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return (torch.tensor(Zva, dtype=torch.float32) @ W + b).numpy()


# SCORING (as in QCML102326_2)

def diagnostic_metrics(score, y):
    """Accuracy, sensitivity, specificity, balanced accuracy and ROC AUC of a score that is
    above 0 for a patient called malignant."""
    called = score > 0
    sensitivity = called[y == 1].mean()
    specificity = 1 - called[y == 0].mean()
    # ROC AUC by counting pairs (Mann-Whitney): ties count half
    malignant, benign = score[y == 1], score[y == 0]
    auc = ((malignant[:, None] > benign[None, :]).mean()
           + 0.5 * (malignant[:, None] == benign[None, :]).mean())
    return [(called == y).mean(), sensitivity, specificity, (sensitivity + specificity) / 2, auc]


def z_score(differences):
    """How many (corrected) standard errors the mean of `differences` is from zero:
    z = mean(d) / sqrt((1/S + n_val/n_train) var(d)) (Nadeau and Bengio, 2003)."""
    d = np.asarray(differences, dtype=float)
    if np.ptp(d) < 1e-12:
        return 0.0 if abs(d.mean()) < 1e-12 else float(np.sign(d.mean()) * np.inf)
    error = np.sqrt((1 / len(d) + N_VAL / N_TRAIN) * d.var(ddof=1))
    return float(d.mean() / error)


METRIC_NAMES = ["accuracy", "sensitivity", "specificity", "balanced accuracy", "ROC AUC"]

print(f"PART A: breast cancer, every classifier on the same {N_SPLITS} splits "
      f"({N_TRAIN} train / {N_VAL} validation)")
scores = []                   # one dict per split: method -> score (above 0 = malignant)
for seed, (train_idx, val_idx) in enumerate(SPLITS):
    Ztr = normalize(features[train_idx], features[train_idx])
    Zva = normalize(features[train_idx], features[val_idx])
    ytr, yva = labels[train_idx], labels[val_idx]
    model_B = train_qcml(Ztr[ytr == 0], CLS_N, CLS_C2, seed=seed)   # benign only
    model_M = train_qcml(Ztr[ytr == 1], CLS_N, CLS_C2, seed=seed)   # malignant only
    sup_model = train_supervised(Ztr, ytr.astype(float), SUP_N,
                                 torch.nn.functional.binary_cross_entropy_with_logits,
                                 lam=SUP_LAMBDA, w=SUP_W, seed=seed, epochs=SUP_EPOCHS)
    centroid_B, centroid_M = Ztr[ytr == 0].mean(0), Ztr[ytr == 1].mean(0)
    scores.append({
        "y": yva,
        "QCML": distance_to_manifold(model_B, Zva) - distance_to_manifold(model_M, Zva),
        "Supervised QCML": supervised_forecast(sup_model, Zva),
        "PCA": pca_distances(Ztr[ytr == 0], Zva) - pca_distances(Ztr[ytr == 1], Zva),
        "Diffusion map": dm_distances(Ztr[ytr == 0], Zva) - dm_distances(Ztr[ytr == 1], Zva),
        "Nearest centroid": ((Zva - centroid_B) ** 2).sum(1) - ((Zva - centroid_M) ** 2).sum(1),
        "Logistic regression": logistic_scores(Ztr, ytr, Zva)})
    print(f"   split {seed} done")

# PCA and the diffusion map at the k and m with the highest mean validation accuracy, as
# in QCML102326_2 (choosing on the validation patients flatters them slightly)
SETTING = {"QCML": f"n = {CLS_N}", "Supervised QCML": f"n = {SUP_N}", "Nearest centroid": "none",
           "Logistic regression": f"penalty {LOGISTIC_PENALTY:g}"}
for method, letter in (("PCA", "k"), ("Diffusion map", "m")):
    accuracy = np.mean([((s[method] > 0).astype(int) == s["y"][:, None]).mean(0) for s in scores], 0)
    best = int(accuracy.argmax())
    for s in scores:
        s[method] = s[method][:, best]
    SETTING[method] = f"{letter} = {COMPONENTS[best]}"
CLASSIFIERS = ["Supervised QCML", "QCML", "Diffusion map", "Logistic regression", "PCA", "Nearest centroid"]
metrics = {m: np.array([diagnostic_metrics(s[m], s["y"]) for s in scores]) for m in CLASSIFIERS}   # (splits, 5)

print("\nvalidation scores, mean ± std over the splits; z of accuracy against supervised QCML and against QCML")
with open("qcml110426_wdbc_results.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["method", "setting"] + [f"{name} {stat}" for name in METRIC_NAMES for stat in ("mean", "std")]
                    + ["accuracy minus supervised QCML", "z against supervised QCML",
                       "accuracy minus QCML", "z against QCML"])
    for m in CLASSIFIERS:
        values = metrics[m]
        against = []
        for reference in ("Supervised QCML", "QCML"):
            d = values[:, 0] - metrics[reference][:, 0]
            against += [d.mean(), z_score(d)]
        print(f"   {m:<20} ({SETTING[m]:<12})" + "".join(f"  {v.mean():.3f} ± {v.std():.3f}" for v in values.T)
              + f"   vs supervised QCML {against[0]:+.3f} (z {against[1]:+.1f})   vs QCML {against[2]:+.3f} (z {against[3]:+.1f})")
        writer.writerow([m, SETTING[m]] + [x for v in values.T for x in (v.mean(), v.std())] + against)
print("saved to qcml110426_wdbc_results.csv")
with open("qcml110426_wdbc_splits.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["split"] + [f"{m} accuracy" for m in CLASSIFIERS])
    for split in range(N_SPLITS):
        writer.writerow([split] + [metrics[m][split, 0] for m in CLASSIFIERS])
print("saved to qcml110426_wdbc_splits.csv")


# PART B: THESIS FIGURES FROM THE SAVED RESERVOIR RESULTS

os.makedirs(PLOT_DIR, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                     "axes.formatter.use_mathtext": True, "font.size": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                     "legend.frameon": False, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.grid": True, "axes.axisbelow": True,
                     "grid.color": "#e1e0d9", "grid.linewidth": 0.5})
QCML_COLOR, DM_COLOR, KRR_COLOR = "#eb6834", "#1baf7a", "#2a78d6"
GRAY, INK, MUTED_INK = "#898781", "#0b0b0b", "#52514e"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(PLOT_DIR, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"   {os.path.join(PLOT_DIR, name)}.pdf")


def read_csv(path):
    with open(path) as file:
        return list(csv.DictReader(file))


print(f"\nPART B: figures saved to {PLOT_DIR}:")

# The final comparison, one figure per horizon (QCML110226)
final = read_csv("qcml110226_readouts.csv")
SETTINGS = ["vacuum only, exact", "vacuum only, 9^5 shots", "40 numbers, exact", "40 numbers, 9^5 shots",
            "all 160, 9^5 shots", "weak record, 1e4 runs", "weak record, 1e6 runs"]
TICKS = ["vacuum,\nexact", "vacuum,\n$9^5$ shots", "40 values,\nexact", "40 values,\n$9^5$ shots",
         "all 160,\n$9^5$ shots", "weak,\n$10^4$ runs", "weak,\n$10^6$ runs"]
READOUTS = [("Diffusion map", "diffusion map", DM_COLOR, "s"), ("Kernel ridge", "kernel ridge regression", KRR_COLOR, "o"),
            ("QCML+", "QCML+", QCML_COLOR, "^")]
for delay in ("1", "20"):
    cells = {row["setting"]: row for row in final if row["delay"] == delay}
    fig, ax = plt.subplots(figsize=(6.3, 2.7), layout="constrained")
    xs = np.arange(len(SETTINGS))
    for k, (name, label, color, marker) in enumerate(READOUTS):
        means = [float(cells[s][f"{name} mean"]) for s in SETTINGS]
        stds = [float(cells[s][f"{name} std"]) for s in SETTINGS]
        ax.errorbar(xs + (k - 1) * 0.2, means, yerr=stds, color=color, marker=marker, ms=5.5,
                    mec="white", mew=0.6, ls="none", capsize=1.5, elinewidth=0.8, label=label)
    ax.set_yscale("log")
    ax.set_xticks(xs, TICKS)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("test NRMSE")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    save(fig, f"results_final_delay_{delay}")

# Experiment time against accuracy, one figure per horizon (QCML102426_2)
cost = read_csv("qcml102426_2_cost.csv")
TIME_TICKS = [(1e-2, "10 ms"), (1, "1 s"), (60, "1 min"), (3600, "1 h"), (86400, "1 day"), (2.63e6, "1 month")]
for delay in ("1", "20"):
    rows_here = [row for row in cost if row["delay"] == delay]
    fig, ax = plt.subplots(figsize=(3.1, 2.6), layout="constrained")
    for prefix, readout, label, color, marker, line in [
            ("theirs", "Linear readout (theirs)", "projective, linear readout", GRAY, "o", ":"),
            ("theirs", "Supervised QCML", "projective, supervised QCML", KRR_COLOR, "s", "-"),
            ("weak", "Supervised QCML", "weak, supervised QCML", QCML_COLOR, "^", "-")]:
        chosen = [row for row in rows_here if row["protocol"].startswith(prefix)]
        times = [float(row["experiment time (s)"]) for row in chosen]
        errors = [float(row[f"{readout} mean"]) for row in chosen]
        ax.plot(times, errors, line, color=color, marker=marker, ms=4.5, mec="white", mew=0.5, lw=1.1,
                label=label)
        if readout == "Supervised QCML":
            # label each point with its number of shots (9^k) or runs (10^k)
            for row, t, e in zip(chosen, times, errors):
                count = row["protocol"].split(",")[1].split()[0]          # "9^5" or "1e+04"
                text = f"${count}$" if prefix == "theirs" else f"$10^{round(np.log10(float(count)))}$"
                ax.annotate(text, (t, e), textcoords="offset points", xytext=(3, 3), fontsize=6.5, color=MUTED_INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([t for t, _ in TIME_TICKS], [name for _, name in TIME_TICKS])
    y_ticks = [0.003, 0.01, 0.03] if delay == "1" else [0.04, 0.06, 0.1, 0.15]
    ax.set_yticks(y_ticks, [f"{v:g}" for v in y_ticks])
    ax.minorticks_off()
    ax.set_xlabel("experiment time")
    ax.set_ylabel("test NRMSE")
    if delay == "1":
        ax.legend(loc="lower left")
    save(fig, f"results_cost_delay_{delay}")


# ========================================================================================
# EXPLANATION
# Notes on the code above, in the order of its sections. A tag [note k] in the code
# marks the section that note k explains.
# ========================================================================================
#
# [note 0] About this script
#
# Part A reruns the all-feature breast-cancer comparison of QCML102326_2 (the same ten
# splits, seeds and settings) and adds the two linear baselines of QCML100626, nearest
# centroid and logistic regression, so that every classifier is scored on the same splits,
# with the same diagnostic measures and the paired z-test of the methods chapter
# (Nadeau and Bengio, 2003). It saves qcml110426_wdbc_results.csv and, per split,
# qcml110426_wdbc_splits.csv.
#
# Part B redraws the reservoir results for the thesis from the saved CSVs of QCML102426_2
# (weak measurement and experiment time) and QCML110226 (the final comparison), in Computer
# Modern like the methods figures of QCML102826: no titles, PDF + PNG, in
# "QCML Plots/qcml110426_plots". Nothing in Part B is recomputed.
#
# Run with python3.13 QCML110426.py. Runs in about 2 minutes.
