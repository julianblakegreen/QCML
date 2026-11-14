# Quantum Cognitive Machine Learning Project
# QCML on the Wisconsin Diagnostic Breast Cancer data (UCI dataset 17)
# What it does and how: see the notes at the end of this file.


import numpy as np
import torch
import matplotlib.pyplot as plt


# Repository set-up: run from this folder, with QCML from 1_Plain_QCML
import os
import sys
_FOLDER = os.path.dirname(os.path.abspath(__file__))
os.chdir(_FOLDER)                                                 # data and outputs live in this folder
sys.path.insert(0, os.path.join(_FOLDER, "..", "1_Plain_QCML"))   # qcml.py and hermitian.py

from qcml import QCML

EPOCHS = 150
BATCH = 64

FIT_N, FIT_C2 = 16, 0.03      # Part 1: fitting the shape of the whole data set
CLS_N, CLS_C2 = 4, 0.1        # Part 2: one manifold per class, for classifying
N_SPLITS = 5                  # how many random train/test splits to average over
TRAIN_FRACTION = 0.7


# Data
rows = np.genfromtxt("wdbc.data", delimiter=",", dtype=str)
labels = (rows[:, 1] == "M").astype(int)          # 1 = malignant, 0 = benign
features = rows[:, 2:].astype(float)              # (569, 30)
N, D = features.shape
print(f"{N} samples, {D} features, {labels.sum()} malignant, {N - labels.sum()} benign")
print(f"widest feature std / narrowest: {features.std(0).max() / features.std(0).min():,.0f}")


def normalize(train_points, points):
    """Z-score `points` using the mean and std of `train_points` (one per feature)."""
    return (points - train_points.mean(0)) / train_points.std(0)


def split_data(seed):
    """Random train/test split of the row numbers."""
    order = np.random.default_rng(seed).permutation(N)
    cut = int(TRAIN_FRACTION * N)
    return order[:cut], order[cut:]


# QCML

def compute_loss(model, points, c2):
    psi = model.ground_state(points)
    L1 = ((model.bloch(psi) - points) ** 2).sum(dim=1).mean()    # position error
    L2 = model.uncertainty(psi).mean()                           # fluctuation
    return L1 + c2 * L2, L1, L2


def train_qcml(points, n, c2, seed=0, label=""):
    """Train a QCML model on `points`, a numpy table with one row per sample."""
    X = torch.tensor(points, dtype=torch.float32)
    M = len(X)
    torch.manual_seed(seed)
    model = QCML(D=points.shape[1], n=n, w=c2)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    history_L1, history_L2 = [], []
    for epoch in range(1, EPOCHS + 1):
        order = torch.randperm(M)
        for start in range(0, M, BATCH):
            loss, _, _ = compute_loss(model, X[order[start:start + BATCH]], c2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            loss, L1, L2 = compute_loss(model, X, c2)
        history_L1.append(L1.item())
        history_L2.append(L2.item())
        if label and (epoch == 1 or epoch % 50 == 0):
            print(f"{label}epoch {epoch:3d}   loss {loss.item():.4f}   "
                  f"L1 {L1.item():.4f}   L2 {L2.item():.4f}")

    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=300,
                              line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        loss, _, _ = compute_loss(model, X, c2)
        loss.backward()
        return loss

    lbfgs.step(closure)
    if label:
        with torch.no_grad():
            loss, L1, L2 = compute_loss(model, X, c2)
        print(f"{label}after L-BFGS   loss {loss.item():.4f}   "
              f"L1 {L1.item():.4f}   L2 {L2.item():.4f}")
    return model, history_L1, history_L2


def reconstruct(model, points):
    """Where the model thinks each sample really belongs."""
    return model.reconstruct(torch.tensor(points, dtype=torch.float32)).numpy()


def fit_error(model, points):
    """Typical distance between a sample and its reconstruction."""
    return float(np.sqrt(((reconstruct(model, points) - points) ** 2).sum(1).mean()))


def distance_to_manifold(model, points):
    """Per sample: squared distance from its reconstruction. How well it fits."""
    X = torch.tensor(points, dtype=torch.float32)
    with torch.no_grad():
        psi = model.ground_state(X)
        return ((model.bloch(psi) - X) ** 2).sum(1).numpy()


# one manifold for the whole data set
print("\nPART 1: fitting one manifold to all the data")
train_idx, test_idx = split_data(0)
Z_train = normalize(features[train_idx], features[train_idx])
Z_test = normalize(features[train_idx], features[test_idx])

model, history_L1, history_L2 = train_qcml(Z_train, FIT_N, FIT_C2, label="   ")

# P1 always answer with the average sample.
mean_baseline = float(np.sqrt((Z_test ** 2).sum(1).mean()))

# PCA, the linear version of the same idea (project onto a flat subspace).
centre = Z_train.mean(0)
_, _, directions = np.linalg.svd(Z_train - centre, full_matrices=False)
pca_ranks = np.arange(1, 13)
pca_errors = []
for k in pca_ranks:
    P = directions[:k]
    projected = (Z_test - centre) @ P.T @ P + centre
    pca_errors.append(float(np.sqrt(((projected - Z_test) ** 2).sum(1).mean())))

print(f"\nreconstruction error on held-out samples (in standard deviations):")
print(f"   answer with the average sample : {mean_baseline:.3f}")
print(f"   QCML (n = {FIT_N})                  : {fit_error(model, Z_test):.3f}"
      f"   [on its training data: {fit_error(model, Z_train):.3f}]")
for k in (2, 5, 8):
    print(f"   PCA with {k} components          : {pca_errors[k - 1]:.3f}")

# Intrinsic dimension: the quantum metric says how fast the quantum state turns
# as each feature is nudged. Its eigenvalues count the directions that matter.
metric = model.quantum_metric(torch.tensor(np.vstack([Z_train, Z_test]), dtype=torch.float32))
metric_eigs = torch.linalg.eigvalsh(metric).clamp(min=0)          # (samples, D)
share = metric_eigs / metric_eigs.sum(-1, keepdim=True)
intrinsic_dim = float((share.sum(-1) ** 2 / (share ** 2).sum(-1)).mean())
print(f"\nintrinsic dimension from the quantum metric: {intrinsic_dim:.2f}"
      f"   (the 30 measurements move along about this many independent directions)")


# P2 one manifold per diagnosis, used as a classifier
print(f"\nPART 2: one manifold per diagnosis, averaged over {N_SPLITS} splits")
qcml_acc, centroid_acc, logistic_acc = [], [], []

for seed in range(N_SPLITS):
    train_idx, test_idx = split_data(seed)
    Ztr = normalize(features[train_idx], features[train_idx])
    Zte = normalize(features[train_idx], features[test_idx])
    ytr, yte = labels[train_idx], labels[test_idx]

    model_B, _, _ = train_qcml(Ztr[ytr == 0], CLS_N, CLS_C2, seed=seed)   # benign only
    model_M, _, _ = train_qcml(Ztr[ytr == 1], CLS_N, CLS_C2, seed=seed)   # malignant only

    # A sample is called malignant if it fits the malignant manifold more closely.
    d_B = distance_to_manifold(model_B, Zte)
    d_M = distance_to_manifold(model_M, Zte)
    qcml_acc.append(float(((d_M < d_B).astype(int) == yte).mean()))

    # Baseline 1: nearest class average.
    cB, cM = Ztr[ytr == 0].mean(0), Ztr[ytr == 1].mean(0)
    nearer_M = ((Zte - cM) ** 2).sum(1) < ((Zte - cB) ** 2).sum(1)
    centroid_acc.append(float((nearer_M.astype(int) == yte).mean()))

    # Baseline 2: logistic regression, the standard linear classifier.
    W = torch.zeros(D, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    Xt = torch.tensor(Ztr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    opt = torch.optim.Adam([W, b], lr=0.05)
    for _ in range(600):
        logits = Xt @ W + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt) + 1e-3 * W.square().sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        predicted = ((torch.tensor(Zte, dtype=torch.float32) @ W + b) > 0).numpy().astype(int)
    logistic_acc.append(float((predicted == yte).mean()))

    print(f"   split {seed}: QCML {qcml_acc[-1]:.3f}   "
          f"nearest centroid {centroid_acc[-1]:.3f}   logistic {logistic_acc[-1]:.3f}")

print(f"\naccuracy over {N_SPLITS} splits:")
for name, values in [("QCML (n = %d)" % CLS_N, qcml_acc),
                     ("nearest centroid", centroid_acc),
                     ("logistic regression", logistic_acc)]:
    print(f"   {name:>22}: {np.mean(values):.3f} +/- {np.std(values):.3f}")



# Plot

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# training curves
# ax = axes[0][0]
epochs = np.arange(1, EPOCHS + 1)
ax1.plot(epochs, history_L1, label="L1  position error")
ax1.plot(epochs, history_L2, label="L2  fluctuation")
ax1.plot(epochs, np.array(history_L1) + FIT_C2 * np.array(history_L2), "k",
        label=f"loss = L1 + {FIT_C2:g}*L2")
ax1.set_yscale("log")
ax1.set_xlabel("epoch")
ax1.set_ylabel("value")
ax1.set_title(f"training, whole data set (n = {FIT_N})")
ax1.legend(fontsize=8)

# how well the data is described, QCML vs PCA
# ax = axes[0][1]
ax2.plot(pca_ranks, pca_errors, "o-", color="#2563eb", label="PCA (flat subspace)")
ax2.axhline(fit_error(model, Z_test), color="red", ls="--",
           label=f"QCML, n = {FIT_N}")
ax2.axhline(mean_baseline, color="grey", ls=":", label="answer with the average")
ax2.set_xlabel("PCA components")
ax2.set_ylabel("error on held-out samples")
ax2.set_title("reconstruction error")
ax2.legend(fontsize=8)

# # quantum metric spectrum
# ax = axes[1][0]
# mean_spectrum = metric_eigs.mean(0).flip(0).numpy()
# ax.plot(np.arange(1, D + 1), mean_spectrum / mean_spectrum.max(), "o-", ms=4, color="#7c3aed")
# ax.set_yscale("log")
# ax.set_xlabel("direction (sorted)")
# ax.set_ylabel("relative eigenvalue")
# ax.set_title(f"quantum metric spectrum\nintrinsic dimension = {intrinsic_dim:.2f}")

# the classifier, on the last split
# ax = axes[1][1]
for value, colour, name in [(0, "#2563eb", "benign"), (1, "#dc2626", "malignant")]:
    sel = yte == value
    ax3.scatter(d_B[sel], d_M[sel], s=12, alpha=0.6, color=colour, label=name)
top = max(d_B.max(), d_M.max()) * 1.05
ax3.plot([0, top], [0, top], "k--", lw=1, label="decision boundary")
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_xlabel("distance to the benign manifold")
ax3.set_ylabel("distance to the malignant manifold")
ax3.set_title(f"classifying held-out samples (n = {CLS_N})")
ax3.legend(fontsize=8)

fig.tight_layout()
fig.savefig("qcml_wdbc.png", dpi=150)
plt.show()


# ========================================================================================
# EXPLANATION
# Notes on the code above, in the order of its sections. A tag [note k] in the code
# marks the section that note k explains.
# ========================================================================================
#
# [note 0] About this script
#
# 569 samples, 30 features measured from cell nuclei in a digitised image,
# each labelled benign (357) or malignant (212).
#
# Part 1 fits one QCML manifold to the data and asks how well it describes it.
# Part 2 fits one manifold per diagnosis and classifies by which one a point fits.
