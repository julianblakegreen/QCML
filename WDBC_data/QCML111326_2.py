# Quantum Cognitive Machine Learning Project
# QCML111326_2: breast cancer with missing features, for all six classifiers
# Runs in about 10 minutes on 8 cores (32 removal settings, one process each).
# What it does and how: see the notes at the end of this file.


import os
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")    # one thread per process (before numpy loads)
import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

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

WORKERS = os.cpu_count()

# Settings, all as in QCML110426 (and so QCML102326_2 and QCML100626)
EPOCHS = 150
BATCH = 64
CLS_N, CLS_C2 = 4, 0.1        # one manifold per diagnosis
SUP_N = 8                     # supervised QCML Hilbert-space dimension
SUP_LAMBDA, SUP_W = 0.1, 0.1  # mixed loss: weight of the reconstruction part and its fluctuation weight
SUP_EPOCHS = 100              # supervised QCML training passes (no L-BFGS: it overfitted)
N_SPLITS = 10
TRAIN_FRACTION = 0.7
PCA_K, DM_M = 2, 15           # QCML110426's choices on all features
DM_NEIGHBOURS = 7
LOGISTIC_STEPS, LOGISTIC_RATE, LOGISTIC_PENALTY = 600, 0.05, 1e-3
TOP_THREE = [28, 13, 12]      # Features 29, 14 and 13, counting columns from 0 (QCML102326_2's pick)
CLASSIFIERS = ["Supervised QCML", "QCML", "Diffusion map", "Logistic regression", "PCA", "Nearest centroid"]

MEASUREMENTS = ["radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity",
                "concave points", "symmetry", "fractal dimension"]
FEATURE_NAMES = [form.format(m) for form in ("mean {}", "{} variation", "largest {}") for m in MEASUREMENTS]

rows = np.genfromtxt("wdbc.data", delimiter=",", dtype=str)
labels = (rows[:, 1] == "M").astype(int)          # 1 = malignant, 0 = benign
features = rows[:, 2:].astype(float)              # (569, 30)
N, D = features.shape


def feature_label(f):
    return f"Feature {f + 1} ({FEATURE_NAMES[f]})"


def normalize(train_points, points):
    """Z-score `points` using the mean and std of `train_points` (one per feature)."""
    return (points - train_points.mean(0)) / train_points.std(0)


def split_data(seed):
    """Stratified random split: 70% of the benign and 70% of the malignant row numbers train,
    the rest are held back for validation."""
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


# QCML (as in QCML110426)

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
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=300, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        loss, _, _ = compute_loss(model, X, c2)
        loss.backward()
        return loss

    lbfgs.step(closure)
    return model


def distance_to_manifold(model, points):
    """Per sample: squared distance from its reconstruction."""
    X = torch.tensor(points, dtype=torch.float32)
    with torch.no_grad():
        psi = model.ground_state(X)
        return ((model.bloch(psi) - X) ** 2).sum(1).numpy()


def ground_state_from(model, values, known):
    """Ground state built from the `known` features only."""
    H = model.observables()[known]
    X = torch.as_tensor(values, dtype=torch.float32)
    I = torch.eye(model.n, dtype=H.dtype)
    E = (torch.einsum("dij,djk->ik", H, H) - 2 * torch.einsum("bd,dij->bij", X.to(H.dtype), H)
         + (X ** 2).sum(1).to(H.dtype)[:, None, None] * I)
    return torch.linalg.eigh(E)[1][:, :, 0]


def expectation(psi, A):
    return torch.einsum("bi,ij,bj->b", psi.conj(), A, psi).real


def train_supervised(inputs, target, n, loss_fn, lam=0.0, w=0.1, seed=0, epochs=EPOCHS):
    """Supervised QCML with the mixed loss, no L-BFGS (as in QCML110426)."""
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
            fluctuation = (expectation(psi, torch.einsum("dij,djk->ik", A, A)) - (position ** 2).sum(1))
            loss = loss + lam * (((position - X[rows]) ** 2).sum(1).mean() + w * fluctuation.mean())
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
    K = inputs.shape[1]
    with torch.no_grad():
        psi = ground_state_from(model, torch.tensor(inputs, dtype=torch.float32), list(range(K)))
        return expectation(psi, model.observables()[K]).numpy()


# PCA AND THE DIFFUSION MAP (as in QCML110426): one model per diagnosis

def pca_distance(train_points, points, k=PCA_K):
    """Squared distance from each sample to the flat k-dimensional PCA subspace of `train_points`."""
    centre = train_points.mean(0)
    _, _, directions = np.linalg.svd(train_points - centre, full_matrices=False)
    offset = points - centre
    return (offset ** 2).sum(1) - ((offset @ directions[:k].T) ** 2).sum(1)


def squared_distances(A, B):
    return np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T, 0)


class DiffusionMap:
    def __init__(self, points, neighbours=DM_NEIGHBOURS):
        self.points, self.neighbours = points, neighbours
        d2 = squared_distances(points, points)
        self.width = np.sqrt(np.sort(d2, axis=1)[:, neighbours])
        K = np.exp(-d2 / np.outer(self.width, self.width))
        self.degree = K.sum(1)
        S = K / np.sqrt(np.outer(self.degree, self.degree))
        eigenvalues, V = np.linalg.eigh(S)
        order = np.argsort(eigenvalues)[::-1]
        self.eigenvalues, V = eigenvalues[order], V[:, order]
        self.right = V / np.sqrt(self.degree)[:, None]
        self.left = V * np.sqrt(self.degree)[:, None]

    def steps(self, points):
        d2 = squared_distances(points, self.points)
        width = np.sqrt(np.sort(d2, axis=1)[:, self.neighbours - 1])
        K = np.exp(-d2 / np.outer(width, self.width))
        return K / K.sum(1, keepdims=True)

    def rebuild(self, values, points, m):
        """Rebuild `values` at `points` from the constant eigenvector plus the first m others."""
        at_points = self.steps(points) @ self.right[:, :m + 1] / self.eigenvalues[:m + 1]
        return at_points @ (self.left[:, :m + 1].T @ values)


def dm_distance(train_points, points, m=DM_M):
    rebuilt = DiffusionMap(train_points).rebuild(train_points, points, m)
    return ((points - rebuilt) ** 2).sum(1)


# THE LINEAR BASELINES (as in QCML110426)

def logistic_scores(Ztr, ytr, Zva):
    """Logistic regression: log-odds u.x + b, trained by Adam on the cross-entropy with a small
    penalty on the weights."""
    W = torch.zeros(Ztr.shape[1], requires_grad=True)
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


def z_score(differences):
    """z = mean(d) / sqrt((1/S + n_val/n_train) var(d)) (Nadeau and Bengio, 2003)."""
    d = np.asarray(differences, dtype=float)
    if np.ptp(d) < 1e-12:
        return 0.0 if abs(d.mean()) < 1e-12 else float(np.sign(d.mean()) * np.inf)
    error = np.sqrt((1 / len(d) + N_VAL / N_TRAIN) * d.var(ddof=1))
    return float(d.mean() / error)


def run_experiment(removed):
    """Per-split validation accuracy of every classifier with the features in `removed` deleted."""
    keep = [f for f in range(D) if f not in removed]
    accuracy = {m: [] for m in CLASSIFIERS}
    for seed, (train_idx, val_idx) in enumerate(SPLITS):
        Ztr = normalize(features[train_idx][:, keep], features[train_idx][:, keep])
        Zva = normalize(features[train_idx][:, keep], features[val_idx][:, keep])
        ytr, yva = labels[train_idx], labels[val_idx]
        model_B = train_qcml(Ztr[ytr == 0], CLS_N, CLS_C2, seed=seed)   # benign only
        model_M = train_qcml(Ztr[ytr == 1], CLS_N, CLS_C2, seed=seed)   # malignant only
        sup_model = train_supervised(Ztr, ytr.astype(float), SUP_N,
                                     torch.nn.functional.binary_cross_entropy_with_logits,
                                     lam=SUP_LAMBDA, w=SUP_W, seed=seed, epochs=SUP_EPOCHS)
        centroid_B, centroid_M = Ztr[ytr == 0].mean(0), Ztr[ytr == 1].mean(0)
        scores = {   # above 0 = called malignant
            "QCML": distance_to_manifold(model_B, Zva) - distance_to_manifold(model_M, Zva),
            "Supervised QCML": supervised_forecast(sup_model, Zva),
            "PCA": pca_distance(Ztr[ytr == 0], Zva) - pca_distance(Ztr[ytr == 1], Zva),
            "Diffusion map": dm_distance(Ztr[ytr == 0], Zva) - dm_distance(Ztr[ytr == 1], Zva),
            "Nearest centroid": ((Zva - centroid_B) ** 2).sum(1) - ((Zva - centroid_M) ** 2).sum(1),
            "Logistic regression": logistic_scores(Ztr, ytr, Zva)}
        for m in CLASSIFIERS:
            accuracy[m].append(((scores[m] > 0).astype(int) == yva).mean())
    return tuple(removed), {m: np.array(v) for m, v in accuracy.items()}


def describe(removed):
    if not removed:
        return "none"
    if list(removed) == TOP_THREE:
        return "Features 29, 14 and 13 together"
    return feature_label(removed[0])


if __name__ == "__main__":
    settings = [()] + [(f,) for f in range(D)] + [tuple(TOP_THREE)]
    print(f"QCML111326_2: {len(settings)} removal settings x {N_SPLITS} splits on {WORKERS} processes", flush=True)
    start = time.time()
    results = {}
    with ProcessPoolExecutor(WORKERS) as pool:
        for done, future in enumerate(as_completed([pool.submit(run_experiment, s) for s in settings]), 1):
            removed, accuracy = future.result()
            results[removed] = accuracy
            if done % 8 == 0 or done == len(settings):
                print(f"   {done}/{len(settings)} done ({(time.time() - start) / 60:.0f} min)", flush=True)

    full = results[()]
    print("\nCHECK against QCML110426 (all features, mean accuracy: this run / saved):")
    with open("qcml110426_wdbc_results.csv") as file:
        saved = {row["method"]: float(row["accuracy mean"]) for row in csv.DictReader(file)}
    print("   " + ", ".join(f"{m} {full[m].mean():.4f} / {saved[m]:.4f}" for m in CLASSIFIERS))

    with open("qcml111326_2_removal.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["removed", "feature number"]
                        + [f"{m} {s}" for m in CLASSIFIERS
                           for s in ("accuracy mean", "accuracy std", "change", "z of change")])
        for removed in settings:
            row = [describe(removed), "" if len(removed) != 1 else removed[0] + 1]
            for m in CLASSIFIERS:
                change = results[removed][m] - full[m]
                row += [results[removed][m].mean(), results[removed][m].std(), change.mean(),
                        z_score(change) if removed else 0.0]
            writer.writerow(row)
    with open("qcml111326_2_removal_splits.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["removed", "split"] + [f"{m} accuracy" for m in CLASSIFIERS])
        for removed in settings:
            for split in range(N_SPLITS):
                writer.writerow([describe(removed), split] + [results[removed][m][split] for m in CLASSIFIERS])
    print("saved to qcml111326_2_removal.csv and qcml111326_2_removal_splits.csv")

    print("\nvalidation accuracy (mean over splits), and the change from all features with its z:")
    singles = np.array([[results[(f,)][m].mean() - full[m].mean() for m in CLASSIFIERS] for f in range(D)])
    print("   " + " " * 34 + "".join(f"{m:>21}" for m in CLASSIFIERS))
    print(f"   {'all features':<34}" + "".join(f"{full[m].mean():21.4f}" for m in CLASSIFIERS))
    top = tuple(TOP_THREE)
    print(f"   {'Features 29, 14, 13 removed':<34}" + "".join(
        f"{results[top][m].mean():12.4f} ({z_score(results[top][m] - full[m]):+.1f})" for m in CLASSIFIERS))
    print(f"   {'one feature removed, average':<34}" + "".join(f"{v:+21.4f}" for v in singles.mean(0)))
    print(f"   {'one feature removed, worst':<34}" + "".join(f"{v:+21.4f}" for v in singles.min(0)))
    print(f"   {'one feature removed, best':<34}" + "".join(f"{v:+21.4f}" for v in singles.max(0)))
    for i, m in enumerate(CLASSIFIERS):
        worst = int(np.argmin(singles[:, i]))
        print(f"   {m:<20} worst single removal: {feature_label(worst)} ({singles[worst, i]:+.4f}, "
              f"z {z_score(results[(worst,)][m] - full[m]):+.1f})")
    print(f"\ntotal {(time.time() - start) / 60:.0f} min")


# ========================================================================================
# EXPLANATION
# Notes on the code above, in the order of its sections. A tag [note k] in the code
# marks the section that note k explains.
# ========================================================================================
#
# [note 0] About this script
#
# QCML102326_2 removed each of the 30 features in turn, and then the three whose removal cost
# the most together, for unsupervised QCML, supervised QCML, PCA and the diffusion map, but it
# saved only a summary. QCML110426 added logistic regression and the nearest centroid on all
# features. Here all six classifiers are scored with every feature removed in turn and with the
# three removed together, on QCML110426's ten splits and settings: PCA keeps k = 2 components and
# the diffusion map m = 15 eigenvectors, both chosen on all features and then held fixed. The
# three features are the ones QCML102326_2 picked, the largest drop in accuracy averaged over
# unsupervised QCML and PCA: Feature 29 (largest symmetry), Feature 14 (area variation) and
# Feature 13 (perimeter variation). They are kept so the result stays comparable; because they
# were picked on the same splits, the drops they cause lean large.
#
# "Removing a feature" deletes its column before anything else happens, in training and in
# validation. Each change in accuracy is paired by split and tested with the corrected z of
# Nadeau and Bengio (2003), as in the methods chapter.
#
# Check: the all-feature run must reproduce qcml110426_wdbc_results.csv.
#
# Runs in about 10 minutes on 8 cores (32 removal settings, one process each). Run with python3.13 QCML111326_2.py. Saves qcml111326_2_removal.csv and qcml111326_2_removal_splits.csv.
