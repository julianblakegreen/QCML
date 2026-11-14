# Quantum Cognitive Machine Learning Project
# QCML on the Wisconsin Diagnostic Breast Cancer data (UCI dataset 17)
# QCML100626: builds on QCML092926_2
# Runs in about 10 minutes.
# What it does and how: see the notes at the end of this file.


import csv
import os
import textwrap

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

torch.set_num_threads(1)      # the matrices are tiny, so extra threads cost more than they save

EPOCHS = 150
BATCH = 64

FIT_N, FIT_C2 = 16, 0.03      # Part 1: fitting the shape of the whole data set
CLS_N, CLS_C2 = 4, 0.1        # Parts 2-4: one manifold per class, for classifying
N_SPLITS = 10                 # how many random train/validation splits to average over
TRAIN_FRACTION = 0.7          # 70% of patients train the models, 30% validate them
PCA_COMPONENTS = np.arange(1, 13)   # the numbers of PCA components k to try
N_REMOVE = 3                  # Part 4: how many features to remove together
PLOT_SPLIT = 0                # which split the classifier scatter plots show
PLOT_DIR = "qcml100626_plots" # every plot is saved here as its own PNG


# Data
rows = np.genfromtxt("wdbc.data", delimiter=",", dtype=str)
labels = (rows[:, 1] == "M").astype(int)          # 1 = malignant, 0 = benign
features = rows[:, 2:].astype(float)              # (569, 30)
N, D = features.shape
print(f"{N} samples, {D} features, {labels.sum()} malignant, {N - labels.sum()} benign")

# The 30 features (from wdbc.names).  [note 1]
MEASUREMENTS = {
    "radius": "distance from the centre of a nucleus to its edge",
    "texture": "variation in grey level inside a nucleus",
    "perimeter": "length of the outline of a nucleus",
    "area": "area of a nucleus",
    "smoothness": "how much the radius wobbles along the outline",
    "compactness": "how far from round (perimeter^2 / area - 1)",
    "concavity": "how deep the dents in the outline are",
    "concave points": "how many dents the outline has",
    "symmetry": "how symmetric a nucleus is",
    "fractal dimension": "how crinkly the outline is",
}
SUMMARIES = [("mean {}", "averaged over the nuclei"),
             ("{} variation", "how much it varies between nuclei"),
             ("largest {}", "average of the 3 largest nuclei")]
FEATURE_NAMES = [form.format(m) for form, _ in SUMMARIES for m in MEASUREMENTS]


def feature_label(f):
    """'Feature 24 (largest area)' for column f, counting the columns from 0."""
    return f"Feature {f + 1} ({FEATURE_NAMES[f]})"


def describe(f):
    """'Feature 24 (largest area): area of a nucleus, average of the 3 largest nuclei'."""
    measurement = list(MEASUREMENTS)[f % 10]
    return f"{feature_label(f)}: {MEASUREMENTS[measurement]}, {SUMMARIES[f // 10][1]}"


print("\nTHE 30 FEATURES: ten measurements of the cell nuclei in each image,")
for measurement, meaning in MEASUREMENTS.items():
    print(f"   {measurement:<18} {meaning}")
print("each summarised over the image three ways:")
print("   features  1-10  mean ...       averaged over the nuclei")
print("   features 11-20  ... variation  how much it varies between nuclei (the standard error)")
print('   features 21-30  largest ...    average of the 3 largest nuclei ("worst" in wdbc.names)')
for f in range(10):
    print("   " + "".join(f"{g + 1:3d} {FEATURE_NAMES[g]:<29}" for g in (f, f + 10, f + 20)))


# NOTE ON NORMALIZATION (z-scoring every feature to mean 0, standard deviation 1)  [note 2]

def normalize(train_points, points):
    """Z-score `points` using the mean and std of `train_points` (one per feature)."""
    return (points - train_points.mean(0)) / train_points.std(0)


# NOTE ON SPLITS  [note 3]

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


# PCA AS A CLASSIFIER  [note 4]

def pca_distances(train_points, points):
    """Squared distance from each sample to the flat PCA subspace of `train_points`,
    for every k in PCA_COMPONENTS at once: shape (samples, len(PCA_COMPONENTS))."""
    centre = train_points.mean(0)
    _, _, directions = np.linalg.svd(train_points - centre, full_matrices=False)
    offset = points - centre
    explained = np.cumsum((offset @ directions[:PCA_COMPONENTS.max()].T) ** 2, axis=1)
    return (offset ** 2).sum(1, keepdims=True) - explained[:, PCA_COMPONENTS - 1]


def pca_accuracy(Z_train, y_train, points, y):
    """Accuracy of the PCA classifier on `points`, one value per k."""
    d_B = pca_distances(Z_train[y_train == 0], points)
    d_M = pca_distances(Z_train[y_train == 1], points)
    return ((d_M < d_B).astype(int) == y[:, None]).mean(0)


def own_and_other(d_B, d_M, y):
    """Reconstruction error (root mean square, in standard deviations) of the patients
    under the model of their own diagnosis, and under the model of the other one.
    d_B, d_M are squared distances, shape (patients,) or (patients, k)."""
    if d_B.ndim == 2:
        y = y[:, None]
    own = np.where(y == 1, d_M, d_B)
    other = np.where(y == 1, d_B, d_M)
    return np.sqrt(own.mean(0)), np.sqrt(other.mean(0))


# VALIDATION SCORING  [note 5]

def z_score(differences):
    """How many (corrected) standard errors the mean of `differences` is from zero.
    If every split gives exactly the same difference there is no spread to measure
    against: z is then 0 for no difference and +/-inf for a perfectly consistent one."""
    d = np.asarray(differences, dtype=float)
    if np.ptp(d) < 1e-12:
        return 0.0 if abs(d.mean()) < 1e-12 else float(np.sign(d.mean()) * np.inf)
    error = np.sqrt((1 / len(d) + N_VAL / N_TRAIN) * d.var(ddof=1))
    return float(d.mean() / error)


def run_experiment(removed=()):
    """Train and score QCML and PCA on every split, with the features in `removed`
    deleted. Returns per-split results: accuracies qcml_train and qcml_val with shape
    (N_SPLITS,), pca_train and pca_val (N_SPLITS, len(PCA_COMPONENTS)); the validation
    patients' reconstruction errors under their own and the other diagnosis's model
    (qcml_own, qcml_other, pca_own, pca_other); and in "plot" the QCML distances of
    the PLOT_SPLIT validation patients, for the scatter plots."""
    keep = [f for f in range(D) if f not in removed]
    scores = {key: [] for key in ("qcml_train", "qcml_val", "pca_train", "pca_val",
                                  "qcml_own", "qcml_other", "pca_own", "pca_other")}
    result = {}
    for seed, (train_idx, val_idx) in enumerate(SPLITS):
        Ztr = normalize(features[train_idx][:, keep], features[train_idx][:, keep])
        Zva = normalize(features[train_idx][:, keep], features[val_idx][:, keep])
        ytr, yva = labels[train_idx], labels[val_idx]

        model_B, _, _ = train_qcml(Ztr[ytr == 0], CLS_N, CLS_C2, seed=seed)   # benign only
        model_M, _, _ = train_qcml(Ztr[ytr == 1], CLS_N, CLS_C2, seed=seed)   # malignant only

        # A patient is called malignant if they fit the malignant model more closely.
        for part, Z, y in [("train", Ztr, ytr), ("val", Zva, yva)]:
            d_B, d_M = distance_to_manifold(model_B, Z), distance_to_manifold(model_M, Z)
            p_B, p_M = pca_distances(Ztr[ytr == 0], Z), pca_distances(Ztr[ytr == 1], Z)
            scores["qcml_" + part].append(float(((d_M < d_B).astype(int) == y).mean()))
            scores["pca_" + part].append(((p_M < p_B).astype(int) == y[:, None]).mean(0))
            if part == "val":
                for method, dist_B, dist_M in [("qcml", d_B, d_M), ("pca", p_B, p_M)]:
                    own, other = own_and_other(dist_B, dist_M, y)
                    scores[method + "_own"].append(own)
                    scores[method + "_other"].append(other)
                if seed == PLOT_SPLIT:
                    result["plot"] = (d_B, d_M, y)
    result.update({key: np.array(values) for key, values in scores.items()})
    return result


def pca_average(result):
    """PCA validation score on each split, averaged over k: the mean height of its curve."""
    return result["pca_val"].mean(1)


# Normalization check: the numbers behind the note on normalization
print("\nNORMALIZATION CHECK")
spread = features.std(0)
share = features.var(0) / features.var(0).sum()
top3 = np.argsort(share)[::-1][:3]
print(f"   raw: widest feature std / narrowest = {spread.max() / spread.min():,.0f}"
      f"   ({feature_label(spread.argmax())} vs {feature_label(spread.argmin())})")
print(f"   raw: features {', '.join(str(f + 1) for f in top3)} (the area features) hold "
      f"{share[top3].sum():.1%} of the total variance")

train_idx, val_idx = SPLITS[0]
Z_train = normalize(features[train_idx], features[train_idx])
Z_val = normalize(features[train_idx], features[val_idx])
for name, Z in [("training  ", Z_train), ("validation", Z_val)]:
    print(f"   z-scored {name} set (split 0): feature means {Z.mean(0).min():+.2f} to "
          f"{Z.mean(0).max():+.2f}, stds {Z.std(0).min():.2f} to {Z.std(0).max():.2f}")

Z_all = normalize(features, features)
beyond = np.abs(Z_all) > 3
skew = (Z_all ** 3).mean(0)
print(f"   {beyond.mean():.2%} of z-scored values lie beyond ±3 (a normal distribution "
      f"would give 0.27%), in {beyond.any(1).sum()} of {N} patients")
print("   most skewed: " + ", ".join(
    f"{feature_label(f)} skew {skew[f]:.1f}, max |z| {np.abs(Z_all[:, f]).max():.0f}"
    for f in np.argsort(skew)[::-1][:3]))
correlated = (np.abs(np.triu(np.corrcoef(features.T), 1)) > 0.95).sum()
print(f"   {correlated} feature pairs correlate above 0.95 (radius / perimeter / area)")

# The PCA classifier is quick, so try it on the raw, z-scored and log-then-z-scored
# features. (Columns that contain zeros get half their smallest non-zero value added
# before the log.)
smallest_positive = np.where(features > 0, features, np.inf).min(0)
log_features = np.log(features + (features == 0).any(0) * smallest_positive / 2)
print(f"   PCA classifier validation accuracy, averaged over k and {N_SPLITS} splits:")
for name, table, scale in [("raw numbers", features, False),
                           ("z-scored", features, True),
                           ("log, then z-scored", log_features, True)]:
    scores = []
    for train_idx, val_idx in SPLITS:
        A, B = table[train_idx], table[val_idx]
        if scale:
            A, B = normalize(A, A), normalize(A, B)
        scores.append(pca_accuracy(A, labels[train_idx], B, labels[val_idx]).mean())
    print(f"      {name:<20}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

times_validated = np.zeros(N, dtype=int)
for _, val_idx in SPLITS:
    times_validated[val_idx] += 1
print(f"\nSPLITS: {N_SPLITS} stratified splits, each {N_TRAIN} train "
      f"({(labels[SPLITS[0][0]] == 0).sum()} benign + {labels[SPLITS[0][0]].sum()} malignant)"
      f" / {N_VAL} validation ({(labels[SPLITS[0][1]] == 0).sum()} + {labels[SPLITS[0][1]].sum()})")
print(f"   across the splits a patient is validated {times_validated.mean():.1f} times on "
      f"average; {(times_validated == 0).sum()} patients are never validated")


# RECONSTRUCTION ERROR (what Part 1 measures)  [note 6]

# P1 one manifold for the whole data set
print("\nPART 1: fitting one manifold to all the data (split 0)")
train_idx, test_idx = SPLITS[0]
Z_train = normalize(features[train_idx], features[train_idx])
Z_test = normalize(features[train_idx], features[test_idx])

model, history_L1, history_L2 = train_qcml(Z_train, FIT_N, FIT_C2, label="   ")

# P1 always answer with the average sample.
mean_baseline = float(np.sqrt((Z_test ** 2).sum(1).mean()))

# PCA, the linear version of the same idea (project onto a flat subspace).
centre = Z_train.mean(0)
_, _, directions = np.linalg.svd(Z_train - centre, full_matrices=False)
pca_errors = []
for k in PCA_COMPONENTS:
    P = directions[:k]
    projected = (Z_test - centre) @ P.T @ P + centre
    pca_errors.append(float(np.sqrt(((projected - Z_test) ** 2).sum(1).mean())))

print("\nreconstruction error on validation patients, in standard deviations")
print("(one model for all patients, diagnoses not used; lower = closer description):")
print(f"   answer with the average sample : {mean_baseline:.3f}")
print(f"   QCML (n = {FIT_N})                  : {fit_error(model, Z_test):.3f}"
      f"   [on its training data: {fit_error(model, Z_train):.3f}]")
for k in (2, 5, 8):
    print(f"   PCA with {k} components          : {pca_errors[k - 1]:.3f}")

# Intrinsic dimension: the quantum metric says how fast the quantum state turns
# as each feature is nudged. Its eigenvalues count the directions that matter.
metric = model.quantum_metric(torch.tensor(np.vstack([Z_train, Z_test]), dtype=torch.float32))
metric_eigs = torch.linalg.eigvalsh(metric).clamp(min=0)          # (samples, D)
eig_share = metric_eigs / metric_eigs.sum(-1, keepdim=True)
intrinsic_dim = float((eig_share.sum(-1) ** 2 / (eig_share ** 2).sum(-1)).mean())
print(f"\nintrinsic dimension from the quantum metric: {intrinsic_dim:.2f}"
      f"   (the 30 measurements move along about this many independent directions)")


# P2 one manifold per diagnosis, used as a classifier, all features
print(f"\nPART 2: one manifold per diagnosis, all {D} features, {N_SPLITS} splits")
baseline = run_experiment()
centroid_acc, logistic_acc = [], []

for seed, (train_idx, val_idx) in enumerate(SPLITS):
    Ztr = normalize(features[train_idx], features[train_idx])
    Zva = normalize(features[train_idx], features[val_idx])
    ytr, yva = labels[train_idx], labels[val_idx]

    # Baseline 1: nearest class average.
    cB, cM = Ztr[ytr == 0].mean(0), Ztr[ytr == 1].mean(0)
    nearer_M = ((Zva - cM) ** 2).sum(1) < ((Zva - cB) ** 2).sum(1)
    centroid_acc.append(float((nearer_M.astype(int) == yva).mean()))

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
        predicted = ((torch.tensor(Zva, dtype=torch.float32) @ W + b) > 0).numpy().astype(int)
    logistic_acc.append(float((predicted == yva).mean()))

    print(f"   split {seed}: QCML train {baseline['qcml_train'][seed]:.3f} "
          f"val {baseline['qcml_val'][seed]:.3f}   PCA val (avg over k) "
          f"{pca_average(baseline)[seed]:.3f}   nearest centroid {centroid_acc[-1]:.3f}   "
          f"logistic {logistic_acc[-1]:.3f}")

best = baseline["pca_val"].mean(0).argmax()      # PCA's best k, picked on validation (flattering)
print(f"\nover {N_SPLITS} splits (mean ± std)        training     validation     "
      f"QCML minus it: mean, z")
for name, val, train in [
        (f"QCML (n = {CLS_N})", baseline["qcml_val"], baseline["qcml_train"]),
        (f"PCA, average over k = 1-{PCA_COMPONENTS.max()}", pca_average(baseline),
         baseline["pca_train"].mean(1)),
        (f"PCA, best k (k = {PCA_COMPONENTS[best]})", baseline["pca_val"][:, best],
         baseline["pca_train"][:, best]),
        ("nearest centroid", np.array(centroid_acc), None),
        ("logistic regression", np.array(logistic_acc), None)]:
    train_text = f"{train.mean():.3f}" if train is not None else "  -  "
    line = f"   {name:<28}:   {train_text}    {val.mean():.3f} ± {val.std():.3f}"
    if not name.startswith("QCML"):
        d = baseline["qcml_val"] - val
        line += f"     {d.mean():+.3f}, z = {z_score(d):+.1f}"
    print(line)

print("\nreconstruction error of the validation patients under each diagnosis's model, in")
print("standard deviations. A patient is diagnosed correctly when the model of their own")
print("diagnosis describes them more closely than the other diagnosis's model does.")
print(f"   {'':<30}" + " ".join(f"{'k=' + str(k):>5}" for k in PCA_COMPONENTS))
for name, key in [("PCA, own diagnosis's model", "pca_own"),
                  ("PCA, other diagnosis's model", "pca_other")]:
    print(f"   {name:<30}" + " ".join(f"{v:5.2f}" for v in baseline[key].mean(0)))
print(f"   QCML (n = {CLS_N}): own diagnosis's model {baseline['qcml_own'].mean():.2f}, "
      f"other diagnosis's model {baseline['qcml_other'].mean():.2f}")


# Results table: one experiment per feature, in the format
#   feature : QCML training accuracy : QCML validation accuracy : PCA training accuracy
#           : PCA validation accuracy, at each number of components k
NAME_WIDTH = 31


def print_table_header():
    ks = " ".join(f"{'k=' + str(k):>5}" for k in PCA_COMPONENTS)
    print(f"{'':{NAME_WIDTH}}   QCML    QCML                    PCA at each number of components")
    print(f"{'feature removed':{NAME_WIDTH}} : train : validation    :       {ks}")


def print_table_rows(name, result):
    """Two lines per experiment: QCML scores and PCA training accuracy, then PCA validation."""
    left = (f"{name:{NAME_WIDTH}} : {result['qcml_train'].mean():.3f} : "
            f"{result['qcml_val'].mean():.3f} ± {result['qcml_val'].std():.3f} : ")
    print(left + "train " + " ".join(f"{v:.3f}" for v in result["pca_train"].mean(0)))
    print(" " * (len(left) - 2) + ": val   " + " ".join(f"{v:.3f}" for v in result["pca_val"].mean(0)))


def compare(result):
    """Per-split changes from the all-feature run (paired: same splits). Returns
    QCML's drop, PCA's drop, their average (importance), and PCA's drop minus
    QCML's drop (QCML's advantage). A drop is positive when accuracy got worse."""
    qcml_drop = baseline["qcml_val"] - result["qcml_val"]
    pca_drop = pca_average(baseline) - pca_average(result)
    return qcml_drop, pca_drop, (qcml_drop + pca_drop) / 2, pca_drop - qcml_drop


# P3 each feature removed in turn  [note 7]
print(f"\nPART 3: each feature removed in turn ({D} experiments, {N_SPLITS} splits each)")
print("accuracy averaged over the splits; validation shown as mean ± std\n")
print_table_header()
print_table_rows(f"none (all {D} features)", baseline)
singles = []
for f in range(D):
    singles.append(run_experiment([f]))
    print_table_rows(f"{f + 1:2d} {FEATURE_NAMES[f]}", singles[-1])

importance = np.array([compare(r)[2].mean() for r in singles])
importance_z = np.array([z_score(compare(r)[2]) for r in singles])
advantage = np.array([compare(r)[3].mean() for r in singles])
advantage_z = np.array([z_score(compare(r)[3]) for r in singles])
by_importance = np.lexsort((-importance_z, -importance))     # biggest first, ties by z
by_advantage = np.lexsort((-advantage_z, -advantage))

print("\nFEATURES RANKED BY IMPORTANCE: the drop in validation accuracy when the feature is")
print("removed, averaged over QCML and PCA*. z compares to the all-feature run: |z| > 2 is")
print("a real change, |z| < 1 is noise.")
print(f"rank  {'feature removed':<30}  QCML drop (z)   PCA* drop (z)   average drop (z)")
for rank, f in enumerate(by_importance, 1):
    qcml_drop, pca_drop, average_drop, _ = compare(singles[f])
    print(f"{rank:4d}  {f + 1:2d} {FEATURE_NAMES[f]:<27}  "
          f"{qcml_drop.mean():+.3f} ({z_score(qcml_drop):+4.1f})  "
          f"{pca_drop.mean():+.3f} ({z_score(pca_drop):+4.1f})   "
          f"{average_drop.mean():+.3f} ({z_score(average_drop):+4.1f})")
print(f"   * PCA averaged over k = 1-{PCA_COMPONENTS.max()}")

gap_all = baseline["qcml_val"] - pca_average(baseline)
print("\nFEATURES RANKED BY QCML ADVANTAGE: PCA's drop minus QCML's drop, so positive means")
print("removing the feature hurts PCA more than it hurts QCML.")
print(f"   (with all features QCML leads PCA* by {gap_all.mean():+.3f}, z = {z_score(gap_all):+.1f})")
print(f"rank  {'feature removed':<30}  QCML drop   PCA* drop   QCML advantage (z)")
for rank, f in enumerate(by_advantage, 1):
    qcml_drop, pca_drop, _, gain = compare(singles[f])
    print(f"{rank:4d}  {f + 1:2d} {FEATURE_NAMES[f]:<27}   {qcml_drop.mean():+.3f}      "
          f"{pca_drop.mean():+.3f}       {gain.mean():+.3f} ({z_score(gain):+4.1f})")
print(f"   * PCA averaged over k = 1-{PCA_COMPONENTS.max()}")


# P4 several features removed together  [note 8]
top_important = [int(f) for f in by_importance[:N_REMOVE]]
top_advantage = [int(f) for f in by_advantage[:N_REMOVE]]
print(f"\nPART 4: {N_REMOVE} features removed together")
print(f"   top {N_REMOVE} important      : {', '.join(feature_label(f) for f in top_important)}")
print(f"   top {N_REMOVE} QCML advantage : {', '.join(feature_label(f) for f in top_advantage)}")
both_important = run_experiment(top_important)
both_advantage = (both_important if set(top_advantage) == set(top_important)
                  else run_experiment(top_advantage))

print()
print_table_header()
print_table_rows(f"none (all {D} features)", baseline)
print_table_rows(f"top {N_REMOVE} important", both_important)
print_table_rows(f"top {N_REMOVE} QCML advantage", both_advantage)
print()
for name, result in [(f"top {N_REMOVE} important", both_important),
                     (f"top {N_REMOVE} QCML advantage", both_advantage)]:
    qcml_drop, pca_drop, average_drop, gain = compare(result)
    print(f"   {name:<22}: QCML drop {qcml_drop.mean():+.3f} (z {z_score(qcml_drop):+.1f})   "
          f"PCA* drop {pca_drop.mean():+.3f} (z {z_score(pca_drop):+.1f})   "
          f"average {average_drop.mean():+.3f} (z {z_score(average_drop):+.1f})   "
          f"QCML advantage {gain.mean():+.3f} (z {z_score(gain):+.1f})")


# Everything from Parts 2-4 in one spreadsheet-friendly file
experiments = ([("none", baseline)]
               + [(feature_label(f), r) for f, r in enumerate(singles)]
               + [(" + ".join(feature_label(f) for f in top_important), both_important),
                  (" + ".join(feature_label(f) for f in top_advantage), both_advantage)])
with open("qcml100626_feature_sweep.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["features removed", "QCML train", "QCML train std", "QCML val",
                     "QCML val std"]
                    + [f"PCA train k={k}" for k in PCA_COMPONENTS]
                    + [f"PCA val k={k}" for k in PCA_COMPONENTS]
                    + [f"PCA val std k={k}" for k in PCA_COMPONENTS]
                    + ["PCA val avg over k", "QCML drop", "QCML drop z", "PCA drop", "PCA drop z",
                       "importance (average drop)", "importance z",
                       "QCML advantage (PCA drop - QCML drop)", "QCML advantage z",
                       "QCML minus PCA", "QCML minus PCA z"])
    for name, r in experiments:
        qcml_drop, pca_drop, average_drop, gain = compare(r)
        gap = r["qcml_val"] - pca_average(r)
        writer.writerow([name, r["qcml_train"].mean(), r["qcml_train"].std(),
                         r["qcml_val"].mean(), r["qcml_val"].std()]
                        + list(r["pca_train"].mean(0)) + list(r["pca_val"].mean(0))
                        + list(r["pca_val"].std(0))
                        + [pca_average(r).mean(), qcml_drop.mean(), z_score(qcml_drop),
                           pca_drop.mean(), z_score(pca_drop),
                           average_drop.mean(), z_score(average_drop),
                           gain.mean(), z_score(gain), gap.mean(), z_score(gap)])
print("\nfull results saved to qcml100626_feature_sweep.csv")


# Plots: one PNG per plot, numbered in reading order, saved in PLOT_DIR

os.makedirs(PLOT_DIR, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "figure.titlesize": 13,
                     "figure.titleweight": "bold", "legend.fontsize": 9,
                     "legend.title_fontsize": 9, "legend.frameon": False,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.grid": True, "axes.axisbelow": True,
                     "grid.color": "#e1e0d9", "grid.linewidth": 0.6})
QCML_COLOUR, PCA_COLOUR = "#eb6834", "#2a78d6"
BENIGN_COLOUR, MALIGNANT_COLOUR = "#2a78d6", "#e34948"
GREY, INK, MUTED_INK = "#898781", "#0b0b0b", "#52514e"


def new_plot(title, subtitle, size=(8.5, 7)):
    """A figure holding a single plot, with a bold title and a grey subtitle."""
    fig, ax = plt.subplots(figsize=size, layout="constrained")
    fig.suptitle(title)
    ax.set_title(subtitle, color=MUTED_INK)
    return fig, ax


def save_plot(fig, ax, name, legend_title=None):
    """Put the legend under the plot, save it as PLOT_DIR/name.png and close it."""
    ax.legend(title=legend_title, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2,
              alignment="left")
    path = os.path.join(PLOT_DIR, f"{name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"   {path}")


def removed_title(heading, removed):
    """Legend title: why the features were picked, then each one and what it measures
    (long descriptions wrap onto a second, indented line)."""
    lines = [heading]
    for f in removed:
        lines += textwrap.wrap(describe(f), width=80, initial_indent="   ",
                               subsequent_indent="      ")
    return "\n".join(lines)


def score_plot(name, title, qcml_val, pca_val, what, legend_title,
               bands=f"± 1 std over the {N_SPLITS} splits"):
    """Validation accuracy against PCA components. QCML has no components, so it is a
    flat line; the bands are ± 1 std over the rows of qcml_val (S,) and pca_val
    (S, len(k)). The all-feature run is drawn in grey for comparison."""
    fig, ax = new_plot(title, f"validation accuracy (bands: {bands})")
    ax.axhline(baseline["qcml_val"].mean(), color=GREY, ls="--", lw=1,
               label=f"QCML, all {D} features")
    ax.plot(PCA_COMPONENTS, baseline["pca_val"].mean(0), color=GREY, lw=1, marker=".",
            label=f"PCA, all {D} features")
    q_mean, q_std = qcml_val.mean(), qcml_val.std()
    p_mean, p_std = pca_val.mean(0), pca_val.std(0)
    ax.axhline(q_mean, color=QCML_COLOUR, lw=2, label=f"QCML, {what}")
    ax.axhspan(q_mean - q_std, q_mean + q_std, color=QCML_COLOUR, alpha=0.1, lw=0)
    ax.plot(PCA_COMPONENTS, p_mean, color=PCA_COLOUR, lw=2, marker="o", ms=4,
            label=f"PCA, {what}")
    ax.fill_between(PCA_COMPONENTS, p_mean - p_std, p_mean + p_std, color=PCA_COLOUR,
                    alpha=0.1, lw=0)
    ax.set_xticks(PCA_COMPONENTS)
    ax.set_xlabel("PCA components (k)")
    ax.set_ylabel("validation accuracy")
    ax.set_ylim(*score_range)
    save_plot(fig, ax, name, legend_title)


def classifier_plot(name, title, result, legend_title=None):
    """QCML's view of the PLOT_SPLIT validation patients: distance to each manifold.
    Below the dashed decision line a patient is nearer the malignant manifold."""
    d_B, d_M, y = result["plot"]
    wrong = (d_M < d_B).astype(int) != y
    fig, ax = new_plot(title, f"validation patients of split {PLOT_SPLIT}, "
                              f"accuracy {1 - wrong.mean():.3f}", size=(8, 8.5))
    for value, colour, marker, name_ in [(0, BENIGN_COLOUR, "o", "benign"),
                                         (1, MALIGNANT_COLOUR, "^", "malignant")]:
        sel = y == value
        ax.scatter(d_B[sel], d_M[sel], s=16, alpha=0.7, color=colour, marker=marker,
                   lw=0, label=name_)
    ax.scatter(d_B[wrong], d_M[wrong], s=80, facecolors="none", edgecolors=INK, lw=0.8,
               label=f"misdiagnosed ({wrong.sum()} of {len(y)})")
    low = min(d_B.min(), d_M.min()) / 1.5
    top = max(d_B.max(), d_M.max()) * 1.5
    ax.plot([low, top], [low, top], color=INK, ls="--", lw=1, label="decision line")
    ax.text(0.97, 0.03, "called malignant", transform=ax.transAxes, ha="right",
            va="bottom", color=MUTED_INK)
    ax.text(0.65, 0.97, "called benign", transform=ax.transAxes, ha="center",
            va="top", color=MUTED_INK)
    ax.set_xlim(low, top)
    ax.set_ylim(low, top)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("distance to the benign manifold")
    ax.set_ylabel("distance to the malignant manifold")
    save_plot(fig, ax, name, legend_title)


most_important, most_advantage = int(by_importance[0]), int(by_advantage[0])
average_qcml = np.array([r["qcml_val"].mean() for r in singles])     # one value per run
average_pca = np.array([r["pca_val"].mean(0) for r in singles])

# every accuracy plot gets the same y range, so they can be compared side by side
shown = [(baseline["qcml_val"], baseline["pca_val"]), (average_qcml, average_pca)] + [
    (r["qcml_val"], r["pca_val"]) for r in (singles[most_important], both_important,
                                            singles[most_advantage], both_advantage)]
low = min(min(q.mean() - q.std(), (p.mean(0) - p.std(0)).min()) for q, p in shown)
high = max(max(q.mean() + q.std(), (p.mean(0) + p.std(0)).max()) for q, p in shown)
score_range = (low - 0.05 * (high - low), high + 0.05 * (high - low))

print(f"\nplots saved to {PLOT_DIR}:")

# 01: Part 1 training curves
fig, ax = new_plot(f"Part 1: training one QCML manifold on all patients (n = {FIT_N})",
                   "loss on the training patients after each epoch")
epochs = np.arange(1, EPOCHS + 1)
ax.plot(epochs, history_L1, color=PCA_COLOUR, label="L1  position error")
ax.plot(epochs, history_L2, color=QCML_COLOUR, label="L2  fluctuation")
ax.plot(epochs, np.array(history_L1) + FIT_C2 * np.array(history_L2), color=INK,
        label=f"loss = L1 + {FIT_C2:g}*L2")
ax.set_yscale("log")
ax.set_xlabel("epoch")
ax.set_ylabel("value (log scale)")
save_plot(fig, ax, "01_part1_training")

# 02: Part 1, how closely one model describes everybody
fig, ax = new_plot("Part 1: how closely ONE model describes all patients",
                   f"reconstruction error of the validation patients of split 0 "
                   f"(lower = closer)\ndiagnoses are not used here, so this is not accuracy")
ax.plot(PCA_COMPONENTS, pca_errors, "o-", color=PCA_COLOUR, lw=2, ms=4,
        label="PCA, one flat subspace for all patients")
ax.axhline(fit_error(model, Z_test), color=QCML_COLOUR, lw=2, ls="--",
           label=f"QCML, one manifold for all patients (n = {FIT_N})")
ax.axhline(mean_baseline, color=GREY, ls=":", label="answer with the average patient")
ax.set_xticks(PCA_COMPONENTS)
ax.set_xlabel("PCA components (k)")
ax.set_ylabel("reconstruction error (standard deviations)")
save_plot(fig, ax, "02_part1_reconstruction_error")

# 03: Part 2, the same error for the two diagnosis models the classifiers use
fig, ax = new_plot("Part 2: reconstruction error under the two diagnosis models",
                   f"validation patients, all {D} features, mean of {N_SPLITS} splits\n"
                   "more PCA components bring BOTH models closer to every patient")
ax.axhline(baseline["qcml_own"].mean(), color=QCML_COLOUR, lw=2,
           label=f"QCML (n = {CLS_N}), model of their own diagnosis")
ax.axhline(baseline["qcml_other"].mean(), color=QCML_COLOUR, lw=2, ls="--",
           label=f"QCML (n = {CLS_N}), model of the other diagnosis")
ax.plot(PCA_COMPONENTS, baseline["pca_own"].mean(0), "o-", color=PCA_COLOUR, lw=2, ms=4,
        label="PCA, model of their own diagnosis")
ax.plot(PCA_COMPONENTS, baseline["pca_other"].mean(0), "o--", color=PCA_COLOUR, lw=2, ms=4,
        mfc="white", label="PCA, model of the other diagnosis")
ax.set_xticks(PCA_COMPONENTS)
ax.set_xlabel("PCA components (k)")
ax.set_ylabel("reconstruction error (standard deviations)")
save_plot(fig, ax, "03_part2_diagnosis_models_error",
          "a patient is diagnosed correctly when the model of their own diagnosis\n"
          "describes them more closely than the model of the other diagnosis")

# 04: Part 2, the QCML classifier with every feature
classifier_plot("04_part2_classifier_all_features",
                f"Part 2: QCML classifier, all {D} features (n = {CLS_N})", baseline)

# 05-11: removing features
score_plot("05_average_one_feature_removed", "Average when one feature is removed",
           average_qcml, average_pca, "one feature removed (average of the runs)",
           f"average of {D} runs, each removing one of the {D} features",
           bands=f"± 1 std across the {D} runs")
score_plot("06_most_important_removed", "Most important feature removed",
           singles[most_important]["qcml_val"], singles[most_important]["pca_val"],
           "without it", removed_title("most important feature, removed:", [most_important]))
classifier_plot("07_most_important_removed_classifier",
                "QCML classifier, most important feature removed", singles[most_important],
                removed_title("most important feature, removed:", [most_important]))
score_plot("08_top3_important_removed", f"Top {N_REMOVE} important features removed",
           both_important["qcml_val"], both_important["pca_val"], f"without these {N_REMOVE}",
           removed_title(f"top {N_REMOVE} important features, removed together:", top_important))
score_plot("09_biggest_qcml_advantage_removed", "Biggest QCML advantage feature removed",
           singles[most_advantage]["qcml_val"], singles[most_advantage]["pca_val"],
           "without it", removed_title("biggest QCML advantage feature, removed:", [most_advantage]))
classifier_plot("10_biggest_qcml_advantage_removed_classifier",
                "QCML classifier, biggest QCML advantage feature removed",
                singles[most_advantage],
                removed_title("biggest QCML advantage feature, removed:", [most_advantage]))
score_plot("11_top3_qcml_advantage_removed", f"Top {N_REMOVE} QCML advantage features removed",
           both_advantage["qcml_val"], both_advantage["pca_val"], f"without these {N_REMOVE}",
           removed_title(f"top {N_REMOVE} QCML advantage features, removed together:",
                         top_advantage))


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
# Part 1 fits one QCML manifold to all the patients and measures how closely it
#        describes them (reconstruction error), compared with PCA.
# Part 2 fits one manifold per diagnosis, classifies by which one a patient fits
#        better, and scores that against a PCA classifier built the same way.
# Part 3 removes each of the 30 features in turn and reruns Part 2, to find the
#        most important feature and the features whose loss hurts QCML less than PCA.
# Part 4 removes 3 features at once: the 3 most important, then the 3 with the
#        biggest QCML advantage.
#
# New since QCML092926_2:
#   - plain-English feature names, and a key to what the 30 features measure
#   - notes on the z-score normalization and whether it suits this data, with a printed check
#   - notes on how the train/validation splits work; they are now stratified
#   - validation scoring: every score is a mean ± standard deviation over the splits,
#     and every comparison comes with a z-score
#   - what reconstruction error is, and why it is not the same thing as accuracy
#   - PCA used as a classifier (one flat subspace per diagnosis), scored at every k
#   - Parts 3 and 4, a results table (also saved as a CSV), and one PNG per plot,
#     numbered in reading order, in the folder qcml100626_plots
#
# Runs in about 10 minutes. Lower N_SPLITS for a quicker, noisier run.
#
# [note 1]
# The 30 features (from wdbc.names). Each image shows many cell nuclei. Ten things are
# measured on every nucleus, and each is summarised over the image three ways:
# features 1-10 are the mean, 11-20 the variation between the nuclei (their standard
# error) and 21-30 the largest value (the average of the 3 largest). wdbc.names calls
# the largest values "worst", so feature 24, "largest area", is its "worst area".
#
# [note 2] NOTE ON NORMALIZATION (z-scoring every feature to mean 0, standard deviation 1)
#
# What it does: from each feature subtract its mean and divide by its standard
# deviation, both measured on the TRAINING patients only. Every number then reads as
# "standard deviations away from the typical training patient".
#
# Why this data needs it: the raw features live on wildly different scales. Largest
# area (feature 24) runs from 185 to 4254 while fractal dimension variation (feature
# 20) runs from 0.0009 to 0.03, and their standard deviations differ by a factor of
# about 215,000. The QCML loss and PCA both add up squared differences over the
# features, so on raw numbers the three area features would be over 99% of everything
# and the other 27 measurements would be ignored. QCML does not even train on the raw
# numbers: the area terms swamp the error Hamiltonian and PyTorch stops with an
# "ill-defined" eigenvector-gradient error (tried while writing this file). After
# z-scoring, every measurement gets an equal vote.
#
# Does it work for this data? Yes, with three caveats (the check below prints the numbers):
#   1. Skew and outliers. z-scoring is a straight-line rescale, so it moves and stretches
#      each feature but cannot change its shape. The variation features (11-20) are
#      strongly right-skewed (area variation, feature 14, has skew 5.4): most patients
#      bunch up near 0 while a few sit 8 to 12 standard deviations out, and because the
#      losses square the errors, those few count heavily. Taking the log first removes
#      most of the skew, but it did not help: QCML validation 0.963 with the log vs
#      0.970 without (10 splits, tried while writing this file), and the PCA classifier
#      drops too (printed below), so the plain z-score is kept.
#   2. Redundancy. radius, perimeter and area measure almost the same thing (15 feature
#      pairs correlate above 0.95), so after z-scoring "size" effectively gets about 9
#      of the 30 votes. This is also why removing any single one of them barely
#      matters in Part 3: its near-copies are still there.
#   3. No peeking. The mean and std come from the training split only and are reused,
#      unchanged, on the validation split. So the validation features are only roughly
#      mean 0 / std 1. That is deliberate: the validation patients must stay unseen.
#
# [note 3] NOTE ON SPLITS
#
# A split divides the 569 patients into two groups: a training set the models learn
# from, and a held-out set that is only used to score them. QCML092926_2 made one by
# shuffling the row numbers with a seeded random generator (np.random.default_rng(seed))
# and cutting the shuffled list at 70%: 398 patients to train, 171 to test. Part 1
# used one split (seed 0); Part 2 repeated it for seeds 0-4 and averaged, so that
# one lucky or unlucky draw of test patients could not decide the result. This is
# called repeated random sub-sampling (or Monte Carlo cross-validation). It is not
# k-fold cross-validation: the held-out sets overlap, so some patients get scored
# several times and some never.
#
# What changes here:
#   - The splits are stratified: benign and malignant patients are shuffled and cut
#     separately, so every split keeps the data's 63% / 37% mix in both groups
#     (250 + 148 train, 107 + 64 validation). Before, the mix drifted from split to
#     split, which added noise unrelated to the models.
#   - 10 splits instead of 5, and the SAME 10 are reused by every experiment, so a run
#     with a feature removed differs from the all-feature run only in that feature.
#   - The held-out 30% is called the validation set. The settings n, c2 and EPOCHS were
#     tuned in earlier iterations by looking at these scores, so it is not a
#     never-touched test set.
#
# [note 4] PCA AS A CLASSIFIER
#
# The QCML classifier fits one curved manifold to the benign training patients and
# one to the malignant ones, then calls a patient malignant if they sit closer to the
# malignant manifold. The PCA classifier does exactly the same with flat subspaces:
# k principal components fitted to each diagnosis, and each patient goes to whichever
# flat subspace reconstructs them better. Same data, same decision rule, so the only
# difference is the shape fitted to each class: curved (QCML) or flat (PCA).
# (At k = 0 it would be the nearest-centroid baseline.)
#
# [note 5] VALIDATION SCORING
#
# training accuracy   : fraction of the 70% training patients diagnosed correctly.
# validation accuracy : fraction of the 30% validation patients diagnosed correctly. The
#                       models never saw these patients, and their features are z-scored
#                       with the training mean and std, so this is the honest number.
# Both are reported as mean ± standard deviation over the N_SPLITS splits. The
# standard deviation is how much the score moves when a different 30% is held out.
#
# To compare two settings A and B, take their validation scores split by split (same
# patients, so the comparison is paired) and z-score the differences d = A - B:
#     z = mean(d) / sqrt((1/S + n_val/n_train) * var(d))         S = N_SPLITS
# A plain standard error would be sqrt(var(d) / S), but the splits share patients, so
# their scores are not independent and that would overstate the evidence. The extra
# n_val/n_train term is the Nadeau & Bengio (2003) correction for exactly this.
# Reading it: |z| below 1 is noise, |z| above 2 is a real difference.
#
# [note 6] RECONSTRUCTION ERROR (what Part 1 measures)
#
# Each model can only describe patients that lie on its own shape. Its
# "reconstruction" of a patient is the nearest point it can describe:
#   - QCML: find the patient's ground state psi and read off the 30 expected
#     measurements <psi|H_d|psi>. These always lie on the model's curved manifold.
#   - PCA with k components: project the patient onto a flat k-dimensional subspace,
#     throwing away everything outside those k directions.
# The reconstruction error is the typical distance between a patient's 30 z-scored
# numbers and their reconstruction, in standard deviations:
#     error = sqrt( average over patients of  sum_d (x_d - reconstruction_d)^2 )
# Lower means the model describes the patients more closely. Answering every patient
# with the average patient gives about 5.3, the scale to compare against.
#
# Reading the Part 1 graph: ONE model is fitted to all the training patients and the
# diagnoses are not used, so this measures description only, not accuracy. PCA's
# error always falls as k grows (at k = 30 nothing is thrown away and it is exactly 0),
# so PCA dropping below QCML at about 6-7 components only says that a flat 6-7
# dimensional subspace describes these patients about as closely as the n = 16 QCML
# manifold, in line with the intrinsic dimension of about 5-6 printed below.
#
# It does NOT mean PCA diagnoses better. Diagnosis (Parts 2-4) uses different models,
# one per diagnosis with n = 4, and a patient is diagnosed correctly when the model of
# their own diagnosis describes them more closely than the other diagnosis's model
# does. Adding PCA components brings BOTH of its models closer to every patient,
# including the patients of the other diagnosis, so PCA's error keeps falling while
# its accuracy does not improve. Part 2 prints both errors, and plot 03 shows them.
#
# [note 7] P3 each feature removed in turn
#
# "Removing a feature" deletes its column before anything else happens: the
# z-scoring, QCML and PCA all see only the other 29 features, in training and in
# validation, on the same splits as the all-feature run. For each feature:
#   QCML drop       = QCML validation accuracy with all features minus without it
#   PCA drop        = the same for PCA, using its accuracy averaged over k = 1-12 (the
#                     mean height of the PCA curve in the plots, so no single k is
#                     cherry-picked)
#   importance      = the average of the two drops. The most important feature is the
#                     one whose removal costs the most accuracy.
#   QCML advantage  = PCA drop minus QCML drop. Positive means removing the feature
#                     hurts PCA more than it hurts QCML.
# Expect small numbers: losing mean radius, say, is cushioned by mean perimeter,
# mean area and largest radius, which carry nearly the same information.
#
# [note 8] P4 several features removed together
#
# The 3 features are picked with the same splits that then score these runs, so a
# feature that looked important partly by luck is also scored on that same luck: the
# z-scores here lean optimistic. Scoring on fresh splits (seeds N_SPLITS and up)
# would remove that bias.
