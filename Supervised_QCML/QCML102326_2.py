# Quantum Cognitive Machine Learning Project
# QCML102326_2: builds on QCML102326 with methods from Di Caro et al. (2025), "Quantum
# Cognition Machine Learning for Forecasting Chromosomal Instability"
# Runs in about 20 minutes.
# What it does and how: see the notes at the end of this file.


import csv
import os
import textwrap

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


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

FIT_N, FIT_C2 = 16, 0.03      # A1: fitting the shape of the whole data set
CLS_N, CLS_C2 = 4, 0.1        # A2-A4: one manifold per class, for classifying
N_SPLITS = 10                 # how many random train/validation splits to average over
TRAIN_FRACTION = 0.7          # 70% of the data train the models, 30% validate them
COMPONENTS = np.arange(1, 21) # PCA components k and diffusion-map eigenvectors m to try
DM_NEIGHBOURS = 7             # diffusion map kernel width (see DIFFUSION MAPS; best of 7, 30, 100)
N_REMOVE = 3                  # A4: how many features to remove together
PLOT_SPLIT = 0                # which split the classifier scatter plot shows
PLOT_DIR = "qcml102326_2_plots"   # every plot is saved here as its own PNG
SUP_N = 8                     # supervised QCML Hilbert-space dimension (best of 2, 4, 8)
SUP_LAMBDA, SUP_W = 0.1, 0.1  # mixed loss: weight of the reconstruction part (best of 0.01,
                              # 0.1, 1) and its fluctuation weight
SUP_EPOCHS = 100              # supervised QCML training passes (no L-BFGS: it overfitted)

MG_SERIES = 10                # Part B: how many random Mackey-Glass series to average over
MG_N, MG_C2 = 16, 0.01        # Part B: QCML settings (the best of n = 4, 8, 16, w = 0.1, 0.01)
MG_SUP_N = 16                 # Part B: supervised QCML Hilbert-space dimension (best of 8, 16)
MG_PCA = np.arange(1, 5)      # Part B: PCA components (the windows have 5 numbers)
MG_DM = np.arange(1, 401)     # Part B: diffusion-map eigenvectors (its error bottoms out
                              # near m = 300 and rises after, so the best is inside the range)
MG_DM_NEIGHBOURS = 300        # Part B: diffusion map kernel width (best of 7, 30, 100, 300)


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
    for epoch in range(1, EPOCHS + 1):
        order = torch.randperm(M)
        for start in range(0, M, BATCH):
            loss, _, _ = compute_loss(model, X[order[start:start + BATCH]], c2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if label and (epoch == 1 or epoch % 50 == 0):
            with torch.no_grad():
                loss, L1, L2 = compute_loss(model, X, c2)
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
    return model


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


def ground_state_from(model, values, known):
    """Ground state built from the `known` features only: their matrices alone make the
    error Hamiltonian. This is how QCML fills in a missing feature (Part B) and how
    supervised QCML reads its inputs. Differentiable, so it can be trained through."""
    H = model.observables()[known]
    X = torch.as_tensor(values, dtype=torch.float32)
    I = torch.eye(model.n, dtype=H.dtype)
    E = (torch.einsum("dij,djk->ik", H, H) - 2 * torch.einsum("bd,dij->bij", X.to(H.dtype), H)
         + (X ** 2).sum(1).to(H.dtype)[:, None, None] * I)
    return torch.linalg.eigh(E)[1][:, :, 0]


# SUPERVISED QCML (Di Caro et al., 2025)  [note 4]

def expectation(psi, A):
    """<psi|A|psi> for a batch of states: shape (batch,)."""
    return torch.einsum("bi,ij,bj->b", psi.conj(), A, psi).real


def train_supervised(inputs, target, n, loss_fn, lam=0.0, w=0.1, seed=0, epochs=EPOCHS,
                     polish=True):
    """Supervised QCML: K input matrices plus a target observable B (the last matrix),
    trained so <psi(x)|B|psi(x)> predicts `target`. lam > 0 adds the mixed-loss
    reconstruction term; polish finishes with L-BFGS on all the points."""
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

    if polish:
        lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=200,
                                  line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad()
            loss = loss_on(slice(None))
            loss.backward()
            return loss

        lbfgs.step(closure)
    return model


def supervised_forecast(model, inputs):
    """<psi(x)|B|psi(x)> for every row of `inputs`, B being the last matrix."""
    K = inputs.shape[1]
    with torch.no_grad():
        psi = ground_state_from(model, torch.tensor(inputs, dtype=torch.float32), list(range(K)))
        return expectation(psi, model.observables()[K]).numpy()


# PCA  [note 5]

def pca_distances(train_points, points, ks=COMPONENTS):
    """Squared distance from each sample to the flat PCA subspace of `train_points`,
    for every k in ks at once: shape (samples, len(ks))."""
    centre = train_points.mean(0)
    _, _, directions = np.linalg.svd(train_points - centre, full_matrices=False)
    offset = points - centre
    explained = np.cumsum((offset @ directions[:ks.max()].T) ** 2, axis=1)
    return (offset ** 2).sum(1, keepdims=True) - explained[:, ks - 1]


# DIFFUSION MAPS  [note 6]

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

    def coordinates(self, points, m=2):
        """Diffusion coordinates 1..m of (new) points."""
        return self.steps(points) @ self.right[:, 1:m + 1]

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


def own_and_other(d_B, d_M, y):
    """Reconstruction error (root mean square, in standard deviations) of the patients
    under the model of their own diagnosis, and under the model of the other one.
    d_B, d_M are squared distances, shape (patients,) or (patients, k)."""
    if d_B.ndim == 2:
        y = y[:, None]
    own = np.where(y == 1, d_M, d_B)
    other = np.where(y == 1, d_B, d_M)
    return np.sqrt(own.mean(0)), np.sqrt(other.mean(0))


# VALIDATION SCORING  [note 7]

def z_score(differences):
    """How many (corrected) standard errors the mean of `differences` is from zero."""
    d = np.asarray(differences, dtype=float)
    if np.ptp(d) < 1e-12:
        return 0.0 if abs(d.mean()) < 1e-12 else float(np.sign(d.mean()) * np.inf)
    error = np.sqrt((1 / len(d) + N_VAL / N_TRAIN) * d.var(ddof=1))
    return float(d.mean() / error)


def run_experiment(removed=()):
    """Train and score QCML, supervised QCML, PCA and diffusion maps on every split, with
    the features in `removed` deleted. Returns per-split validation accuracies (qcml_val
    and sup_val (N_SPLITS,), pca_val and dm_val (N_SPLITS, len(COMPONENTS))), each
    unsupervised method's reconstruction error under the patient's own and the other
    diagnosis's model, in "plot" the QCML distances of the PLOT_SPLIT validation patients
    for the scatter plot, and in "split_scores" every classifier's raw score per split
    (above 0 = called malignant) for the diagnostic metrics."""
    keep = [f for f in range(D) if f not in removed]
    scores = {f"{method}_{part}": [] for method in ("qcml", "pca", "dm")
              for part in ("val", "own", "other")}
    scores["sup_val"] = []
    result = {"split_scores": []}
    for seed, (train_idx, val_idx) in enumerate(SPLITS):
        Ztr = normalize(features[train_idx][:, keep], features[train_idx][:, keep])
        Zva = normalize(features[train_idx][:, keep], features[val_idx][:, keep])
        ytr, yva = labels[train_idx], labels[val_idx]

        model_B = train_qcml(Ztr[ytr == 0], CLS_N, CLS_C2, seed=seed)   # benign only
        model_M = train_qcml(Ztr[ytr == 1], CLS_N, CLS_C2, seed=seed)   # malignant only

        # A patient is called malignant if the malignant model describes them better.
        distances = {
            "qcml": (distance_to_manifold(model_B, Zva), distance_to_manifold(model_M, Zva)),
            "pca": (pca_distances(Ztr[ytr == 0], Zva), pca_distances(Ztr[ytr == 1], Zva)),
            "dm": (dm_distances(Ztr[ytr == 0], Zva), dm_distances(Ztr[ytr == 1], Zva)),
        }
        for method, (d_B, d_M) in distances.items():
            y = yva if d_B.ndim == 1 else yva[:, None]
            scores[method + "_val"].append(((d_M < d_B).astype(int) == y).mean(0))
            own, other = own_and_other(d_B, d_M, yva)
            scores[method + "_own"].append(own)
            scores[method + "_other"].append(other)

        # Supervised QCML, trained on the diagnosis with the mixed loss.
        sup_model = train_supervised(Ztr, ytr.astype(float), SUP_N,
                                     torch.nn.functional.binary_cross_entropy_with_logits,
                                     lam=SUP_LAMBDA, w=SUP_W, seed=seed, epochs=SUP_EPOCHS,
                                     polish=False)
        sup_score = supervised_forecast(sup_model, Zva)
        scores["sup_val"].append(((sup_score > 0).astype(int) == yva).mean())

        result["split_scores"].append({"y": yva, "sup": sup_score,
                                       **{method: d_B - d_M for method, (d_B, d_M)
                                          in distances.items()}})
        if seed == PLOT_SPLIT:
            result["plot"] = (*distances["qcml"], yva)
            result["sup_plot"] = (sup_score, yva)
    result.update({key: np.array(values) for key, values in scores.items()})
    return result


times_validated = np.zeros(N, dtype=int)
for _, val_idx in SPLITS:
    times_validated[val_idx] += 1
print(f"\nSPLITS: {N_SPLITS} stratified splits, each {N_TRAIN} train / {N_VAL} validation;"
      f" a patient is validated {times_validated.mean():.1f} times on average")


# RECONSTRUCTION ERROR (what A1 measures)  [note 8]

print("\nPART A1: one model for all the patients (split 0)")
train_idx, test_idx = SPLITS[0]
Z_train = normalize(features[train_idx], features[train_idx])
Z_test = normalize(features[train_idx], features[test_idx])

model = train_qcml(Z_train, FIT_N, FIT_C2, label="   ")
qcml_fit = fit_error(model, Z_test)
mean_baseline = float(np.sqrt((Z_test ** 2).sum(1).mean()))     # always answer the average
pca_errors = np.sqrt(pca_distances(Z_train, Z_test).mean(0))
dm_errors = np.sqrt(dm_distances(Z_train, Z_test).mean(0))

print("\nreconstruction error on validation patients, in standard deviations:")
print(f"   answer with the average patient : {mean_baseline:.3f}")
print(f"   QCML (n = {FIT_N})                   : {qcml_fit:.3f}")
print(f"   {'components':<32}: " + " ".join(f"{k:5d}" for k in COMPONENTS))
print(f"   {'PCA':<32}: " + " ".join(f"{v:5.2f}" for v in pca_errors))
print(f"   {'diffusion map':<32}: " + " ".join(f"{v:5.2f}" for v in dm_errors))

# Intrinsic dimension: the quantum metric says how fast the quantum state turns as each
# feature is nudged; the participation ratio of its eigenvalues counts the directions
# that matter.
metric = model.quantum_metric(torch.tensor(np.vstack([Z_train, Z_test]), dtype=torch.float32))
metric_eigs = torch.linalg.eigvalsh(metric).clamp(min=0)
eig_share = metric_eigs / metric_eigs.sum(-1, keepdim=True)
intrinsic_dim = float((eig_share.sum(-1) ** 2 / (eig_share ** 2).sum(-1)).mean())
print(f"\nintrinsic dimension from the quantum metric: {intrinsic_dim:.2f}")


# A2 one model per diagnosis, all features  [note 9]
print(f"\nPART A2: one model per diagnosis, all {D} features, {N_SPLITS} splits")
baseline = run_experiment()
k_index = int(baseline["pca_val"].mean(0).argmax())
m_index = int(baseline["dm_val"].mean(0).argmax())
k_best, m_best = int(COMPONENTS[k_index]), int(COMPONENTS[m_index])
print(f"   best PCA setting: k = {k_best};  best diffusion-map setting: m = {m_best}")


def scores_of(result):
    """Per-split validation accuracy of QCML, supervised QCML, PCA (k_best) and the
    diffusion map (m_best)."""
    return (result["qcml_val"], result["sup_val"], result["pca_val"][:, k_index],
            result["dm_val"][:, m_index])


qcml_all, sup_all, pca_all, dm_all = scores_of(baseline)
print(f"\nvalidation accuracy over {N_SPLITS} splits (mean ± std)     QCML minus it: mean, z")
print(f"   QCML (n = {CLS_N})               : {qcml_all.mean():.3f} ± {qcml_all.std():.3f}")
for name, val in [(f"supervised QCML (n = {SUP_N})", sup_all), (f"PCA (k = {k_best})", pca_all),
                  (f"diffusion map (m = {m_best})", dm_all)]:
    d = qcml_all - val
    print(f"   {name:<28}: {val.mean():.3f} ± {val.std():.3f}     {d.mean():+.3f}, z = {z_score(d):+.1f}")
d = sup_all - dm_all
print(f"   supervised QCML minus diffusion map: {d.mean():+.3f}, z = {z_score(d):+.1f}")


def compare(result):
    """Per-split changes from the all-feature run (paired: same splits). Returns QCML's
    drop, PCA's drop, their average (importance), and PCA's drop minus QCML's drop
    (QCML's advantage). A drop is positive when accuracy got worse. Importance and
    advantage use unsupervised QCML, as in QCML102326, so the rankings stay comparable."""
    qcml, _, pca, _ = scores_of(result)
    qcml_drop, pca_drop = qcml_all - qcml, pca_all - pca
    return qcml_drop, pca_drop, (qcml_drop + pca_drop) / 2, pca_drop - qcml_drop


# A3 each feature removed in turn  [note 10]
print(f"\nPART A3: each feature removed in turn ({D} experiments, {N_SPLITS} splits each)")
print(f"   {'feature removed':<31}  QCML   sup. QCML  PCA (k={k_best})  DM (m={m_best})")
singles = []
for f in range(D):
    singles.append(run_experiment([f]))
    q, s, p, m = (s.mean() for s in scores_of(singles[-1]))
    print(f"   {f + 1:2d} {FEATURE_NAMES[f]:<28}  {q:.3f}   {s:.3f}      {p:.3f}       {m:.3f}")

importance = np.array([compare(r)[2].mean() for r in singles])
importance_z = np.array([z_score(compare(r)[2]) for r in singles])
advantage = np.array([compare(r)[3].mean() for r in singles])
advantage_z = np.array([z_score(compare(r)[3]) for r in singles])
by_importance = np.lexsort((-importance_z, -importance))     # biggest first, ties by z
by_advantage = np.lexsort((-advantage_z, -advantage))

print("\nmost important features (average drop of QCML and PCA, z):")
for f in by_importance[:5]:
    print(f"   {feature_label(f):<40} {importance[f]:+.3f} ({importance_z[f]:+.1f})")
print("biggest QCML advantage (PCA drop minus QCML drop, z):")
for f in by_advantage[:5]:
    print(f"   {feature_label(f):<40} {advantage[f]:+.3f} ({advantage_z[f]:+.1f})")


# A4 three features removed together. They are picked on the same splits that then score
# them, so any difference here leans optimistic.
top_important = [int(f) for f in by_importance[:N_REMOVE]]
top_advantage = [int(f) for f in by_advantage[:N_REMOVE]]
print(f"\nPART A4: {N_REMOVE} features removed together")
both_important = run_experiment(top_important)
both_advantage = (both_important if set(top_advantage) == set(top_important)
                  else run_experiment(top_advantage))


# The breast-cancer results table: validation accuracy of the four classifiers at every
# removal step. For the single-feature average, ± is the spread across the 30 runs.
def wrapped(text, width=46):
    return "\n".join(textwrap.wrap(text, width))


def removed_names(fs):
    return ", ".join(feature_label(f) for f in fs)


averages = np.array([[s.mean() for s in scores_of(r)] for r in singles])   # (30, 4)
wdbc_rows = [(f"All {D} features", [s for s in scores_of(baseline)]),
             (f"One feature removed (average of the {D} runs)", list(averages.T)),
             (wrapped(f"Most important removed: {feature_label(int(by_importance[0]))}"),
              list(scores_of(singles[by_importance[0]]))),
             (wrapped(f"Top {N_REMOVE} important removed: {removed_names(top_important)}"),
              list(scores_of(both_important))),
             (wrapped(f"Biggest QCML advantage removed: {feature_label(int(by_advantage[0]))}"),
              list(scores_of(singles[by_advantage[0]]))),
             (wrapped(f"Top {N_REMOVE} QCML advantage removed: {removed_names(top_advantage)}"),
              list(scores_of(both_advantage)))]
wdbc_header = ["Features removed", f"QCML\n(n = {CLS_N})", f"Supervised QCML\n(n = {SUP_N})",
               f"PCA\n(k = {k_best})", f"Diffusion map\n(m = {m_best})"]
one_line = lambda text: text.replace("\n", " ")

print("\nBREAST-CANCER RESULTS: validation accuracy, mean ± std")
with open("qcml102326_2_wdbc_table.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["features removed"] + [f"{one_line(h)} {stat}" for h in wdbc_header[1:]
                                            for stat in ("mean", "std")])
    for label, cells in wdbc_rows:
        print(f"   {one_line(label)}")
        print("      " + "   ".join(f"{one_line(h)}: {c.mean():.3f} ± {c.std():.3f}"
                                    for h, c in zip(wdbc_header[1:], cells)))
        writer.writerow([one_line(label)] + [v for c in cells for v in (c.mean(), c.std())])
print("saved to qcml102326_2_wdbc_table.csv")


# DIAGNOSTIC SCORES (as reported by Di Caro et al.), all features  [note 11]
def diagnostic_metrics(score, y):
    called = score > 0
    sensitivity = called[y == 1].mean()
    specificity = 1 - called[y == 0].mean()
    # ROC AUC by counting pairs (Mann-Whitney): ties count half
    malignant, benign = score[y == 1], score[y == 0]
    auc = ((malignant[:, None] > benign[None, :]).mean()
           + 0.5 * (malignant[:, None] == benign[None, :]).mean())
    return [(called == y).mean(), sensitivity, specificity, (sensitivity + specificity) / 2, auc]


METRIC_NAMES = ["Accuracy", "Sensitivity", "Specificity", "Balanced\naccuracy", "ROC AUC"]
metric_columns = {"qcml": None, "sup": None, "pca": k_index, "dm": m_index}
metrics = {method: np.array([diagnostic_metrics(split[method] if column is None
                                                else split[method][:, column], split["y"])
                             for split in baseline["split_scores"]])          # (splits, 5)
           for method, column in metric_columns.items()}
metric_rows = [(one_line(h), list(metrics[method].T))
               for h, method in zip(wdbc_header[1:], metric_columns)]

print(f"\nBREAST-CANCER DIAGNOSTIC SCORES, all {D} features (mean ± std over {N_SPLITS} splits)")
print("   " + " " * 30 + "".join(f"{one_line(m):>19}" for m in METRIC_NAMES))
with open("qcml102326_2_wdbc_metrics.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["method"] + [f"{one_line(m).lower()} {stat}" for m in METRIC_NAMES
                                  for stat in ("mean", "std")])
    for name, cells in metric_rows:
        print(f"   {name:<30}" + "".join(f"{c.mean():>11.3f} ± {c.std():.3f}" for c in cells))
        writer.writerow([name] + [v for c in cells for v in (c.mean(), c.std())])
print("saved to qcml102326_2_wdbc_metrics.csv")


# PART B: MACKEY-GLASS  [note 12]
LAGS = np.array([-18, -12, -6, 0, 6])
KNOWN, TARGET = [0, 1, 2, 3], 4


def mackey_glass(seed, length=1500, burn=500, tau=17, beta=0.2, gamma=0.1, power=10, dt=0.1):
    rng = np.random.default_rng(seed)
    lag, steps = round(tau / dt), round((length + burn) / dt)
    x = np.empty(lag + steps + 1)
    x[:lag + 1] = rng.uniform(0.5, 1.5)                          # random constant history
    rate = lambda now, past: beta * past / (1 + past ** power) - gamma * now
    for i in range(lag, lag + steps):
        past0, past1 = x[i - lag], x[i - lag + 1]
        past_half = (past0 + past1) / 2
        k1 = rate(x[i], past0)
        k2 = rate(x[i] + dt / 2 * k1, past_half)
        k3 = rate(x[i] + dt / 2 * k2, past_half)
        k4 = rate(x[i] + dt * k3, past1)
        x[i + 1] = x[i] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    every = round(1 / dt)
    return x[lag::every][burn:burn + length]


def windows(series):
    t = np.arange(-LAGS.min(), len(series) - LAGS.max())
    return np.stack([series[t + lag] for lag in LAGS], axis=1)


def pca_predict(train, values, known, ks):
    """For every k in ks: the full points on the flat subspace that best match `values`
    in the `known` columns. Shape (len(ks), points, columns)."""
    centre = train.mean(0)
    _, _, directions = np.linalg.svd(train - centre, full_matrices=False)
    out = []
    for k in ks:
        V = directions[:k]
        shares = np.linalg.lstsq(V[:, known].T, (values - centre[known]).T, rcond=None)[0]
        out.append(centre + shares.T @ V)
    return np.array(out)


print(f"\nPART B: Mackey-Glass, {MG_SERIES} random series, predicting x(t+6)")
mg = {key: [] for key in ("qcml_pred", "qcml_recon", "sup_pred", "pca_pred", "pca_recon",
                          "dm_pred", "dm_recon")}
for seed in range(MG_SERIES):
    series = mackey_glass(seed)
    W = windows(series)
    cut = int(TRAIN_FRACTION * len(W))
    train, val = W[:cut], W[cut - LAGS.min() + LAGS.max():]
    Ztr, Zva = normalize(train, train), normalize(train, val)
    mu, sd = train.mean(0), train.std(0)
    truth = val[:, TARGET]

    def error(z_prediction):                       # z-scored target -> normalised RMSE (per setting)
        return np.sqrt(((z_prediction * sd[TARGET] + mu[TARGET] - truth) ** 2).mean(-1)) / series.std()

    def recon_error(rebuilt):                      # rebuilt windows -> RMS distance
        return np.sqrt(((rebuilt - Zva) ** 2).sum(-1).mean(-1))

    qcml_model = train_qcml(Ztr, MG_N, MG_C2, seed=seed)
    with torch.no_grad():
        psi = ground_state_from(qcml_model, Zva[:, KNOWN], KNOWN)
        mg["qcml_pred"].append(error(qcml_model.bloch(psi)[:, TARGET].numpy()))
    mg["qcml_recon"].append(fit_error(qcml_model, Zva))

    sup_model = train_supervised(Ztr[:, KNOWN], Ztr[:, TARGET], MG_SUP_N,
                                 torch.nn.functional.mse_loss, seed=seed)
    mg["sup_pred"].append(error(supervised_forecast(sup_model, Zva[:, KNOWN])))

    mg["pca_pred"].append(error(pca_predict(Ztr, Zva[:, KNOWN], KNOWN, MG_PCA)[:, :, TARGET]))
    mg["pca_recon"].append(np.sqrt(pca_distances(Ztr, Zva, MG_PCA).mean(0)))

    dm_inputs = DiffusionMap(Ztr[:, KNOWN], MG_DM_NEIGHBOURS)
    mg["dm_pred"].append(error(dm_inputs.rebuild(Ztr[:, [TARGET]], Zva[:, KNOWN], MG_DM)[:, :, 0]))
    mg["dm_recon"].append(recon_error(DiffusionMap(Ztr, MG_DM_NEIGHBOURS).rebuild(Ztr, Zva, MG_DM)))
    if seed == 0:
        mg_plot_map, mg_plot_points, mg_plot_x = DiffusionMap(Ztr, MG_DM_NEIGHBOURS), Ztr, train[:, 3]
    print(f"   series {seed}: QCML {mg['qcml_pred'][-1]:.4f}   supervised QCML "
          f"{mg['sup_pred'][-1]:.4f}   PCA best {np.min(mg['pca_pred'][-1]):.4f}   "
          f"diffusion map best {np.min(mg['dm_pred'][-1]):.4f}")
mg = {key: np.array(values) for key, values in mg.items()}
mg_k = int(MG_PCA[mg["pca_pred"].mean(0).argmin()])
mg_m = int(MG_DM[mg["dm_pred"].mean(0).argmin()])
mg_rows = [("QCML, unsupervised", [f"n = {MG_N}, w = {MG_C2:g}", mg["qcml_pred"], mg["qcml_recon"]]),
           ("Supervised QCML", [f"n = {MG_SUP_N}", mg["sup_pred"], "–"]),
           ("PCA", [f"k = {mg_k}", mg["pca_pred"][:, mg_k - 1], mg["pca_recon"][:, mg_k - 1]]),
           ("Diffusion map", [f"m = {mg_m}", mg["dm_pred"][:, mg_m - 1], mg["dm_recon"][:, mg_m - 1]])]
mg_header = ["Method", "Setting", "Prediction error x(t+6)", "Reconstruction error"]
d = mg["sup_pred"] - mg["dm_pred"][:, mg_m - 1]
# The series are independent, so no overlap correction: z = mean(d) / (std(d) / sqrt(S)).
print(f"\nsupervised QCML minus diffusion map, prediction error: {d.mean():+.4f}, "
      f"z = {d.mean() / (d.std(ddof=1) / np.sqrt(MG_SERIES)):+.1f}")

mean_std = lambda c: ("–", "–") if isinstance(c, str) else (c.mean(), c.std())
print("\nMACKEY-GLASS RESULTS (validation, mean ± std over the series):")
with open("qcml102326_2_mackey_glass_table.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["method", "setting", "prediction error mean", "prediction error std",
                     "reconstruction error mean", "reconstruction error std"])
    for name, (setting, pred, recon) in mg_rows:
        recon_text = recon if isinstance(recon, str) else f"{recon.mean():.4f} ± {recon.std():.4f}"
        print(f"   {name:<19} {setting:<16} prediction {pred.mean():.4f} ± {pred.std():.4f}"
              f"   reconstruction {recon_text}")
        writer.writerow([name, setting, *mean_std(pred), *mean_std(recon)])
print("saved to qcml102326_2_mackey_glass_table.csv")


# Plots: one PNG per plot, numbered in reading order, saved in PLOT_DIR

os.makedirs(PLOT_DIR, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "figure.titlesize": 13,
                     "figure.titleweight": "bold", "legend.fontsize": 9,
                     "legend.title_fontsize": 9, "legend.frameon": False,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.grid": True, "axes.axisbelow": True,
                     "grid.color": "#e1e0d9", "grid.linewidth": 0.6})
QCML_COLOUR, PCA_COLOUR, DM_COLOUR = "#eb6834", "#2a78d6", "#1baf7a"
BENIGN_COLOUR, MALIGNANT_COLOUR = "#2a78d6", "#e34948"
GREY, INK, MUTED_INK, RULE = "#898781", "#0b0b0b", "#52514e", "#c3c2b7"
SEQUENTIAL = LinearSegmentedColormap.from_list("blues", ["#b7d3f6", "#104281"])


def new_plot(title, subtitle, size=(8.5, 7)):
    """A figure holding a single plot, with a bold title and a grey subtitle."""
    fig, ax = plt.subplots(figsize=size, layout="constrained")
    fig.suptitle(title)
    ax.set_title(subtitle, color=MUTED_INK)
    return fig, ax


def save_figure(fig, name):
    path = os.path.join(PLOT_DIR, f"{name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"   {path}")


def save_plot(fig, ax, name, legend_title=None):
    """Put the legend under the plot, then save and close it."""
    ax.legend(title=legend_title, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2,
              alignment="left")
    save_figure(fig, name)


def table_plot(name, title, subtitle, header, rows, note="", label_width=0.44, size=11,
               digits=3):
    """A results table as a PNG. rows: (label, cells); a cell is text or an array of
    per-run scores, shown as mean ± std. The label column takes `label_width` of the
    `size`-inch width and the value columns share the rest."""
    show = lambda c: c if isinstance(c, str) else f"{np.mean(c):.{digits}f} ± {np.std(c):.{digits}f}"
    lines = [max(h.count("\n") + 1 for h in header)] + [label.count("\n") + 1 for label, _ in rows]
    line, pad, top = 0.22, 0.18, 1.0                        # inches
    note = "\n".join(textwrap.wrap(note, int(size * 14)))      # fits the page width
    note_height = 0.3 + 0.16 * note.count("\n") + 0.2 if note else 0.25
    height = top + sum(n * line + pad for n in lines) + note_height
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
            fig.text(label_width + width * (j + 0.5), y_of(y + h / 2), show(cell), ha="center",
                     va="center", fontweight=weight, linespacing=1.4)
        y += h
        fig.add_artist(plt.Line2D([0.02, 0.98], [y_of(y)] * 2, color=INK if i == 0 else RULE,
                                  lw=1 if i == 0 else 0.6))
    if note:
        fig.text(0.02, y_of(y + 0.2), note, fontsize=8.5, color=MUTED_INK, va="top",
                 linespacing=1.4)
    save_figure(fig, name)


print(f"\nplots saved to {PLOT_DIR}:")

# 01: one model for everybody
fig, ax = new_plot("Breast cancer: how closely ONE model describes all patients",
                   "reconstruction error of the validation patients of split 0 (lower = closer)\n"
                   "diagnoses are not used here, so this is not accuracy")
ax.axhline(qcml_fit, color=QCML_COLOUR, lw=2, ls="--", label=f"QCML (n = {FIT_N})")
ax.plot(COMPONENTS, pca_errors, "o-", color=PCA_COLOUR, lw=2, ms=4, label="PCA, k components")
ax.plot(COMPONENTS, dm_errors, "s-", color=DM_COLOUR, lw=2, ms=4,
        label="diffusion map, m eigenvectors")
ax.axhline(mean_baseline, color=GREY, ls=":", label="answer with the average patient")
ax.set_xticks(COMPONENTS)
ax.set_xlabel("PCA components (k) or diffusion-map eigenvectors (m)")
ax.set_ylabel("reconstruction error (standard deviations)")
save_plot(fig, ax, "01_wdbc_reconstruction_one_model")

# 02: the same error for the two diagnosis models each classifier uses
fig, ax = new_plot("Breast cancer: reconstruction error under the two diagnosis models",
                   f"validation patients, all {D} features, mean of {N_SPLITS} splits")
for method, colour, marker, title in [("pca", PCA_COLOUR, "o", "PCA"),
                                      ("dm", DM_COLOUR, "s", "diffusion map")]:
    ax.plot(COMPONENTS, baseline[method + "_own"].mean(0), marker + "-", color=colour, lw=2,
            ms=4, label=f"{title}, model of their own diagnosis")
    ax.plot(COMPONENTS, baseline[method + "_other"].mean(0), marker + "--", color=colour, lw=2,
            ms=4, mfc="white", label=f"{title}, model of the other diagnosis")
ax.axhline(baseline["qcml_own"].mean(), color=QCML_COLOUR, lw=2,
           label=f"QCML (n = {CLS_N}), model of their own diagnosis")
ax.axhline(baseline["qcml_other"].mean(), color=QCML_COLOUR, lw=2, ls="--",
           label=f"QCML (n = {CLS_N}), model of the other diagnosis")
ax.set_xticks(COMPONENTS)
ax.set_xlabel("PCA components (k) or diffusion-map eigenvectors (m)")
ax.set_ylabel("reconstruction error (standard deviations)")
save_plot(fig, ax, "02_wdbc_reconstruction_diagnosis_models",
          "a patient is diagnosed correctly when the model of their own diagnosis\n"
          "describes them more closely than the model of the other diagnosis")

# 03: the QCML classifier and its decision line
d_B, d_M, y = baseline["plot"]
wrong = (d_M < d_B).astype(int) != y
fig, ax = new_plot(f"Breast cancer: QCML classifier, all {D} features (n = {CLS_N})",
                   f"validation patients of split {PLOT_SPLIT}, accuracy {1 - wrong.mean():.3f}",
                   size=(8, 8.5))
for value, colour, marker, name in [(0, BENIGN_COLOUR, "o", "benign"),
                                    (1, MALIGNANT_COLOUR, "^", "malignant")]:
    sel = y == value
    ax.scatter(d_B[sel], d_M[sel], s=16, alpha=0.7, color=colour, marker=marker, lw=0, label=name)
ax.scatter(d_B[wrong], d_M[wrong], s=80, facecolors="none", edgecolors=INK, lw=0.8,
           label=f"misdiagnosed ({wrong.sum()} of {len(y)})")
low, high = min(d_B.min(), d_M.min()) / 1.5, max(d_B.max(), d_M.max()) * 1.5
ax.plot([low, high], [low, high], color=INK, ls="--", lw=1, label="decision line")
ax.text(0.97, 0.03, "called malignant", transform=ax.transAxes, ha="right", va="bottom",
        color=MUTED_INK)
ax.text(0.65, 0.97, "called benign", transform=ax.transAxes, ha="center", va="top",
        color=MUTED_INK)
ax.set_xlim(low, high)
ax.set_ylim(low, high)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("distance to the benign manifold")
ax.set_ylabel("distance to the malignant manifold")
save_plot(fig, ax, "03_wdbc_qcml_classifier")

# 04: the supervised QCML classifier and its decision line. Each validation patient is
# one dot at their score <psi|B|psi>; the rows only separate the two diagnoses, and the
# small vertical jitter keeps dots from hiding each other.
sup_score, y = baseline["sup_plot"]
wrong = (sup_score > 0).astype(int) != y
jitter = np.random.default_rng(0).uniform(-0.18, 0.18, len(y))
fig, ax = new_plot(f"Breast cancer: supervised QCML classifier, all {D} features (n = {SUP_N})",
                   f"validation patients of split {PLOT_SPLIT}, accuracy {1 - wrong.mean():.3f}",
                   size=(8.5, 5.5))
for value, colour, marker, name in [(0, BENIGN_COLOUR, "o", "benign"),
                                    (1, MALIGNANT_COLOUR, "^", "malignant")]:
    sel = y == value
    ax.scatter(sup_score[sel], value + jitter[sel], s=16, alpha=0.7, color=colour,
               marker=marker, lw=0, label=name)
ax.scatter(sup_score[wrong], y[wrong] + jitter[wrong], s=80, facecolors="none", edgecolors=INK,
           lw=0.8, label=f"misdiagnosed ({wrong.sum()} of {len(y)})")
ax.axvline(0, color=INK, ls="--", lw=1, label="decision line")
ax.text(0.98, 0.5, "called malignant →", transform=ax.transAxes, ha="right", va="center",
        color=MUTED_INK)
ax.text(0.02, 0.5, "← called benign", transform=ax.transAxes, ha="left", va="center",
        color=MUTED_INK)
ax.set_yticks([0, 1], ["benign", "malignant"])
ax.set_ylim(-0.6, 1.6)
ax.grid(axis="y", visible=False)
ax.set_xlabel("score <ψ|B|ψ>: the log-odds of malignant")
ax.set_ylabel("true diagnosis")
save_plot(fig, ax, "04_wdbc_supervised_qcml_classifier")

# 05: the breast-cancer results table
table_plot("05_wdbc_results_table", "Breast cancer: validation accuracy as features are removed",
           f"mean ± std over {N_SPLITS} splits; PCA and the diffusion map at their best "
           "setting on all features", wdbc_header, wdbc_rows, label_width=0.36, size=12,
           note=f"For the one-feature row, ± is the spread across the {D} runs. Importance = "
                "accuracy drop averaged over QCML and PCA; QCML advantage = PCA's drop "
                "minus QCML's drop (both use unsupervised QCML).")

# 06: the breast-cancer diagnostic scores
table_plot("06_wdbc_diagnostic_metrics", "Breast cancer: diagnostic scores, all features",
           f"validation patients, mean ± std over {N_SPLITS} splits; malignant counts as "
           "positive", ["Method"] + METRIC_NAMES, metric_rows, label_width=0.2, size=12,
           note="Sensitivity = malignant called malignant; specificity = benign called "
                "benign; balanced accuracy = their average; ROC AUC = chance a malignant "
                "patient scores above a benign one.")

# 07: the breast-cancer diffusion map (all patients, z-scored together)
all_map = DiffusionMap(normalize(features, features))
coords = all_map.coordinates(all_map.points)
fig, ax = new_plot("Breast cancer: diffusion map of all 569 patients",
                   "the first two diffusion coordinates; nearby points are easy to walk "
                   "between in the data", size=(8, 7.5))
for value, colour, marker, name in [(0, BENIGN_COLOUR, "o", "benign"),
                                    (1, MALIGNANT_COLOUR, "^", "malignant")]:
    sel = labels == value
    ax.scatter(coords[sel, 0], coords[sel, 1], s=16, alpha=0.7, color=colour, marker=marker,
               lw=0, label=name)
ax.set_xlabel("diffusion coordinate 1")
ax.set_ylabel("diffusion coordinate 2")
save_plot(fig, ax, "07_wdbc_diffusion_map")

# 08: Mackey-Glass reconstruction errors
fig, ax = new_plot("Mackey–Glass: how closely each method describes the windows",
                   f"reconstruction error of the validation windows, mean of {MG_SERIES} "
                   "series (log scale)")
ax.axhline(mg["qcml_recon"].mean(), color=QCML_COLOUR, lw=2, ls="--",
           label=f"QCML (n = {MG_N})")
ax.plot(MG_PCA, mg["pca_recon"].mean(0), "o-", color=PCA_COLOUR, lw=2, ms=4,
        label="PCA, k components")
ax.plot(MG_DM, mg["dm_recon"].mean(0), "-", color=DM_COLOUR, lw=2,
        label="diffusion map, m eigenvectors")
ax.set_yscale("log")
ax.set_xlabel("PCA components (k) or diffusion-map eigenvectors (m)")
ax.set_ylabel("reconstruction error (standard deviations)")
save_plot(fig, ax, "08_mackey_glass_reconstruction")

# 09: the Mackey-Glass results table
table_plot("09_mackey_glass_results_table", "Mackey–Glass: predicting x(t+6)",
           f"validation, mean ± std over {MG_SERIES} random series; PCA and the diffusion map "
           "at their best setting", mg_header, mg_rows, digits=4,
           note="Prediction error = RMS error / the series' standard deviation (0 is perfect, "
                "1 = guessing the average). Reconstruction error in standard deviations; "
                "supervised QCML only predicts, so it has none.")

# 10: the Mackey-Glass diffusion map (training windows of series 0)
coords = mg_plot_map.coordinates(mg_plot_points)
fig, ax = new_plot("Mackey–Glass: diffusion map of the training windows (series 0)",
                   "the first two diffusion coordinates, coloured by the current value x(t)",
                   size=(8, 7.5))
dots = ax.scatter(coords[:, 0], coords[:, 1], c=mg_plot_x, cmap=SEQUENTIAL, s=10, lw=0)
fig.colorbar(dots, ax=ax, label="x(t)", shrink=0.8)
ax.set_xlabel("diffusion coordinate 1")
ax.set_ylabel("diffusion coordinate 2")
save_figure(fig, "10_mackey_glass_diffusion_map")


# ========================================================================================
# EXPLANATION
# Notes on the code above, in the order of its sections. A tag [note k] in the code
# marks the section that note k explains.
# ========================================================================================
#
# [note 0] About this script
#
# PART A  Wisconsin Diagnostic Breast Cancer data (UCI dataset 17): 569 tumours, 30
#         features measured from cell nuclei in a digitised image, each labelled benign
#         (357) or malignant (212).
#   A1 fits one model to all the patients and measures how closely it describes them
#      (reconstruction error): QCML against PCA and a diffusion map.
#   A2 classifies patients: unsupervised QCML, PCA and the diffusion map fit one model
#      per diagnosis and pick the one that describes a patient better; supervised QCML
#      is trained on the diagnosis directly.
#   A3 removes each of the 30 features in turn to find the most important features and
#      the ones whose loss hurts QCML less than PCA, then A4 removes three at once.
#   Output: a table of the four classifiers' accuracy at every removal step, and a
#   table of their diagnostic scores with all features.
# PART B  The Mackey-Glass chaotic time series: how well QCML (unsupervised and
#         supervised), PCA and a diffusion map predict the series 6 steps ahead.
#
# New since QCML102326 (from Di Caro et al.):
#   - supervised QCML: a learned target observable B whose expected value <psi|B|psi>
#     is the prediction, trained directly on the prediction error (SUPERVISED QCML)
#   - for breast cancer, a mixed loss: the classification loss plus QCML's own
#     reconstruction loss, which stops supervised QCML memorising the training patients
#   - Hilbert-space dimensions chosen by validation, as Di Caro et al. recommend
#   - their diagnostic scores for the all-feature run: sensitivity, specificity,
#     balanced accuracy and ROC AUC
#   - a wider diffusion-map range for Mackey-Glass (up to 400 eigenvectors; QCML102326
#     stopped at 40, while it was still improving), so its score is not capped
#   - plots in qcml102326_2_plots, one PNG each:
#       01-02 reconstruction errors, 03 the unsupervised QCML classifier with its
#       decision line, 04 the supervised QCML classifier with its decision line,
#       05 the breast-cancer results table, 06 the breast-cancer diagnostic scores,
#       07 the breast-cancer diffusion map, 08 Mackey-Glass reconstruction errors,
#       09 the Mackey-Glass results table, 10 the Mackey-Glass diffusion map
#   - CSVs named qcml102326_2_*.csv
#
# Runs in about 20 minutes. Lower N_SPLITS or MG_SERIES for a quicker, noisier run.
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
# deviation, both measured on the TRAINING data only. Every number then reads as
# "standard deviations away from the typical training example".
#
# Why the breast-cancer data need it: the raw features live on wildly different scales.
# Largest area (feature 24) runs from 185 to 4254 while fractal dimension variation
# (feature 20) runs from 0.0009 to 0.03, and their standard deviations differ by a
# factor of about 215,000. QCML, PCA and the diffusion map all add up squared
# differences over the features, so on raw numbers the three area features would be
# over 99% of everything and the other 27 measurements would be ignored. QCML does not
# even train on the raw numbers: the area terms swamp the error Hamiltonian and PyTorch
# stops with an "ill-defined" eigenvector-gradient error. After z-scoring, every
# measurement gets an equal vote.
#
# Caveats (QCML100626 printed the numbers):
#   1. Skew and outliers. z-scoring is a straight-line rescale, so it cannot change a
#      feature's shape. The variation features (11-20) are strongly right-skewed, so a
#      few patients sit 8 to 12 standard deviations out and count heavily in squared
#      errors. Taking the log first did not help (QCML validation 0.963 vs 0.970).
#   2. Redundancy. radius, perimeter and area measure almost the same thing (15 feature
#      pairs correlate above 0.95), so removing any single one of them barely matters.
#   3. No peeking. The mean and std come from the training split only and are reused,
#      unchanged, on the validation split, which must stay unseen.
#
# [note 3] NOTE ON SPLITS
#
# A split divides the 569 patients into a training set the models learn from and a
# held-out validation set that is only used to score them. Each split is stratified:
# benign and malignant patients are shuffled and cut separately, so every split keeps
# the data's 63% / 37% mix (250 + 148 train, 107 + 64 validation). The same 10 splits
# (seeds 0-9) are reused by every experiment, so a run with a feature removed differs
# from the all-feature run only in that feature. This is repeated random sub-sampling,
# not k-fold cross-validation: the held-out sets overlap. The settings n, w and EPOCHS
# were tuned in earlier iterations on these scores, so the validation set is not a
# never-touched test set. (Part B splits each series in time instead; see MACKEY-GLASS.)
#
# [note 4] SUPERVISED QCML (Di Caro et al., 2025)
#
# Unsupervised QCML learns the shape of the data and is then asked a question it was not
# trained for (which manifold is nearer, or what the missing feature is). Di Caro et al.
# train it on the question itself. Next to the K input matrices H_1..H_K there is one
# more learned Hermitian matrix B, the "target observable". A sample x is turned into a
# ground state psi(x) using the input matrices only (as in ground_state_from), and the
# prediction is B's expected value in that state:
#     prediction(x) = <psi(x)| B |psi(x)>
# All K + 1 matrices are trained together to make the predictions right: squared error
# for Mackey-Glass, and for breast cancer the binary cross-entropy of the diagnosis,
# with <psi|B|psi> as the log-odds of malignant (called malignant when it is above 0).
#
# Trained on the diagnosis alone, it memorised the 398 training patients (perfect on
# them, 0.971 on validation). Di Caro et al. regularise it with a mixed loss: the
# prediction loss plus lambda times QCML's own loss (position error + w * fluctuation),
# so the input matrices must still describe the data and cannot bend freely around
# single patients. lambda = SUP_LAMBDA and the dimension n were chosen on the validation
# scores, as Di Caro et al. do by cross-validation.
#
# [note 5] PCA
#
# PCA with k components fits a flat k-dimensional subspace through the data; a point's
# reconstruction is its projection onto that subspace. As a classifier it is built like
# QCML: k components fitted to each diagnosis, and a patient goes to whichever flat
# subspace reconstructs them better, so the only difference from QCML is the shape fitted
# to each class (flat vs curved).
#
# [note 6] DIFFUSION MAPS
#
# A diffusion map is a nonlinear manifold learner built from a random walk on the data.
# Every pair of points gets a similarity exp(-distance^2 / (s_i s_j)): large for near
# neighbours, tiny for far ones. Each point's width s is its distance to its
# DM_NEIGHBOURS-th nearest neighbour, so the width adapts to how crowded the data are
# around it. (One shared width, such as the median distance, let a few extreme patients
# cut themselves off from the rest; the first diffusion coordinates then described those
# few patients and nothing else.) Turned into step probabilities, the similarities give
# a random walk that mostly hops between neighbours, so it spreads along the data's own
# shape. The walk's slowest-changing patterns (the top eigenvectors of its step matrix)
# are smooth coordinates along that shape, and plotting the first two gives the
# "diffusion map" picture. A wider neighbourhood suits smooth, evenly sampled data
# (Mackey-Glass uses MG_DM_NEIGHBOURS).
#
# Like PCA, it can rebuild a point from its first m coordinates (m eigenvectors plus the
# constant one). A new point is placed by averaging the eigenvectors over its training
# neighbours (the Nystrom extension), and anything known on the training points, such as
# the features themselves or a value to predict, is rebuilt from the same m eigenvectors
# (geometric harmonics). At m = 0 every point is rebuilt as the weighted average training
# point; more eigenvectors follow the data more closely. Unlike PCA and QCML it keeps the
# whole training set and works locally, through each point's neighbours.
#
# As a classifier it is built exactly like the other two: one diffusion map per
# diagnosis, and a patient goes to whichever reconstructs them more closely.
#
# [note 7] VALIDATION SCORING
#
# validation accuracy: the fraction of the 30% validation patients diagnosed correctly,
# reported as mean ± standard deviation over the N_SPLITS splits. To compare two settings
# A and B, take their scores split by split (paired) and z-score the differences d = A - B:
#     z = mean(d) / sqrt((1/S + n_val/n_train) * var(d))         S = N_SPLITS
# The n_val/n_train term (Nadeau & Bengio, 2003) corrects for the splits sharing patients.
# Reading it: |z| below 1 is noise, |z| above 2 is a real difference.
#
# [note 8] RECONSTRUCTION ERROR (what A1 measures)
#
# Each model can only describe points that lie on its own shape. Its "reconstruction"
# of a patient is the nearest point it can describe:
#   - QCML: the expected measurements <psi|H_d|psi> of the patient's ground state psi.
#   - PCA with k components: the projection onto a flat k-dimensional subspace.
#   - Diffusion map with m eigenvectors: the patient rebuilt from m eigenvectors.
# The reconstruction error is the typical distance between a patient's 30 z-scored
# numbers and their reconstruction, in standard deviations:
#     error = sqrt( average over patients of  sum_d (x_d - reconstruction_d)^2 )
# Lower means a closer description; answering every patient with the average patient
# gives about 5.3. ONE model is fitted to all training patients and the diagnoses are
# not used, so this measures description, not accuracy. Diagnosis (A2) needs each
# patient described clearly better by the model of their own diagnosis than by the
# other one; plot 02 shows both errors for the three methods.
#
# [note 9] A2 one model per diagnosis, all features
#
# PCA and the diffusion map are scored at one setting each: the number of components
# (or eigenvectors) with the highest mean validation accuracy on all features. That
# setting is then kept fixed for every removal step. Picking it on the validation
# patients flatters PCA and the diffusion map slightly, so a QCML lead is conservative.
#
# [note 10] A3 each feature removed in turn
#
# "Removing a feature" deletes its column before anything else happens, in training and
# in validation, on the same splits as the all-feature run. For each feature:
#   importance     = the drop in validation accuracy, averaged over QCML and PCA
#   QCML advantage = PCA's drop minus QCML's drop: positive means removing the feature
#                    hurts PCA more than it hurts QCML
#
# [note 11] DIAGNOSTIC SCORES (as reported by Di Caro et al.), all features
#
# Accuracy alone hides which mistakes a classifier makes. With malignant as "positive":
#   sensitivity       = share of malignant patients called malignant (missed cancers = 1 - it)
#   specificity       = share of benign patients called benign (false alarms = 1 - it)
#   balanced accuracy = (sensitivity + specificity) / 2, which does not reward a classifier
#                       for favouring benign, the larger class (63% of patients)
#   ROC AUC           = the chance that a random malignant patient gets a higher score than
#                       a random benign one. It judges the score itself, at every possible
#                       threshold rather than only at 0: 1 is perfect ranking, 0.5 is chance.
# A classifier's score is how much nearer the malignant model is (benign distance minus
# malignant distance) for QCML, PCA and the diffusion map, and <psi|B|psi> for supervised
# QCML; above 0 means called malignant.
#
# [note 12] PART B: MACKEY-GLASS
#
# dx/dt = beta x(t - tau) / (1 + x(t - tau)^p) - gamma x(t), with beta = 0.2,
# gamma = 0.1, p = 10 and delay tau = 17, the standard chaotic setting. Mackey and Glass
# introduced it for physiological control such as blood-cell production: the delayed
# term makes the present depend on the past, and at tau = 17 the series never repeats.
# Each of MG_SERIES series starts from a different random constant history, so each
# follows its own chaotic path. The equation is stepped with Runge-Kutta (dt = 0.1), the
# first 500 time units are dropped so the start does not matter, and the series is
# sampled once per time unit.
#
# The task is the classic benchmark: from x(t-18), x(t-12), x(t-6) and x(t), predict
# x(t+6). Every method learns the shape of the 5-number windows
# (x(t-18), x(t-12), x(t-6), x(t), x(t+6)) on the first 70% of a series and predicts on
# the last 30% (split in time, with a 24-step gap so no window straddles the two):
#   - QCML: the ground state is found from the four known values only, and the fifth
#     matrix's expectation value in that state is the prediction.
#   - Supervised QCML: the same prediction, <psi|B|psi> with B the fifth matrix, but all
#     five matrices are trained on the squared prediction error itself rather than on
#     describing the windows (see SUPERVISED QCML). No mixed loss here: with 1000+
#     smooth training windows it did not overfit. It does not describe whole windows,
#     so it has no reconstruction error.
#   - PCA with k components: the point on the flat subspace that best matches the four
#     known values (least squares); its fifth coordinate is the prediction.
#   - Diffusion map with m eigenvectors: built on the four known values; x(t+6) is
#     rebuilt from the training windows' values, as in DIFFUSION MAPS.
# Prediction error = root-mean-square error divided by the series' standard deviation
# (0 is perfect; 1 is no better than always guessing the average). Reconstruction error
# = how closely each method describes whole windows, as in Part A1 (standard deviations).
