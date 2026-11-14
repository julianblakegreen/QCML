# Quantum Cognitive Machine Learning Project
# QCML102426: QCML on quantum reservoir computing data (Dudas et al., 2023, "Quantum
# reservoir computing implementation on coherently coupled quantum oscillators", npj
# Quantum Information 9, 64; their code and saved results are published with the paper)
# Runs in about 20 minutes.
# What it does and how: see the notes at the end of this file.


import csv
import os
import textwrap
import time
from math import factorial

import numpy as np
import scipy.sparse as sp
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply


# Repository set-up: run from this folder, with QCML from 1_Plain_QCML
import os
import sys
_FOLDER = os.path.dirname(os.path.abspath(__file__))
os.chdir(_FOLDER)                                                 # data and outputs live in this folder
sys.path.insert(0, os.path.join(_FOLDER, "..", "1_Plain_QCML"))   # qcml.py and hermitian.py

from qcml import QCML

torch.set_num_threads(1)      # the matrices are tiny, so extra threads cost more than they save

PLOT_DIR = "qcml102426_plots" # every plot is saved here as its own PNG
DATA_DIR = "dudas_data"       # their saved results, exported by dudas_export.jl
RUN_CHECKS = True             # Part A's slow checks against the full quantum simulation (~2 min)


# THE RESERVOIR (as in main_fig7.jl)  [note 1]

W_A, W_B, COUPLING = 10e9, 9e9, 7e8
KAPPA_A, KAPPA_B = 17e6, 21e6
INTERVAL, RESOLUTION = 100e-9, 300     # each input lasts 100 ns, split into 300 time steps


def reservoir(inputs, drive, offset=0.0, max_photons=3, samples=10, g=COUPLING,
              kappa_a=KAPPA_A, kappa_b=KAPPA_B):
    """Measurement record of the reservoir driven by `inputs`, starting empty: one row per
    input, holding P(n_a, n_b) for n_a, n_b = 0..max_photons at each of `samples` times."""
    M = np.array([[-(1j * W_A + kappa_a / 2), -1j * g],
                  [-1j * g, -(1j * W_B + kappa_b / 2)]])
    step = INTERVAL / RESOLUTION
    # As in their code: the k-th sample is taken after (k * RESOLUTION/samples - 1) steps,
    # and the next input starts after RESOLUTION - 1 steps.
    to_sample = [expm(M * (k * (RESOLUTION // samples) - 1) * step) for k in range(1, samples + 1)]
    to_next = expm(M * (RESOLUTION - 1) * step)
    n = np.arange(max_photons + 1)
    n_factorial = np.array([factorial(int(i)) for i in n], dtype=float)
    poisson = lambda amp: np.exp(-abs(amp) ** 2) * abs(amp) ** (2 * n) / n_factorial
    amplitudes = np.zeros(2, complex)                      # empty reservoir
    record = np.zeros((len(inputs), samples * (max_photons + 1) ** 2))
    for i, x in enumerate(inputs):
        push = -drive * (x + offset) * np.sqrt(2 * np.array([kappa_a, kappa_b]))
        rest = -np.linalg.solve(M, push)                   # where the amplitudes settle
        row = []
        for to in to_sample:
            alpha, beta = to @ (amplitudes - rest) + rest
            row.append(np.outer(poisson(alpha), poisson(beta)).ravel())   # n_a outer, n_b inner
        record[i] = np.concatenate(row)
        amplitudes = to_next @ (amplitudes - rest) + rest
    return record


def master_equation_reservoir(inputs, drive, offset=0.0, max_photons=3, samples=10, levels=9):
    """The same record from the full quantum master equation, as their code computes it
    (photon numbers 0..levels-1 in each oscillator). Slow: a few seconds per input."""
    lower = sp.diags(np.sqrt(np.arange(1, levels)), 1, format="csr")
    one = sp.identity(levels, format="csr")
    a, b = sp.kron(lower, one, format="csr"), sp.kron(one, lower, format="csr")
    dim = levels ** 2
    big_one = sp.identity(dim, format="csr")
    H0 = W_A * a.T @ a + W_B * b.T @ b + COUPLING * (a.T @ b + a @ b.T)
    loss = sum(rate * (sp.kron(J, J) - 0.5 * sp.kron(big_one, J.T @ J) - 0.5 * sp.kron((J.T @ J).T, big_one))
               for rate, J in [(KAPPA_A, a), (KAPPA_B, b)])
    rho = np.zeros(dim * dim, complex)
    rho[0] = 1                                             # empty reservoir
    step, hop = INTERVAL / RESOLUTION, RESOLUTION // samples
    record = []
    for x in inputs:
        u = drive * (x + offset)
        H = H0 + 1j * u * (np.sqrt(2 * KAPPA_A) * (a - a.T) + np.sqrt(2 * KAPPA_B) * (b - b.T))
        L = (-1j * (sp.kron(big_one, H) - sp.kron(H.T, big_one)) + loss).tocsc()
        row, done = [], 0.0
        for k in range(1, samples + 1):
            t = (k * hop - 1) * step
            rho = expm_multiply(L * (t - done), rho)
            done = t
            P = np.real(rho.reshape(dim, dim, order="F").diagonal()).reshape(levels, levels)
            row.append((P / P.sum())[:max_photons + 1, :max_photons + 1].ravel())
        if (RESOLUTION - 1) * step > done:
            rho = expm_multiply(L * ((RESOLUTION - 1) * step - done), rho)
        record.append(np.concatenate(row))
    return np.array(record)


# THE MACKEY-GLASS INPUT (as in main_fig7.jl)  [note 2]

MG_DRIVE = 5e5                # drive per unit of input in their Mackey-Glass runs
WASHOUT = 30                  # as in their scoring, skip the first inputs of each run while
                              # the reservoir fills up


def mackey_glass_dudas(length=2204, tau=17, beta=0.2, gamma=0.1, power=10, dt=0.002):
    lag, steps = round(tau / dt), round((length - 1) / dt)
    x = np.ones(lag + steps + 1)                       # x = 1 before t = 0
    x[lag] = 0.0                                       # x(0) = 0
    rate = lambda now, past: beta * past / (1 + past ** power) - gamma * now
    for i in range(lag, lag + steps):
        past0, past1 = x[i - lag], x[i - lag + 1]
        past_half = (past0 + past1) / 2
        k1 = rate(x[i], past0)
        k2 = rate(x[i] + dt / 2 * k1, past_half)
        k3 = rate(x[i] + dt / 2 * k2, past_half)
        k4 = rate(x[i] + dt * k3, past1)
        x[i + 1] = x[i] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x[lag::round(1 / dt)]                       # x(0), x(1), ...


def nrmse(prediction, target):
    """Their error measure: root-mean-square error divided by the target's range
    (max - min). 0 is perfect; their Fig. 7 plots its log10."""
    return np.sqrt(((prediction - target) ** 2).mean()) / (target.max() - target.min())


def linear_readout(train_record, train_target, record):
    """Their readout: weights from the Moore-Penrose pseudo-inverse of the training
    record (no constant term), applied to `record`. The cutoff for tiny singular values
    matches Julia's pinv."""
    cutoff = min(train_record.shape) * np.finfo(float).eps
    return record @ (np.linalg.pinv(train_record, rtol=cutoff) @ train_target)


# PART A: rebuild the data and check it against theirs

print("PART A: rebuilding the reservoir data and checking it against Dudas et al.")
load = lambda name: np.loadtxt(os.path.join(DATA_DIR, name + ".csv"), delimiter=",")

# A1 their saved Fig. 3 measurements: the sine/square task (each 8-input block is one
# period of a square or a sine wave; their saved labels Y say which), drive 1.2e6 with
# offset 0.6, 36 neurons (n_a, n_b = 0..5), one sample per input.
their_record = load("fig3_X")
square = np.array([1, 1, 1, 1, -1, -1, -1, -1]) / 2
sine = np.array([0, 0.7071, 1, 0.7071, 0, -0.7071, -1, -0.7071]) / 2
sinesquare = np.concatenate([sine if label == 1 else square for label in load("fig3_Y")[::8]])
ours = reservoir(sinesquare, 1.2e6, offset=0.6, max_photons=5, samples=1)
print(f"   Fig. 3 data, coherent-state shortcut, all {len(ours)} inputs: largest difference "
      f"{np.abs(ours - their_record).max():.1e} (their 8-photon cutoff matters at this "
      "stronger drive)")
if RUN_CHECKS:
    start = time.time()
    exact = master_equation_reservoir(sinesquare[:8], 1.2e6, offset=0.6, max_photons=5, samples=1)
    print(f"   Fig. 3 data, full master equation, first 8 inputs: largest difference "
          f"{np.abs(exact - their_record[:8]).max():.1e}  ({time.time() - start:.0f} s)")

# A2 the shortcut against the full master equation at the Mackey-Glass drive
series = mackey_glass_dudas()
if RUN_CHECKS:
    start = time.time()
    piece = series[300:310]
    gap = np.abs(master_equation_reservoir(piece, MG_DRIVE) - reservoir(piece, MG_DRIVE)).max()
    print(f"   Mackey-Glass drive, shortcut vs full master equation, 10 inputs: largest "
          f"difference {gap:.1e}  ({time.time() - start:.0f} s)")

# A3 the measurement records of the Mackey-Glass task
TRAIN_INPUTS = np.arange(1000)                 # x(0..999)
TEST_INPUTS = np.arange(1100, 2100)            # x(1100..2099)
record_train = reservoir(series[TRAIN_INPUTS], MG_DRIVE)
record_test = reservoir(series[TEST_INPUTS], MG_DRIVE)
print(f"   Mackey-Glass records: {record_train.shape[0]} training and {record_test.shape[0]} "
      f"test inputs, {record_train.shape[1]} numbers each")

# A4 their Fig. 7: error against delay with their readout, for three reservoirs.  [note 3]
DELAYS = np.arange(1, 101)
FIG7 = [("g = 700 MHz, κ = 17, 21 MHz", "fig7_coupling_error", COUPLING, KAPPA_A, KAPPA_B),
        ("g = 0", "fig7_no_coupling_error", 0.0, KAPPA_A, KAPPA_B),
        ("κ = 170, 210 MHz", "fig7_κ10_error", COUPLING, 10 * KAPPA_A, 10 * KAPPA_B)]
fig7_curves = []
for label, file, g, kappa_a, kappa_b in FIG7:
    train = reservoir(series[TRAIN_INPUTS], MG_DRIVE, g=g, kappa_a=kappa_a, kappa_b=kappa_b)
    test = reservoir(series[TEST_INPUTS], MG_DRIVE, g=g, kappa_a=kappa_a, kappa_b=kappa_b)
    ours = np.array([nrmse(linear_readout(train, series[TRAIN_INPUTS + d], test)[WASHOUT - 1:],
                           series[TEST_INPUTS + d][WASHOUT - 1:]) for d in DELAYS])
    theirs = load(file)
    fig7_curves.append((label, theirs, ours))
    print(f"   Fig. 7, {label:<28} log10 error at delay 1, 20, 100:  theirs "
          + " ".join(f"{np.log10(theirs[d - 1]):+.2f}" for d in (1, 20, 100)) + "   ours "
          + " ".join(f"{np.log10(ours[d - 1]):+.2f}" for d in (1, 20, 100))
          + f"   mean gap {np.abs(np.log10(ours / theirs)).mean():.2f}")


# THE METHODS (QCML, supervised QCML, PCA and the diffusion map as in QCML102326_2)  [note 4]

EPOCHS = 150
BATCH = 64


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


# SUPERVISED QCML (Di Caro et al., 2025)  [note 5]

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


# QCML TRAINED TO FILL IN (the same idea applied to missing measurements)  [note 6]

def train_filler(points, measured, n, seed=0, epochs=EPOCHS, polish=True):
    """QCML trained to rebuild every column of `points` from the `measured` columns."""
    X = torch.tensor(points, dtype=torch.float32)
    torch.manual_seed(seed)
    model = QCML(D=points.shape[1], n=n)
    measured = list(measured)

    def loss_on(rows):
        psi = ground_state_from(model, X[rows][:, measured], measured)
        return ((model.bloch(psi) - X[rows]) ** 2).sum(1).mean()

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


def qcml_fill(model, values, measured):
    """Every column, as the expected values of the ground state built from the measured
    columns' matrices only."""
    with torch.no_grad():
        return model.bloch(ground_state_from(model, values, list(measured))).numpy()


# PCA (as in QCML102326 Part B)  [note 7]

def squared_distances(A, B):
    return np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T, 0)


class DiffusionMap:
    def __init__(self, points, neighbours):
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

    def rebuild_errors(self, values, points, truth, max_m):
        """Mean squared error of rebuild(values, points, m) against `truth`, for every
        m = 1..max_m, without storing all the rebuilds."""
        J = max_m + 1
        at_points = self.steps(points) @ self.right[:, :J] / self.eigenvalues[:J]
        shares = self.left[:, :J].T @ values
        residual = np.outer(at_points[:, 0], shares[0]) - truth
        errors = np.empty(max_m)
        for j in range(1, J):
            residual += np.outer(at_points[:, j], shares[j])
            errors[j - 1] = (residual ** 2).mean()
        return errors


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



# PARTS B AND C: WEAKER MEASUREMENTS  [note 8]

def neuron(n_a, n_b):
    return n_a * 4 + n_b                       # position of P(n_a, n_b) within a sample


BUDGETS = [("all 16 neurons", [neuron(i, j) for i in range(4) for j in range(4)]),
           ("4 neurons (0 or 1 photon)", [neuron(0, 0), neuron(0, 1), neuron(1, 0), neuron(1, 1)]),
           ("vacuum only", [neuron(0, 0)])]
BUDGETS = [(label, np.array([k * 16 + c for k in range(10) for c in neurons]))
           for label, neurons in BUDGETS]
SHOTS = [("exact", None), ("9^9 shots", 9 ** 9), ("9^5 shots", 9 ** 5)]
NOISE_DRAWS = 3
PREDICT_DELAYS = [1, 20]      # their Fig. 7 shows delays 1-100 and illustrates 20

FILL_N, FILL_C2 = 16, 0.01    # plain QCML (best of n = 16, 24, 32 and w = 0.01, 0.001, 0.0001)
FILLER_N = 16                 # QCML trained to fill in (best of n = 8, 16). L-BFGS polishing
                              # only on exact records: on noisy ones it fits the noise
                              # (pilot, vacuum only at 9^5 shots: error 0.118 with, 0.092 without)
SUP_N = 8                     # supervised QCML (best of n = 8, 16); polishing helps here even
                              # with noise (pilot: 0.0064 with, 0.0089 without)
DM_NEIGHBOUR_OPTIONS = (30, 100, 300)
DM_MAX_M = 600

TRAIN_ROWS = np.arange(WASHOUT, 1000)          # training inputs used (after the washout)
FIT_ROWS, VAL_ROWS = np.arange(WASHOUT, 700), np.arange(700, 1000)
TEST_ROWS = np.arange(WASHOUT - 1, 1000)       # their scoring starts at the 30th test input
EXAMPLE = (149, np.arange(200, 260))           # plot 03: P(1,1) at the 10th sample, 60 inputs


def measure(record, shots, rng):
    """A finite number of shots: each probability p gets Gaussian noise of size
    sqrt(p (1 - p) / shots), as in their Fig. 5."""
    if shots is None:
        return record
    return record + rng.standard_normal(record.shape) * np.sqrt(np.abs(record * (1 - record) / shots))


rms = lambda a: float(np.sqrt(np.mean(a ** 2)))


def best_pca(train, values, val_truth, measured, target_columns, ks):
    """The k whose fill-in of `target_columns` from the `measured` ones is closest on the
    validation block."""
    filled = pca_predict(train, values, list(measured), ks)
    return int(ks[np.argmin([rms(f[:, target_columns] - val_truth) for f in filled])])


def best_diffusion_map(fit_inputs, fit_values, val_inputs, val_truth):
    """(neighbours, m) whose rebuild is closest on the validation block."""
    best = (np.inf, None, None)
    for neighbours in DM_NEIGHBOUR_OPTIONS:
        errors = DiffusionMap(fit_inputs, neighbours).rebuild_errors(fit_values, val_inputs,
                                                                     val_truth, DM_MAX_M)
        if errors.min() < best[0]:
            best = (errors.min(), neighbours, int(errors.argmin()) + 1)
    return best[1], best[2]


FILL_METHODS = ["PCA", "Diffusion map", "QCML", "QCML trained to fill in"]
PREDICT_METHODS = ["Linear readout (theirs)", "PCA", "Diffusion map", "Supervised QCML"]
spread = record_train[TRAIN_ROWS].std(0)       # each number's spread, for fill-in errors
fill = {}                                      # (shots, budget) -> {method: [error per draw]}
predict = {}                                   # (shots, budget, delay) -> {method: [NRMSE per draw]}
example = {}

print(f"\nPARTS B AND C: {len(BUDGETS)} measurement budgets x {len(SHOTS)} noise levels x "
      f"{NOISE_DRAWS} draws")
for shots_label, shots in SHOTS:
    for draw in range(NOISE_DRAWS):
        start = time.time()
        rng = np.random.default_rng(draw)
        train, test = measure(record_train, shots, rng), measure(record_test, shots, rng)
        mu, sd = train[TRAIN_ROWS].mean(0), train[TRAIN_ROWS].std(0)
        Z, Z_test = (train - mu) / sd, (test - mu) / sd
        plain_qcml = train_qcml(Z[TRAIN_ROWS], FILL_N, FILL_C2, seed=draw)
        notes = []
        for budget_label, measured in BUDGETS:
            hidden = np.setdiff1d(np.arange(160), measured)
            known_test = Z_test[TEST_ROWS][:, measured]

            # PART B: fill in the unmeasured numbers.  [note 9]
            if len(hidden):
                truth = record_test[TEST_ROWS][:, hidden]
                fill_error = lambda z: rms(z[:, hidden] * sd[hidden] + mu[hidden] - truth) / rms(spread[hidden])
                k = best_pca(Z[FIT_ROWS], Z[VAL_ROWS][:, measured], Z[VAL_ROWS][:, hidden],
                             measured, hidden, np.arange(1, len(measured) + 1))
                neighbours, m = best_diffusion_map(Z[FIT_ROWS][:, measured], Z[FIT_ROWS][:, hidden],
                                                   Z[VAL_ROWS][:, measured], Z[VAL_ROWS][:, hidden])
                filler = train_filler(Z[TRAIN_ROWS], measured, FILLER_N, seed=draw, polish=shots is None)
                filled = {
                    "PCA": pca_predict(Z[TRAIN_ROWS], known_test, list(measured), [k])[0],
                    "Diffusion map": DiffusionMap(Z[TRAIN_ROWS][:, measured], neighbours).rebuild(
                        Z[TRAIN_ROWS], known_test, np.array([m]))[0],
                    "QCML": qcml_fill(plain_qcml, known_test, measured),
                    "QCML trained to fill in": qcml_fill(filler, known_test, measured),
                }
                scores = fill.setdefault((shots_label, budget_label), {name: [] for name in FILL_METHODS})
                for name, z in filled.items():
                    scores[name].append(fill_error(z))
                notes.append(f"fill {len(measured)}: PCA k={k}, DM {neighbours}/{m}")
                if shots is None and draw == 0 and len(measured) == 10:
                    column, inputs = EXAMPLE
                    rows = np.searchsorted(TEST_ROWS, inputs)
                    example = {name: z[rows, column] * sd[column] + mu[column] for name, z in filled.items()}
                    example["exact"] = record_test[inputs, column]

            # PART C: predict x(t + delay) from the measured numbers only
            for delay in PREDICT_DELAYS:
                y, y_test = series[TRAIN_INPUTS + delay], series[TEST_INPUTS + delay][TEST_ROWS]
                y_mu, y_sd = y[TRAIN_ROWS].mean(), y[TRAIN_ROWS].std()
                Y = (y - y_mu) / y_sd
                score = lambda z: nrmse(z * y_sd + y_mu, y_test)
                joint = np.column_stack([Z, Y])
                k = best_pca(joint[FIT_ROWS], joint[VAL_ROWS][:, measured], Y[VAL_ROWS, None],
                             measured, [160], np.arange(1, len(measured) + 2))
                neighbours, m = best_diffusion_map(Z[FIT_ROWS][:, measured], Y[FIT_ROWS, None],
                                                   Z[VAL_ROWS][:, measured], Y[VAL_ROWS, None])
                supervised = train_supervised(Z[TRAIN_ROWS][:, measured], Y[TRAIN_ROWS], SUP_N,
                                              torch.nn.functional.mse_loss, seed=draw)
                results = {
                    "Linear readout (theirs)": nrmse(linear_readout(
                        train[TRAIN_ROWS][:, measured], y[TRAIN_ROWS], test[TEST_ROWS][:, measured]), y_test),
                    "PCA": score(pca_predict(joint[TRAIN_ROWS], known_test, list(measured), [k])[0][:, 160]),
                    "Diffusion map": score(DiffusionMap(Z[TRAIN_ROWS][:, measured], neighbours).rebuild(
                        Y[TRAIN_ROWS, None], known_test, np.array([m]))[0][:, 0]),
                    "Supervised QCML": score(supervised_forecast(supervised, known_test)),
                }
                scores = predict.setdefault((shots_label, budget_label, delay),
                                            {name: [] for name in PREDICT_METHODS})
                for name, value in results.items():
                    scores[name].append(value)
                notes.append(f"delay {delay}, {len(measured)}: PCA k={k}, DM {neighbours}/{m}")
        print(f"   {shots_label}, draw {draw + 1}: done in {time.time() - start:.0f} s   "
              + ";  ".join(notes))

fill = {key: {name: np.array(v) for name, v in scores.items()} for key, scores in fill.items()}
predict = {key: {name: np.array(v) for name, v in scores.items()} for key, scores in predict.items()}

print("\nPART B: filling in the unmeasured numbers (error as a fraction of their typical spread, "
      "mean ± std over draws)")
fill_rows = []
for shots_label, _ in SHOTS:
    for budget_label, measured in BUDGETS:
        if (shots_label, budget_label) not in fill:
            continue
        cells = [fill[(shots_label, budget_label)][name] for name in FILL_METHODS]
        label = f"{budget_label}: {len(measured)} of 160 measured\n{shots_label}"
        fill_rows.append((label, cells))
        print(f"   {label.replace(chr(10), ', '):<58}  " + "   ".join(
            f"{name} {c.mean():.4f} ± {c.std():.4f}" for name, c in zip(FILL_METHODS, cells)))

print("\nPART C: predicting x(t + delay) from the measured numbers (NRMSE, mean ± std over draws)")
predict_rows = {delay: [] for delay in PREDICT_DELAYS}
for delay in PREDICT_DELAYS:
    for shots_label, _ in SHOTS:
        for budget_label, measured in BUDGETS:
            scores = predict[(shots_label, budget_label, delay)]
            cells = [scores[name] for name in PREDICT_METHODS]
            label = f"{budget_label}: {len(measured)} measured\n{shots_label}"
            predict_rows[delay].append((label, cells))
            gap = scores["Supervised QCML"] - scores["Diffusion map"]
            print(f"   delay {delay:>2}, {label.replace(chr(10), ', '):<50}  " + "   ".join(
                f"{name.split(' (')[0]} {c.mean():.4f}" for name, c in zip(PREDICT_METHODS, cells))
                + f"   supervised QCML - DM {gap.mean():+.4f} ({(gap < 0).sum()} of {len(gap)} draws better)")

with open("qcml102426_fill_in_table.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["measurement"] + [f"{name} {stat}" for name in FILL_METHODS for stat in ("mean", "std")])
    for label, cells in fill_rows:
        writer.writerow([label.replace("\n", ", ")] + [v for c in cells for v in (c.mean(), c.std())])
with open("qcml102426_prediction_table.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["delay", "measurement"] + [f"{name} {stat}" for name in PREDICT_METHODS
                                                for stat in ("mean", "std")])
    for delay in PREDICT_DELAYS:
        for label, cells in predict_rows[delay]:
            writer.writerow([delay, label.replace("\n", ", ")] + [v for c in cells for v in (c.mean(), c.std())])
print("saved to qcml102426_fill_in_table.csv and qcml102426_prediction_table.csv")


# Plots: one PNG per plot, numbered in reading order, saved in PLOT_DIR

os.makedirs(PLOT_DIR, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "figure.titlesize": 13,
                     "figure.titleweight": "bold", "legend.fontsize": 9,
                     "legend.title_fontsize": 9, "legend.frameon": False,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.grid": True, "axes.axisbelow": True,
                     "grid.color": "#e1e0d9", "grid.linewidth": 0.6})
QCML_COLOUR, PCA_COLOUR, DM_COLOUR = "#eb6834", "#2a78d6", "#1baf7a"
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


METHOD_STYLE = {"Linear readout (theirs)": (GREY, "D"), "PCA": (PCA_COLOUR, "o"),
                "Diffusion map": (DM_COLOUR, "s"), "Supervised QCML": (QCML_COLOUR, "^"),
                "QCML": (QCML_COLOUR, "v"), "QCML trained to fill in": (QCML_COLOUR, "^")}
print(f"\nplots saved to {PLOT_DIR}:")

# 01: their Fig. 7 against this simulation
fig, ax = new_plot("Mackey–Glass with their readout: Dudas et al. vs this simulation",
                   "their Fig. 7 (solid, their saved results) and the rebuilt data (dashed); "
                   "all 16 neurons, 10 samples, exact")
for (label, theirs, ours), colour in zip(fig7_curves, [QCML_COLOUR, PCA_COLOUR, DM_COLOUR]):
    ax.plot(DELAYS, np.log10(theirs), color=colour, lw=2, label=f"{label}, Dudas et al.")
    ax.plot(DELAYS, np.log10(ours), color=colour, lw=2, ls="--", label=f"{label}, rebuilt")
ax.set_xlabel("delay (how many inputs ahead x is predicted)")
ax.set_ylabel("log10 NRMSE")
save_plot(fig, ax, "01_fig7_reproduction")

# 02: what the reservoir record looks like
inputs = np.arange(200, 230)
fig, ax = new_plot("The reservoir record: input and the vacuum probability",
                   "test run, 30 inputs (100 ns each); P(0,0) is read 10 times per input",
                   size=(10, 6))
times = (inputs[:, None] + (np.arange(1, 11) * 30 - 1)[None, :] / RESOLUTION).ravel() * 0.1
ax.plot(times, record_test[inputs][:, :160:16].ravel(), color=PCA_COLOUR, lw=1.5, marker="o",
        ms=3, label="P(0,0): probability the reservoir is empty (measured)")
ax.set_xlabel("time (μs)")
ax.set_ylabel("P(0,0)")
right = ax.twinx()
right.step(np.append(inputs, inputs[-1] + 1) * 0.1, np.append(series[TEST_INPUTS][inputs],
           series[TEST_INPUTS][inputs[-1]]), where="post", color=INK, lw=1.5,
           label="input x(t), held 100 ns")
right.set_ylabel("input x(t)")
right.grid(False)
handles = ax.get_legend_handles_labels()[0] + right.get_legend_handles_labels()[0]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
save_figure(fig, "02_reservoir_record")

# 03: one unmeasured number filled in
column, inputs = EXAMPLE
fig, ax = new_plot("Filling in an unmeasured number: P(1,1) from the vacuum probability alone",
                   f"test run, exact records; only P(0,0) measured (10 of 160 numbers); "
                   f"P(1,1) at the 10th sample", size=(10, 6))
ax.plot(inputs, example["exact"], color=INK, lw=3, alpha=0.35, label="exact (not measured)")
for name in ["PCA", "Diffusion map", "QCML trained to fill in"]:
    colour, marker = METHOD_STYLE[name]
    ax.plot(inputs, example[name], color=colour, lw=1.5, marker=marker, ms=3, label=name)
ax.set_xlabel("test input")
ax.set_ylabel("P(1,1)")
save_plot(fig, ax, "03_fill_in_example")

# 04: the fill-in table
table_plot("04_fill_in_table", "Filling in the unmeasured numbers",
           f"test run; error = RMS distance from the exact probabilities / their typical "
           f"spread; mean ± std over {NOISE_DRAWS} draws", ["Measured"] + [
               name.replace(" trained", "\ntrained") for name in FILL_METHODS], fill_rows,
           label_width=0.3, size=12, digits=4,
           note="Settings: PCA's k and the diffusion map's neighbours and m chosen on a "
                "validation block of the training run. QCML: one model of all 160 numbers, "
                "filling in from a ground state of the measured numbers' matrices. QCML "
                "trained to fill in: trained for this measurement budget.")

# 05-06: prediction error against the number of measured numbers, delay 20
counts = [len(measured) for _, measured in BUDGETS]
for number, (shots_label, _) in [("05", SHOTS[0]), ("06", SHOTS[-1])]:
    fig, ax = new_plot(f"Predicting x(t + 20) from fewer measurements ({shots_label})",
                       f"test run, NRMSE (log scale), mean ± std over {NOISE_DRAWS} draws")
    for name in PREDICT_METHODS:
        colour, marker = METHOD_STYLE[name]
        values = [predict[(shots_label, label, 20)][name] for label, _ in BUDGETS]
        ax.errorbar(counts, [v.mean() for v in values], yerr=[v.std() for v in values],
                    color=colour, marker=marker, lw=2, ms=6, capsize=3, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(counts, [f"{c}\n{label}" for c, (label, _) in zip(counts, BUDGETS)])
    ax.set_xlabel("numbers measured per input")
    ax.set_ylabel("NRMSE")
    save_plot(fig, ax, f"{number}_prediction_{'exact' if shots_label == 'exact' else 'noisy'}")

# 07-08: the prediction tables
for number, delay in [("07", 1), ("08", 20)]:
    table_plot(f"{number}_prediction_table_delay_{delay}",
               f"Predicting x(t + {delay}) from the measured numbers",
               f"test run, NRMSE (their error measure), mean ± std over {NOISE_DRAWS} draws",
               ["Measured"] + [name.replace(" (", "\n(") for name in PREDICT_METHODS],
               predict_rows[delay], label_width=0.3, size=12, digits=4,
               note="Linear readout: their pseudo-inverse readout on the measured numbers. "
                    "PCA and the diffusion map: settings chosen on a validation block of the "
                    "training run. Supervised QCML: n = 8, the target observable B predicts "
                    "x(t + delay).")

# 09: the diffusion map of the reservoir records
records = (record_train[TRAIN_ROWS] - record_train[TRAIN_ROWS].mean(0)) / spread
coords = DiffusionMap(records, 100).coordinates(records)
fig, ax = new_plot("Diffusion map of the reservoir records (training run, exact)",
                   "the first two diffusion coordinates of the 160-number records, coloured by "
                   "the input x(t)", size=(8, 7.5))
dots = ax.scatter(coords[:, 0], coords[:, 1], c=series[TRAIN_INPUTS][TRAIN_ROWS], cmap=SEQUENTIAL,
                  s=10, lw=0)
fig.colorbar(dots, ax=ax, label="x(t)", shrink=0.8)
ax.set_xlabel("diffusion coordinate 1")
ax.set_ylabel("diffusion coordinate 2")
save_figure(fig, "09_reservoir_diffusion_map")


# ========================================================================================
# EXPLANATION
# Notes on the code above, in the order of its sections. A tag [note k] in the code
# marks the section that note k explains.
# ========================================================================================
#
# [note 0] About this script
#
# Dudas et al. send a signal into two coupled quantum oscillators (the "reservoir") and
# read it out by measuring the probability of each photon-number state. A linear readout
# trained on those measurements predicts the signal. Their Mackey-Glass task (their
# Fig. 7): from the measurements taken while x(t) is sent in, predict x(t + delay).
#
# The problem this file looks at: measuring a quantum state destroys it, so every
# measurement means replaying the input. A model that can fill in the measurements that
# were NOT taken would let the reservoir be measured less (weaker measurements) without
# losing information. Here the reservoir measurements are treated as data, and QCML,
# PCA and a diffusion map each learn their shape.
#
# PART A  Rebuild the data: simulate the reservoir exactly as their code does and check
#         it against their saved results (their Fig. 3 measurements and Fig. 7 errors).
# PART B  Missing measurements: measure only some of the 160 numbers per input and fill
#         in the rest with QCML, PCA and a diffusion map.
# PART C  Predict the Mackey-Glass signal from the measured numbers only.
#
# Runs in about 20 minutes. No Julia needed: dudas_export.jl already copied their saved
# results into dudas_data/ as CSV files.
#
# [note 1] THE RESERVOIR (as in main_fig7.jl)
#
# Two oscillators, a and b (think of two microwave resonators), each holding photons:
#   frequencies   w_a = 10e9, w_b = 9e9 (per second), coupled with strength g = 700e6, so
#                 photons hop between them
#   losses        each leaks photons out at rate kappa_a = 17e6, kappa_b = 21e6, so the
#                 reservoir slowly forgets old inputs (its "fading memory")
#   input         each value x(t) of the signal sets how hard both oscillators are driven,
#                 eps = 5e5 * x(t), and is held for 100 ns before the next value
#   measurement   the probability P(n_a, n_b) of finding n_a photons in a and n_b in b, for
#                 n_a, n_b = 0..3: 16 "neurons". They are read 10 times per input (every
#                 10 ns), so each input gives a record of 160 numbers.
# The paper's readout is a weighted sum of the 160 numbers, with the weights found by
# least squares (a pseudo-inverse) on 1000 training inputs, then tested on 1000 others.
#
# How it is simulated. Their code solves the full quantum master equation with up to 8
# photons per oscillator (81 basis states). Here a shortcut is exact: the oscillators'
# energy is quadratic, the drive is linear and the only loss is photons leaking out, so a
# reservoir that starts empty stays in a "coherent state" (the quantum version of a
# classical oscillation). Two complex amplitudes alpha and beta then describe it fully:
#   d alpha/dt = -(i w_a + kappa_a/2) alpha - i g beta - sqrt(2 kappa_a) eps_a
#   d beta/dt  = -(i w_b + kappa_b/2) beta - i g alpha - sqrt(2 kappa_b) eps_b
# a linear equation solved exactly over each 100 ns step. The photon counts are then
# Poisson distributed:
#   P(n_a, n_b) = exp(-|alpha|^2) |alpha|^(2 n_a) / n_a!  *  exp(-|beta|^2) |beta|^(2 n_b) / n_b!
# Part A checks this against the full master equation and against their saved results.
#
# [note 2] THE MACKEY-GLASS INPUT (as in main_fig7.jl)
#
# dx/dt = 0.2 x(t - 17) / (1 + x(t - 17)^10) - 0.1 x(t), started from x = 1 before t = 0
# and x(0) = 0, and read once per time unit. The reservoir is trained on inputs x(0..999)
# and tested on x(1100..2099), each run starting from an empty reservoir; the target for
# input x(t) is x(t + delay). Their solver's exact numbers were not saved, and the series
# is chaotic, so small integration differences grow: this series follows the same
# equation but not digit for digit theirs (step 0.002 here; with step 0.01 the test half
# differs by up to 0.4). Its statistics, and so the task, are the same.
#
# [note 3]
# A4 their Fig. 7: error against delay with their readout, for three reservoirs. The
# curves differ a little from theirs because the chaotic series is not digit for digit
# theirs (see THE MACKEY-GLASS INPUT). The kappa = 170, 210 MHz reservoir differs more at
# short delays: ten times the loss also means a sqrt(10) times stronger drive in their
# code, the reservoir then holds up to 3.3 photons, and their 8-photon cutoff starts to
# clip it. At the main settings it never holds more than 0.33 photons, so the cutoff
# plays no part there (A2).
#
# [note 4] THE METHODS (QCML, supervised QCML, PCA and the diffusion map as in QCML102326_2)
#
# Every method learns from the full training records (all 160 numbers measured on the
# 1000 training inputs: a one-off calibration, where replaying the input is affordable)
# and is then given only the numbers measured on each test input.
#
# [note 5] SUPERVISED QCML (Di Caro et al., 2025)
#
# Unsupervised QCML learns the shape of the data and is then asked a question it was not
# trained for (which manifold is nearer, or what the missing feature is). Di Caro et al.
# train it on the question itself. Next to the K input matrices H_1..H_K there is one
# more learned Hermitian matrix B, the "target observable". A sample x is turned into a
# ground state psi(x) using the input matrices only (as in ground_state_from), and the
# prediction is B's expected value in that state:
#     prediction(x) = <psi(x)| B |psi(x)>
# All K + 1 matrices are trained together on the squared prediction error. Here the
# inputs are the measured numbers of the reservoir record and the target is x(t + delay).
#
# [note 6] QCML TRAINED TO FILL IN (the same idea applied to missing measurements)
#
# Plain QCML learns all 160 numbers together, then fills in the unmeasured ones from a
# ground state built from the measured numbers' matrices alone (ground_state_from), a
# job it was never trained for. Since we choose which numbers to measure, QCML can be
# trained for exactly that job: the ground state is built from the measured numbers'
# matrices only, and the loss asks every matrix's expected value to match the full
# record, so the unmeasured numbers' matrices learn to predict them. It is supervised
# QCML with one target observable per unmeasured number.
#
# [note 7]
# PCA (as in QCML102326 Part B): the point on the flat k-dimensional subspace of the
# training records that best matches the measured numbers (least squares); its other
# coordinates are the filled-in numbers or the prediction.
#
# DIFFUSION MAP (as in QCML102326): built on the measured numbers of the training
# records, with each point's kernel width its distance to its k-th nearest neighbour.
# Anything known on the training records (the unmeasured numbers, or the target) is
# rebuilt at a new record from the walk's first m eigenvectors (geometric harmonics).
#
# [note 8] PARTS B AND C: WEAKER MEASUREMENTS
#
# Measurement budgets. Each budget reads some of the 16 neurons at all 10 samples:
#   all 16 neurons                   160 numbers per input (their setting)
#   4 neurons, 0 or 1 photon each     40 numbers: P(0,0), P(0,1), P(1,0), P(1,1)
#   vacuum only                       10 numbers: P(0,0), "is the reservoir empty?"
# Measurement noise. A probability is estimated by repeating the experiment "shots"
# times, so it carries noise of size sqrt(p (1 - p) / shots), exactly as their Fig. 5
# adds it (main_fig5.jl), to the training and the test records alike. Fewer shots means
# a weaker, noisier measurement. Levels: exact (their Fig. 7), 9^9 and 9^5 shots (two of
# the three in their Fig. 5; at 9^5 the noise is a quarter of a typical number's spread).
# Each noisy level is drawn NOISE_DRAWS times (different noise, different QCML start),
# and results are mean ± std over the draws.
#
# Choosing settings fairly. PCA's k and the diffusion map's neighbours and m are chosen
# on a validation block inside the training run (fit on inputs 30-699, score on 700-999),
# then refitted on all training inputs and scored on the test run. QCML's settings were
# chosen in pilot runs (listed below). Every score is on the test run (inputs from the
# 30th on, as in their scoring), never used for choosing.
#
# [note 9]
# PART B: fill in the unmeasured numbers. Error = RMS distance of the filled-in
# probabilities from the exact (noise-free) ones, divided by the typical spread
# of the unmeasured probabilities (the RMS of their standard deviations over the
# training run). 0 is perfect; 1 is as far off as the numbers typically vary.
# (Dividing each number by its own spread instead lets the tiny high-photon
# probabilities, around 1e-9 and swamped by shot noise, dominate the average.)
