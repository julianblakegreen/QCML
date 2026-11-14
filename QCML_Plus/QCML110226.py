# Quantum Cognitive Machine Learning Project
# QCML110226: builds on QCML102426_5. The same comparison, with noise weights an experimenter
# could measure.
# Runs in about 10 minutes on 8 cores: the 42 jobs (7 settings x 3 draws x 2 delays) run
# side by side, one process each.
# What it does and how: see the notes at the end of this file.


import os
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")    # one thread per process, since the jobs run
                                                        # side by side (set before numpy loads)
import csv
import itertools
import textwrap
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import factorial

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import expm


# Repository set-up: run from this folder, with QCML from 1_Plain_QCML
import os
import sys
_FOLDER = os.path.dirname(os.path.abspath(__file__))
os.chdir(_FOLDER)                                                 # data and outputs live in this folder
sys.path.insert(0, os.path.join(_FOLDER, "..", "1_Plain_QCML"))   # qcml.py and hermitian.py

from qcml import QCML

torch.set_num_threads(1)      # the matrices are tiny, so extra threads cost more than they save

PLOT_DIR = os.path.join("QCML Plots", "qcml110226_plots")   # every plot is saved here as its own PNG
WORKERS = os.cpu_count()


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


# WEAK PHOTON-NUMBER MEASUREMENT  [note 2]

MOMENTS = [m for m in itertools.product(range(5), repeat=4) if sum(m) <= 4]   # (p, q, r, s)
MOMENT_INDEX = {m: i for i, m in enumerate(MOMENTS)}


def moment_generator(push_a, push_b, dephasing):
    """The matrix A of d<M>/dt = A <M> for M = a+^p a^q b+^r b^s, all p + q + r + s <= 4."""
    A = np.zeros((len(MOMENTS), len(MOMENTS)), complex)

    def add(row, value, moment):
        if value != 0 and min(moment) >= 0:
            A[row, MOMENT_INDEX[moment]] += value

    for row, (p, q, r, s) in enumerate(MOMENTS):
        add(row, 1j * (W_A * (p - q) + W_B * (r - s))                  # free rotation
            - KAPPA_A / 2 * (p + q) - KAPPA_B / 2 * (r + s)              # photon loss
            - dephasing / 2 * ((p - q) ** 2 + (r - s) ** 2), (p, q, r, s))   # dephasing
        add(row, 1j * COUPLING * r, (p + 1, q, r - 1, s))                # photons hopping
        add(row, -1j * COUPLING * q, (p, q - 1, r, s + 1))               # between a and b
        add(row, 1j * COUPLING * p, (p - 1, q, r + 1, s))
        add(row, -1j * COUPLING * s, (p, q + 1, r, s - 1))
        add(row, -push_a * p, (p - 1, q, r, s))                          # the drive
        add(row, -push_a * q, (p, q - 1, r, s))
        add(row, -push_b * r, (p, q, r - 1, s))
        add(row, -push_b * s, (p, q, r, s - 1))
    return A


def photon_numbers(inputs, drive, dephasing, offset=0.0, windows=10):
    """Window averages of <N_a>, <N_b>, <N_a^2>, <N_b^2> for each of `windows` equal windows
    of every input, with dephasing at rate `dephasing`: shape (inputs, windows, 4). With
    windows=None, the snapshots at their code's 10 sample times instead (for checks)."""
    n = len(MOMENTS)
    moments = np.zeros(n, complex)
    moments[MOMENT_INDEX[(0, 0, 0, 0)]] = 1                              # empty reservoir
    picks = [MOMENT_INDEX[m] for m in [(1, 1, 0, 0), (0, 0, 1, 1), (2, 2, 0, 0), (0, 0, 2, 2)]]
    as_numbers = lambda m: np.array([m[picks[0]], m[picks[1]], m[picks[2]] + m[picks[0]],
                                     m[picks[3]] + m[picks[1]]]).real
    step = INTERVAL / RESOLUTION
    span = (RESOLUTION - 1) * step                                       # as in their code
    out = []
    for x in inputs:
        u = drive * (x + offset)
        A = moment_generator(np.sqrt(2 * KAPPA_A) * u, np.sqrt(2 * KAPPA_B) * u, dephasing)
        row = []
        if windows is None:
            done = 0.0
            for k in range(1, 11):
                t = (k * 30 - 1) * step
                moments = expm(A * (t - done)) @ moments
                done = t
                row.append(as_numbers(moments))
            moments = expm(A * (span - done)) @ moments
        else:
            # one exponential of [[A, 0], [1, 0]] gives both the step and its integral
            tau = span / windows
            B = np.zeros((2 * n, 2 * n), complex)
            B[:n, :n], B[n:, :n] = A, np.eye(n)
            both = expm(B * tau)
            forward, integral = both[:n, :n], both[n:, :n]
            for _ in range(windows):
                row.append(as_numbers(integral @ moments) / tau)
                moments = forward @ moments
        out.append(row)
    return np.array(out)


def weak_record(numbers, gamma, runs, rng):
    """The measured record: window-mean photon numbers plus noise, averaged over `runs`
    runs. numbers from photon_numbers; returns (inputs, 20)."""
    mean = numbers[:, :, :2]
    spread = np.maximum(numbers[:, :, 2:] - mean ** 2, 0)
    tau = (RESOLUTION - 1) * INTERVAL / RESOLUTION / numbers.shape[1]
    noise = np.sqrt((spread + 1 / (4 * ETA * gamma * tau)) / runs)
    return (mean + rng.standard_normal(mean.shape) * noise).reshape(len(numbers), -1)


# THE MACKEY-GLASS INPUT (as in main_fig7.jl)  [note 3]

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



# THE METHODS (as in QCML102426: QCML, supervised QCML, QCML trained to fill in, PCA
# and the diffusion map)

EPOCHS = 150
BATCH = 64


# QCML

def ground_state_from(model, values, known):
    """Ground state built from the `known` features only: their matrices alone make the
    error Hamiltonian. This is how supervised QCML reads its inputs. Differentiable, so it can be trained through."""
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


# DIFFUSION MAP (as in QCML102326)  [note 5]

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


# SETTINGS

ETA = 0.5                                       # detector efficiency of the weak record
SUP_N = 8                                       # supervised QCML Hilbert-space dimension
HISTORY = {1: 4, 20: 6}                         # previous inputs whose records are added (as in
                                                # QCML102426_3), the same for every method
PREDICT_DELAYS = [1, 20]
NOISE_DRAWS = 3
SETTINGS = [("vacuum only, exact", "projective", 10, None),
            ("vacuum only, 9^5 shots", "projective", 10, 9 ** 5),
            ("40 numbers, exact", "projective", 40, None),
            ("40 numbers, 9^5 shots", "projective", 40, 9 ** 5),
            ("all 160, 9^5 shots", "projective", 160, 9 ** 5),
            ("weak record, 1e4 runs", "weak", 1e4, 1e9),
            ("weak record, 1e6 runs", "weak", 1e6, 1e9)]
NOISIEST = ["vacuum only, 9^5 shots", "40 numbers, 9^5 shots", "all 160, 9^5 shots",
            "weak record, 1e4 runs"]            # the abstract's "8 noisiest" cells: these at both delays
EXAMPLE = ("vacuum only, 9^5 shots", 20)        # plot 07: one test run with QCML+'s uncertainty
DM_NEIGHBOUR_OPTIONS = (30, 100, 300)
DM_MAX_M = 600
KERNEL_WIDTHS = (0.25, 0.5, 1, 2, 4)            # x the median distance between records
KERNEL_STRENGTHS = np.logspace(-8, 1, 19)
TRAIN_ROWS = np.arange(WASHOUT, 1000)
FIT_ROWS, VAL_ROWS = np.arange(WASHOUT, 700), np.arange(700, 1000)
TEST_ROWS = np.arange(WASHOUT - 1, 1000)
TRAIN_INPUTS, TEST_INPUTS = np.arange(1000), np.arange(1100, 2100)
mse = torch.nn.functional.mse_loss


def neuron(n_a, n_b):
    return n_a * 4 + n_b                        # position of P(n_a, n_b) within a sample


BUDGETS = {160: [neuron(i, j) for i in range(4) for j in range(4)],
           40: [neuron(0, 0), neuron(0, 1), neuron(1, 0), neuron(1, 1)],
           10: [neuron(0, 0)]}
BUDGETS = {count: np.array([k * 16 + c for k in range(10) for c in neurons])
           for count, neurons in BUDGETS.items()}


# INPUTS (as in QCML102426_3)

def projective(record, shots, rng):
    """Their measurement with a finite number of shots (their Fig. 5)."""
    if shots is None:
        return record
    return record + rng.standard_normal(record.shape) * np.sqrt(np.abs(record * (1 - record) / shots))


def with_history(record, previous):
    """Each row followed by the rows of the `previous` inputs before it."""
    return np.hstack([np.vstack([np.repeat(record[:1], k, 0), record[:len(record) - k]])
                      for k in range(previous + 1)])


def standardise(train, test, noise=None):
    """z-score with the training run; with `noise`, weight each column by the share of
    its spread that is signal."""
    mu, sd = train[TRAIN_ROWS].mean(0), train[TRAIN_ROWS].std(0)
    Z, Z_test = (train - mu) / sd, (test - mu) / sd
    if noise is not None:
        weight = np.sqrt(np.clip(1 - noise / sd ** 2, 0, 1))
        Z, Z_test = Z * weight, Z_test * weight
    return Z, Z_test


# NOISE MEASURED FROM THE DATA (new in QCML110226)  [note 6]

def measure(kind, size, strength, draw):
    """One draw of the experiment: the training and test records (QCML102426_5's) and each
    column's noise variance measured from the two halves of the training run."""
    rng = np.random.default_rng(draw)
    second = np.random.default_rng(100 + draw)      # the halves' extra draw, kept apart so the
                                                    # records stay QCML102426_5's
    if kind == "projective":
        train, test = projective(exact_train, strength, rng), projective(exact_test, strength, rng)
        g = projective(exact_train, strength, second) - exact_train
        columns = BUDGETS[size]
        train, test, g = train[:, columns], test[:, columns], g[:, columns]
    else:
        train = weak_record(weak_numbers[0], strength, size, rng)
        test = weak_record(weak_numbers[1], strength, size, rng)
        g = weak_record(weak_numbers[0], strength, size, second) - weak_numbers[0][:, :, :2].reshape(len(train), -1)
    A, B = train + g, train - g                     # the two halves' records
    return train, test, (((A - B) / 2)[TRAIN_ROWS] ** 2).mean(0)


def qcml_readout(Z, y, Z_test, seed=0):
    """Supervised QCML (n = SUP_N), not validated: settings from QCML102426's pilots."""
    mu, sd = y[TRAIN_ROWS].mean(), y[TRAIN_ROWS].std()
    model = train_supervised(Z[TRAIN_ROWS], (y[TRAIN_ROWS] - mu) / sd, SUP_N, mse, seed=seed)
    return supervised_forecast(model, Z_test) * sd + mu


def target(delay):
    return series[TRAIN_INPUTS + delay], series[TEST_INPUTS + delay][TEST_ROWS]


series = mackey_glass_dudas()
exact_train, exact_test = reservoir(series[TRAIN_INPUTS], MG_DRIVE), reservoir(series[TEST_INPUTS], MG_DRIVE)
weak_numbers = (photon_numbers(series[TRAIN_INPUTS], MG_DRIVE, 1e9),
                photon_numbers(series[TEST_INPUTS], MG_DRIVE, 1e9))


# QCML102426_5 ADDITIONS: input representations, validated readouts, QCML+

REPRESENTATIONS = ["raw", 10, 30]               # improved inputs as they are, or their first k
                                                # principal components (fitted on the training run)
QCML_PLUS_OPTIONS = [(10, 8), (10, 16), (30, 8), (30, 16)]   # (components, Hilbert-space dimension)
QCML_PLUS_EPOCHS, QCML_PLUS_POLISH = 400, 600   # longer training for the final QCML+ model


def compress(Z, Z_test, k):
    """The first k principal components of the training rows (all rows if k is 'raw')."""
    if k == "raw":
        return Z, Z_test
    centre = Z[TRAIN_ROWS].mean(0)
    _, _, directions = np.linalg.svd(Z[TRAIN_ROWS] - centre, full_matrices=False)
    return (Z - centre) @ directions[:k].T, (Z_test - centre) @ directions[:k].T


def diffusion_map_fit(Z, y, Z_test):
    """(test predictions, validation predictions from the FIT-rows model, validation error)."""
    mu, sd = y[TRAIN_ROWS].mean(), y[TRAIN_ROWS].std()
    Y = (y - mu) / sd
    best = (np.inf, None)
    for neighbours in DM_NEIGHBOUR_OPTIONS:
        errors = DiffusionMap(Z[FIT_ROWS], neighbours).rebuild_errors(
            Y[FIT_ROWS, None], Z[VAL_ROWS], Y[VAL_ROWS, None], DM_MAX_M)
        if errors.min() < best[0]:
            best = (errors.min(), (neighbours, int(errors.argmin()) + 1))
    neighbours, m = best[1]
    on_val = DiffusionMap(Z[FIT_ROWS], neighbours).rebuild(Y[FIT_ROWS, None], Z[VAL_ROWS], np.array([m]))[0][:, 0]
    on_test = DiffusionMap(Z[TRAIN_ROWS], neighbours).rebuild(Y[TRAIN_ROWS, None], Z_test, np.array([m]))[0][:, 0]
    return on_test * sd + mu, on_val * sd + mu, best[0]


def kernel_ridge_fit(Z, y, Z_test):
    """(test predictions, validation error) for Gaussian-kernel ridge, settings on validation."""
    distances = lambda A, B: np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T, 0)
    median = np.sqrt(np.median(distances(Z[FIT_ROWS], Z[FIT_ROWS])))
    d_fit, d_val = distances(Z[FIT_ROWS], Z[FIT_ROWS]), distances(Z[VAL_ROWS], Z[FIT_ROWS])
    offset, sd = y[FIT_ROWS].mean(), y[TRAIN_ROWS].std()
    best = (np.inf, None)
    for width in KERNEL_WIDTHS:
        scale = 2 * (width * median) ** 2
        values, vectors = np.linalg.eigh(np.exp(-d_fit / scale))
        shares, at_val = vectors.T @ (y[FIT_ROWS] - offset), np.exp(-d_val / scale) @ vectors
        for strength in KERNEL_STRENGTHS:
            error = np.mean((at_val @ (shares / (values + strength)) + offset - y[VAL_ROWS]) ** 2) / sd ** 2
            if error < best[0]:
                best = (error, (width, strength))
    width, strength = best[1]
    scale = 2 * (width * median) ** 2
    offset = y[TRAIN_ROWS].mean()
    weights = np.linalg.solve(np.exp(-distances(Z[TRAIN_ROWS], Z[TRAIN_ROWS]) / scale)
                              + strength * np.eye(len(TRAIN_ROWS)), y[TRAIN_ROWS] - offset)
    return np.exp(-distances(Z_test, Z[TRAIN_ROWS]) / scale) @ weights + offset, best[0]


# QCML+ AND ITS UNCERTAINTY  [note 7]

def train_qcml_plus(Z, Y, n, seed, epochs, polish):
    torch.manual_seed(seed)
    X, T = torch.tensor(Z, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
    K = X.shape[1]
    model = QCML(D=K + 1, n=n)
    known = list(range(K))

    def loss_on(rows):
        psi = ground_state_from(model, X[rows], known)
        return ((expectation(psi, model.observables()[K]) - T[rows]) ** 2).mean()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(epochs):
        order = torch.randperm(len(X))
        for start in range(0, len(X), BATCH):
            loss = loss_on(order[start:start + BATCH])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=polish, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        loss = loss_on(slice(None))
        loss.backward()
        return loss

    lbfgs.step(closure)
    return model


def qcml_plus_predict(model, Z):
    """(<psi|B|psi>, Var_psi(B)) for each row of Z."""
    K = Z.shape[1]
    with torch.no_grad():
        psi = ground_state_from(model, torch.tensor(Z, dtype=torch.float32), list(range(K)))
        B = model.observables()[K]
        mean = expectation(psi, B)
        return mean.numpy().astype(float), (expectation(psi, B @ B) - mean ** 2).clamp(min=0).numpy().astype(float)


def qcml_plus(Z, y, Z_test, seed):
    """QCML+: components and n chosen on validation (standard training on the FIT rows),
    then the chosen model retrained on all training rows with longer training. Returns the
    final model's test predictions, the chosen FIT model's validation and test predictions
    with their variances (for the blend weight and the uncertainty check, so that the
    variance is calibrated and checked on the same model), and the choice."""
    mu, sd = y[TRAIN_ROWS].mean(), y[TRAIN_ROWS].std()
    Y = (y - mu) / sd
    best = (np.inf, None)
    for k, n in QCML_PLUS_OPTIONS:
        Zk, Zk_test = compress(Z, Z_test, k)
        model = train_qcml_plus(Zk[FIT_ROWS], Y[FIT_ROWS], n, seed, EPOCHS, 200)
        mean, var = qcml_plus_predict(model, Zk[VAL_ROWS])
        error = np.mean((mean - Y[VAL_ROWS]) ** 2)
        if error < best[0]:
            best = (error, (k, n), (mean, var), qcml_plus_predict(model, Zk_test))
    _, (k, n), on_val, on_test = best
    Zk, Zk_test = compress(Z, Z_test, k)
    model = train_qcml_plus(Zk[TRAIN_ROWS], Y[TRAIN_ROWS], n, seed, QCML_PLUS_EPOCHS, QCML_PLUS_POLISH)
    mean, _ = qcml_plus_predict(model, Zk_test)
    rescale = lambda pair: (pair[0] * sd + mu, pair[1] * sd ** 2)
    return mean * sd + mu, rescale(on_val), rescale(on_test), (k, n)


def evaluate(train, test, noise, delay, seed):
    """Every method on one setting: test NRMSE, the choices made, and QCML+'s uncertainty."""
    y, y_test = target(delay)
    y_val = y[VAL_ROWS]
    previous = HISTORY[delay]
    Z, Z_test = standardise(with_history(train, previous), with_history(test, previous),
                            None if noise is None else np.tile(noise, previous + 1))
    Z_test = Z_test[TEST_ROWS]
    out, choices = {}, {}
    # diffusion map and kernel ridge: the representation with the lowest validation error
    represented = {r: compress(Z, Z_test, r) for r in REPRESENTATIONS}
    dm = min(((diffusion_map_fit(represented[r][0], y, represented[r][1]), r) for r in REPRESENTATIONS),
             key=lambda t: t[0][2])
    (dm_test, dm_val, _), choices["Diffusion map"] = dm
    out["Diffusion map"] = nrmse(dm_test, y_test)
    kr = min(((kernel_ridge_fit(represented[r][0], y, represented[r][1]), r) for r in REPRESENTATIONS),
             key=lambda t: t[0][1])
    out["Kernel ridge"], choices["Kernel ridge"] = nrmse(kr[0][0], y_test), kr[1]
    out["QCML (as before)"] = nrmse(qcml_readout(Z, y, Z_test, seed), y_test)
    q_test, (q_val, q_val_var), (f_test, f_var), choices["QCML+"] = qcml_plus(Z, y, Z_test, seed)
    out["QCML+"] = nrmse(q_test, y_test)
    # blend of QCML+ and the diffusion map, weight chosen on validation
    alphas = np.linspace(0, 1, 21)
    alpha = alphas[int(np.argmin([np.mean((a * q_val + (1 - a) * dm_val - y_val) ** 2) for a in alphas]))]
    out["QCML+ & diffusion map"] = nrmse(alpha * q_test + (1 - alpha) * dm_test, y_test)
    choices["blend weight on QCML+"] = alpha
    # QCML+'s uncertainty, on the chosen FIT-rows model: one scale factor for its variance
    # from the validation block, then checked on the test run
    factor = np.mean((q_val - y_val) ** 2) / max(np.mean(q_val_var), 1e-30)
    sigma = np.sqrt(factor * f_var)
    error = np.abs(f_test - y_test)
    confident = sigma <= np.median(sigma)
    spread = y_test.max() - y_test.min()
    uncertainty = {"correlation": float(np.corrcoef(sigma, error)[0, 1]),
                   "within 1 sigma": float(np.mean(error < sigma)),
                   "within 2 sigma": float(np.mean(error < 2 * sigma)),
                   "NRMSE, confident half": float(np.sqrt(np.mean(error[confident] ** 2)) / spread),
                   "NRMSE, unsure half": float(np.sqrt(np.mean(error[~confident] ** 2)) / spread)}
    return out, choices, uncertainty, (y_test, f_test, sigma)


METHODS = ["Diffusion map", "Kernel ridge", "QCML (as before)", "QCML+", "QCML+ & diffusion map"]


def run_job(job):
    """One setting, noise draw and delay: evaluate's results (the jobs run side by side)."""
    setting, draw, delay = job
    train, test, noise = measure(*SETTINGS[setting][1:], draw)
    return job, evaluate(train, test, noise, delay, draw)


# Plots: one PNG per plot, numbered in reading order, saved in PLOT_DIR

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "figure.titlesize": 13,
                     "figure.titleweight": "bold", "legend.fontsize": 9,
                     "legend.title_fontsize": 9, "legend.frameon": False,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.grid": True, "axes.axisbelow": True,
                     "grid.color": "#e1e0d9", "grid.linewidth": 0.6})
QCML_COLOUR, PCA_COLOUR, DM_COLOUR = "#eb6834", "#2a78d6", "#1baf7a"
GREY, INK, MUTED_INK, RULE = "#898781", "#0b0b0b", "#52514e", "#c3c2b7"


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


MARKERS = {"Diffusion map": (DM_COLOUR, "s"), "Kernel ridge": (PCA_COLOUR, "o"),
           "QCML (as before)": (GREY, "v"), "QCML+": (QCML_COLOUR, "^"), "QCML+ & diffusion map": (INK, "*")}
LABELS = [label for label, *_ in SETTINGS]


# THE COMPARISON

if __name__ == "__main__":
    jobs = list(itertools.product(range(len(SETTINGS)), range(NOISE_DRAWS), PREDICT_DELAYS))
    print(f"QCML+ against the diffusion map and kernel ridge, with measured noise weights: {len(SETTINGS)} "
          f"settings x {len(PREDICT_DELAYS)} delays x {NOISE_DRAWS} draws, {len(jobs)} jobs on {WORKERS} "
          "processes", flush=True)
    start = time.time()
    results, choices, uncertainty = {}, {}, {}
    example = None
    with ProcessPoolExecutor(WORKERS) as pool:
        for done, future in enumerate(as_completed([pool.submit(run_job, job) for job in jobs]), 1):
            (setting, draw, delay), (scores, picked, unsure, arrays) = future.result()
            label = SETTINGS[setting][0]
            if (label, delay) == EXAMPLE and draw == 0:
                example = arrays
            for store, values in [(results, scores), (choices, picked), (uncertainty, unsure)]:
                cell = store.setdefault((label, delay), {})
                for name, value in values.items():
                    cell.setdefault(name, [None] * NOISE_DRAWS)[draw] = value
            if done % 6 == 0 or done == len(jobs):
                print(f"   {done}/{len(jobs)} jobs done ({(time.time() - start) / 60:.0f} min)", flush=True)
    results = {key: {name: np.array(v) for name, v in cell.items()} for key, cell in results.items()}

    print("\nThe measured noise weights on the training run (draw 1), per measured column:")
    with open("qcml110226_noise_weights.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["setting", "columns", "weight, median", "weight, smallest", "columns weighted below 0.5",
                         "columns weighted 0"])
        for label, kind, size, strength in SETTINGS:
            train, _, noise = measure(kind, size, strength, 0)
            weight = np.sqrt(np.clip(1 - noise / train[TRAIN_ROWS].var(0), 0, 1))
            row = [len(weight), np.median(weight), weight.min(), int((weight < 0.5).sum()), int((weight == 0).sum())]
            print(f"   {label:<24} median {row[1]:.3f}, smallest {row[2]:.3f}   below 0.5: {row[3]} of {row[0]}"
                  f"   weighted 0: {row[4]}")
            writer.writerow([label] + row)
    print("saved to qcml110226_noise_weights.csv")

    print("\nTest-run NRMSE, mean over draws (* = best); QCML+ vs the diffusion map and vs kernel ridge: "
          "ratio of means, and draws where QCML+ was better")
    with open("qcml110226_readouts.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["delay", "setting"] + [f"{m} {s}" for m in METHODS for s in ("mean", "std")]
                        + ["QCML+ / diffusion map", "draws QCML+ beat diffusion map", "QCML+ / kernel ridge",
                           "draws QCML+ beat kernel ridge", "QCML+ choices", "diffusion map inputs",
                           "kernel ridge inputs"])
        for delay in PREDICT_DELAYS:
            print(f"\n   delay {delay}:" + "".join(f"{m:>23}" for m in METHODS))
            for label, *_ in SETTINGS:
                cell, chosen = results[(label, delay)], choices[(label, delay)]
                best = min(METHODS, key=lambda m: cell[m].mean())
                vs = lambda m: (cell["QCML+"].mean() / cell[m].mean(), int((cell["QCML+"] < cell[m]).sum()))
                (r_dm, w_dm), (r_kr, w_kr) = vs("Diffusion map"), vs("Kernel ridge")
                print(f"   {label:<24}" + "".join(f"{cell[m].mean():22.4f}{'*' if m == best else ' '}" for m in METHODS)
                      + f"   vs DM {r_dm:.2f} ({w_dm}/{NOISE_DRAWS})   vs KRR {r_kr:.2f} ({w_kr}/{NOISE_DRAWS})"
                      + f"   QCML+ {chosen['QCML+']}")
                writer.writerow([delay, label] + [v for m in METHODS for v in (cell[m].mean(), cell[m].std())]
                                + [r_dm, w_dm, r_kr, w_kr, chosen["QCML+"], chosen["Diffusion map"],
                                   chosen["Kernel ridge"]])
    print("saved to qcml110226_readouts.csv")

    print("\nQCML+'s uncertainty (the quantum variance of B, rescaled on validation), mean over draws:")
    UNCERTAINTY_KEYS = ["correlation", "within 1 sigma", "within 2 sigma", "NRMSE, confident half", "NRMSE, unsure half"]
    with open("qcml110226_uncertainty.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["delay", "setting"] + UNCERTAINTY_KEYS)
        for delay in PREDICT_DELAYS:
            for label, *_ in SETTINGS:
                cell = uncertainty[(label, delay)]
                values = [np.mean(cell[k]) for k in UNCERTAINTY_KEYS]
                print(f"   delay {delay:>2}, {label:<24} corr(sigma, |error|) {values[0]:.2f}   within 1 sigma "
                      f"{values[1]:.0%}, 2 sigma {values[2]:.0%} (ideal 68%, 95%)   NRMSE confident half "
                      f"{values[3]:.4f}, unsure half {values[4]:.4f}")
                writer.writerow([delay, label] + values)
    print("saved to qcml110226_uncertainty.csv")

    # the same comparison with QCML102426_5's weights, from its saved results
    old = None
    if os.path.exists("qcml102426_5_readouts.csv"):
        with open("qcml102426_5_readouts.csv") as file:
            old = {(int(row["delay"]), row["setting"]): row for row in csv.DictReader(file)}
        print("\nAgainst QCML102426_5 (noise weights from the model): new / old mean test NRMSE")
        with open("qcml110226_vs_102426_5.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["delay", "setting"] + [f"{m} {s}" for m in METHODS for s in ("old", "new", "new / old")])
            for delay in PREDICT_DELAYS:
                print(f"\n   delay {delay}:" + "".join(f"{m:>23}" for m in METHODS))
                for label in LABELS:
                    pairs = [(float(old[(delay, label)][f"{m} mean"]), results[(label, delay)][m].mean())
                             for m in METHODS]
                    print(f"   {label:<24}" + "".join(f"{new / was:23.3f}" for was, new in pairs))
                    writer.writerow([delay, label] + [v for was, new in pairs for v in (was, new, new / was)])
        print("saved to qcml110226_vs_102426_5.csv")

    beats = lambda rival, labels: sum(results[(l, d)]["QCML+"].mean() < results[(l, d)][rival].mean()
                                      for l in labels for d in PREDICT_DELAYS)
    print(f"\nThe abstract's numbers: QCML+ beats the diffusion map in {beats('Diffusion map', LABELS)} of "
          f"{2 * len(LABELS)} cells and kernel ridge in {beats('Kernel ridge', NOISIEST)} of the "
          f"{2 * len(NOISIEST)} noisiest; vacuum only, 9^5 shots, delay 20: QCML+ "
          f"{results[('vacuum only, 9^5 shots', 20)]['QCML+'].mean():.4f}")

    # Plots
    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f"\nplots saved to {PLOT_DIR}:")

    # 01-02: every method on every setting
    for number, delay in [("01", 1), ("02", 20)]:
        fig, ax = new_plot(f"QCML+ against the diffusion map and kernel ridge: x(t + {delay})",
                           f"test run NRMSE (log scale), mean ± std over {NOISE_DRAWS} draws; noise weights "
                           "measured from two halves", size=(11, 7))
        xs = np.arange(len(LABELS))
        for k, name in enumerate(METHODS):
            colour, marker = MARKERS[name]
            values = [results[(label, delay)][name] for label in LABELS]
            ax.errorbar(xs + (k - 2) * 0.13, [v.mean() for v in values], yerr=[v.std() for v in values],
                        color=colour, marker=marker, ls="none", ms=8 if marker == "*" else 7, capsize=2, label=name)
        ax.set_yscale("log")
        ax.set_xticks(xs, [label.replace(", ", ",\n") for label in LABELS])
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("NRMSE")
        save_plot(fig, ax, f"{number}_methods_delay_{delay}")

    # 03: QCML+ relative to the diffusion map and to kernel ridge
    fig, ax = new_plot("QCML+ relative to the diffusion map and kernel ridge",
                       f"ratio of mean test NRMSE (below 1: QCML+ better); {NOISE_DRAWS} draws; measured noise weights",
                       size=(9.5, 7))
    ys = np.arange(len(LABELS))[::-1]
    for rival, marker in [("Diffusion map", "s"), ("Kernel ridge", "o")]:
        for delay, colour in [(1, PCA_COLOUR), (20, QCML_COLOUR)]:
            ratios = [results[(l, delay)]["QCML+"].mean() / results[(l, delay)][rival].mean() for l in LABELS]
            ax.plot(ratios, ys + (0.12 if rival == "Kernel ridge" else -0.12), marker, color=colour, ms=8,
                    mfc=colour if rival == "Diffusion map" else "white", label=f"vs {rival.lower()}, delay {delay}")
    ax.axvline(1, color=INK, lw=1)
    ax.set_xscale("log")
    ax.set_yticks(ys, LABELS)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("QCML+ NRMSE / rival's NRMSE")
    save_plot(fig, ax, "03_qcml_plus_ratios")

    # 04-05: tables
    for number, delay in [("04", 1), ("05", 20)]:
        rows = []
        for label in LABELS:
            cell = results[(label, delay)]
            best = min(METHODS, key=lambda m: cell[m].mean())
            rows.append((label, [f"{cell[m].mean():.4f} ± {cell[m].std():.4f}" + (" *" if m == best else "")
                                 for m in METHODS]))
        table_plot(f"{number}_methods_table_delay_{delay}", f"QCML+ against the diffusion map and kernel ridge: x(t + {delay})",
                   f"test run NRMSE, mean ± std over {NOISE_DRAWS} draws; * = best",
                   ["Setting"] + [m.replace(" & ", " &\n") for m in METHODS], rows, label_width=0.2, size=15,
                   note="All methods get the records of the current and last "
                        f"{HISTORY[delay]} inputs, each column weighted by the share of its spread that is signal, "
                        "with the noise measured from two halves of the shots or runs. Diffusion map and kernel "
                        "ridge choose raw or PCA-compressed inputs (10 or 30 components) and their settings on "
                        "validation; QCML+ chooses components and n (8 or 16) on validation, then trains longer. "
                        "Blend weight on validation.")

    # 06: the uncertainty table
    rows = [(f"delay {d}, {label}", [f"{np.mean(uncertainty[(label, d)][k]):.2f}" if i == 0 else
                                      f"{np.mean(uncertainty[(label, d)][k]):.0%}" if i < 3 else
                                      f"{np.mean(uncertainty[(label, d)][k]):.4f}"
                                      for i, k in enumerate(UNCERTAINTY_KEYS)])
            for d in PREDICT_DELAYS for label in LABELS]
    table_plot("06_uncertainty_table", "Does QCML+'s quantum variance flag its own errors?",
               f"test run, mean over {NOISE_DRAWS} draws; QCML+ trained on the fit rows; sigma = sqrt(Var(B)), "
               "rescaled on validation",
               ["Delay, setting", "corr(sigma,\n|error|)", "Error within\n1 sigma", "Error within\n2 sigma",
                "NRMSE, most\nconfident half", "NRMSE, least\nconfident half"], rows, label_width=0.32, size=13,
               note="Ideal coverage: 68% within 1 sigma, 95% within 2 sigma. If sigma tracks the error, the "
                    "confident half (lower sigma) has the lower error.")

    # 07: one test run with QCML+'s uncertainty
    if example is not None:
        y_test, q_test, sigma = example
        show = np.arange(200, 320)
        fig, ax = new_plot(f"QCML+ with its uncertainty: {EXAMPLE[0]}, x(t + {EXAMPLE[1]})",
                           "test run, draw 1, QCML+ trained on the fit rows; band = ± 2 sigma from the quantum "
                           "variance of B", size=(10, 6))
        ax.fill_between(show, (q_test - 2 * sigma)[show], (q_test + 2 * sigma)[show], color=QCML_COLOUR,
                        alpha=0.2, lw=0, label="QCML+ ± 2 sigma")
        ax.plot(show, y_test[show], color=INK, lw=1.5, label="true x(t + 20)")
        ax.plot(show, q_test[show], color=QCML_COLOUR, lw=1.5, label="QCML+ prediction")
        ax.set_xlabel("test input")
        ax.set_ylabel("x")
        save_plot(fig, ax, "07_uncertainty_example")

    # 08: the change from QCML102426_5's weights
    if old is not None:
        rows = [(f"{label}, delay {delay}",
                 [f"{float(old[(delay, label)]['QCML+ mean']):.4f}", f"{results[(label, delay)]['QCML+'].mean():.4f}"]
                 + [f"{results[(label, delay)][m].mean() / float(old[(delay, label)][f'{m} mean']):.3f}" for m in METHODS])
                for delay in PREDICT_DELAYS for label in LABELS]
        table_plot("08_against_model_weights", "Noise weights measured from the data against weights from the model",
                   f"test run NRMSE, mean over {NOISE_DRAWS} draws, of QCML+ under each, then every method's "
                   "measured / model ratio (QCML102426_5 used the model)",
                   ["Setting, delay", "QCML+\nmodel", "QCML+\nmeasured"] + [m.replace(" & ", " &\n") for m in METHODS],
                   rows, label_width=0.24, size=16,
                   note="Same records, settings and readouts in both runs; only each column's noise variance "
                        "differs. Model: p(1 - p)/shots, or (Var N + 1/(4 eta Gamma tau))/runs with Var N from the "
                        "simulator. Measured: the mean of ((A - B)/2)^2 over the training inputs, A and B the "
                        "records of two halves of the shots or runs. The exact settings have no noise, so their "
                        "weights are 1 in both.")


# ========================================================================================
# EXPLANATION
# Notes on the code above, in the order of its sections. A tag [note k] in the code
# marks the section that note k explains.
# ========================================================================================
#
# [note 0] About this script
#
# Every readout's inputs weight each measured column by the share of its spread that is signal,
# sqrt(1 - noise variance / column variance) (see standardise). QCML102426_5 took that noise
# variance from the measurement model, and for the weak record the model needs Var N from the
# simulator's exact solution, which no experimenter sees. Here the noise variance is measured
# from the data instead, by splitting the shots (or runs) into two halves (see measure). Nothing
# else changes: the records, settings, readouts, blend and uncertainty check are QCML102426_5's,
# so any difference between the two runs comes from the weights. The end of the run compares
# the results with QCML102426_5's.
#
# Runs in about 10 minutes on 8 cores: the 42 jobs (7 settings x 3 draws x 2 delays) run side
# by side, one process each. Run with python3.13 QCML110226.py.
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
# QCML102426 checked this against the full master equation and their saved results.
#
# [note 2] WEAK PHOTON-NUMBER MEASUREMENT
#
# The photon number of each oscillator is monitored continuously during a run, with
# measurement strength GAMMA (per second) and detector efficiency ETA:
#   back-action  the measurement dephases each oscillator in its photon-number basis at
#                rate GAMMA: the Lindblad term GAMMA D[N], the same dephasing their Fig. 4
#                uses (there 100e6 per second, as noise rather than a measurement)
#   record       each 10 ns window of a run gives that window's mean photon number plus
#                noise of variance  Var(N) + 1 / (4 ETA GAMMA tau)  (tau = window length):
#                the photon number's own quantum spread, which a strong measurement
#                resolves, plus the detector's imprecision, large when GAMMA is small
#   runs         R runs of the whole input sequence are averaged, dividing the variance by R
# A record has 20 numbers per input (N_a and N_b in each of the 10 windows) against their
# 160 probabilities. Approximations: the noise is Gaussian and independent between windows
# and oscillators (in a real run the photon number found in one window carries over to the
# next), and the averaged record follows the unconditional (run-averaged) dynamics.
#
# How it is simulated. With dephasing the reservoir no longer stays in a coherent state,
# but its averages <a+^p a^q b+^r b^s> still obey closed linear equations: the energy is at
# most quadratic in a and b and the drive is linear, so no average's equation involves a
# higher power than its own. The 70 averages up to 4th order give <N> and Var(N) exactly,
# with no photon cutoff, from one 140 x 140 matrix exponential per input.
#
# One surprise (QCML102426_2): dephasing HEATS this reservoir. Their drive is constant within
# each input, far below the oscillators' 10 GHz, so it mostly cancels itself; dephasing
# scrambles the phase that does the cancelling, and photons pile up (about 5 per
# oscillator at GAMMA = 1e9). At the strong drive of their Fig. 4, 100 MHz dephasing pushes
# the reservoir past their 8-photon cutoff, so their saved dephasing results are clipped.
#
# [note 3] THE MACKEY-GLASS INPUT (as in main_fig7.jl)
#
# dx/dt = 0.2 x(t - 17) / (1 + x(t - 17)^10) - 0.1 x(t), started from x = 1 before t = 0
# and x(0) = 0, and read once per time unit. The reservoir is trained on inputs x(0..999)
# and tested on x(1100..2099), each run starting from an empty reservoir; the target for
# input x(t) is x(t + delay). Their solver's exact numbers were not saved, and the series
# is chaotic, so small integration differences grow: this series follows the same
# equation but not digit for digit theirs (step 0.002 here; with step 0.01 the test half
# differs by up to 0.4). Its statistics, and so the task, are the same.
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
# All K + 1 matrices are trained together on the squared prediction error. Here the
# inputs are the measured numbers of the reservoir record and the target is x(t + delay).
#
# [note 5]
# DIFFUSION MAP (as in QCML102326): built on the measured numbers of the training
# records, with each point's kernel width its distance to its k-th nearest neighbour.
# Anything known on the training records (the unmeasured numbers, or the target) is
# rebuilt at a new record from the walk's first m eigenvectors (geometric harmonics).
#
# [note 6] NOISE MEASURED FROM THE DATA (new in QCML110226)
#
# Every record is an average over repetitions: shots of the projective measurement, or runs
# of the weak one. The experimenter splits the repetitions into two halves and keeps each
# half's record, A and B. The readouts get the usual record, their average (A + B)/2. Both
# halves measure the same true values, so A - B holds noise only, and since each half has
# twice the full record's noise variance, (A - B)/2 has exactly the full record's. Each
# column's noise variance is the mean of ((A - B)/2)^2 over the training inputs: no formula,
# and nothing the experimenter cannot see. (QCML102426_5 used p(1 - p)/shots and
# (Var N + 1/(4 eta Gamma tau))/runs, the second with Var N from the simulator's exact solution.)
#
# The simulation makes the halves from the full record and a second, independent draw g of
# its noise: A = record + g, B = record - g. For Gaussian noise these are exactly two
# independent half-records, and the full record stays QCML102426_5's.
#
# [note 7] QCML+ AND ITS UNCERTAINTY
#
# The same supervised QCML, returning with each prediction <psi|B|psi> the quantum variance
# of the target observable in that state, Var(B) = <psi|B^2|psi> - <psi|B|psi>^2. A sharp
# ground state (small Var) means the measured record pins the state down; a broad one means
# the model is unsure. Training on this variance (a Gaussian likelihood) wrecked the
# accuracy in the pilot, so QCML+ is trained on squared error as before and the variance is
# only read off afterwards, rescaled by one factor fitted on the validation block.
