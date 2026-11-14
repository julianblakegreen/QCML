# Quantum Cognitive Machine Learning Project
# QCML102426_4: builds on QCML102426_3. Is it QCML, or any nonlinear readout?
# Runs in about 12 minutes.
# What it does and how: see the notes at the end of this file.


import csv
import itertools
import os
import textwrap
import time
from math import factorial

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.linalg import expm


# Repository set-up: run from this folder, with QCML from 1_Plain_QCML
import os
import sys
_FOLDER = os.path.dirname(os.path.abspath(__file__))
os.chdir(_FOLDER)                                                 # data and outputs live in this folder
sys.path.insert(0, os.path.join(_FOLDER, "..", "1_Plain_QCML"))   # qcml.py and hermitian.py

from qcml import QCML

torch.set_num_threads(1)      # the matrices are tiny, so extra threads cost more than they save

PLOT_DIR = "qcml102426_4_plots"   # every plot is saved here as its own PNG


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
METHODS = ["Ridge", "Diffusion map", "Kernel ridge", "Neural network", "QCML"]
RIDGE_STRENGTHS = np.logspace(-6, 4, 41)
DM_NEIGHBOUR_OPTIONS = (30, 100, 300)
DM_MAX_M = 600
KERNEL_WIDTHS = (0.25, 0.5, 1, 2, 4)            # x the median distance between records
KERNEL_STRENGTHS = np.logspace(-8, 1, 19)
NET_WIDTHS, NET_DECAYS = (64, 128, 256), (0.0, 1e-4)
NET_MAX_EPOCHS, NET_PATIENCE = 1000, 40
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


def projective_noise(train, shots):
    if shots is None:
        return np.zeros(train.shape[1])
    return np.mean(np.abs(train[TRAIN_ROWS] * (1 - train[TRAIN_ROWS])) / shots, 0)


def weak_noise(numbers, gamma, runs):
    tau = (RESOLUTION - 1) * INTERVAL / RESOLUTION / numbers.shape[1]
    spread = np.maximum(numbers[TRAIN_ROWS, :, 2:] - numbers[TRAIN_ROWS, :, :2] ** 2, 0)
    return ((spread + 1 / (4 * ETA * gamma * tau)) / runs).mean(0).ravel()


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


# THE READOUTS. Each takes inputs Z (training run, one row per input), target y and the
# test rows' inputs, chooses its settings on the validation block, refits on all training
# rows and returns the test predictions.

def ridge(Z, y, Z_test):
    """Linear, with a constant term; strength chosen on validation."""
    def fit(rows, strength):
        centre, offset = Z[rows].mean(0), y[rows].mean()
        A = Z[rows] - centre
        weights = np.linalg.solve(A.T @ A + strength * np.eye(A.shape[1]), A.T @ (y[rows] - offset))
        return lambda X: (X - centre) @ weights + offset
    errors = [np.mean((fit(FIT_ROWS, s)(Z[VAL_ROWS]) - y[VAL_ROWS]) ** 2) for s in RIDGE_STRENGTHS]
    return fit(TRAIN_ROWS, RIDGE_STRENGTHS[int(np.argmin(errors))])(Z_test)


def diffusion_map(Z, y, Z_test):
    """Geometric harmonics (as in QCML102426); neighbours and m chosen on validation."""
    mu, sd = y[TRAIN_ROWS].mean(), y[TRAIN_ROWS].std()
    Y = (y - mu) / sd
    best = (np.inf, None)
    for neighbours in DM_NEIGHBOUR_OPTIONS:
        errors = DiffusionMap(Z[FIT_ROWS], neighbours).rebuild_errors(
            Y[FIT_ROWS, None], Z[VAL_ROWS], Y[VAL_ROWS, None], DM_MAX_M)
        if errors.min() < best[0]:
            best = (errors.min(), (neighbours, int(errors.argmin()) + 1))
    neighbours, m = best[1]
    return DiffusionMap(Z[TRAIN_ROWS], neighbours).rebuild(Y[TRAIN_ROWS, None], Z_test,
                                                           np.array([m]))[0][:, 0] * sd + mu


# KERNEL RIDGE REGRESSION  [note 6]

def kernel_ridge(Z, y, Z_test):
    distances = lambda A, B: np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T, 0)
    median = np.sqrt(np.median(distances(Z[FIT_ROWS], Z[FIT_ROWS])))
    d_fit, d_val = distances(Z[FIT_ROWS], Z[FIT_ROWS]), distances(Z[VAL_ROWS], Z[FIT_ROWS])
    offset = y[FIT_ROWS].mean()
    best = (np.inf, None)
    for width in KERNEL_WIDTHS:
        scale = 2 * (width * median) ** 2
        values, vectors = np.linalg.eigh(np.exp(-d_fit / scale))      # all strengths at once
        shares, at_val = vectors.T @ (y[FIT_ROWS] - offset), np.exp(-d_val / scale) @ vectors
        for strength in KERNEL_STRENGTHS:
            error = np.mean((at_val @ (shares / (values + strength)) + offset - y[VAL_ROWS]) ** 2)
            if error < best[0]:
                best = (error, (width, strength))
    width, strength = best[1]
    scale = 2 * (width * median) ** 2
    offset = y[TRAIN_ROWS].mean()
    weights = np.linalg.solve(np.exp(-distances(Z[TRAIN_ROWS], Z[TRAIN_ROWS]) / scale)
                              + strength * np.eye(len(TRAIN_ROWS)), y[TRAIN_ROWS] - offset)
    return np.exp(-distances(Z_test, Z[TRAIN_ROWS]) / scale) @ weights + offset


# NEURAL NETWORK  [note 7]

def neural_network(Z, y, Z_test, seed=0):
    mu, sd = y[TRAIN_ROWS].mean(), y[TRAIN_ROWS].std()
    X, Y = torch.tensor(Z, dtype=torch.float32), torch.tensor((y - mu) / sd, dtype=torch.float32)

    def train(rows, width, decay, epochs, validate=False):
        torch.manual_seed(seed)
        net = torch.nn.Sequential(torch.nn.Linear(Z.shape[1], width), torch.nn.Tanh(),
                                  torch.nn.Linear(width, width), torch.nn.Tanh(), torch.nn.Linear(width, 1))
        optimizer = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=decay)
        rows, val = torch.tensor(rows), torch.tensor(VAL_ROWS)
        best, best_epoch, waited = np.inf, 1, 0
        for epoch in range(1, epochs + 1):
            order = rows[torch.randperm(len(rows))]
            for start in range(0, len(order), BATCH):
                batch = order[start:start + BATCH]
                loss = ((net(X[batch])[:, 0] - Y[batch]) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if validate:
                with torch.no_grad():
                    error = ((net(X[val])[:, 0] - Y[val]) ** 2).mean().item()
                if error < best:
                    best, best_epoch, waited = error, epoch, 0
                elif (waited := waited + 1) >= NET_PATIENCE:
                    break
        return net, best, best_epoch

    results = [(train(FIT_ROWS, w, d, NET_MAX_EPOCHS, validate=True)[1:], w, d)
               for w in NET_WIDTHS for d in NET_DECAYS]
    (_, epochs), width, decay = min(results, key=lambda r: r[0][0])
    net, _, _ = train(TRAIN_ROWS, width, decay, epochs)
    with torch.no_grad():
        return net(torch.tensor(Z_test, dtype=torch.float32))[:, 0].numpy() * sd + mu


def qcml_readout(Z, y, Z_test, seed=0):
    """Supervised QCML (n = SUP_N), not validated: settings from QCML102426's pilots."""
    mu, sd = y[TRAIN_ROWS].mean(), y[TRAIN_ROWS].std()
    model = train_supervised(Z[TRAIN_ROWS], (y[TRAIN_ROWS] - mu) / sd, SUP_N, mse, seed=seed)
    return supervised_forecast(model, Z_test) * sd + mu


def target(delay):
    return series[TRAIN_INPUTS + delay], series[TEST_INPUTS + delay][TEST_ROWS]


def all_readouts(train, test, noise, delay, seed):
    """Test-run NRMSE of every readout on the improved inputs, and how long each took."""
    y, y_test = target(delay)
    previous = HISTORY[delay]
    Z, Z_test = standardise(with_history(train, previous), with_history(test, previous),
                            None if noise is None else np.tile(noise, previous + 1))
    Z_test = Z_test[TEST_ROWS]
    readouts = {"Ridge": lambda: ridge(Z, y, Z_test), "Diffusion map": lambda: diffusion_map(Z, y, Z_test),
                "Kernel ridge": lambda: kernel_ridge(Z, y, Z_test),
                "Neural network": lambda: neural_network(Z, y, Z_test, seed),
                "QCML": lambda: qcml_readout(Z, y, Z_test, seed)}
    scores, seconds = {}, {}
    for name, run in readouts.items():
        start = time.time()
        scores[name] = nrmse(run(), y_test)
        seconds[name] = time.time() - start
    return scores, seconds


series = mackey_glass_dudas()
exact_train, exact_test = reservoir(series[TRAIN_INPUTS], MG_DRIVE), reservoir(series[TEST_INPUTS], MG_DRIVE)
weak_numbers = (photon_numbers(series[TRAIN_INPUTS], MG_DRIVE, 1e9),
                photon_numbers(series[TEST_INPUTS], MG_DRIVE, 1e9))


# THE COMPARISON

print(f"Every readout on the same inputs: {len(SETTINGS)} settings x {len(PREDICT_DELAYS)} delays x "
      f"{NOISE_DRAWS} draws")
results, timing = {}, {name: [] for name in METHODS}
for label, kind, size, strength in SETTINGS:
    start = time.time()
    for draw in range(NOISE_DRAWS):
        rng = np.random.default_rng(draw)
        if kind == "projective":
            train, test = projective(exact_train, strength, rng), projective(exact_test, strength, rng)
            noise = projective_noise(train, strength)
            columns = BUDGETS[size]
            train, test, noise = train[:, columns], test[:, columns], noise[columns]
        else:
            train = weak_record(weak_numbers[0], strength, size, rng)
            test = weak_record(weak_numbers[1], strength, size, rng)
            noise = weak_noise(weak_numbers[0], strength, size)
        for delay in PREDICT_DELAYS:
            scores, seconds = all_readouts(train, test, noise, delay, draw)
            cell = results.setdefault((label, delay), {name: [] for name in METHODS})
            for name in METHODS:
                cell[name].append(scores[name])
                timing[name].append(seconds[name])
    print(f"   {label} done ({time.time() - start:.0f} s)")
results = {key: {name: np.array(v) for name, v in cell.items()} for key, cell in results.items()}

# reference: the input history itself, no reservoir (exact inputs, one draw)
reference = {}
for delay in PREDICT_DELAYS:
    reference[delay], _ = all_readouts(series[TRAIN_INPUTS, None], series[TEST_INPUTS, None], None, delay, 0)

print("\nTest-run NRMSE, mean over draws (the best in each row marked *); QCML vs the best other "
      "readout: ratio of means, and in how many draws QCML was better")
with open("qcml102426_4_readouts.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["delay", "setting"] + [f"{m} {s}" for m in METHODS for s in ("mean", "std")]
                    + ["best other", "QCML / best other", "draws QCML better"])
    for delay in PREDICT_DELAYS:
        print(f"\n   delay {delay}:" + "".join(f"{m:>16}" for m in METHODS))
        rows = [(label, results[(label, delay)]) for label, *_ in SETTINGS]
        rows.append(("no reservoir: input history", {m: np.array([v]) for m, v in reference[delay].items()}))
        for label, cell in rows:
            best = min(METHODS, key=lambda m: cell[m].mean())
            others = [m for m in METHODS if m != "QCML"]
            rival = min(others, key=lambda m: cell[m].mean())
            wins = int((cell["QCML"] < cell[rival]).sum())
            print(f"   {label:<30}" + "".join(f"{cell[m].mean():15.4f}{'*' if m == best else ' '}" for m in METHODS)
                  + f"   QCML/{rival} {cell['QCML'].mean() / cell[rival].mean():.2f} ({wins} of {len(cell['QCML'])})")
            writer.writerow([delay, label] + [v for m in METHODS for v in (cell[m].mean(), cell[m].std())]
                            + [rival, cell["QCML"].mean() / cell[rival].mean(), wins])
print("saved to qcml102426_4_readouts.csv")
print("\nseconds per fit, median: " + "   ".join(f"{m} {np.median(timing[m]):.1f}" for m in METHODS))


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


MARKERS = {"Ridge": (GREY, "D"), "Diffusion map": (DM_COLOUR, "s"), "Kernel ridge": (PCA_COLOUR, "o"),
           "Neural network": (INK, "v"), "QCML": (QCML_COLOUR, "^")}
LABELS = [label for label, *_ in SETTINGS]
print(f"\nplots saved to {PLOT_DIR}:")

# 01-02: every readout on every setting
for number, delay in [("01", 1), ("02", 20)]:
    fig, ax = new_plot(f"Every readout on the same inputs: x(t + {delay})",
                       f"test run NRMSE (log scale), mean ± std over {NOISE_DRAWS} draws; inputs: the "
                       f"records of the last {HISTORY[delay]} inputs, noise-aware weights", size=(11, 7))
    xs = np.arange(len(LABELS))
    for k, name in enumerate(METHODS):
        colour, marker = MARKERS[name]
        values = [results[(label, delay)][name] for label in LABELS]
        ax.errorbar(xs + (k - 2) * 0.12, [v.mean() for v in values], yerr=[v.std() for v in values],
                    color=colour, marker=marker, ls="none", ms=7, capsize=2, label=name)
    ax.set_yscale("log")
    ax.set_xticks(xs, [label.replace(", ", ",\n") for label in LABELS])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("NRMSE")
    save_plot(fig, ax, f"{number}_readouts_delay_{delay}")

# 03: QCML against the best other readout
fig, ax = new_plot("QCML against the best other readout on the same inputs",
                   f"ratio of mean test NRMSE (below 1: QCML better); {NOISE_DRAWS} draws",
                   size=(9, 6.5))
ys = np.arange(len(LABELS))[::-1]
for delay, marker, colour in [(1, "o", PCA_COLOUR), (20, "^", QCML_COLOUR)]:
    ratios = []
    for label in LABELS:
        cell = results[(label, delay)]
        rival = min([m for m in METHODS if m != "QCML"], key=lambda m: cell[m].mean())
        ratios.append(cell["QCML"].mean() / cell[rival].mean())
    ax.plot(ratios, ys, marker, color=colour, ms=9, label=f"delay {delay}")
ax.axvline(1, color=INK, lw=1)
ax.set_xscale("log")
ax.set_yticks(ys, LABELS)
ax.grid(axis="y", visible=False)
ax.set_xlabel("QCML NRMSE / best other readout's NRMSE")
save_plot(fig, ax, "03_qcml_vs_best_other")

# 04-05: tables, best in each row marked *
for number, delay in [("04", 1), ("05", 20)]:
    rows = []
    for label in LABELS + ["no reservoir: input history"]:
        cell = (results[(label, delay)] if label in LABELS
                else {m: np.array([v]) for m, v in reference[delay].items()})
        best = min(METHODS, key=lambda m: cell[m].mean())
        rows.append((label, [f"{cell[m].mean():.4f} ± {cell[m].std():.4f}" + (" *" if m == best else "")
                             for m in METHODS]))
    table_plot(f"{number}_readouts_table_delay_{delay}", f"Every readout on the same inputs: x(t + {delay})",
               f"test run NRMSE, mean ± std over {NOISE_DRAWS} draws (no-reservoir row: one draw); * = best",
               ["Setting"] + METHODS, rows, label_width=0.24, size=14,
               note=f"Inputs for every readout: the records of the current and last {HISTORY[delay]} inputs, "
                    "each number weighted by its share of signal. Ridge, diffusion map, kernel ridge and "
                    "the neural network choose their settings on a validation block of the training run; "
                    "QCML uses n = 8 from earlier pilots.")


# ========================================================================================
# EXPLANATION
# Notes on the code above, in the order of its sections. A tag [note k] in the code
# marks the section that note k explains.
# ========================================================================================
#
# [note 0] About this script
#
# In QCML102426_3, "improved QCML" (the records of the last few inputs added, with
# noise-aware weights) beat ridge regression and, with shot noise, every method given all
# 160 probabilities. But ridge is linear and the other nonlinear methods never got the same
# inputs. Here every method gets exactly the improved QCML inputs and has its settings
# chosen on the validation block:
#   - ridge regression (linear)
#   - diffusion map (geometric harmonics, as before)
#   - kernel ridge regression: ridge on a Gaussian similarity between records, the
#     standard nonlinear version of ridge
#   - a small neural network: two hidden layers
#   - supervised QCML (n = 8, settings from the earlier pilots, not validated)
# Settings: vacuum only (10 numbers) and 4 neurons (40) measured, exact and at 9^5 shots;
# all 160 at 9^5 shots; the weak photon-number record at strength 1e9 with 1e4 and 1e6
# runs. Delays 1 and 20, three noise draws each. A reference: the same methods on the
# input history itself, with no reservoir.
#
# Runs in about 12 minutes. The simulation is QCML102426_2's.
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
# [note 6] KERNEL RIDGE REGRESSION
#
# Ridge regression on similarities instead of the numbers themselves: a prediction is a
# weighted sum of exp(-|z - z_i|^2 / (2 width^2)) over the training records z_i, with the
# weights fitted by ridge. The width (a multiple of the median distance between records)
# and the ridge strength are chosen on validation.
#
# [note 7] NEURAL NETWORK
#
# Two hidden layers of tanh units, trained with Adam on squared error. Width and weight
# decay are chosen on validation, with early stopping (training stops when the
# validation error has not improved for NET_PATIENCE passes); the network is then retrained
# on all training rows for the chosen number of passes.
