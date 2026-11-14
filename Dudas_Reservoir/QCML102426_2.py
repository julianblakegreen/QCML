# Quantum Cognitive Machine Learning Project
# QCML102426_2: builds on QCML102426 with a weak, continuous measurement of the reservoir
# Runs in about 20-40 minutes.
# What it does and how: see the notes at the end of this file.


import csv
import itertools
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

PLOT_DIR = "qcml102426_2_plots"   # every plot is saved here as its own PNG
DATA_DIR = "dudas_data"       # their saved results, exported by dudas_export.jl
RUN_CHECKS = True             # Part A's slow checks against the full master equation (~3 min)


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


def master_equation(inputs, drive, offset=0.0, samples=10, levels=9, dephasing=0.0):
    """Joint photon-number distribution P(n_a, n_b) of the reservoir at each sample, from
    the full quantum master equation as their code computes it (photon numbers
    0..levels-1 per oscillator), optionally with photon-number dephasing at rate
    `dephasing` on each oscillator. Shape (inputs, samples, levels, levels). Slow: a few
    seconds per input."""
    lower = sp.diags(np.sqrt(np.arange(1, levels)), 1, format="csr")
    one = sp.identity(levels, format="csr")
    a, b = sp.kron(lower, one, format="csr"), sp.kron(one, lower, format="csr")
    N_a, N_b = (a.T @ a).tocsr(), (b.T @ b).tocsr()
    dim = levels ** 2
    big_one = sp.identity(dim, format="csr")
    lindblad = lambda J: sp.kron(J, J) - 0.5 * sp.kron(big_one, J.T @ J) - 0.5 * sp.kron((J.T @ J).T, big_one)
    H0 = W_A * N_a + W_B * N_b + COUPLING * (a.T @ b + a @ b.T)
    loss = KAPPA_A * lindblad(a) + KAPPA_B * lindblad(b) + dephasing * (lindblad(N_a) + lindblad(N_b))
    rho = np.zeros(dim * dim, complex)
    rho[0] = 1                                             # empty reservoir
    step, hop = INTERVAL / RESOLUTION, RESOLUTION // samples
    out = np.zeros((len(inputs), samples, levels, levels))
    for i, x in enumerate(inputs):
        u = drive * (x + offset)
        H = H0 + 1j * u * (np.sqrt(2 * KAPPA_A) * (a - a.T) + np.sqrt(2 * KAPPA_B) * (b - b.T))
        L = (-1j * (sp.kron(big_one, H) - sp.kron(H.T, big_one)) + loss).tocsc()
        done = 0.0
        for k in range(1, samples + 1):
            t = (k * hop - 1) * step
            rho = expm_multiply(L * (t - done), rho)
            done = t
            P = np.real(rho.reshape(dim, dim, order="F").diagonal()).reshape(levels, levels)
            out[i, k - 1] = P / P.sum()
        if (RESOLUTION - 1) * step > done:
            rho = expm_multiply(L * ((RESOLUTION - 1) * step - done), rho)
    return out


def photon_moments(P):
    """<N_a>, <N_b>, Var N_a, Var N_b from joint distributions P(..., n_a, n_b)."""
    n = np.arange(P.shape[-1])
    p_a, p_b = P.sum(-1), P.sum(-2)
    mean_a, mean_b = p_a @ n, p_b @ n
    return np.stack([mean_a, mean_b, p_a @ n ** 2 - mean_a ** 2, p_b @ n ** 2 - mean_b ** 2], -1)


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
    windows=None, the snapshots at their code's 10 sample times instead (for Part A)."""
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


def linear_readout(train_record, train_target, record, constant=False):
    """Their readout: weights from the Moore-Penrose pseudo-inverse of the training
    record, applied to `record` (cutoff for tiny singular values as in Julia's pinv).
    Their 160 probabilities include P(0,0), close to 1 throughout, which works as a
    constant term; the 20-number weak record has none, so constant=True adds one."""
    if constant:
        train_record = np.column_stack([train_record, np.ones(len(train_record))])
        record = np.column_stack([record, np.ones(len(record))])
    cutoff = min(train_record.shape) * np.finfo(float).eps
    return record @ (np.linalg.pinv(train_record, rtol=cutoff) @ train_target)


# PART A: check the simulation

print("PART A: checking the simulation of the weakly measured reservoir")
load = lambda name: np.loadtxt(os.path.join(DATA_DIR, name + ".csv"), delimiter=",")
series = mackey_glass_dudas()
TRAIN_INPUTS = np.arange(1000)                 # x(0..999), as in QCML102426
TEST_INPUTS = np.arange(1100, 2100)            # x(1100..2099)

if RUN_CHECKS:
    # A1 their Fig. 4 dephasing results: the sine/square inputs (from their saved labels),
    # drive 1.2e6 with offset 0.6, photon-number dephasing 100e6 per second
    square = np.array([1, 1, 1, 1, -1, -1, -1, -1]) / 2
    sine = np.array([0, 0.7071, 1, 0.7071, 0, -0.7071, -1, -0.7071]) / 2
    inputs = np.concatenate([sine if label == 1 else square for label in load("fig4_dephasing_Y")[::8]])
    start = time.time()
    P = master_equation(inputs[:6], 1.2e6, offset=0.6, samples=1, dephasing=100e6)
    gap = np.abs(P[:, 0, :6, :6].reshape(6, -1) - load("fig4_dephasing_X")[:6]).max()
    print(f"   their Fig. 4 dephasing records vs the master equation, 6 inputs: largest "
          f"difference {gap:.1e}  ({time.time() - start:.0f} s)")
    exact = photon_numbers(inputs[:1], 1.2e6, 100e6, offset=0.6, windows=None)[0, 9]
    for levels in (9, 14):
        P = master_equation(inputs[:1], 1.2e6, offset=0.6, samples=1, levels=levels, dephasing=100e6)
        print(f"   their Fig. 4 drive, first input, mean photons in a: master equation with "
              f"{levels - 1}-photon cutoff {photon_moments(P)[0, 0, 0]:.3f}; moments (no cutoff) "
              f"{exact[0]:.3f}")

# A2 the moments against the coherent-state shortcut (no dephasing) and against the full
# master equation (with dephasing) at the Mackey-Glass drive
piece = series[300:304]
coherent = reservoir(piece, MG_DRIVE)                                  # P(n_a, n_b) records
mean_a = coherent[:, 4::16] / coherent[:, 0::16]                       # P(1,0)/P(0,0) = <N_a>
gap = np.abs(photon_numbers(piece, MG_DRIVE, 0.0, windows=None)[:, :, 0] - mean_a).max()
print(f"   moments vs the coherent-state shortcut, no dephasing: largest difference {gap:.1e}")
if RUN_CHECKS:
    for gamma, levels in [(1e6, 9), (1e8, 12)]:
        start = time.time()
        full = photon_moments(master_equation(piece, MG_DRIVE, levels=levels, dephasing=gamma))
        ours = photon_numbers(piece, MG_DRIVE, gamma, windows=None)
        ours[..., 2:] -= ours[..., :2] ** 2                            # <N^2> -> Var N
        print(f"   moments vs master equation ({levels - 1}-photon cutoff), dephasing "
              f"{gamma:.0e}: largest difference in <N>, Var N {np.abs(ours - full).max():.1e}"
              f"  ({time.time() - start:.0f} s)")


# THE METHODS (as in QCML102426: QCML, supervised QCML, QCML trained to fill in, PCA
# and the diffusion map)

EPOCHS = 150
BATCH = 64


# QCML

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


# QCML TRAINED TO FILL IN (the same idea applied to missing measurements)  [note 5]

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


# PCA (as in QCML102326 Part B)  [note 6]

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



# PARTS B-D: SETTINGS  [note 7]

ETA = 0.5                                           # detector efficiency
GAMMAS = np.array([1e5, 1e6, 1e7, 3e7, 1e8, 3e8, 1e9])   # measurement strengths, per second
RUNS = np.array([1e2, 1e4, 1e6, 1e8])               # runs of the input sequence averaged
NOISE_DRAWS = 2                                     # independent noise draws per setting
PREDICT_DELAYS = [1, 20]
FILL_RUNS = 1e6                                     # Part C: runs averaged
SHOTS = 9 ** np.array([3, 5, 7, 9])                 # Part D: their protocol's shots per number
REPLAY = 10                                         # Part D: inputs replayed before each of their
                                                    # measurements (the reservoir forgets within
                                                    # about 10: its amplitude falls by e^-8.5)
INPUT_TIME = INTERVAL                               # each input lasts 100 ns
SUP_N = 8                                           # supervised QCML (as in QCML102426)
FILLER_N = 16                                       # QCML trained to fill in (as in QCML102426;
                                                    # no L-BFGS polish, as the records are noisy)
DM_NEIGHBOUR_OPTIONS = (30, 100, 300)
DM_MAX_M = 600
TRAIN_ROWS = np.arange(WASHOUT, 1000)
FIT_ROWS, VAL_ROWS = np.arange(WASHOUT, 700), np.arange(700, 1000)
TEST_ROWS = np.arange(WASHOUT - 1, 1000)
PREDICT_METHODS = ["Linear readout (theirs)", "PCA", "Diffusion map", "Supervised QCML"]
FILL_METHODS = ["PCA", "Diffusion map", "QCML trained to fill in"]

rms = lambda a: float(np.sqrt(np.mean(a ** 2)))


def best_pca(train, values, val_truth, measured, target_columns, ks):
    """The k whose fill-in of `target_columns` from the `measured` ones is closest on the
    validation block."""
    filled = pca_predict(train, values, list(measured), ks)
    return int(ks[np.argmin([rms(f[:, target_columns] - val_truth) for f in filled])])


def best_diffusion_map(fit_inputs, fit_values, val_inputs, val_truth):
    """((neighbours, m), validation error) of the diffusion map closest on the validation
    block."""
    best = (np.inf, None)
    for neighbours in DM_NEIGHBOUR_OPTIONS:
        errors = DiffusionMap(fit_inputs, neighbours).rebuild_errors(fit_values, val_inputs,
                                                                     val_truth, DM_MAX_M)
        if errors.min() < best[0]:
            best = (errors.min(), (neighbours, int(errors.argmin()) + 1))
    return best[1], float(np.sqrt(best[0]))


def predict_all(train, test, delay, seed, constant):
    """Test-run NRMSE of every method predicting x(t + delay) from the records `train` and
    `test` (one row per input), and the diffusion map's validation error."""
    y, y_test = series[TRAIN_INPUTS + delay], series[TEST_INPUTS + delay][TEST_ROWS]
    y_mu, y_sd = y[TRAIN_ROWS].mean(), y[TRAIN_ROWS].std()
    Y = (y - y_mu) / y_sd
    score = lambda z: nrmse(z * y_sd + y_mu, y_test)
    mu, sd = train[TRAIN_ROWS].mean(0), train[TRAIN_ROWS].std(0)
    Z, Z_test = (train - mu) / sd, (test[TEST_ROWS] - mu) / sd
    columns = np.arange(train.shape[1])
    joint = np.column_stack([Z, Y])
    k = best_pca(joint[FIT_ROWS], joint[VAL_ROWS][:, columns], Y[VAL_ROWS, None], columns,
                 [len(columns)], np.arange(1, len(columns) + 2))
    (neighbours, m), dm_validation = best_diffusion_map(Z[FIT_ROWS], Y[FIT_ROWS, None],
                                                        Z[VAL_ROWS], Y[VAL_ROWS, None])
    supervised = train_supervised(Z[TRAIN_ROWS], Y[TRAIN_ROWS], SUP_N,
                                  torch.nn.functional.mse_loss, seed=seed)
    return {"Linear readout (theirs)": nrmse(linear_readout(train[TRAIN_ROWS], y[TRAIN_ROWS],
                                                            test[TEST_ROWS], constant), y_test),
            "PCA": score(pca_predict(joint[TRAIN_ROWS], Z_test, list(columns), [k])[0][:, -1]),
            "Diffusion map": score(DiffusionMap(Z[TRAIN_ROWS], neighbours).rebuild(
                Y[TRAIN_ROWS, None], Z_test, np.array([m]))[0][:, 0]),
            "Supervised QCML": score(supervised_forecast(supervised, Z_test))}, dm_validation


def projective(record, shots, rng):
    """Their measurement with a finite number of shots: each probability p gets Gaussian
    noise of size sqrt(p (1 - p) / shots), as in their Fig. 5."""
    return record + rng.standard_normal(record.shape) * np.sqrt(np.abs(record * (1 - record) / shots))


def as_arrays(results):
    return {key: {name: np.array(v) for name, v in scores.items()} for key, scores in results.items()}


mean_guess = {delay: nrmse(np.full(len(TEST_ROWS), series[TRAIN_INPUTS + delay][TRAIN_ROWS].mean()),
                           series[TEST_INPUTS + delay][TEST_ROWS]) for delay in PREDICT_DELAYS}


# PART B: measurement strength x number of runs

print(f"\nPART B: weak photon-number records, {len(GAMMAS)} strengths x {len(RUNS)} run counts "
      f"x {NOISE_DRAWS} draws (efficiency {ETA})")
start = time.time()
numbers = {gamma: (photon_numbers(series[TRAIN_INPUTS], MG_DRIVE, gamma),
                   photon_numbers(series[TEST_INPUTS], MG_DRIVE, gamma)) for gamma in GAMMAS}
print(f"   photon numbers simulated ({time.time() - start:.0f} s)")
for gamma in GAMMAS:
    test_numbers = numbers[gamma][1][TEST_ROWS]
    print(f"   strength {gamma:.0e}: mean photons a {test_numbers[:, :, 0].mean():.3f}, "
          f"b {test_numbers[:, :, 1].mean():.3f}")

strength_map, dm_validation = {}, {}
for draw in range(NOISE_DRAWS):
    start = time.time()
    for runs in RUNS:
        for gamma in GAMMAS:
            rng = np.random.default_rng(draw)
            train = weak_record(numbers[gamma][0], gamma, runs, rng)
            test = weak_record(numbers[gamma][1], gamma, runs, rng)
            for delay in PREDICT_DELAYS:
                scores, validation = predict_all(train, test, delay, draw, constant=True)
                cell = strength_map.setdefault((delay, runs, gamma), {name: [] for name in PREDICT_METHODS})
                for name, value in scores.items():
                    cell[name].append(value)
                dm_validation.setdefault((delay, runs, gamma), []).append(validation)
    print(f"   draw {draw + 1} done ({time.time() - start:.0f} s)")
strength_map = as_arrays(strength_map)
chosen_gamma = {(delay, runs): GAMMAS[np.argmin([np.mean(dm_validation[(delay, runs, g)]) for g in GAMMAS])]
                for delay in PREDICT_DELAYS for runs in RUNS}

with open("qcml102426_2_strength_map.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["delay", "runs", "strength"] + [f"{name} {stat}" for name in PREDICT_METHODS
                                                     for stat in ("mean", "std")])
    for (delay, runs, gamma), scores in strength_map.items():
        writer.writerow([delay, runs, gamma] + [v for name in PREDICT_METHODS
                                                for v in (scores[name].mean(), scores[name].std())])
for delay in PREDICT_DELAYS:
    print(f"\n   delay {delay} (NRMSE, mean of {NOISE_DRAWS} draws; predicting the average gives "
          f"{mean_guess[delay]:.3f}):")
    print("   runs   strength  " + "".join(f"{name.split(' (')[0]:>18}" for name in PREDICT_METHODS))
    for runs in RUNS:
        for gamma in GAMMAS:
            mark = "  <- chosen" if chosen_gamma[(delay, runs)] == gamma else ""
            print(f"   {runs:.0e}  {gamma:.0e}  " + "".join(
                f"{strength_map[(delay, runs, gamma)][name].mean():18.4f}" for name in PREDICT_METHODS) + mark)
print("saved to qcml102426_2_strength_map.csv")


# PART C: rebuild their destructive measurement from the weak record  [note 8]

print(f"\nPART C: rebuilding their 160 probabilities from the weak record ({FILL_RUNS:.0e} runs)")
exact_train, exact_test = reservoir(series[TRAIN_INPUTS], MG_DRIVE), reservoir(series[TEST_INPUTS], MG_DRIVE)
spread = exact_train[TRAIN_ROWS].std(0)
fill, fill_readout = {}, {}
for gamma in GAMMAS:
    start = time.time()
    for draw in range(NOISE_DRAWS):
        rng = np.random.default_rng(draw)
        train = weak_record(numbers[gamma][0], gamma, FILL_RUNS, rng)
        test = weak_record(numbers[gamma][1], gamma, FILL_RUNS, rng)
        mu, sd = train[TRAIN_ROWS].mean(0), train[TRAIN_ROWS].std(0)
        p_mu, p_sd = exact_train[TRAIN_ROWS].mean(0), exact_train[TRAIN_ROWS].std(0)
        Z, Z_test = (train - mu) / sd, (test[TEST_ROWS] - mu) / sd
        P = (exact_train - p_mu) / p_sd
        joint = np.column_stack([Z, P])
        measured, hidden = np.arange(Z.shape[1]), np.arange(Z.shape[1], joint.shape[1])
        k = best_pca(joint[FIT_ROWS], joint[VAL_ROWS][:, measured], joint[VAL_ROWS][:, hidden],
                     measured, hidden, np.arange(1, len(measured) + 1))
        (neighbours, m), _ = best_diffusion_map(Z[FIT_ROWS], P[FIT_ROWS], Z[VAL_ROWS], P[VAL_ROWS])
        filler = train_filler(joint[TRAIN_ROWS], measured, FILLER_N, seed=draw, polish=False)
        dm = DiffusionMap(Z[TRAIN_ROWS], neighbours)
        fillers = {"PCA": lambda z: pca_predict(joint[TRAIN_ROWS], z, list(measured), [k])[0][:, hidden],
                   "Diffusion map": lambda z: dm.rebuild(P[TRAIN_ROWS], z, np.array([m]))[0],
                   "QCML trained to fill in": lambda z: qcml_fill(filler, z, measured)[:, hidden]}
        for name, fill_from in fillers.items():
            probabilities = fill_from(Z_test) * p_sd + p_mu
            train_probabilities = fill_from(Z[TRAIN_ROWS]) * p_sd + p_mu
            fill.setdefault(gamma, {n: [] for n in FILL_METHODS})[name].append(
                rms(probabilities - exact_test[TEST_ROWS]) / rms(spread))
            for delay in PREDICT_DELAYS:
                weights = (np.linalg.pinv(train_probabilities, rtol=1e-6)
                           @ series[TRAIN_INPUTS + delay][TRAIN_ROWS])
                fill_readout.setdefault((gamma, delay), {n: [] for n in FILL_METHODS})[name].append(
                    nrmse(probabilities @ weights, series[TEST_INPUTS + delay][TEST_ROWS]))
    print(f"   strength {gamma:.0e}: fill-in error " + "   ".join(
        f"{name} {np.mean(fill[gamma][name]):.4f}" for name in FILL_METHODS)
        + "   their readout on the QCML fill-in, delay 1 / 20: " + " / ".join(
        f"{np.mean(fill_readout[(gamma, d)]['QCML trained to fill in']):.4f}" for d in PREDICT_DELAYS)
        + f"  ({time.time() - start:.0f} s)")
fill, fill_readout = as_arrays(fill), as_arrays(fill_readout)
with open("qcml102426_2_fill_in.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["strength"] + [f"fill-in {name} {stat}" for name in FILL_METHODS for stat in ("mean", "std")]
                    + [f"readout delay {d} on {name} {stat}" for d in PREDICT_DELAYS
                       for name in FILL_METHODS for stat in ("mean", "std")])
    for gamma in GAMMAS:
        writer.writerow([gamma] + [v for name in FILL_METHODS for v in (fill[gamma][name].mean(), fill[gamma][name].std())]
                        + [v for d in PREDICT_DELAYS for name in FILL_METHODS
                           for v in (fill_readout[(gamma, d)][name].mean(), fill_readout[(gamma, d)][name].std())])
print("saved to qcml102426_2_fill_in.csv")


# PART D: cost  [note 9]

print(f"\nPART D: cost, their projective protocol at {len(SHOTS)} shot counts")
weak_time = lambda runs: runs * (len(TRAIN_INPUTS) + len(TEST_INPUTS)) * INPUT_TIME
projective_time = lambda shots: shots * 10 * (len(TRAIN_INPUTS) + len(TEST_INPUTS)) * REPLAY * INPUT_TIME
theirs = {}
for shots in SHOTS:
    start = time.time()
    for draw in range(NOISE_DRAWS):
        rng = np.random.default_rng(draw)
        train, test = projective(exact_train, shots, rng), projective(exact_test, shots, rng)
        for delay in PREDICT_DELAYS:
            scores, _ = predict_all(train, test, delay, draw, constant=False)
            cell = theirs.setdefault((delay, shots), {name: [] for name in PREDICT_METHODS})
            for name, value in scores.items():
                cell[name].append(value)
    print(f"   {shots:.1e} shots done ({time.time() - start:.0f} s)")
theirs = as_arrays(theirs)

cost_rows = {delay: [] for delay in PREDICT_DELAYS}
with open("qcml102426_2_cost.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["delay", "protocol", "experiment time (s)"] + [f"{name} {stat}" for name in PREDICT_METHODS
                                                                   for stat in ("mean", "std")])
    for delay in PREDICT_DELAYS:
        print(f"\n   delay {delay}: protocol, experiment time, NRMSE of each method")
        for runs in RUNS:
            gamma = chosen_gamma[(delay, runs)]
            label = f"weak, {runs:.0e} runs (strength {gamma:.0e})"
            scores = strength_map[(delay, runs, gamma)]
            cost_rows[delay].append((label, weak_time(runs), scores))
        for shots in SHOTS:
            label = f"theirs, 9^{round(np.log(shots) / np.log(9))} shots"
            cost_rows[delay].append((label, projective_time(shots), theirs[(delay, shots)]))
        for label, seconds, scores in cost_rows[delay]:
            print(f"   {label:<38} {seconds:10.3g} s  " + "   ".join(
                f"{name.split(' (')[0]} {scores[name].mean():.4f}" for name in PREDICT_METHODS))
            writer.writerow([delay, label, seconds] + [v for name in PREDICT_METHODS
                                                       for v in (scores[name].mean(), scores[name].std())])
print("saved to qcml102426_2_cost.csv")


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
                "QCML trained to fill in": (QCML_COLOUR, "^")}
RUN_COLOURS = SEQUENTIAL(np.linspace(0.15, 1, len(RUNS)))
print(f"\nplots saved to {PLOT_DIR}:")

# 01: what the measurement does to the reservoir, and the noise of one run
fig, ax = new_plot("Back-action: measuring the photon number heats this reservoir",
                   "test run averages; noise = one run's record noise in a 10 ns window "
                   f"(efficiency {ETA})")
tau = (RESOLUTION - 1) * INTERVAL / RESOLUTION / 10
photons_a = [numbers[g][1][TEST_ROWS][:, :, 0].mean() for g in GAMMAS]
photons_b = [numbers[g][1][TEST_ROWS][:, :, 1].mean() for g in GAMMAS]
variation = [numbers[g][1][TEST_ROWS][:, :, 0].std() for g in GAMMAS]
one_run = [np.sqrt(np.mean(numbers[g][1][TEST_ROWS][:, :, 2] - numbers[g][1][TEST_ROWS][:, :, 0] ** 2)
                   + 1 / (4 * ETA * g * tau)) for g in GAMMAS]
ax.plot(GAMMAS, photons_a, "o-", color=PCA_COLOUR, lw=2, label="mean photons in a")
ax.plot(GAMMAS, photons_b, "s-", color=DM_COLOUR, lw=2, label="mean photons in b")
ax.plot(GAMMAS, variation, "o--", color=PCA_COLOUR, lw=1.5, mfc="white",
        label="how much the photons in a vary with the input (the signal)")
ax.plot(GAMMAS, one_run, "^-", color=GREY, lw=2, label="noise of one run's record (a)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("measurement strength Γ (per second)")
ax.set_ylabel("photons")
save_plot(fig, ax, "01_backaction_and_noise")

# 02: a weak record
gamma, runs, inputs = 1e8, 1e4, np.arange(200, 230)
record = weak_record(numbers[gamma][1], gamma, runs, np.random.default_rng(0)).reshape(-1, 10, 2)
times = (inputs[:, None] + (np.arange(10)[None, :] + 0.5) / 10).ravel() * INPUT_TIME * 1e6
fig, ax = new_plot("A weak record: photons in oscillator a, 10 windows per input",
                   f"test run, strength Γ = {gamma:.0e} per second, {runs:.0e} runs averaged",
                   size=(10, 6))
ax.plot(times, record[inputs, :, 0].ravel(), "o", color=PCA_COLOUR, ms=3, label="measured record")
ax.plot(times, numbers[gamma][1][inputs, :, 0].ravel(), color=INK, lw=1.5, label="exact mean photon number")
ax.set_xlabel("time (μs)")
ax.set_ylabel("photons in a")
save_plot(fig, ax, "02_weak_record_example")

# 03: prediction error against measurement strength, all methods, 1e6 runs
for number, delay in [("03", 20)]:
    fig, ax = new_plot(f"Predicting x(t + {delay}) from the weak record ({FILL_RUNS:.0e} runs)",
                       f"test run NRMSE (log scale), mean ± std over {NOISE_DRAWS} draws")
    for name in PREDICT_METHODS:
        colour, marker = METHOD_STYLE[name]
        values = [strength_map[(delay, FILL_RUNS, g)][name] for g in GAMMAS]
        label = name + (" + constant" if name.startswith("Linear") else "")
        ax.errorbar(GAMMAS, [v.mean() for v in values], yerr=[v.std() for v in values], color=colour,
                    marker=marker, lw=2, ms=6, capsize=3, label=label)
    ax.axhline(mean_guess[delay], color=GREY, ls=":", label="always predicting the average")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("measurement strength Γ (per second)")
    ax.set_ylabel("NRMSE")
    save_plot(fig, ax, f"{number}_prediction_vs_strength")

# 04: supervised QCML across strength and runs
delay = 20
fig, ax = new_plot(f"Supervised QCML: predicting x(t + {delay}) across strength and runs",
                   f"test run NRMSE (log scale), mean over {NOISE_DRAWS} draws")
for runs, colour in zip(RUNS, RUN_COLOURS):
    ax.plot(GAMMAS, [strength_map[(delay, runs, g)]["Supervised QCML"].mean() for g in GAMMAS],
            "^-", color=colour, lw=2, ms=6, label=f"{runs:.0e} runs")
ax.axhline(mean_guess[delay], color=GREY, ls=":", label="always predicting the average")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("measurement strength Γ (per second)")
ax.set_ylabel("NRMSE")
save_plot(fig, ax, "04_supervised_qcml_strength_and_runs")

# 05: cost comparison
fig, ax = new_plot(f"Error against experiment time: weak record vs their protocol (x(t + {delay}))",
                   "total reservoir time, training + test runs; weak: strength chosen on validation")
for name in ["Linear readout (theirs)", "Diffusion map", "Supervised QCML"]:
    colour, marker = METHOD_STYLE[name]
    weak = [strength_map[(delay, r, chosen_gamma[(delay, r)])][name] for r in RUNS]
    ax.errorbar([weak_time(r) for r in RUNS], [v.mean() for v in weak], yerr=[v.std() for v in weak],
                color=colour, marker=marker, lw=2, ms=6, capsize=3,
                label=f"weak record, {name}" + (" + constant" if name.startswith("Linear") else ""))
    projective_scores = [theirs[(delay, s)][name] for s in SHOTS]
    ax.errorbar([projective_time(s) for s in SHOTS], [v.mean() for v in projective_scores],
                yerr=[v.std() for v in projective_scores], color=colour, marker=marker, lw=2, ms=6,
                ls="--", mfc="white", capsize=3, label=f"their protocol, {name}")
ax.axhline(mean_guess[delay], color=GREY, ls=":", label="always predicting the average")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("experiment time (seconds)")
ax.set_ylabel("NRMSE")
save_plot(fig, ax, "05_cost_comparison")

# 06-07: prediction tables, strength chosen on validation for each number of runs
for number, delay in [("06", 1), ("07", 20)]:
    rows = [(f"{runs:.0e} runs, Γ = {chosen_gamma[(delay, runs)]:.0e}",
             [strength_map[(delay, runs, chosen_gamma[(delay, runs)])][name] for name in PREDICT_METHODS])
            for runs in RUNS]
    table_plot(f"{number}_prediction_table_delay_{delay}", f"Predicting x(t + {delay}) from the weak record",
               f"test run NRMSE, mean ± std over {NOISE_DRAWS} draws; predicting the average gives "
               f"{mean_guess[delay]:.3f}",
               ["Runs averaged, strength"] + ["Linear readout\n(theirs + constant)", "PCA", "Diffusion map",
                                              "Supervised QCML"], rows, label_width=0.3, size=12, digits=4,
               note="The strength Γ (per second) for each number of runs is the one with the lowest "
                    "diffusion-map validation error. Records: 20 numbers per input (photons in a "
                    "and b, 10 windows each).")

# 08: rebuilding their probabilities from the weak record
rows = [(f"Γ = {gamma:.0e}", [fill[gamma][name] for name in FILL_METHODS]
         + [fill_readout[(gamma, 20)][name] for name in FILL_METHODS]) for gamma in GAMMAS]
table_plot("08_fill_in_table", "Rebuilding their 160 probabilities from the weak record",
           f"test run, {FILL_RUNS:.0e} runs; mean ± std over {NOISE_DRAWS} draws",
           ["Strength"] + [f"Fill-in error\n{name.replace(' trained to fill in', '')}" for name in FILL_METHODS]
           + [f"Their readout on\n{name.replace(' trained to fill in', '')} fill-in, d = 20" for name in FILL_METHODS],
           rows, label_width=0.12, size=14, digits=4,
           note="Fill-in error: RMS distance from the exact probabilities (undisturbed reservoir, "
                "unlimited shots) / their typical spread. Readout: their pseudo-inverse readout trained "
                "on the filled-in training probabilities, applied to the filled-in test ones (NRMSE). "
                "QCML: trained to fill in.")

# 09: cost table
for delay in [20]:
    rows = [(label, [f"{seconds:.3g} s"] + [scores[name] for name in PREDICT_METHODS])
            for label, seconds, scores in cost_rows[delay]]
    table_plot("09_cost_table", f"Cost and error: weak record vs their protocol (x(t + {delay}))",
               f"test run NRMSE, mean ± std over {NOISE_DRAWS} draws",
               ["Protocol", "Experiment\ntime", "Linear readout\n(theirs)", "PCA", "Diffusion map",
                "Supervised QCML"], rows, label_width=0.3, size=13, digits=4,
               note=f"Experiment time: weak = runs x 2000 inputs x 100 ns; theirs = shots x 10 samples x "
                    f"2000 inputs, each after replaying {REPLAY} inputs. Weak records use their readout "
                    "with a constant added.")


# ========================================================================================
# EXPLANATION
# Notes on the code above, in the order of its sections. A tag [note k] in the code
# marks the section that note k explains.
# ========================================================================================
#
# [note 0] About this script
#
# QCML102426 used the measurement of Dudas et al. (2023): the photon-number probabilities
# of their two-oscillator reservoir, measured projectively. That measurement destroys the
# state, so every one of the 10 samples of every input needs its own replay of the input,
# repeated for every shot. Here the reservoir is instead measured weakly and
# continuously: the photon number in each oscillator is monitored during the run (as a
# qubit coupled to a resonator would), which disturbs the reservoir only a little and
# needs no replays, but gives a noisy record. Measuring more weakly disturbs less and
# gives more noise. The question: with QCML reading the record, which measurement
# strength and how many runs work best, and how does that compare with their protocol?
#
# PART A  Check the simulation of the weakly measured reservoir: against their own
#         dephasing results (their Fig. 4) and against the full master equation.
# PART B  Measurement strength x number of runs: predict Mackey-Glass from the weak record
#         with their readout, PCA, a diffusion map and supervised QCML.
# PART C  Rebuild what their destructive measurement would have given (the 160
#         photon-number probabilities) from the weak record: PCA, diffusion map, QCML.
# PART D  Cost: prediction error against total experiment time, weak vs their protocol.
#
# Runs in about 20-40 minutes. Needs dudas_data/ from dudas_export.jl (already there).
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
# One surprise (Part A): dephasing HEATS this reservoir. Their drive is constant within
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
# [note 5] QCML TRAINED TO FILL IN (the same idea applied to missing measurements)
#
# Plain QCML learns all 160 numbers together, then fills in the unmeasured ones from a
# ground state built from the measured numbers' matrices alone (ground_state_from), a
# job it was never trained for. Since we choose which numbers to measure, QCML can be
# trained for exactly that job: the ground state is built from the measured numbers'
# matrices only, and the loss asks every matrix's expected value to match the full
# record, so the unmeasured numbers' matrices learn to predict them. It is supervised
# QCML with one target observable per unmeasured number.
#
# [note 6]
# PCA (as in QCML102326 Part B): the point on the flat k-dimensional subspace of the
# training records that best matches the measured numbers (least squares); its other
# coordinates are the filled-in numbers or the prediction.
#
# DIFFUSION MAP (as in QCML102326): built on the measured numbers of the training
# records, with each point's kernel width its distance to its k-th nearest neighbour.
# Anything known on the training records (the unmeasured numbers, or the target) is
# rebuilt at a new record from the walk's first m eigenvectors (geometric harmonics).
#
# [note 7] PARTS B-D: SETTINGS
#
# Every method learns from the training run (inputs 30-999) and is scored on the test run
# (from the 30th input, as in their scoring), never used for choosing. PCA's k and the
# diffusion map's neighbours and m are chosen on a validation block inside the training run
# (fit on inputs 30-699, score on 700-999); QCML's settings come from QCML102426's pilots.
# The measurement strength used for each number of runs (Parts B and D) is the one whose
# diffusion-map validation error is lowest, so it is chosen without the test run.
#
# [note 8] PART C: rebuild their destructive measurement from the weak record
#
# Target: the 160 photon-number probabilities their projective measurement gives with
# unlimited shots, on the undisturbed reservoir (the coherent-state shortcut). Each method
# learns, on the training run, how they relate to the weak record (taken on a separate,
# weakly measured run of the same inputs), and fills them in on the test run from the
# record alone. Fill-in error: RMS distance from the exact probabilities, divided by their
# typical spread, as in QCML102426. Their readout is then trained on the filled-in
# training probabilities and applied to the filled-in test ones: the full pipeline of a
# weakly measured reservoir. (Trained on the exact probabilities instead, its weights are
# so large that the small fill-in errors blow its predictions up.) Singular values below
# 1e-6 of the largest are cut, since QCML computes in single precision.
#
# [note 9] PART D: cost
#
# Total time the reservoir runs, for the training and the test run together (2000 inputs):
#   weak         runs x 2000 inputs x 100 ns
#   theirs       shots x 10 samples x 2000 inputs measurements, each after replaying the last
#                REPLAY inputs: shots x 10 x 2000 x REPLAY x 100 ns
# Their protocol is scored with all 160 probabilities and finite shots, as in their Fig. 5.
