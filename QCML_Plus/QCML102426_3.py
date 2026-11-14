# Quantum Cognitive Machine Learning Project
# QCML102426_3: builds on QCML102426 and QCML102426_2 (the Dudas et al. reservoir)
# Runs in about 30 minutes.
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

PLOT_DIR = "qcml102426_3_plots"   # every plot is saved here as its own PNG


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


# PCA (as in QCML102326)  [note 5]

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



# SETTINGS  [note 6]

ETA = 0.5                                       # detector efficiency of the weak record
SUP_N = 8                                       # supervised QCML Hilbert-space dimension
HISTORY = {1: 4, 20: 6}                         # improved methods: previous inputs whose
                                                # records are added, per delay (best of 1, 2,
                                                # 4, 6 in a pilot; Part C shows all four)
PREDICT_DELAYS = [1, 20]
NOISE_DRAWS = 2
SHOTS = [("exact", None), ("9^9 shots", 9 ** 9), ("9^5 shots", 9 ** 5)]
WEAK_GAMMAS = [1e8, 1e9]                        # Part B: measurement strengths (per second)
WEAK_RUNS = [1e2, 1e4, 1e6]                     # Part B: runs averaged (results level off by 1e6)
RIDGE_STRENGTHS = np.logspace(-6, 4, 41)
DM_NEIGHBOUR_OPTIONS = (30, 100, 300)
DM_MAX_M = 600
TRAIN_ROWS = np.arange(WASHOUT, 1000)
FIT_ROWS, VAL_ROWS = np.arange(WASHOUT, 700), np.arange(700, 1000)
TEST_ROWS = np.arange(WASHOUT - 1, 1000)
TRAIN_INPUTS, TEST_INPUTS = np.arange(1000), np.arange(1100, 2100)
mse = torch.nn.functional.mse_loss


def neuron(n_a, n_b):
    return n_a * 4 + n_b                        # position of P(n_a, n_b) within a sample


BUDGETS = [("all 160", [neuron(i, j) for i in range(4) for j in range(4)]),
           ("40: 4 neurons", [neuron(0, 0), neuron(0, 1), neuron(1, 0), neuron(1, 1)]),
           ("10: vacuum only", [neuron(0, 0)])]
BUDGETS = [(label, np.array([k * 16 + c for k in range(10) for c in neurons]))
           for label, neurons in BUDGETS]


# TOOLS

def projective(record, shots, rng):
    """Their measurement with a finite number of shots (their Fig. 5): each probability p
    gets Gaussian noise of size sqrt(p (1 - p) / shots)."""
    if shots is None:
        return record
    return record + rng.standard_normal(record.shape) * np.sqrt(np.abs(record * (1 - record) / shots))


def projective_noise(train, shots):
    """Each column's shot-noise variance, estimated from the measured training values (an
    experimenter knows the shot count, so knows this)."""
    if shots is None:
        return np.zeros(train.shape[1])
    return np.mean(np.abs(train[TRAIN_ROWS] * (1 - train[TRAIN_ROWS])) / shots, 0)


def weak_noise(numbers, gamma, runs):
    """Each column's noise variance in the weak record (known from the measurement model)."""
    tau = (RESOLUTION - 1) * INTERVAL / RESOLUTION / numbers.shape[1]
    spread = np.maximum(numbers[TRAIN_ROWS, :, 2:] - numbers[TRAIN_ROWS, :, :2] ** 2, 0)
    return ((spread + 1 / (4 * ETA * gamma * tau)) / runs).mean(0).ravel()


def with_history(record, previous):
    """Each row followed by the rows of the `previous` inputs before it (time-delay inputs).
    Rows before the start repeat the first row; they fall in the washout anyway."""
    return np.hstack([np.vstack([np.repeat(record[:1], k, 0), record[:len(record) - k]])
                      for k in range(previous + 1)])


def standardise(train, test, noise=None):
    """z-score with the training run's mean and spread. With `noise` (each column's noise
    variance), each column is also multiplied by sqrt(1 - noise / variance), the share of
    its spread that is signal, so columns that are mostly noise count for little."""
    mu, sd = train[TRAIN_ROWS].mean(0), train[TRAIN_ROWS].std(0)
    Z, Z_test = (train - mu) / sd, (test - mu) / sd
    if noise is not None:
        weight = np.sqrt(np.clip(1 - noise / sd ** 2, 0, 1))
        Z, Z_test = Z * weight, Z_test * weight
    return Z, Z_test


def ridge(Z, y, Z_test):
    """Ridge regression with a constant term: weights minimising squared error plus
    strength x |weights|^2. The strength is chosen on the validation block, then refitted
    on all training rows. Returns predictions for the test rows."""
    def fit(rows, strength):
        centre, offset = Z[rows].mean(0), y[rows].mean()
        A = Z[rows] - centre
        weights = np.linalg.solve(A.T @ A + strength * np.eye(A.shape[1]), A.T @ (y[rows] - offset))
        return lambda X: (X - centre) @ weights + offset
    errors = [np.mean((fit(FIT_ROWS, s)(Z[VAL_ROWS]) - y[VAL_ROWS]) ** 2) for s in RIDGE_STRENGTHS]
    return fit(TRAIN_ROWS, RIDGE_STRENGTHS[int(np.argmin(errors))])(Z_test[TEST_ROWS])


def supervised_qcml(Z, y, Z_test, n=SUP_N, seeds=(0,), **options):
    """Supervised QCML on z-scored inputs and target; the average over `seeds` (an
    ensemble when there are several). Returns predictions for the test rows."""
    mu, sd = y[TRAIN_ROWS].mean(), y[TRAIN_ROWS].std()
    predictions = [supervised_forecast(train_supervised(Z[TRAIN_ROWS], (y[TRAIN_ROWS] - mu) / sd,
                                                        n, mse, seed=s, **options), Z_test[TEST_ROWS])
                   for s in seeds]
    return np.mean(predictions, 0) * sd + mu


def target(delay):
    return series[TRAIN_INPUTS + delay], series[TEST_INPUTS + delay][TEST_ROWS]


def compare(train, test, noise, delay, seed, extras=False):
    """Test-run NRMSE of ridge and QCML, plain and improved (time-delay inputs plus
    noise-aware weights), predicting x(t + delay) from the records `train` and `test`.
    With extras, also their readout, PCA and the diffusion map on the plain records."""
    y, y_test = target(delay)
    score = lambda p: nrmse(p, y_test)
    plain = standardise(train, test)
    previous = HISTORY[delay]
    improved = standardise(with_history(train, previous), with_history(test, previous),
                           np.tile(noise, previous + 1))
    out = {"Ridge": score(ridge(plain[0], y, plain[1])),
           "Ridge, improved": score(ridge(improved[0], y, improved[1])),
           "QCML": score(supervised_qcml(plain[0], y, plain[1], seeds=(seed,))),
           "QCML, improved": score(supervised_qcml(improved[0], y, improved[1], seeds=(seed,)))}
    if extras:
        Z, Z_test = plain
        mu, sd = y[TRAIN_ROWS].mean(), y[TRAIN_ROWS].std()
        Y = (y - mu) / sd
        joint = np.column_stack([Z, Y])
        columns = np.arange(Z.shape[1])
        filled = pca_predict(joint[FIT_ROWS], Z[VAL_ROWS], list(columns), np.arange(1, len(columns) + 2))
        k = 1 + int(np.argmin([np.mean((f[:, -1] - Y[VAL_ROWS]) ** 2) for f in filled]))
        out["PCA"] = score(pca_predict(joint[TRAIN_ROWS], Z_test[TEST_ROWS], list(columns), [k])[0][:, -1] * sd + mu)
        best = (np.inf, None)
        for neighbours in DM_NEIGHBOUR_OPTIONS:
            errors = DiffusionMap(Z[FIT_ROWS], neighbours).rebuild_errors(
                Y[FIT_ROWS, None], Z[VAL_ROWS], Y[VAL_ROWS, None], DM_MAX_M)
            if errors.min() < best[0]:
                best = (errors.min(), (neighbours, int(errors.argmin()) + 1))
        neighbours, m = best[1]
        out["Diffusion map"] = score(DiffusionMap(Z[TRAIN_ROWS], neighbours).rebuild(
            Y[TRAIN_ROWS, None], Z_test[TEST_ROWS], np.array([m]))[0][:, 0] * sd + mu)
        out["Linear readout (theirs)"] = score(linear_readout(train[TRAIN_ROWS], y[TRAIN_ROWS],
                                                              test[TEST_ROWS]))
    return out


def collect(store, key, results):
    cell = store.setdefault(key, {})
    for name, value in results.items():
        cell.setdefault(name, []).append(value)


def as_arrays(store):
    return {key: {name: np.array(v) for name, v in cell.items()} for key, cell in store.items()}


series = mackey_glass_dudas()
exact_train, exact_test = reservoir(series[TRAIN_INPUTS], MG_DRIVE), reservoir(series[TEST_INPUTS], MG_DRIVE)


# PART A: projective measurement with missing probabilities

print(f"PART A: ridge vs QCML, {len(BUDGETS)} measurement budgets x {len(SHOTS)} noise levels x "
      f"{NOISE_DRAWS} draws")
part_a = {}
for shots_label, shots in SHOTS:
    for draw in range(NOISE_DRAWS):
        start = time.time()
        rng = np.random.default_rng(draw)
        train, test = projective(exact_train, shots, rng), projective(exact_test, shots, rng)
        noise = projective_noise(train, shots)
        for budget_label, measured in BUDGETS:
            for delay in PREDICT_DELAYS:
                results = compare(train[:, measured], test[:, measured], noise[measured], delay, draw,
                                  extras=len(measured) == 160)
                collect(part_a, (shots_label, budget_label, delay), results)
        print(f"   {shots_label}, draw {draw + 1} done ({time.time() - start:.0f} s)")
part_a = as_arrays(part_a)

FULL = BUDGETS[0][0]
best_full = {}                                   # (shots, delay) -> (method, NRMSE per draw)
for shots_label, _ in SHOTS:
    for delay in PREDICT_DELAYS:
        cell = part_a[(shots_label, FULL, delay)]
        name = min(cell, key=lambda n: cell[n].mean())
        best_full[(shots_label, delay)] = (name, cell[name])

ratio = lambda a, b: a.mean() / b.mean()
print("\n   NRMSE, mean over draws. Ratios above 1 mean the first is worse.")
with open("qcml102426_3_missing_data.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["delay", "noise", "measured", "ridge", "ridge improved", "QCML", "QCML improved",
                     "QCML all 160", "QCML improved all 160", "best with all 160", "best method",
                     "QCML missing / ridge missing", "QCML all / ridge missing",
                     "QCML missing / best all", "QCML improved missing / best all"])
    for delay in PREDICT_DELAYS:
        print(f"\n   delay {delay}")
        for shots_label, _ in SHOTS:
            full = part_a[(shots_label, FULL, delay)]
            name, best = best_full[(shots_label, delay)]
            print(f"   {shots_label}: with all 160, ridge {full['Ridge'].mean():.4f}, QCML "
                  f"{full['QCML'].mean():.4f}, QCML improved {full['QCML, improved'].mean():.4f}; best "
                  f"{name} {best.mean():.4f}")
            for budget_label, measured in BUDGETS[1:]:
                cell = part_a[(shots_label, budget_label, delay)]
                print(f"      {budget_label:<16} ridge {cell['Ridge'].mean():.4f}  QCML {cell['QCML'].mean():.4f}"
                      f"  QCML improved {cell['QCML, improved'].mean():.4f}   QCML/ridge "
                      f"{ratio(cell['QCML'], cell['Ridge']):.2f}   QCML(all)/ridge(missing) "
                      f"{ratio(full['QCML'], cell['Ridge']):.2f}   QCML(missing)/best(all) "
                      f"{ratio(cell['QCML'], best):.2f}   improved/best(all) "
                      f"{ratio(cell['QCML, improved'], best):.2f}")
                writer.writerow([delay, shots_label, len(measured)] + [
                    cell[m].mean() for m in ("Ridge", "Ridge, improved", "QCML", "QCML, improved")]
                    + [full["QCML"].mean(), full["QCML, improved"].mean(), best.mean(), name,
                       ratio(cell["QCML"], cell["Ridge"]), ratio(full["QCML"], cell["Ridge"]),
                       ratio(cell["QCML"], best), ratio(cell["QCML, improved"], best)])
print("saved to qcml102426_3_missing_data.csv")


# PART B: the weak photon-number record (all 160 probabilities missing)

print(f"\nPART B: ridge vs QCML on weak records, strengths {WEAK_GAMMAS}, runs {WEAK_RUNS}")
part_b = {}
for gamma in WEAK_GAMMAS:
    start = time.time()
    numbers = (photon_numbers(series[TRAIN_INPUTS], MG_DRIVE, gamma),
               photon_numbers(series[TEST_INPUTS], MG_DRIVE, gamma))
    for runs in WEAK_RUNS:
        noise = weak_noise(numbers[0], gamma, runs)
        for draw in range(NOISE_DRAWS):
            rng = np.random.default_rng(draw)
            train, test = weak_record(numbers[0], gamma, runs, rng), weak_record(numbers[1], gamma, runs, rng)
            for delay in PREDICT_DELAYS:
                collect(part_b, (gamma, runs, delay), compare(train, test, noise, delay, draw))
    print(f"   strength {gamma:.0e} done ({time.time() - start:.0f} s)")
part_b = as_arrays(part_b)
B_METHODS = ["Ridge", "Ridge, improved", "QCML", "QCML, improved"]
with open("qcml102426_3_weak_record.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["delay", "strength", "runs"] + [f"{m} {s}" for m in B_METHODS for s in ("mean", "std")])
    for delay in PREDICT_DELAYS:
        print(f"\n   delay {delay}: best with all 160 projective probabilities: exact "
              f"{best_full[('exact', delay)][1].mean():.4f}, 9^5 shots {best_full[('9^5 shots', delay)][1].mean():.4f}")
        for gamma in WEAK_GAMMAS:
            for runs in WEAK_RUNS:
                cell = part_b[(gamma, runs, delay)]
                print(f"   strength {gamma:.0e}, {runs:.0e} runs:  " + "   ".join(
                    f"{m} {cell[m].mean():.4f}" for m in B_METHODS))
                writer.writerow([delay, gamma, runs] + [v for m in B_METHODS for v in (cell[m].mean(), cell[m].std())])
print("saved to qcml102426_3_weak_record.csv")


# PART C: screening the improvements (one draw)  [note 7]

def train_multi(inputs, targets, n=SUP_N, seed=0, epochs=EPOCHS):
    """Supervised QCML with one target observable per column of `targets`, all read from
    the same quantum state."""
    X, Y = torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)
    K = X.shape[1]
    torch.manual_seed(seed)
    model = QCML(D=K + Y.shape[1], n=n)

    def loss_on(rows):
        psi = ground_state_from(model, X[rows], list(range(K)))
        out = torch.einsum("bi,dij,bj->bd", psi.conj(), model.observables()[K:], psi).real
        return ((out - Y[rows]) ** 2).mean()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(epochs):
        order = torch.randperm(len(X))
        for start in range(0, len(X), BATCH):
            loss = loss_on(order[start:start + BATCH])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        loss = loss_on(slice(None))
        loss.backward()
        return loss

    lbfgs.step(closure)
    return model, K


def multi_forecast(model, K, inputs):
    with torch.no_grad():
        psi = ground_state_from(model, torch.tensor(inputs, dtype=torch.float32), list(range(K)))
        return torch.einsum("bi,dij,bj->bd", psi.conj(), model.observables()[K:], psi).real.numpy()


def state_features(model, K, inputs):
    """Real and imaginary parts of rho = |psi><psi|: every expectation value <psi|B|psi>
    is a linear function of them, so ridge on them fits B in closed form."""
    with torch.no_grad():
        psi = ground_state_from(model, torch.tensor(inputs, dtype=torch.float32), list(range(K))).numpy()
    rho = psi[:, :, None] * psi[:, None, :].conj()
    upper, strict = np.triu_indices(psi.shape[1]), np.triu_indices(psi.shape[1], 1)
    return np.hstack([rho[:, upper[0], upper[1]].real, rho[:, strict[0], strict[1]].imag])


SCREEN_SETTINGS = ["vacuum only, 9^5 shots", "all 160, 9^5 shots", "weak, 1e8 x 1e4 runs", "weak, 1e9 x 1e6 runs"]
MULTI_DELAYS = [0, 1, 2, 5, 10, 20]
HISTORY_OPTIONS = [1, 2, 4, 6]


def screening_data(setting):
    """(train, test, noise variance, full training probabilities or None) for a setting."""
    rng = np.random.default_rng(0)
    if setting.startswith("weak"):
        gamma, runs = (1e8, 1e4) if "1e8" in setting else (1e9, 1e6)
        numbers = (photon_numbers(series[TRAIN_INPUTS], MG_DRIVE, gamma),
                   photon_numbers(series[TEST_INPUTS], MG_DRIVE, gamma))
        return (weak_record(numbers[0], gamma, runs, rng), weak_record(numbers[1], gamma, runs, rng),
                weak_noise(numbers[0], gamma, runs), None)
    train, test = projective(exact_train, 9 ** 5, rng), projective(exact_test, 9 ** 5, rng)
    measured = BUDGETS[2][1] if setting.startswith("vacuum") else BUDGETS[0][1]
    return train[:, measured], test[:, measured], projective_noise(train, 9 ** 5)[measured], train


print("\nPART C: screening ways to improve QCML (one draw)")
screen = {}
for setting in SCREEN_SETTINGS:
    start = time.time()
    train, test, noise, all_train = screening_data(setting)
    plain = standardise(train, test)
    weighted = standardise(train, test, noise)
    multi_targets = np.column_stack([(target(d)[0] - target(d)[0][TRAIN_ROWS].mean()) / target(d)[0][TRAIN_ROWS].std()
                                     for d in MULTI_DELAYS])
    multi_model, K = train_multi(plain[0][TRAIN_ROWS], multi_targets[TRAIN_ROWS])
    multi_out = multi_forecast(multi_model, K, plain[1][TEST_ROWS])
    multi_states = state_features(multi_model, K, plain[0]), state_features(multi_model, K, plain[1])
    for delay in PREDICT_DELAYS:
        y, y_test = target(delay)
        mu, sd = y[TRAIN_ROWS].mean(), y[TRAIN_ROWS].std()
        Y = (y - mu) / sd
        score = lambda p: nrmse(p, y_test)
        r = {"QCML (as before)": score(supervised_qcml(plain[0], y, plain[1])),
             "Ensemble of 5": score(supervised_qcml(plain[0], y, plain[1], seeds=range(5))),
             "Hilbert space n = 16": score(supervised_qcml(plain[0], y, plain[1], n=16)),
             "Mixed loss, lambda 0.01": score(supervised_qcml(plain[0], y, plain[1], lam=0.01, w=0.1)),
             "Mixed loss, lambda 0.1": score(supervised_qcml(plain[0], y, plain[1], lam=0.1, w=0.1)),
             "Noise-aware weights": score(supervised_qcml(weighted[0], y, weighted[1]))}
        base = train_supervised(plain[0][TRAIN_ROWS], Y[TRAIN_ROWS], SUP_N, mse, seed=0)
        states = state_features(base, plain[0].shape[1], plain[0]), state_features(base, plain[0].shape[1], plain[1])
        r["States + ridge readout"] = score(ridge(states[0], y, states[1]))
        r["Several delays as targets"] = score(multi_out[:, MULTI_DELAYS.index(delay)] * sd + mu)
        r["Several delays, states + ridge"] = score(ridge(multi_states[0], y, multi_states[1]))
        if all_train is not None:
            extra = (all_train - all_train[TRAIN_ROWS].mean(0)) / all_train[TRAIN_ROWS].std(0)
            boost = np.sqrt(extra.shape[1])            # the target counts as much as all extras together
            model, K = train_multi(plain[0][TRAIN_ROWS], np.column_stack([Y * boost, extra])[TRAIN_ROWS])
            r["Privileged information"] = score(multi_forecast(model, K, plain[1][TEST_ROWS])[:, 0] / boost * sd + mu)
        for previous in HISTORY_OPTIONS:
            Z, Z_test = standardise(with_history(train, previous), with_history(test, previous),
                                    np.tile(noise, previous + 1))
            r[f"History of {previous} + weights"] = score(supervised_qcml(Z, y, Z_test))
            r[f"Ridge, history of {previous} + weights"] = score(ridge(Z, y, Z_test))
            if previous == HISTORY[delay]:
                r["Improved (history + weights), ensemble of 3"] = score(supervised_qcml(Z, y, Z_test, seeds=range(3)))
        r["Ridge (as before)"] = score(ridge(plain[0], y, plain[1]))
        # control: the input itself, no reservoir
        inputs = with_history(series[TRAIN_INPUTS, None], HISTORY[delay]), with_history(series[TEST_INPUTS, None], HISTORY[delay])
        Z, Z_test = standardise(*inputs)
        r["No reservoir: QCML on the inputs"] = score(supervised_qcml(Z, y, Z_test))
        r["No reservoir: ridge on the inputs"] = score(ridge(Z, y, Z_test))
        screen[(setting, delay)] = r
    print(f"   {setting} done ({time.time() - start:.0f} s)")

names = list(dict.fromkeys(name for cell in screen.values() for name in cell))
SCREEN_ROWS = ([n for n in names if not n.startswith(("Ridge", "No reservoir"))]
               + [n for n in names if n.startswith("Ridge")] + [n for n in names if n.startswith("No reservoir")])
with open("qcml102426_3_screening.csv", "w", newline="") as file:
    writer = csv.writer(file)
    columns = [(s, d) for d in PREDICT_DELAYS for s in SCREEN_SETTINGS]
    writer.writerow(["technique"] + [f"{s}, delay {d}" for s, d in columns])
    for row in SCREEN_ROWS:
        writer.writerow([row] + [screen[c].get(row, "") for c in columns])
        print(f"   {row:<42}" + "".join(f"{screen[c][row]:9.4f}" if row in screen[c] else "        -" for c in columns))
print("saved to qcml102426_3_screening.csv")


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


STYLE = {"Ridge": (GREY, "D", "--"), "Ridge, improved": (GREY, "D", "-"),
         "QCML": (QCML_COLOUR, "^", "--"), "QCML, improved": (QCML_COLOUR, "^", "-")}
SETTING_COLOURS = [PCA_COLOUR, DM_COLOUR, QCML_COLOUR, INK]
print(f"\nplots saved to {PLOT_DIR}:")
counts = [len(m) for _, m in BUDGETS]

# 01-02: missing data, delay 20
for number, shots_label in [("01", "exact"), ("02", "9^5 shots")]:
    delay = 20
    fig, ax = new_plot(f"Missing probabilities: ridge vs QCML, x(t + {delay}), {shots_label}",
                       f"test run NRMSE (log scale), {NOISE_DRAWS} draws; dashed = as before, "
                       "solid = improved")
    for name, (colour, marker, line) in STYLE.items():
        values = [part_a[(shots_label, label, delay)][name] for label, _ in BUDGETS]
        ax.errorbar(counts, [v.mean() for v in values], yerr=[v.std() for v in values], color=colour,
                    marker=marker, ls=line, lw=2, ms=6, capsize=3, mfc="white" if line == "--" else colour,
                    label=name)
    name, best = best_full[(shots_label, delay)]
    ax.axhline(best.mean(), color=DM_COLOUR, ls=":", lw=2, label=f"best with all 160: {name}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(counts, [label.replace(": ", "\n") for label, _ in BUDGETS])
    ax.set_xlabel("probabilities measured per input")
    ax.set_ylabel("NRMSE")
    save_plot(fig, ax, f"{number}_missing_data_{'exact' if shots_label == 'exact' else 'noisy'}")

# 03-04: missing-data values
for number, delay in [("03", 1), ("04", 20)]:
    rows = []
    for shots_label, _ in SHOTS:
        name, best = best_full[(shots_label, delay)]
        for budget_label, _ in BUDGETS:
            cell = part_a[(shots_label, budget_label, delay)]
            rows.append((f"{shots_label}, {budget_label}", [cell[m] for m in STYLE]
                         + [f"{best.mean():.4f} ({name.split(' (')[0]})" if budget_label == FULL else ""]))
    table_plot(f"{number}_missing_data_table_delay_{delay}", f"Missing probabilities: x(t + {delay})",
               f"test run NRMSE, mean ± std over {NOISE_DRAWS} draws",
               ["Noise, probabilities measured", "Ridge", "Ridge,\nimproved", "QCML", "QCML,\nimproved",
                "Best with\nall 160"], rows, label_width=0.27, size=14, digits=4,
               note="Improved = the records of the last "
                    f"{HISTORY[delay]} inputs added, and each number weighted by its share of signal "
                    "(known shot noise). Best with all 160: the lowest of their readout, ridge, PCA, "
                    "diffusion map and QCML (both versions) with every probability measured.")

# 05: the three comparisons as ratios
rows = []
for delay in PREDICT_DELAYS:
    for shots_label, _ in SHOTS:
        full = part_a[(shots_label, FULL, delay)]
        _, best = best_full[(shots_label, delay)]
        for budget_label, _ in BUDGETS[1:]:
            cell = part_a[(shots_label, budget_label, delay)]
            rows.append((f"delay {delay}, {shots_label}, {budget_label}",
                         [f"{ratio(cell['QCML'], cell['Ridge']):.2f}",
                          f"{ratio(cell['QCML, improved'], cell['Ridge, improved']):.2f}",
                          f"{ratio(full['QCML'], cell['Ridge']):.2f}",
                          f"{ratio(cell['QCML'], best):.2f}",
                          f"{ratio(cell['QCML, improved'], best):.2f}"]))
table_plot("05_missing_data_ratios", "The comparisons as error ratios (below 1: QCML better)",
           f"test run NRMSE ratios, means over {NOISE_DRAWS} draws",
           ["Delay, noise, measured", "QCML / ridge\n(both missing)", "Improved QCML /\nimproved ridge",
            "QCML all 160 /\nridge missing", "QCML missing /\nbest all 160", "Improved QCML missing\n/ best all 160"],
           rows, label_width=0.3, size=14,
           note="All ratios compare NRMSE on the same test run. 'Missing' = only the measured "
                "probabilities are given (40: four neurons; 10: vacuum only).")

# 06: weak record against runs
delay, gamma = 20, 1e9
fig, ax = new_plot(f"Weak photon-number record: ridge vs QCML, x(t + {delay})",
                   f"strength {gamma:.0e} per second; test run NRMSE, mean ± std over {NOISE_DRAWS} draws")
for name, (colour, marker, line) in STYLE.items():
    values = [part_b[(gamma, r, delay)][name] for r in WEAK_RUNS]
    ax.errorbar(WEAK_RUNS, [v.mean() for v in values], yerr=[v.std() for v in values], color=colour,
                marker=marker, ls=line, lw=2, ms=6, capsize=3, mfc="white" if line == "--" else colour,
                label=name)
for shots_label, style in [("exact", ":"), ("9^5 shots", "-.")]:
    name, best = best_full[(shots_label, delay)]
    ax.axhline(best.mean(), color=DM_COLOUR, ls=style, lw=2,
               label=f"best with all 160, {shots_label}: {name}")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("runs averaged")
ax.set_ylabel("NRMSE")
save_plot(fig, ax, "06_weak_record")

# 07-08: weak-record tables
for number, delay in [("07", 1), ("08", 20)]:
    rows = [(f"strength {g:.0e}, {r:.0e} runs", [part_b[(g, r, delay)][m] for m in STYLE])
            for g in WEAK_GAMMAS for r in WEAK_RUNS]
    table_plot(f"{number}_weak_record_table_delay_{delay}", f"Weak photon-number record: x(t + {delay})",
               f"test run NRMSE, mean ± std over {NOISE_DRAWS} draws; best with all 160 projective "
               f"probabilities: {best_full[('exact', delay)][1].mean():.4f} exact, "
               f"{best_full[('9^5 shots', delay)][1].mean():.4f} at 9^5 shots",
               ["Strength, runs", "Ridge", "Ridge,\nimproved", "QCML", "QCML,\nimproved"], rows,
               label_width=0.3, size=12, digits=4,
               note=f"Records: 20 numbers per input. Improved = the records of the last {HISTORY[delay]} "
                    "inputs added, with noise-aware weights; ridge gets the same inputs.")

# 09: how much history helps (screening, delay 20)
delay = 20
fig, ax = new_plot(f"Adding the records of previous inputs (x(t + {delay}))",
                   "screening, one draw; solid = QCML, dashed = ridge, both with noise-aware weights")
for setting, colour in zip(SCREEN_SETTINGS, SETTING_COLOURS):
    cell = screen[(setting, delay)]
    xs = [0] + HISTORY_OPTIONS
    ax.plot(xs, [cell["Noise-aware weights"]] + [cell[f"History of {p} + weights"] for p in HISTORY_OPTIONS],
            "^-", color=colour, lw=2, label=f"QCML, {setting}")
    ax.plot(xs, [cell["Ridge (as before)"]] + [cell[f"Ridge, history of {p} + weights"] for p in HISTORY_OPTIONS],
            "D--", color=colour, lw=1.5, mfc="white", label=f"ridge, {setting}")
ax.set_yscale("log")
ax.set_xlabel("previous inputs whose records are added")
ax.set_ylabel("NRMSE")
save_plot(fig, ax, "09_history_length")

# 10: the screening table
columns = [(s, d) for d in PREDICT_DELAYS for s in SCREEN_SETTINGS]
rows = [(row, [f"{screen[c][row]:.4f}" if row in screen[c] else "–" for c in columns]) for row in SCREEN_ROWS]
table_plot("10_screening_table", "Screening ways to improve QCML (one draw, test run NRMSE)",
           "settings: vacuum only / all 160 at 9^5 shots; weak record 1e8 x 1e4 runs / 1e9 x 1e6 runs",
           ["Technique"] + [f"{s.replace(', ', chr(10))}\ndelay {d}" for s, d in columns], rows,
           label_width=0.25, size=18,
           note="History of k: the records of the last k inputs added to the current one. Privileged "
                "information applies only where all 160 probabilities were measured in training. No "
                "reservoir: the input history itself, a control.")


# ========================================================================================
# EXPLANATION
# Notes on the code above, in the order of its sections. A tag [note k] in the code
# marks the section that note k explains.
# ========================================================================================
#
# [note 0] About this script
#
# Two additions:
# 1. Ridge regression, the standard reservoir-computing readout, as the baseline, with
#    and without missing data. "Missing data" means measuring only some of the 160
#    photon-number probabilities (QCML102426's budgets) or measuring only the weak
#    photon-number record (QCML102426_2, where all 160 are missing). Compared:
#      - QCML with missing data vs ridge with missing data
#      - QCML with all data vs ridge with missing data
#      - QCML with missing data vs the best method with all data
# 2. Ways to improve QCML for reservoir readout, screened on four settings (Part C):
#      - time-delay inputs: the records of the last few inputs, not just the current one.
#        The reservoir forgets within 1-3 inputs (its photons leak out), and predicting 20
#        inputs ahead needs a longer memory, so the readout supplies it
#      - noise-aware weights: the shot noise of each measured number is known, so numbers
#        that are mostly noise are shrunk instead of z-scored to full weight
#      - QCML's quantum state as features for a ridge readout (fitting the target
#        observable B in closed form, reservoir-computing style)
#      - several delays as targets of one model, sharing its quantum state
#      - "privileged information": the unmeasured probabilities, measured during
#        calibration only, as extra training targets
#      - the mixed loss of Di Caro et al. (as for breast cancer), ensembles of 5 models,
#        a larger Hilbert space (n = 16)
#    plus a control: QCML and ridge given the input history itself, with no reservoir.
#    The winners (time-delay inputs + noise-aware weights) make "improved QCML" in
#    Parts A and B, and ridge gets exactly the same inputs ("improved ridge").
#
# PART A  Projective measurement with missing probabilities: ridge vs QCML.
# PART B  The weak photon-number record: ridge vs QCML.
# PART C  The screening of the improvements.
#
# Runs in about 30 minutes. The simulation is QCML102426_2's (checked there against the
# full master equation and their saved results); needs dudas_data/ (already there).
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
# PCA (as in QCML102326): the point on the flat k-dimensional subspace of the
# training records that best matches the measured numbers (least squares); its other
# coordinates are the filled-in numbers or the prediction.
#
# DIFFUSION MAP (as in QCML102326): built on the measured numbers of the training
# records, with each point's kernel width its distance to its k-th nearest neighbour.
# Anything known on the training records (the unmeasured numbers, or the target) is
# rebuilt at a new record from the walk's first m eigenvectors (geometric harmonics).
#
# [note 6] SETTINGS
#
# Every method learns from the training run (inputs 30-999) and is scored on the test run
# (from its 30th input, as in their scoring). Ridge's strength is chosen on a validation
# block inside the training run (fit on inputs 30-699, score on 700-999), as are PCA's k
# and the diffusion map's neighbours and m; QCML's settings come from pilot runs.
#
# [note 7] PART C: screening the improvements (one draw)
#
# Four settings: vacuum only and all 160 probabilities at 9^5 shots (QCML102426's noisiest
# level), and the weak record at strength 1e8 with 1e4 runs and at 1e9 with 1e6 runs.
