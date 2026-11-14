# Dudas_Python_Weak.py

import csv
import itertools
import os
import time
from math import factorial

import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

HERE = os.path.dirname(os.path.abspath(__file__))
THESIS = HERE                                    # the thesis CSVs and dudas_data sit beside this file
DATA_DIR = os.path.join(THESIS, "dudas_data")       # in the thesis scripts: DATA_DIR = "dudas_data"
RUN_SLOW_CHECKS = True
load = lambda name: np.loadtxt(os.path.join(DATA_DIR, name + ".csv"), delimiter=",")



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


# Their Qmodel itself

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


# Their Qmodel and, with dephasing, their Cmodel (fig4): the same master equation with
# photon-number dephasing at rate `dephasing` on each oscillator. Part 2 reuses it to check
# the weak-measurement simulation.

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

# THE MACKEY-GLASS INPUT (as in main_fig7.jl)  [note 0]

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


TRAIN_INPUTS = np.arange(1000)                 # x(0..999)
TEST_INPUTS = np.arange(1100, 2100)            # x(1100..2099)



def linear_readout(train_record, train_target, record):
    """Their readout: weights from the Moore-Penrose pseudo-inverse of the training
    record (no constant term), applied to `record`. The cutoff for tiny singular values
    matches Julia's pinv."""
    cutoff = min(train_record.shape) * np.finfo(float).eps
    return record @ (np.linalg.pinv(train_record, rtol=cutoff) @ train_target)


# Their Fig. 5 measurement with a finite number of shots.
# ---- copied unchanged from 3_QCML_Plus/QCML110226.py

def projective(record, shots, rng):
    """Their measurement with a finite number of shots (their Fig. 5)."""
    if shots is None:
        return record
    return record + rng.standard_normal(record.shape) * np.sqrt(np.abs(record * (1 - record) / shots))


# ---- Part 1 checks  [note 1]
# ---- copied unchanged from 5_Dudas_Reservoir_Simulation/QCML102426.py
DELAYS = np.arange(1, 101)
FIG7 = [("g = 700 MHz, κ = 17, 21 MHz", "fig7_coupling_error", COUPLING, KAPPA_A, KAPPA_B),
        ("g = 0", "fig7_no_coupling_error", 0.0, KAPPA_A, KAPPA_B),
        ("κ = 170, 210 MHz", "fig7_κ10_error", COUPLING, 10 * KAPPA_A, 10 * KAPPA_B)]


def their_error(prediction, target):
    """Their Fig. 7 error exactly as main_fig7.jl lines 152-157 compute it: squared errors summed
    from the 30th test input, divided by all 1000 inputs, over the range of all 1000 targets."""
    return np.sqrt(((prediction[29:] - target[29:]) ** 2).sum() / len(prediction)) / (target.max() - target.min())


def part1_checks(series):
    print("PART 1: their code in Python, checked against their saved results")
    # Fig. 3: their saved sine/square measurements (A1 of QCML102426). The inputs are rebuilt
    # from their saved labels: each 8-input block is one period of a square or a sine wave.
    their_record = load("fig3_X")
    square = np.array([1, 1, 1, 1, -1, -1, -1, -1]) / 2
    sine = np.array([0, 0.7071, 1, 0.7071, 0, -0.7071, -1, -0.7071]) / 2
    sinesquare = np.concatenate([sine if label == 1 else square for label in load("fig3_Y")[::8]])
    ours = reservoir(sinesquare, 1.2e6, offset=0.6, max_photons=5, samples=1)
    print(f"   Fig. 3 data, coherent-state shortcut, all {len(ours)} inputs: largest difference "
          f"{np.abs(ours - their_record).max():.1e} (their 8-photon cutoff matters at this "
          "stronger drive)")
    if RUN_SLOW_CHECKS:
        start = time.time()
        exact = master_equation_reservoir(sinesquare[:8], 1.2e6, offset=0.6, max_photons=5, samples=1)
        print(f"   Fig. 3 data, full master equation, first 8 inputs: largest difference "
              f"{np.abs(exact - their_record[:8]).max():.1e}  ({time.time() - start:.0f} s)")
        # Fig. 4: their saved dephasing measurements, their Cmodel (A1 of QCML102426_2)
        inputs = np.concatenate([sine if label == 1 else square for label in load("fig4_dephasing_Y")[::8]])
        start = time.time()
        P = master_equation(inputs[:6], 1.2e6, offset=0.6, samples=1, dephasing=100e6)
        gap = np.abs(P[:, 0, :6, :6].reshape(6, -1) - load("fig4_dephasing_X")[:6]).max()
        print(f"   their Fig. 4 dephasing records vs the master equation, 6 inputs: largest "
              f"difference {gap:.1e}  ({time.time() - start:.0f} s)")
        # the shortcut against the full master equation at the Mackey-Glass drive (A2 of QCML102426)
        start = time.time()
        piece = series[300:310]
        gap = np.abs(master_equation_reservoir(piece, MG_DRIVE) - reservoir(piece, MG_DRIVE)).max()
        print(f"   Mackey-Glass drive, shortcut vs full master equation, 10 inputs: largest "
              f"difference {gap:.1e}  ({time.time() - start:.0f} s)")
    # Fig. 7: error against delay with their readout, for three reservoirs (A4 of QCML102426)
    for label, file, g, kappa_a, kappa_b in FIG7:
        train = reservoir(series[TRAIN_INPUTS], MG_DRIVE, g=g, kappa_a=kappa_a, kappa_b=kappa_b)
        test = reservoir(series[TEST_INPUTS], MG_DRIVE, g=g, kappa_a=kappa_a, kappa_b=kappa_b)
        predictions = [linear_readout(train, series[TRAIN_INPUTS + d], test) for d in DELAYS]
        ours = np.array([nrmse(p[WASHOUT - 1:], series[TEST_INPUTS + d][WASHOUT - 1:])
                         for p, d in zip(predictions, DELAYS)])
        formula = np.array([their_error(p, series[TEST_INPUTS + d]) for p, d in zip(predictions, DELAYS)])
        theirs = load(file)
        print(f"   Fig. 7, {label:<28} log10 error at delay 1, 20, 100:  theirs "
              + " ".join(f"{np.log10(theirs[d - 1]):+.2f}" for d in (1, 20, 100)) + "   ours "
              + " ".join(f"{np.log10(ours[d - 1]):+.2f}" for d in (1, 20, 100))
              + f"   mean gap {np.abs(np.log10(ours / theirs)).mean():.2f}"
              + f"   (with their exact formula {np.abs(np.log10(formula / theirs)).mean():.2f})")




ETA = 0.5                                       # detector efficiency of the weak record


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


# Their readout with an optional constant term. This replaces Part 1's linear_readout, and with
# constant=False (the default) it computes exactly the same thing.
# ---- copied unchanged from 5_Dudas_Reservoir_Simulation/QCML102426_2.py

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


# ---- copied unchanged from 5_Dudas_Reservoir_Simulation/QCML102426_2.py
REPLAY = 10                                         # Part D: inputs replayed before each of their
                                                    # measurements (the reservoir forgets within
                                                    # about 10: its amplitude falls by e^-8.5)
INPUT_TIME = INTERVAL                               # each input lasts 100 ns


weak_time = lambda runs: runs * (len(TRAIN_INPUTS) + len(TEST_INPUTS)) * INPUT_TIME
projective_time = lambda shots: shots * 10 * (len(TRAIN_INPUTS) + len(TEST_INPUTS)) * REPLAY * INPUT_TIME


# ---- Part 2 checks  [note 2]

def part2_checks(series):
    print("\nPART 2: the same reservoir, measured weakly")
    # the moments against the coherent-state shortcut (no dephasing) and against the full master
    # equation (with dephasing) at the Mackey-Glass drive
    piece = series[300:304]
    coherent = reservoir(piece, MG_DRIVE)                                  # P(n_a, n_b) records
    mean_a = coherent[:, 4::16] / coherent[:, 0::16]                       # P(1,0)/P(0,0) = <N_a>
    gap = np.abs(photon_numbers(piece, MG_DRIVE, 0.0, windows=None)[:, :, 0] - mean_a).max()
    print(f"   moments vs the coherent-state shortcut, no dephasing: largest difference {gap:.1e}")
    if RUN_SLOW_CHECKS:
        for gamma, levels in [(1e6, 9), (1e8, 12)]:
            start = time.time()
            full = photon_moments(master_equation(piece, MG_DRIVE, levels=levels, dephasing=gamma))
            ours = photon_numbers(piece, MG_DRIVE, gamma, windows=None)
            ours[..., 2:] -= ours[..., :2] ** 2                            # <N^2> -> Var N
            print(f"   moments vs master equation ({levels - 1}-photon cutoff), dephasing "
                  f"{gamma:.0e}: largest difference in <N>, Var N {np.abs(ours - full).max():.1e}"
                  f"  ({time.time() - start:.0f} s)")
        # their Fig. 4 drive with 100e6 dephasing: their cutoff against the cutoff-free moments
        square = np.array([1, 1, 1, 1, -1, -1, -1, -1]) / 2
        sine = np.array([0, 0.7071, 1, 0.7071, 0, -0.7071, -1, -0.7071]) / 2
        inputs = np.concatenate([sine if label == 1 else square for label in load("fig4_dephasing_Y")[::8]])
        exact = photon_numbers(inputs[:1], 1.2e6, 100e6, offset=0.6, windows=None)[0, 9]
        for levels in (9, 14):
            P = master_equation(inputs[:1], 1.2e6, offset=0.6, samples=1, levels=levels, dephasing=100e6)
            print(f"   their Fig. 4 drive, first input, mean photons in a: master equation with "
                  f"{levels - 1}-photon cutoff {photon_moments(P)[0, 0, 0]:.3f}; moments (no cutoff) "
                  f"{exact[0]:.3f}")
    # back-action and noise against strength, test run (the numbers behind plot 01 of QCML102426_2)
    tau = (RESOLUTION - 1) * INTERVAL / RESOLUTION / 10
    numbers = {}
    print("   strength   mean photons a   Var N_a   detector noise variance per run and window")
    for gamma in (1e5, 1e7, 1e8, 1e9):
        numbers[gamma] = (photon_numbers(series[TRAIN_INPUTS], MG_DRIVE, gamma),
                          photon_numbers(series[TEST_INPUTS], MG_DRIVE, gamma))
        test = numbers[gamma][1][TEST_ROWS]
        print(f"   {gamma:8.0e}   {test[:, :, 0].mean():14.3f}   {np.mean(test[:, :, 2] - test[:, :, 0] ** 2):7.3f}"
              f"   {1 / (4 * ETA * gamma * tau):10.3g}")
    # their readout with a constant term on the weak record (Part B of QCML102426_2, 2 draws)
    saved = {}
    with open(os.path.join(THESIS, "qcml102426_2_strength_map.csv")) as file:
        for row in csv.DictReader(file):
            saved[(int(row["delay"]), float(row["runs"]), float(row["strength"]))] = float(row["Linear readout (theirs) mean"])
    for gamma in (1e8, 1e9):
        for runs in (1e4, 1e6):
            out = []
            for delay in (1, 20):
                scores = []
                for draw in range(2):
                    rng = np.random.default_rng(draw)
                    train = weak_record(numbers[gamma][0], gamma, runs, rng)
                    test = weak_record(numbers[gamma][1], gamma, runs, rng)
                    y, y_test = series[TRAIN_INPUTS + delay], series[TEST_INPUTS + delay][TEST_ROWS]
                    scores.append(nrmse(linear_readout(train[TRAIN_ROWS], y[TRAIN_ROWS], test[TEST_ROWS],
                                                       constant=True), y_test))
                out.append(f"delay {delay}: {np.mean(scores):.4f} (thesis CSV {saved[(delay, runs, gamma)]:.4f})")
            print(f"   their readout on the weak record, strength {gamma:.0e}, {runs:.0e} runs:  " + "   ".join(out))
    # the time cost of the two protocols (Part D of QCML102426_2)
    print("   experiment time: weak " + ", ".join(f"{r:.0e} runs {weak_time(r):.3g} s" for r in (1e4, 1e6))
          + ";  theirs " + ", ".join(f"9^{k} shots {projective_time(9 ** k):.3g} s" for k in (5, 7, 9)))





# ---- copied unchanged from 3_QCML_Plus/QCML110226.py
TRAIN_ROWS = np.arange(WASHOUT, 1000)
FIT_ROWS, VAL_ROWS = np.arange(WASHOUT, 700), np.arange(700, 1000)
TEST_ROWS = np.arange(WASHOUT - 1, 1000)
# ---- copied unchanged from 4_Baseline_Methods/QCML102426_4.py
RIDGE_STRENGTHS = np.logspace(-6, 4, 41)


# ---- copied unchanged from 3_QCML_Plus/QCML110226.py

def neuron(n_a, n_b):
    return n_a * 4 + n_b                        # position of P(n_a, n_b) within a sample


BUDGETS = {160: [neuron(i, j) for i in range(4) for j in range(4)],
           40: [neuron(0, 0), neuron(0, 1), neuron(1, 0), neuron(1, 1)],
           10: [neuron(0, 0)]}
BUDGETS = {count: np.array([k * 16 + c for k in range(10) for c in neurons])
           for count, neurons in BUDGETS.items()}


# ---- copied unchanged from 3_QCML_Plus/QCML110226.py

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


# The noise variances from formulas, used up to QCML102426_5 and replaced by measure() in
# QCML110226 (an experimenter does not know Var N, which weak_noise takes from the simulation).
# ---- copied unchanged from 3_QCML_Plus/QCML102426_5.py

def projective_noise(train, shots):
    if shots is None:
        return np.zeros(train.shape[1])
    return np.mean(np.abs(train[TRAIN_ROWS] * (1 - train[TRAIN_ROWS])) / shots, 0)


def weak_noise(numbers, gamma, runs):
    tau = (RESOLUTION - 1) * INTERVAL / RESOLUTION / numbers.shape[1]
    spread = np.maximum(numbers[TRAIN_ROWS, :, 2:] - numbers[TRAIN_ROWS, :, :2] ** 2, 0)
    return ((spread + 1 / (4 * ETA * gamma * tau)) / runs).mean(0).ravel()



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


# Ridge regression: their readout with a constant term and a penalty on the weights, the
# strength chosen on the validation block (the baseline of QCML102426_3 and QCML102426_4).
# ---- copied unchanged from 4_Baseline_Methods/QCML102426_4.py

def ridge(Z, y, Z_test):
    """Linear, with a constant term; strength chosen on validation."""
    def fit(rows, strength):
        centre, offset = Z[rows].mean(0), y[rows].mean()
        A = Z[rows] - centre
        weights = np.linalg.solve(A.T @ A + strength * np.eye(A.shape[1]), A.T @ (y[rows] - offset))
        return lambda X: (X - centre) @ weights + offset
    errors = [np.mean((fit(FIT_ROWS, s)(Z[VAL_ROWS]) - y[VAL_ROWS]) ** 2) for s in RIDGE_STRENGTHS]
    return fit(TRAIN_ROWS, RIDGE_STRENGTHS[int(np.argmin(errors))])(Z_test)


# ---- copied unchanged from 3_QCML_Plus/QCML102426_3.py
HISTORY = {1: 4, 20: 6}                         # improved methods: previous inputs whose
                                                # records are added, per delay (best of 1, 2,
                                                # 4, 6 in a pilot; Part C shows all four)
# ---- copied unchanged from 3_QCML_Plus/QCML110226.py
SETTINGS = [("vacuum only, exact", "projective", 10, None),
            ("vacuum only, 9^5 shots", "projective", 10, 9 ** 5),
            ("40 numbers, exact", "projective", 40, None),
            ("40 numbers, 9^5 shots", "projective", 40, 9 ** 5),
            ("all 160, 9^5 shots", "projective", 160, 9 ** 5),
            ("weak record, 1e4 runs", "weak", 1e4, 1e9),
            ("weak record, 1e6 runs", "weak", 1e6, 1e9)]


# ---- Part 3 checks  [note 3]

def part3_checks(series):
    print("\nPART 3: the thesis' other changes")
    print("   budgets: " + ", ".join(f"{size} numbers = columns {BUDGETS[size][:4].tolist()}..." for size in (160, 40, 10)))
    saved = {}
    with open(os.path.join(THESIS, "qcml102426_3_missing_data.csv")) as file:
        for row in csv.DictReader(file):
            saved[(int(row["delay"]), row["noise"], int(row["measured"]))] = (float(row["ridge"]), float(row["ridge improved"]))
    for size in (40, 10):
        for delay in (1, 20):
            plain, improved = [], []
            for draw in range(2):
                rng = np.random.default_rng(draw)
                train, test = projective(exact_train, 9 ** 5, rng), projective(exact_test, 9 ** 5, rng)
                noise = projective_noise(train, 9 ** 5)
                columns = BUDGETS[size]
                train, test, noise = train[:, columns], test[:, columns], noise[columns]
                y, y_test = series[TRAIN_INPUTS + delay], series[TEST_INPUTS + delay][TEST_ROWS]
                Z, Z_test = standardise(train, test)
                plain.append(nrmse(ridge(Z, y, Z_test[TEST_ROWS]), y_test))
                previous = HISTORY[delay]
                Z, Z_test = standardise(with_history(train, previous), with_history(test, previous),
                                        np.tile(noise, previous + 1))
                improved.append(nrmse(ridge(Z, y, Z_test[TEST_ROWS]), y_test))
            thesis = saved[(delay, "9^5 shots", size)]
            print(f"   ridge, {size} numbers at 9^5 shots, delay {delay:>2}: plain {np.mean(plain):.4f} (thesis "
                  f"{thesis[0]:.4f}), with history and weights {np.mean(improved):.4f} (thesis {thesis[1]:.4f})")
    saved = {}
    with open(os.path.join(THESIS, "qcml110226_noise_weights.csv")) as file:
        for row in csv.DictReader(file):
            saved[row["setting"]] = (float(row["weight, median"]), float(row["weight, smallest"]),
                                     int(row["columns weighted below 0.5"]), int(row["columns weighted 0"]))
    print("   noise weights on the training run (first draw): measured from halves | from the old formulas | thesis CSV")
    for label, kind, size, strength in SETTINGS:
        train, _, noise = measure(kind, size, strength, 0)
        variance = train[TRAIN_ROWS].var(0)
        weight = np.sqrt(np.clip(1 - noise / variance, 0, 1))
        formula = (projective_noise(train, strength) if kind == "projective"
                   else weak_noise(weak_numbers[0], strength, size))
        old = np.sqrt(np.clip(1 - formula / variance, 0, 1))
        s = saved[label]
        print(f"   {label:<24} median {np.median(weight):.4f}, below 0.5: {int((weight < 0.5).sum()):>2}, zero: "
              f"{int((weight == 0).sum()):>2} | median {np.median(old):.4f}, below 0.5: {int((old < 0.5).sum()):>2}, zero: "
              f"{int((old == 0).sum()):>2} | median {s[0]:.4f}, below 0.5: {s[2]:>2}, zero: {s[3]:>2}")


# ==========================================================================================

if __name__ == "__main__":
    start = time.time()
    # the records every thesis comparison uses (copied from 3_QCML_Plus/QCML110226.py;
    # only the indentation differs, since here they sit under the main guard)
    series = mackey_glass_dudas()
    exact_train, exact_test = reservoir(series[TRAIN_INPUTS], MG_DRIVE), reservoir(series[TEST_INPUTS], MG_DRIVE)
    weak_numbers = (photon_numbers(series[TRAIN_INPUTS], MG_DRIVE, 1e9),
                    photon_numbers(series[TEST_INPUTS], MG_DRIVE, 1e9))
    part1_checks(series)
    part2_checks(series)
    part3_checks(series)
    print(f"\ndone in {time.time() - start:.0f} s")


# ========================================================================================
# EXPLANATION
# Notes on the code above, in the order of its sections. A tag [note k] in the code
# marks the section that note k explains.
# ========================================================================================
#
# [note 0] THE MACKEY-GLASS INPUT (as in main_fig7.jl)
#
# dx/dt = 0.2 x(t - 17) / (1 + x(t - 17)^10) - 0.1 x(t), started from x = 1 before t = 0
# and x(0) = 0, and read once per time unit. The reservoir is trained on inputs x(0..999)
# and tested on x(1100..2099), each run starting from an empty reservoir; the target for
# input x(t) is x(t + delay). Their solver's exact numbers were not saved, and the series
# is chaotic, so small integration differences grow: this series follows the same
# equation but not digit for digit theirs (step 0.002 here; with step 0.01 the test half
# differs by up to 0.4). Its statistics, and so the task, are the same.
#
# [note 1]
# ---- Part 1 checks: Part A of QCML102426.py (A1, A2, A4) and the first check of Part A of
# QCML102426_2.py, wrapped in a function. New here: their exact error formula, printed next to
# the thesis one, so the small difference between the two can be seen.
#
# [note 2]
# ---- Part 2 checks: Part A of QCML102426_2.py (A2 and the second half of A1), wrapped in a
# function, then the photon numbers and noise against strength (plot 01 of QCML102426_2), their
# readout on weak records as in Part B of QCML102426_2 (reproducing the "Linear readout" column
# of qcml102426_2_strength_map.csv), and the time cost of Part D.
#
# [note 3]
# ---- Part 3 checks: ridge on the budgets at 9^5 shots, plain and with time-delay inputs and the
# formula weights (Part A of QCML102426_3, reproducing qcml102426_3_missing_data.csv), then
# the measured noise weights of QCML110226 against the formula ones (reproducing
# qcml110226_noise_weights.csv).
#
# ==========================================================================================
# PART 1
#
# Which Julia each piece replaces (line numbers in fig*/main_fig*.jl of the code of Dudas et al.):
#   fig7 lines 14-37    Fock bases (0..8 photons), a, b, the parameters, H_0, H_drive, c_ops
#                       -> the constants below, inside reservoir() and master_equation_reservoir()
#   fig7 lines 41-80    Qmodel: hold each input for 100 ns, solve the master equation, read
#                       P(n_a, n_b) at the sample times, carry the state to the next input
#                       -> reservoir() (exact coherent-state shortcut, used for all thesis data)
#                       -> master_equation_reservoir() (the same master equation, used to check it)
#   fig4 lines 83-120   Cmodel: the same with photon-number dephasing (c_ops [a, b, Na, Nb])
#                       -> master_equation(..., dephasing=100e6)
#   fig7 lines 82-117   createDataset / createX / createY: the Mackey-Glass delay equation,
#                       inputs x(0..999) and x(1100..2099), targets x(t + delay)
#                       -> mackey_glass_dudas(), TRAIN_INPUTS, TEST_INPUTS
#   fig7 lines 145-158  W = pinv(X) * Y and their error
#                       -> linear_readout(), nrmse()
#   fig5 lines 211-225  finite shots: Gaussian noise sqrt(p (1 - p) / shots) on each probability
#                       -> projective()
#   fig3 lines 82-108   the sine/square inputs -> rebuilt from their saved labels in part1_checks()
# Their drives eA and eB are always equal, so one `drive` stands for both.

# THE RESERVOIR (as in main_fig7.jl)
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



# ==========================================================================================
# PART 2: THE SAME RESERVOIR, MEASURED WEAKLY
#
# Their code has no weak measurement: it measures projectively, which destroys the state, so
# every sample of every input needs its own replay of the input. Part 2 keeps their reservoir
# and input unchanged and replaces the measurement:
#   - back-action: photon-number dephasing GAMMA D[N] on each oscillator, the same Lindblad
#     term as their Cmodel (fig4 lines 84 and 98: c_ops Na, Nb at rate κϕ), here set by the
#     measurement strength
#   - record: the photon number <N> of each oscillator averaged over each of the 10 windows of
#     an input, plus noise (see weak_record), averaged over R runs of the whole input sequence
#   - simulation: closed moment equations (moment_generator, photon_numbers), checked against
#     master_equation() of Part 1
#   - readout: their pseudo-inverse readout, given a constant term
#   - cost: experiment time of the weak protocol against their replays (weak_time, projective_time)

# WEAK PHOTON-NUMBER MEASUREMENT
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

# PART D: cost
#
# Total time the reservoir runs, for the training and the test run together (2000 inputs):
#   weak         runs x 2000 inputs x 100 ns
#   theirs       shots x 10 samples x 2000 inputs measurements, each after replaying the last
#                REPLAY inputs: shots x 10 x 2000 x REPLAY x 100 ns
# Their protocol is scored with all 160 probabilities and finite shots, as in their Fig. 5.

# ==========================================================================================
# PART 3: THE OTHER CHANGES THE THESIS MADE TO THEIR PIPELINE
#
#   - rows: training also skips the first 30 inputs (they train on all 1000), and a validation
#     block (inputs 700-999 of the training run) chooses every readout's settings
#   - budgets: measuring only some of the 16 photon-number probabilities (40 or 10 numbers per
#     input instead of 160)
#   - time-delay inputs: each input's record is joined by the records of the last few inputs
#   - noise-aware weights: each measured number is shrunk by the share of its spread that is
#     noise; that noise was first computed from formulas (projective_noise, weak_noise) and is
#     now measured from two halves of the shots or runs (measure)
#   - ridge regression: the standard reservoir-computing version of their readout

# NOISE MEASURED FROM THE DATA (new in QCML110226)
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
