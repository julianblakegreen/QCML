# Quantum Cognitive Machine Learning Project
# QCML111326: accuracy against the number of repetitions, for every readout
# Runs in about 45 minutes on 8 cores (168 jobs, one process each).
# What it does and how: see the notes at the end of this file.


import os
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")    # one thread per process (before numpy loads)
import csv
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import QCML110226 as base     # the reservoir, records and readouts (simulates the reservoir on import)

# Repository set-up: run from this folder
import os
import sys
_FOLDER = os.path.dirname(os.path.abspath(__file__))
os.chdir(_FOLDER)                                                 # data and outputs live in this folder


WORKERS = os.cpu_count()
GAMMA = 1e9                                      # measurement strength of the weak record
RUNS = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]       # Part A
SHOTS = [9 ** k for k in range(3, 10)]           # Part B
SETS = {10: "vacuum only", 40: "40 values", 160: "all 160"}
RIDGE_STRENGTHS = np.logspace(-6, 4, 41)         # as in QCML102426_4
DELAYS = base.PREDICT_DELAYS
DRAWS = base.NOISE_DRAWS
weak_time = lambda runs: runs * 2000 * 100e-9
projective_time = lambda shots: shots * 10 * 2000 * 10 * 100e-9
WEAK_READOUTS = ["Ridge regression", "Diffusion map", "Kernel ridge", "Supervised QCML", "QCML+"]
PROJECTIVE_READOUTS = ["Ridge regression", "Diffusion map", "Kernel ridge", "QCML+"]


def ridge_fit(Z, y, Z_test):
    """The linear readout: ridge regression with a constant term (as in QCML102426_4), its
    strength chosen on validation. Returns (test predictions, validation error)."""
    def fit(rows, strength):
        centre, offset = Z[rows].mean(0), y[rows].mean()
        A = Z[rows] - centre
        weights = np.linalg.solve(A.T @ A + strength * np.eye(A.shape[1]), A.T @ (y[rows] - offset))
        return lambda X: (X - centre) @ weights + offset
    errors = [np.mean((fit(base.FIT_ROWS, s)(Z[base.VAL_ROWS]) - y[base.VAL_ROWS]) ** 2)
              for s in RIDGE_STRENGTHS]
    best = int(np.argmin(errors))
    return fit(base.TRAIN_ROWS, RIDGE_STRENGTHS[best])(Z_test), errors[best] / y[base.TRAIN_ROWS].var()


def evaluate(train, test, noise, delay, seed, readouts):
    """Every readout in `readouts` on one record, exactly as QCML110226's evaluate (no blend):
    {readout: (test NRMSE, validation error in units of the target's variance, seconds, choice)}.
    Supervised QCML is not validated (n = 8 from the pilots), so its validation error is nan."""
    y, y_test = base.target(delay)
    previous = base.HISTORY[delay]
    Z, Z_test = base.standardise(base.with_history(train, previous), base.with_history(test, previous),
                                 None if noise is None else np.tile(noise, previous + 1))
    Z_test = Z_test[base.TEST_ROWS]
    variance = y[base.TRAIN_ROWS].var()
    out = {}

    clock = time.perf_counter()
    prediction, validation = ridge_fit(Z, y, Z_test)
    out["Ridge regression"] = (base.nrmse(prediction, y_test), validation, time.perf_counter() - clock, "raw")

    clock = time.perf_counter()
    represented = {r: base.compress(Z, Z_test, r) for r in base.REPRESENTATIONS}
    compress_time = time.perf_counter() - clock
    clock = time.perf_counter()
    (dm_test, _, dm_validation), choice = min(
        ((base.diffusion_map_fit(represented[r][0], y, represented[r][1]), r) for r in base.REPRESENTATIONS),
        key=lambda t: t[0][2])
    out["Diffusion map"] = (base.nrmse(dm_test, y_test), dm_validation,
                            time.perf_counter() - clock + compress_time, str(choice))

    clock = time.perf_counter()
    (kr_test, kr_validation), choice = min(
        ((base.kernel_ridge_fit(represented[r][0], y, represented[r][1]), r) for r in base.REPRESENTATIONS),
        key=lambda t: t[0][1])
    out["Kernel ridge"] = (base.nrmse(kr_test, y_test), kr_validation,
                           time.perf_counter() - clock + compress_time, str(choice))

    if "Supervised QCML" in readouts:
        clock = time.perf_counter()
        out["Supervised QCML"] = (base.nrmse(base.qcml_readout(Z, y, Z_test, seed), y_test), np.nan,
                                  time.perf_counter() - clock, "n = 8")

    clock = time.perf_counter()
    q_test, (q_val, _), _, choice = base.qcml_plus(Z, y, Z_test, seed)
    out["QCML+"] = (base.nrmse(q_test, y_test), float(np.mean((q_val - y[base.VAL_ROWS]) ** 2) / variance),
                    time.perf_counter() - clock, f"k = {choice[0]}, n = {choice[1]}")
    return out


def run_job(job):
    """One record: ("weak", runs, None, delay, draw) or ("projective", shots, size, delay, draw)."""
    kind, count, size, delay, draw = job
    if kind == "weak":
        train, test, noise = base.measure("weak", count, GAMMA, draw)
        return job, evaluate(train, test, noise, delay, draw, WEAK_READOUTS)
    train, test, noise = base.measure("projective", size, count, draw)
    return job, evaluate(train, test, noise, delay, draw, PROJECTIVE_READOUTS)


def interpolate_time(times, errors, level):
    """The experiment time at which a curve of (time, error), falling with time, first reaches
    `level`, interpolated linearly in log time and log error. None if it never does."""
    times, errors = np.log(np.asarray(times)), np.log(np.asarray(errors))
    if errors[0] <= np.log(level):
        return float(np.exp(times[0]))
    for i in range(1, len(times)):
        if errors[i] <= np.log(level):
            share = (np.log(level) - errors[i - 1]) / (errors[i] - errors[i - 1])
            return float(np.exp(times[i - 1] + share * (times[i] - times[i - 1])))
    return None


if __name__ == "__main__":
    jobs = ([("weak", runs, None, delay, draw) for runs in RUNS for delay in DELAYS for draw in range(DRAWS)]
            + [("projective", shots, size, delay, draw) for shots in SHOTS for size in SETS
               for delay in DELAYS for draw in range(DRAWS)])
    jobs.sort(key=lambda job: job[0] != "weak")          # the longer weak jobs first
    start = time.time()
    if "--saved" in sys.argv:                            # redo the tables from the saved raw results
        with open("qcml111326_raw.pkl", "rb") as file:
            results = pickle.load(file)
    else:
        print(f"QCML111326: {len(jobs)} jobs on {WORKERS} processes "
              f"(weak: {len(RUNS)} run counts; projective: {len(SHOTS)} shot counts x {len(SETS)} measured "
              f"sets; x {len(DELAYS)} delays x {DRAWS} draws)", flush=True)
        results = {}
        with ProcessPoolExecutor(WORKERS) as pool:
            for done, future in enumerate(as_completed([pool.submit(run_job, job) for job in jobs]), 1):
                job, out = future.result()
                results[job] = out
                if done % 12 == 0 or done == len(jobs):
                    print(f"   {done}/{len(jobs)} jobs done ({(time.time() - start) / 60:.0f} min)", flush=True)
        with open("qcml111326_raw.pkl", "wb") as file:
            pickle.dump(results, file)

    # PART A: the weak record
    weak = {}            # (runs, delay, readout) -> per-draw test NRMSE
    with open("qcml111326_weak.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["delay", "runs", "experiment time (s)", "readout", "NRMSE mean", "NRMSE std"]
                        + [f"NRMSE draw {d + 1}" for d in range(DRAWS)]
                        + ["fit seconds, median", "choices"])
        for delay in DELAYS:
            for runs in RUNS:
                for name in WEAK_READOUTS:
                    cells = [results[("weak", runs, None, delay, d)][name] for d in range(DRAWS)]
                    values = np.array([c[0] for c in cells])
                    weak[(runs, delay, name)] = values
                    writer.writerow([delay, runs, weak_time(runs), name, values.mean(), values.std()]
                                    + list(values) + [np.median([c[2] for c in cells]),
                                                      "; ".join(c[3] for c in cells)])
    print("\nPART A: weak record (Gamma = 1e9), test NRMSE, mean over draws")
    for delay in DELAYS:
        print(f"\n   delay {delay}:   runs  time      " + "".join(f"{n:>18}" for n in WEAK_READOUTS))
        for runs in RUNS:
            print(f"   {runs:12.0e}  {weak_time(runs):8.3g} s " + "".join(
                f"{weak[(runs, delay, n)].mean():18.5f}" for n in WEAK_READOUTS))
    print("saved to qcml111326_weak.csv")

    # PART B: the projective record, each readout on the measured set it validates best
    projective, chosen_sets = {}, {}
    with open("qcml111326_projective_sets.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["delay", "shots", "measured set", "readout", "NRMSE mean", "NRMSE std",
                         "validation error mean", "fit seconds, median", "choices"])
        for delay in DELAYS:
            for shots in SHOTS:
                for size, label in SETS.items():
                    for name in PROJECTIVE_READOUTS:
                        cells = [results[("projective", shots, size, delay, d)][name] for d in range(DRAWS)]
                        values = np.array([c[0] for c in cells])
                        writer.writerow([delay, shots, label, name, values.mean(), values.std(),
                                         np.mean([c[1] for c in cells]), np.median([c[2] for c in cells]),
                                         "; ".join(c[3] for c in cells)])
    with open("qcml111326_projective.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["delay", "shots", "experiment time (s)", "readout", "NRMSE mean", "NRMSE std"]
                        + [f"NRMSE draw {d + 1}" for d in range(DRAWS)] + ["measured set chosen per draw"])
        for delay in DELAYS:
            for shots in SHOTS:
                for name in PROJECTIVE_READOUTS:
                    values, picked = [], []
                    for d in range(DRAWS):
                        size = min(SETS, key=lambda s: results[("projective", shots, s, delay, d)][name][1])
                        values.append(results[("projective", shots, size, delay, d)][name][0])
                        picked.append(SETS[size])
                    values = np.array(values)
                    projective[(shots, delay, name)], chosen_sets[(shots, delay, name)] = values, picked
                    writer.writerow([delay, shots, projective_time(shots), name, values.mean(), values.std()]
                                    + list(values) + ["; ".join(picked)])
    print("\nPART B: projective record, each readout on its best measured set (by validation), "
          "test NRMSE, mean over draws")
    for delay in DELAYS:
        print(f"\n   delay {delay}:   shots   time      " + "".join(f"{n:>18}" for n in PROJECTIVE_READOUTS))
        for shots in SHOTS:
            print(f"   {shots:12.3g}  {projective_time(shots):9.3g} s " + "".join(
                f"{projective[(shots, delay, n)].mean():18.5f}" for n in PROJECTIVE_READOUTS)
                + "   sets: " + ", ".join(f"{n.split()[0]} {'/'.join(s.split()[0] for s in chosen_sets[(shots, delay, n)])}"
                                         for n in PROJECTIVE_READOUTS))
    print("saved to qcml111326_projective.csv and qcml111326_projective_sets.csv")

    # Check against QCML110226
    print("\nCHECK against QCML110226 (this run / saved, mean test NRMSE):")
    with open("qcml110226_readouts.csv") as file:
        saved = {(int(r["delay"]), r["setting"]): r for r in csv.DictReader(file)}
    pairs = [("weak record, 1e4 runs", ("weak", 1e4, None)), ("weak record, 1e6 runs", ("weak", 1e6, None)),
             ("vacuum only, 9^5 shots", ("projective", 9 ** 5, 10)), ("40 numbers, 9^5 shots", ("projective", 9 ** 5, 40)),
             ("all 160, 9^5 shots", ("projective", 9 ** 5, 160))]
    for delay in DELAYS:
        for label, (kind, count, size) in pairs:
            row = saved[(delay, label)]
            here = lambda name: np.mean([results[(kind, count, size, delay, d)][name][0] for d in range(DRAWS)])
            names = [("Diffusion map", "Diffusion map"), ("Kernel ridge", "Kernel ridge"), ("QCML+", "QCML+")]
            if kind == "weak":
                names.append(("Supervised QCML", "QCML (as before)"))
            print(f"   delay {delay:>2}, {label:<24}" + "".join(
                f"   {mine} {here(mine) / float(row[f'{theirs} mean']):.4f}" for mine, theirs in names))

    # The comparisons the results chapter quotes
    print("\nEXPERIMENT TIME to match QCML+ on the weak record with 1e4 runs (2 s):")
    for delay in DELAYS:
        level = weak[(1e4, delay, "QCML+")].mean()
        print(f"\n   delay {delay}: QCML+ weak 1e4 runs = {level:.5f}")
        for name in PROJECTIVE_READOUTS:
            curve = [projective[(s, delay, name)].mean() for s in SHOTS]
            needed = interpolate_time([projective_time(s) for s in SHOTS], curve, level)
            print(f"      projective, {name:<18}: " + ("never within 9^9 shots" if needed is None else
                  f"{needed:.4g} s = {needed / 3600:.3g} h ({needed / 2:.3g} times longer)"))
        for name in WEAK_READOUTS:
            curve = [weak[(r, delay, name)].mean() for r in RUNS]
            needed = interpolate_time([weak_time(r) for r in RUNS], curve, level)
            print(f"      weak,       {name:<18}: " + ("never within 1e8 runs" if needed is None else
                  f"{needed:.4g} s ({needed / 2:.3g} times longer, {needed / 2e-4:.3g} runs)"))

    print("\nFIT TIME per readout (median seconds over all jobs of each protocol):")
    for kind, names in [("weak", WEAK_READOUTS), ("projective", PROJECTIVE_READOUTS)]:
        print(f"   {kind}: " + ", ".join(
            f"{n} {np.median([out[n][2] for job, out in results.items() if job[0] == kind]):.1f}" for n in names))
    for name in WEAK_READOUTS:
        by_runs = [np.median([results[("weak", r, None, d, k)][name][2] for d in DELAYS for k in range(DRAWS)])
                   for r in RUNS]
        print(f"   weak, {name:<18} median seconds by runs: " + ", ".join(f"{t:.1f}" for t in by_runs))
    print(f"\ntotal {(time.time() - start) / 60:.0f} min")


# ========================================================================================
# EXPLANATION
# Notes on the code above, in the order of its sections. A tag [note k] in the code
# marks the section that note k explains.
# ========================================================================================
#
# [note 0] About this script
#
# Builds on QCML110226 and imports its reservoir, records, noise weights (measured from two
# halves) and readouts unchanged. QCML110226 scored the readouts at two numbers of weak runs
# (1e4 and 1e6) and one number of projective shots (9^5). Here both are swept:
#   Part A  the weak record at Gamma = 1e9 with R = 1e2, 1e3, ..., 1e8 runs, read by ridge
#           regression (the linear readout), the diffusion map, kernel ridge regression,
#           supervised QCML and QCML+, all on the improved inputs (history + noise weights);
#   Part B  the projective record with 9^3, 9^4, ..., 9^9 shots, for each measured set
#           (vacuum only, the 40 values, all 160), read by ridge regression, the diffusion
#           map, kernel ridge regression and QCML+. A projective shot reports the photon
#           numbers of both oscillators, so all 16 probabilities come from the same shots and
#           the measured set costs no extra time: each readout therefore takes the set with
#           its lowest validation error, as an experimenter would.
# No blend. Each readout's fitting time is recorded, to show that more runs lengthen the
# experiment but not the computation (the averaged record has the same size).
#
# Experiment time (as in QCML102426_2):
#   weak        runs x 2000 inputs x 100 ns                      (1e4 runs = 2 s)
#   projective  shots x 10 samples x 2000 inputs x 10 replays x 100 ns   (9^5 shots = 20 min)
#
# Check: the cells QCML110226 already scored (weak 1e4 and 1e6 runs, projective 9^5 shots)
# must reproduce its saved numbers.
#
# Runs in about 45 minutes on 8 cores (168 jobs, one process each). Run with python3.13 QCML111326.py. Saves qcml111326_weak.csv, qcml111326_projective.csv and
# qcml111326_projective_sets.csv.
