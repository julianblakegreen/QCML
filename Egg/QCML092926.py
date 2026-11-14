# Quantum Cognitive Machine Learning Project
# Julian Green 9/22/26
# Egg-Model

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

# The three numbers need to be defined to create the egg
L = 3.0   # length:    the egg runs from x = -L/2 to x = +L/2
B = 2.0   # breadth:   the height at its widest point
w = 0.2   # shift:     the position of its widest point

N = 2500                   # number of points
VARIANCE = 0.01            # noise variance
SIGMA = np.sqrt(VARIANCE)  # noise standard deviation
SEED = 0                   # noise seed

# Chunks cut out of the data (Run 2). Change these to take out more or bigger pieces.
CHUNK_SEED = 1             # seed for WHERE the chunks land (separate from the data seed)
N_CHUNKS = 3               # how many pieces to cut out
CHUNK_RADIUS = 0.35        # radius of each piece, in egg units

# Make points which we calculate then plot, 500 used to make graph appear smooth 
theta_line = np.linspace(0.0, 2.0 * np.pi, 500)

# formula for an egg
x = (L / 2) * np.cos(theta_line)
y = (B / 2) * L * np.sin(theta_line) / np.sqrt(L**2 + 4 * w * L * np.cos(theta_line) + 4 * w**2)

# function for random numbers for noise
rng = np.random.default_rng(SEED)

# select random angles for where each random point falls
theta = rng.uniform(0.0, 2.0 * np.pi, size=N)
 
# start with points exactly on the egg
x_clean = (L / 2) * np.cos(theta)
y_clean = (B / 2) * L * np.sin(theta) / np.sqrt(L**2 + 4 * w * L * np.cos(theta) + 4 * w**2)

# Shift points off the egg by a random normal distribution amount
noise = rng.normal(loc=0.0, scale=SIGMA, size=(N, 2))
x_noisy = x_clean + noise[:, 0]
y_noisy = y_clean + noise[:, 1]

# The data, as one table: 2500 rows (points), 2 columns (x and y)
data = np.column_stack([x_noisy, y_noisy])
clean = np.column_stack([x_clean, y_clean])   # true positions, for scoring only


## Settings for the model
n = 8          # matrix size
C1 = 1.0       # position error weight
C2 = 0.1       # quantum fluctuation weight
EPOCHS = 100
BATCH = 250


def compute_loss(model, points):
    """Loss for a batch of points: C1*L1 + C2*L2. Also returns L1 and L2."""
    psi = model.ground_state(points)                             # ground state for every point
    L1 = ((model.bloch(psi) - points) ** 2).sum(dim=1).mean()    # average squared distance
    L2 = model.uncertainty(psi).mean()                           # average fluctuation
    return C1 * L1 + C2 * L2, L1, L2


def train_qcml(points, label=""):
    """Train a fresh model on `points` (a numpy table of shape (M, 2)).

    Returns the model, the normalization numbers, and the loss history.
    """
    # Data Normalization: mean 0, variance 1 (one scale, so the shape is kept)
    mean = points.mean(axis=0)
    scale = np.sqrt(((points - mean) ** 2).sum(axis=1).mean())
    X = torch.tensor((points - mean) / scale, dtype=torch.float32)
    M = len(X)

    torch.manual_seed(SEED)                       # same starting matrices every run
    model = QCML(D=2, n=n, w=C2 / C1)             # ONE model holding two matrices, A1 and A2

    # Part 1: Adam, small steps on shuffled batches of points
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    history_L1, history_L2 = [], []
    for epoch in range(1, EPOCHS + 1):
        order = torch.randperm(M)                 # shuffle the points
        for start in range(0, M, BATCH):
            batch = X[order[start:start + BATCH]]
            loss, L1, L2 = compute_loss(model, batch)
            optimizer.zero_grad()                 # clear the old gradients
            loss.backward()                       # compute new gradients
            optimizer.step()                      # small step for all 128 numbers

        with torch.no_grad():                     # after each epoch, check on ALL points
            loss, L1, L2 = compute_loss(model, X)
        history_L1.append(L1.item())
        history_L2.append(L2.item())
        if epoch == 1 or epoch % 50 == 0:
            print(f"{label}epoch {epoch:3d}   loss {loss.item():.5f}   "
                  f"L1 {L1.item():.5f}   L2 {L2.item():.5f}")

    # Part 2: L-BFGS polish, using all points at once
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=300,
                              line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        loss, L1, L2 = compute_loss(model, X)
        loss.backward()
        return loss

    lbfgs.step(closure)
    with torch.no_grad():
        loss, L1, L2 = compute_loss(model, X)
    print(f"{label}after L-BFGS   loss {loss.item():.5f}   "
          f"L1 {L1.item():.5f}   L2 {L2.item():.5f}")
    return model, mean, scale, history_L1, history_L2


def reconstruct(model, points, mean, scale):
    """Where the model thinks each point really belongs, back in egg units."""
    X = torch.tensor((points - mean) / scale, dtype=torch.float32)
    return model.reconstruct(X).numpy() * scale + mean          # <Omega|A_k|Omega>


# A dense copy of the true egg, used only to measure how close results are to it
theta_dense = np.linspace(0.0, 2.0 * np.pi, 2000, endpoint=False)
egg_curve = np.column_stack([
    (L / 2) * np.cos(theta_dense),
    (B / 2) * L * np.sin(theta_dense) / np.sqrt(L**2 + 4 * w * L * np.cos(theta_dense) + 4 * w**2)])


def distance_to_egg(points):
    """Distance from each point to the nearest spot on the true egg."""
    gaps = points[:, None, :] - egg_curve[None, :, :]
    return np.sqrt((gaps ** 2).sum(axis=2)).min(axis=1)


# Full Data

print("RUN 1: all the data")
model, mean, scale, history_L1, history_L2 = train_qcml(data)
recon = reconstruct(model, data, mean, scale)

print(f"\n                               noisy data    QCML")
print(f"average distance to the egg      {distance_to_egg(data).mean():.3f}       "
      f"{distance_to_egg(recon).mean():.3f}")
print(f"RMS error vs the clean points    {np.sqrt(((data - clean)**2).sum(1).mean()):.3f}       "
      f"{np.sqrt(((recon - clean)**2).sum(1).mean()):.3f}")


# random chunks cut out of the data
chunk_rng = np.random.default_rng(CHUNK_SEED)          # its own generator, separate from rng
chunk_angles = chunk_rng.uniform(0.0, 2.0 * np.pi, size=N_CHUNKS)   # where each chunk sits

# chunk centres: pick random spots ON the egg, so every chunk removes real points
chunk_centres = np.column_stack([
    (L / 2) * np.cos(chunk_angles),
    (B / 2) * L * np.sin(chunk_angles) / np.sqrt(L**2 + 4 * w * L * np.cos(chunk_angles) + 4 * w**2)])

# a point is cut if it lies within CHUNK_RADIUS of ANY chunk centre
gaps = data[:, None, :] - chunk_centres[None, :, :]                 # (N, N_CHUNKS, 2)
inside_chunk = (np.sqrt((gaps ** 2).sum(axis=2)) < CHUNK_RADIUS).any(axis=1)
keep = ~inside_chunk                                                # True = QCML may see it

print(f"\nRUN 2: {N_CHUNKS} chunks of radius {CHUNK_RADIUS} cut out "
      f"({keep.sum()} of {N} points kept, {100 * keep.mean():.0f}%)")
if keep.sum() < 100:
    raise SystemExit("Almost nothing left to train on: use fewer or smaller chunks.")

model_cut, mean_cut, scale_cut, _, _ = train_qcml(data[keep], label="   [cut] ")

# Send EVERY point through the model, including the ones inside the chunks.
# Where it puts those is its guess for the missing pieces.
recon_cut = reconstruct(model_cut, data, mean_cut, scale_cut)

print(f"\naverage distance to the egg, points it trained on : {distance_to_egg(recon_cut[keep]).mean():.3f}")
print(f"average distance to the egg, points in the chunks : {distance_to_egg(recon_cut[~keep]).mean():.3f}")


# Plots
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

# full-data result 
ax = axes[0]
ax.scatter(x_noisy, y_noisy, s=6, alpha=0.3, color="#94a3b8", label="noisy points")
ax.scatter(recon[:, 0], recon[:, 1], s=6, alpha=0.6, color="red", label="QCML reconstruction")
ax.plot(x, y, "k--", lw=1.2, label="egg formula")
ax.set_title("all data")

# the partial-data result 
ax = axes[1]
ax.scatter(x_noisy[keep], y_noisy[keep], s=6, alpha=0.3, color="#94a3b8",
           label="points given to QCML")
ax.scatter(x_noisy[~keep], y_noisy[~keep], s=6, alpha=0.3, color="#fca5a5",
           label="points cut out")
ax.scatter(recon_cut[keep, 0], recon_cut[keep, 1], s=6, alpha=0.6, color="red",
           label="reconstruction (trained region)")
ax.scatter(recon_cut[~keep, 0], recon_cut[~keep, 1], s=6, alpha=0.8, color="#1d4ed8",
           label="prediction (inside chunks)")
ax.plot(x, y, "k--", lw=1.2, label="egg formula")
for centre in chunk_centres:                       # show where the chunks were
    ax.add_patch(plt.Circle(centre, CHUNK_RADIUS, fill=False, ls=":", lw=1.0, color="black"))
ax.set_title(f"{N_CHUNKS} chunks of radius {CHUNK_RADIUS} removed")

for ax in axes[:2]:
    ax.set_aspect("equal")
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=7)

#  training curves (full-data run) 
ax = axes[2]
epochs = np.arange(1, EPOCHS + 1)
ax.plot(epochs, history_L1, label="L1  position error")
ax.plot(epochs, history_L2, label="L2  fluctuation")
ax.plot(epochs, C1 * np.array(history_L1) + C2 * np.array(history_L2), "k",
        label=f"loss = {C1:g}*L1 + {C2:g}*L2")
ax.set_yscale("log")
ax.set_xlabel("epoch")
ax.set_ylabel("value (normalized units)")
ax.set_title("training, all data")
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig("qcml_egg_results.png", dpi=150)


# the og fuzzy egg picture 
plt.figure(figsize=(6, 6))
plt.scatter(x_noisy, y_noisy, s=6, alpha=0.5, color="blue", label="noisy points")
plt.plot(x, y, "k--", label="egg formula")

plt.gca().set_aspect("equal")   # same scale on x and y, so the egg isn't stretched
plt.xlim(-2.0, 2.0)
plt.ylim(-2.0, 2.0)
plt.xlabel("x")
plt.ylabel("y")
plt.title("fuzzy egg")
plt.legend(loc="upper right")

plt.savefig("fuzzy_egg.png", dpi=150)
plt.show()
