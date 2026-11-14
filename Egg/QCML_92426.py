# Quantum Cognitive Machine Learning Project
# Julian Green 9/22/26
# Egg-Model

import numpy as np
import torch
import matplotlib as plt

# Repository set-up: run from this folder, with QCML from 1_Plain_QCML
import os
import sys
_FOLDER = os.path.dirname(os.path.abspath(__file__))
os.chdir(_FOLDER)                                                 # data and outputs live in this folder
sys.path.insert(0, os.path.join(_FOLDER, "..", "1_Plain_QCML"))   # qcml.py and hermitian.py

from qcml import QCML

# We need to define 3 key numbers for the egg
L = 3.0 # length, the egg plot runs from -L/2 to L/2
H = 2.0 # height at its tallest point
W = .4  # widest point, the x position of the eggs widest point

N = 2500                        # Number of points
Variance = 0.01                 # Noise Variance
Sigma = np.sqrt(Variance)       # Standard Deviation
Seed = 0                        # Randomness set according to this preset number

# Make space to hold the points
theta_line = np.linspace(0.0,2.0 * np.pi, 500)

# Formula for an egg
x = (L / 2) * np.cos(theta_line)
y = (H / 2) * L * np.sin(theta_line) / np.sqrt(L**2 + 4 * W * L * np.cos(theta_line) + 4 * W**2)

# function for random numbers for noise
rng = np.random.default_rng(Seed)

# select random angles for where each random point falls
theta = rng.uniform(0.0, 2.0 * np.pi, size=N)

# start with points that are exactly on the egg
x_clean = (L/2) * np.cos(theta)
y_clean = (H / 2) * L * np.sin(theta_line) / np.sqrt(L**2 + 4 * W * L * np.cos(theta_line) + 4 * W**2)

# Shift points off the egg by a random amount given by a normal distribution
noise = rng.normal(loc = 0.0, scale = Sigma, size = (N,2))
x_noisy = x_clean + noise[:, 0]
y_noisy = y_clean + noise[:, 1]

# Organize data into a table
data = np.column_stack([x_noisy,y_noisy])
clean = np.column_stack([x_clean, y_clean]) 

# Settings for the Model
n = 8 # matrix size
C1 = 1.0 # position error weight
C2 = .1 # quantum fluctuation weight
EPOCHS = 100 # Runs through training the data set
BATCH = 250 # Data points processed at once

def compute_loss(model, points):
    """Calculates Loss, L1 & L2, for a batch of points using C1*L1 + C2*L2"""
    psi = model.ground_state(points)                            # ground state for every point
    L1 = (model.bloch(psi) - points ** 2).sum(dim=1).mean()     # average distance squared
    L2 = model.uncertainty(psi).mean()                          # avergae fluctuation
    return C1*L1 + C2*L2