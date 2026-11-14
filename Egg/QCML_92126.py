# Quantum Cognitive Machine Learning Project
# Julian Green 9/21/26
# Egg-Model

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Repository set-up: run from this folder
import os
import sys
_FOLDER = os.path.dirname(os.path.abspath(__file__))
os.chdir(_FOLDER)                                                 # data and outputs live in this folder


# The three numbers need to be defined to create the egg
L = 3.0   # length:    the egg runs from x = -L/2 to x = +L/2
B = 2.0   # breadth:   the height at its widest point
w = 0.2   # shift:     the position of its widest point

N = 2500                   # number of points
VARIANCE = 0.01            # noise variance
SIGMA = np.sqrt(VARIANCE)  # noise standard deviation
SEED = 0                   # noise seed

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


# Data Normalization
data = np.column_stack([x_noisy, y_noisy]) # Store data in a column          
mean = data.mean(axis=0)   # find the average of x and y
scale = np.sqrt(((data - mean) ** 2).sum(axis=1).mean()) # find average distance each point is from average of x and y
X = torch.tensor((data - mean) / scale, dtype=torch.float32) # shift to center around 0


## Making Hermitian Matrix (from hermitian.py)

class HermitianMatrices(nn.Module):
    """A stack of D learnable n x n Hermitian matrices.

    Args:
        D: number of matrices (e.g. one observable per feature).
        n: dimension of each matrix.
        real: if True, produce real symmetric matrices instead of complex
            Hermitian ones. n(n+1)/2 free params each instead of n^2.
        scale: std of the initial entries. The default 1/sqrt(n) puts the
            spectrum on an O(1) scale (Wigner semicircle radius ~2).

    Returns from forward(): tensor of shape (D, n, n), dtype complex64
    (or float32 when real=True).
    """

    def __init__(self, D: int, n: int, real: bool = False, scale: float | None = None):
        super().__init__()
        self.D, self.n, self.real = D, n, real
        if scale is None:
            scale = n ** -0.5
        
        # One real tensor holds every free parameter. For the complex case the
        # lower triangle (with diagonal) feeds the real part and the strict
        # upper triangle feeds the imaginary part -- exactly n^2 reals each.

        self.raw = nn.Parameter(torch.randn(D, n, n) * scale) # starting random numbers

    def forward(self) -> torch.Tensor:
        raw = self.raw # the random #s!

        lower = raw.tril()                      # lower triangle incl. diagonal
        reals = lower + lower.tril(-1).mT         # symmetric: mirror off-diagonals on top right (lower.tril(-1).mT) and then add

        if self.real:
            return reals # no need for imag parts
        
        upper = raw.triu(1)                     # look at upper triangle for random #s to become imag part
        imags = upper - upper.mT                 # sum of imag parts: upper.mT takes neg. mirror so antisymmetric, zero diagonal
        return torch.complex(reals, imags) # sym is real part, skew is imag

    def eigvals(self) -> torch.Tensor:
        """Real eigenvalues, shape (D, n), ascending. Differentiable."""
        return torch.linalg.eigvalsh(self())

    def extra_repr(self) -> str:
        return f"D={self.D}, n={self.n}, real={self.real}"


if __name__ == "__main__":
    torch.manual_seed(0)

    for real in (False, True):
        H_mod = HermitianMatrices(D=2, n=4, real=real)
        H = H_mod()
        # Hermitian check: H == H^dagger
        err = (H - H.mH).abs().max().item()
        print(f"real={real}  shape={tuple(H.shape)}  dtype={H.dtype}  "
              f"max |H - H^H| = {err:.2e}")
        print("  eigenvalues:", H_mod.eigvals().detach().numpy().round(3).tolist())

    # Gradients reach the single underlying parameter, and the matrices stay
    # Hermitian after an optimizer step.
    H_mod = HermitianMatrices(D=2, n=4)
    opt = torch.optim.Adam(H_mod.parameters(), lr=1e-2)
    loss = H_mod.eigvals().pow(2).sum()         # arbitrary scalar objective
    opt.zero_grad()
    loss.backward()
    opt.step()
    H = H_mod()
    print(f"after step: grad norm={H_mod.raw.grad.norm():.4f}  "
          f"max |H - H^H| = {(H - H.mH).abs().max().item():.2e}")
##

# Plot
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

