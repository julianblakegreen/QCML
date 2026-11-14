"""Learnable Hermitian (or real-symmetric) matrices.

Parameterizes D matrices of size n x n that are Hermitian *by construction*,
so they stay Hermitian at every optimizer step -- no projection or penalty
needed. Gradients flow through the symmetrization, which is linear.
"""

import torch
import torch.nn as nn


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
