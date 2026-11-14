"""QCML: represent data points as quantum states via learned Hermitian observables."""

import torch
import torch.nn as nn

from hermitian import HermitianMatrices


class QCML(nn.Module):
    def __init__(self, D: int, n: int, w: float = 1.0, scale: float | None = None):
        super().__init__()
        self.D, self.n, self.w = D, n, float(w)
        self.observables = HermitianMatrices(D, n, scale=scale)

    def error_hamiltonian(self, x: torch.Tensor) -> torch.Tensor:
        """E(x) for a batch. x: (B, D) real -> (B, n, n) complex Hermitian.

        Expanded as sum_d H_d^2 - 2 x_d H_d + x_d^2 I so the matrix-matrix
        products happen once per step rather than once per data point.
        """
        H = self.observables()                                  # (D, n, n)
        I = torch.eye(self.n, dtype=H.dtype, device=H.device)
        H_sq = torch.einsum("dij,djk->ik", H, H)                # (n, n)
        xH = torch.einsum("bd,dij->bij", x.to(H.dtype), H)      # (B, n, n)
        x_sq = (x ** 2).sum(-1).to(H.dtype)[:, None, None]      # (B, 1, 1)
        return H_sq - 2.0 * xH + x_sq * I

    def ground_state(self, x: torch.Tensor) -> torch.Tensor:
        """phi_0(x): lowest eigenvector of E(x). (B, D) -> (B, n) complex.

        Differentiable, so gradients reach the observables through it.
        """
        return torch.linalg.eigh(self.error_hamiltonian(x))[1][:, :, 0]

    def bloch(self, psi: torch.Tensor) -> torch.Tensor:
        """B(psi)_d = <psi|H_d|psi>. (B, n) -> (B, D) real."""
        return torch.einsum("bi,dij,bj->bd", psi.conj(), self.observables(), psi).real

    def uncertainty(self, psi: torch.Tensor) -> torch.Tensor:
        """sigma^2 = sum_d Var(H_d) at psi. (B, n) -> (B,) real."""
        H = self.observables()
        H_sq = torch.einsum("dij,djk->ik", H, H)                # sum_d H_d^2
        second = torch.einsum("bi,ij,bj->b", psi.conj(), H_sq, psi).real
        return second - (self.bloch(psi) ** 2).sum(-1)

    def per_point_loss(self, x: torch.Tensor) -> torch.Tensor:
        """||B(phi_0(x)) - x||^2 + w * sigma^2(x), per data point. (B,) real."""
        psi = self.ground_state(x)
        return ((self.bloch(psi) - x) ** 2).sum(-1) + self.w * self.uncertainty(psi)

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """Mean per-point loss over the batch. This is the training objective."""
        return self.per_point_loss(x).mean()

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """B(phi_0(x)) -- the denoised version of x. (B, D) real."""
        return self.bloch(self.ground_state(x))

    @torch.no_grad()
    def quantum_metric(self, x: torch.Tensor) -> torch.Tensor:
        """Quantum (Fubini-Study) metric of phi_0 at each x: (B, D, D).

        Measures how fast the represented state rotates as x is nudged in each
        direction, so the eigenvalue spectrum reveals intrinsic dimension.
        Uses first-order perturbation theory rather than autograd through eigh:
        dE/dx_i = -2(H_i - x_i I), and <k|dE/dx_i|0> = -2<k|H_i|0> for k != 0.
        """
        energies, vecs = torch.linalg.eigh(self.error_hamiltonian(x))
        H = self.observables()
        psi0 = vecs[:, :, 0]
        M = torch.einsum("bnk,dnm,bm->bdk", vecs.conj(), H, psi0)   # <k|H_d|0>
        gaps = energies[:, 0:1] - energies
        weight = torch.zeros_like(gaps)
        weight[:, 1:] = 4.0 / gaps[:, 1:] ** 2                      # 4 from the -2
        return torch.einsum("bdk,bek,bk->bde", M.conj(), M, weight.to(M.dtype)).real

    def extra_repr(self) -> str:
        return f"D={self.D}, n={self.n}, w={self.w}"


def train(model, X, epochs=300, batch_size=256, lr=1e-2, log_every=50, seed=0):
    """Plain minibatch Adam. Nothing QCML-specific here."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    N = X.shape[0]
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(N, generator=g)
        total = 0.0
        for i in range(0, N, batch_size):
            xb = X[perm[i:i + batch_size]]
            loss = model.loss(xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * xb.shape[0]
        if log_every and (epoch % log_every == 0 or epoch == 1):
            print(f"  epoch {epoch:4d}   loss {total / N:.5f}")
    return model


# ========================================================================================
# EXPLANATION
# ========================================================================================
#
# [note 0] From the module docstring
#
# Pipeline
# --------
# 1. Learn D Hermitian matrices H_1..H_D on an n-dimensional space (hermitian.py).
# 2. For a data point x, build the error Hamiltonian
#        E(x) = sum_d (H_d - x_d * I)^2
#    Its ground state phi_0(x) is the quantum state representing x.
# 3. Map that state back to feature space with the Bloch map
#        B(psi)_d = <psi| H_d |psi>
# 4. Train on
#        L = sum_x [ ||B(phi_0(x)) - x||^2  +  w * sigma^2(x) ]
#    where sigma^2(x) = sum_d Var_{phi_0(x)}(H_d) is the total quantum uncertainty.
#
# The weight w trades reconstruction accuracy against state sharpness. At w = 1
# the bracket collapses to <phi_0|E(x)|phi_0>, the ground-state energy, which is
# standard QCML; other w reweight the two halves of that identity.
