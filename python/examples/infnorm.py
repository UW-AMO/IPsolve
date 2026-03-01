"""
Infinity-norm regularisation  (mirrors 515Examples/infNorm.m).

Solves the compressive sensing problem

    min  0.5‖b − Ax‖²  +  λ‖x‖∞

where the ℓ∞ norm is the process regulariser.

The infinity-norm penalty promotes *spread-out* solutions (opposite of ℓ₁)
and is included in the 515 examples to demonstrate the simplex-constraint
PLQ representation.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ipsolve import solve, infnorm


def main():
    rng = np.random.default_rng(0)

    # Problem dimensions (kept small – infnorm dual space is 2n + 1, making
    # the KKT system larger than for ℓ₁)
    m, n, k = 60, 30, 5

    # True sparse signal
    x0 = np.zeros(n)
    perm = rng.permutation(n)
    x0[perm[:k]] = np.sign(rng.standard_normal(k))

    # Measurement matrix (well-conditioned via SVD truncation)
    A = rng.standard_normal((m, n))
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    A = U @ np.diag(np.ones_like(s)) @ Vt  # orthonormalised rows

    # Noisy observations
    b = A @ x0 + 0.005 * rng.standard_normal(m)

    # Regularisation parameter
    lam = np.max(np.abs(A.T @ b)) / 10

    # Solve:  min 0.5‖b − Ax‖² + λ‖x‖∞
    print("=" * 60)
    print(f"Inf-norm regularisation:  m={m}, n={n}, k={k}, λ={lam:.4f}")
    print("=" * 60)
    x_ip = solve(A, b, proc=infnorm(lam=lam), opt_tol=1e-7)

    # Report
    print(f"\n‖x_true‖∞  = {np.max(np.abs(x0)):.4f}")
    print(f"‖x_IP‖∞    = {np.max(np.abs(x_ip)):.4f}")
    print(f"‖x_true - x_IP‖ / ‖x_true‖ = {np.linalg.norm(x_ip - x0) / np.linalg.norm(x0):.4e}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stem(np.arange(n), x0, linefmt="b-", markerfmt="bo", basefmt=" ", label="truth")
    ax.stem(np.arange(n), x_ip, linefmt="r-", markerfmt="r.", basefmt=" ", label="∞-norm")
    ax.legend()
    ax.set_title(f"Inf-norm recovery  (λ={lam:.3f})")
    ax.set_xlabel("index")
    fig.tight_layout()
    fig.savefig("infnorm_example.png", dpi=120)
    print("Saved infnorm_example.png")


if __name__ == "__main__":
    main()
