"""
Lasso / Elastic Net example.

Solves:  min 0.5 ||z - Hx||^2 + λ ||x||_1

Generates a sparse signal, measures with a random matrix, and recovers.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ipsolve import solve


def main():
    rng = np.random.default_rng(0)

    # Problem dimensions
    m, n, k = 500, 2000, 50  # measurements, variables, sparsity

    # True sparse signal
    x0 = np.zeros(n)
    perm = rng.permutation(n)
    x0[perm[:k]] = rng.standard_normal(k)

    # Measurement matrix  (well-conditioned via QR)
    A = rng.standard_normal((m, n))
    Q, _ = np.linalg.qr(A.T, mode="reduced")
    A = Q.T

    # Noisy measurements
    z = A @ x0 + 0.005 * rng.standard_normal(m)

    # Regularisation parameter
    lam = np.max(np.abs(A.T @ z)) / 10

    # Solve with IPsolve
    x_lasso = solve(A, z, meas="l2", proc="l1", proc_lambda=lam, silent=False)

    # Report
    nnz_true = np.sum(np.abs(x0) > 1e-8)
    nnz_rec = np.sum(np.abs(x_lasso) > 1e-4)
    err = np.linalg.norm(x_lasso - x0) / max(np.linalg.norm(x0), 1e-12)
    print(f"\nTrue nnz: {nnz_true},  Recovered nnz: {nnz_rec},  Rel error: {err:.4e}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stem(np.arange(n), x0, linefmt="b-", markerfmt="bo", basefmt=" ", label="truth")
    ax.stem(np.arange(n), x_lasso, linefmt="r-", markerfmt="r.", basefmt=" ", label="lasso")
    ax.legend()
    ax.set_title(f"Lasso recovery  (λ={lam:.3f})")
    fig.tight_layout()
    fig.savefig("lasso_example.png", dpi=120)
    print("Saved lasso_example.png")


if __name__ == "__main__":
    main()
