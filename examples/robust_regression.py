"""
Robust Regression example – Huber loss.

Solves:  min  Σ huber(z_i - (Hx)_i, κ)

Compares with ordinary least squares.
Demonstrates robustness to outliers.
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ipsolve import solve, huber


def main():
    rng = np.random.default_rng(1)

    m, n = 200, 20
    x_true = rng.standard_normal(n)
    H = rng.standard_normal((m, n))
    z = H @ x_true + 0.1 * rng.standard_normal(m)

    # Add outliers
    n_outliers = 20
    idx = rng.choice(m, n_outliers, replace=False)
    z[idx] += 10 * rng.standard_normal(n_outliers)

    # LS solve
    x_ls = solve(H, z, meas="l2", silent=True)

    # Huber solve
    x_huber = solve(H, z, meas=huber(kappa=1.0), silent=True)

    # Compare
    err_ls = np.linalg.norm(x_ls - x_true)
    err_hub = np.linalg.norm(x_huber - x_true)
    print(f"LS error:    {err_ls:.4f}")
    print(f"Huber error: {err_hub:.4f}")
    print(f"Improvement: {err_ls / err_hub:.1f}x")

    # Plot residuals
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].stem(np.arange(m), z - H @ x_ls, linefmt="b-", markerfmt="b.", basefmt=" ")
    axes[0].set_title(f"LS residuals  (||x-x*||={err_ls:.3f})")
    axes[1].stem(np.arange(m), z - H @ x_huber, linefmt="r-", markerfmt="r.", basefmt=" ")
    axes[1].set_title(f"Huber residuals  (||x-x*||={err_hub:.3f})")
    for ax in axes:
        ax.set_xlabel("measurement index")
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/robust_regression.png", dpi=120)
    print("Saved robust_regression.png")


if __name__ == "__main__":
    main()
