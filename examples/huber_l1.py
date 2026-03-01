"""
Huber + L1 example (robust sparse regression).

Solves:  min  Σ huber(zᵢ − (Hx)ᵢ, κ)  +  λ ||x||₁

Compares IPsolve with CVXPY.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ipsolve import solve, huber, l1

try:
    import cvxpy
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False


def main():
    rng = np.random.default_rng(7)

    m, n, k = 300, 100, 15
    x0 = np.zeros(n)
    perm = rng.permutation(n)
    x0[perm[:k]] = rng.standard_normal(k)

    H = rng.standard_normal((m, n))
    z = H @ x0 + 0.1 * rng.standard_normal(m)

    # Add sparse large noise (outliers)
    n_out = 30
    idx = rng.choice(m, n_out, replace=False)
    z[idx] += 10 * rng.standard_normal(n_out)

    kappa = 1.0
    lam = 0.5

    # IPsolve
    x_ip = solve(H, z, meas=huber(kappa=kappa), proc=l1(lam=lam))

    def huber_val(v, k):
        return np.sum(np.where(np.abs(v) > k, np.abs(v) - 0.5 * k, 0.5 * v ** 2 / k))

    obj_ip = huber_val(z - H @ x_ip, kappa) + lam * np.sum(np.abs(x_ip))
    print(f"\nIPsolve objective: {obj_ip:.6f}")
    print(f"Recovery error:    {np.linalg.norm(x_ip - x0):.4f}")

    if HAS_CVXPY:
        x_cv = cvxpy.Variable(n)
        prob = cvxpy.Problem(cvxpy.Minimize(
            0.5 * cvxpy.sum(cvxpy.huber(z - H @ x_cv, M=kappa))
            + lam * cvxpy.norm1(x_cv)
        ))
        prob.solve(solver=cvxpy.CLARABEL, verbose=False)
        obj_cv = prob.value
        print(f"CVXPY objective:   {obj_cv:.6f}")
        print(f"Relative diff:     {abs(obj_cv - obj_ip) / max(obj_cv, 1e-12):.2e}")

    # Plot: true vs recovered signal + residuals
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.stem(np.arange(n), x0, linefmt="b-", markerfmt="bo", basefmt=" ", label="truth")
    ax.stem(np.arange(n), x_ip, linefmt="r-", markerfmt="r.", basefmt=" ", label="Huber+L1")
    ax.legend()
    ax.set_title("Signal recovery")
    ax.set_xlabel("index")

    ax = axes[1]
    r = z - H @ x_ip
    ax.stem(np.arange(m), r, linefmt="g-", markerfmt="g.", basefmt=" ")
    ax.axhline(kappa, ls="--", c="k", lw=0.8, label=f"κ = {kappa}")
    ax.axhline(-kappa, ls="--", c="k", lw=0.8)
    ax.set_title("Residuals")
    ax.set_xlabel("measurement index")
    ax.legend()
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/huber_l1_example.png", dpi=120)
    print("Saved huber_l1_example.png")


if __name__ == "__main__":
    main()
