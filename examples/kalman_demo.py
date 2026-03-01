"""
Kalman Smoothing demo  (mirrors 515Examples/KalmanDemo.m).

Demonstrates state estimation with the PLQ framework using:

1. **Standard Kalman smoother** — ℓ₂ measurement + ℓ₂ process
2. **Robust Kalman smoother** — Huber measurement + Huber process
3. **Constrained** variants of both (box constraints on the state)

The process model is a random-walk-on-velocity model with block-tridiagonal
structure, handled through the ``K_proc`` / ``k_proc`` interface.
"""

import os
import numpy as np
import scipy.sparse as sp
from ipsolve import solve, huber, l2


def blktridiag(Amd, Asub, Asup, N):
    """Build a sparse block tridiagonal matrix (N blocks).

    Parameters
    ----------
    Amd : (p, q) – main diagonal block (repeated N times)
    Asub : (p, q) – sub-diagonal block (repeated N−1 times)
    Asup : (p, q) – super-diagonal block (repeated N−1 times)
    N : int – number of block rows / columns

    Returns
    -------
    A : (N*p, N*q) sparse CSC matrix
    """
    p, q = Amd.shape
    blocks = [[None] * N for _ in range(N)]
    Z = sp.csc_matrix((p, q))
    for i in range(N):
        for j in range(N):
            if i == j:
                blocks[i][j] = sp.csc_matrix(Amd)
            elif i == j + 1:
                blocks[i][j] = sp.csc_matrix(Asub)
            elif i == j - 1:
                blocks[i][j] = sp.csc_matrix(Asup)
            else:
                blocks[i][j] = Z
    return sp.bmat(blocks, format="csc")


def main():
    np.random.seed(1234)

    # ── Signal parameters ──────────────────────────────────────────────
    N = 100           # number of time points
    n = 2             # state dimension (position, velocity)
    m = 1             # measurements per time point
    num_periods = 4
    # MATLAB: dt = numP*2*pi/(N*mult) with numP=mult → 2*pi/N
    dt = 2 * np.pi / N

    # True signal
    t = np.arange(1, N + 1) * dt
    x_true = np.exp(np.sin(4 * t))     # f(t) = exp(sin(4t))

    # Noise and outliers
    sigma = 0.05        # measurement noise std
    gamma = 0.1          # process noise multiplier
    out_frac = 0.10      # outlier fraction
    out_mag = 5.0        # outlier magnitude

    n_out = int(N * out_frac)
    gauss_noise = sigma * np.random.randn(N)
    outliers = np.zeros(N)
    out_idx = np.random.permutation(N)[:n_out]
    outliers[out_idx] = out_mag * np.random.randn(n_out)

    z_raw = x_true + gauss_noise + outliers

    # ── Measurement model ──────────────────────────────────────────────
    # H_mat picks out the second state component (velocity -> position map)
    # and scales by 1/sigma for normalisation.
    rinvk = 1.0 / (sigma ** 2)
    rh = np.sqrt(rinvk) * np.array([[0.0, 1.0]])      # (1, 2)
    H_mat = sp.kron(sp.eye(N, format="csc"), sp.csc_matrix(rh), format="csc")
    meas = np.sqrt(rinvk) * z_raw   # scaled measurements

    # ── Process model ──────────────────────────────────────────────────
    # Random walk on velocity:  x_{k+1} = G_k x_k + w_k
    qk = gamma * np.array([[dt, dt**2 / 2],
                            [dt**2 / 2, dt**3 / 3]])
    sqrtQinv = np.linalg.inv(np.sqrt(qk))  # MATLAB uses element-wise sqrt
    gk = sqrtQinv @ np.array([[1.0, 0.0],
                               [dt,  1.0]])

    G_mat = blktridiag(sqrtQinv @ np.eye(n), -gk, np.zeros((n, n)), N)

    # Override first block to anchor the initial state
    init_weight = 10.0
    G_mat[:n, :n] = init_weight * sp.eye(n)
    w = np.zeros(n * N)
    w[:n] = init_weight * x_true[:1]  # anchor to true initial position (only position)

    # ── Box constraints for constrained estimates ──────────────────────
    # State 2 (position) in [exp(-1), exp(1)]
    con_block = np.array([[0.0, 1.0],
                            [0.0, -1.0]])
    con_rhs_block = np.array([np.exp(1), -np.exp(-1)])
    A_con = sp.kron(sp.eye(N, format="csc"), sp.csc_matrix(con_block), format="csc")
    a_con = np.tile(con_rhs_block, N)

    # ── 1. Standard Kalman smoother (ℓ₂ / ℓ₂) ─────────────────────────
    print("=" * 70)
    print("1. Standard Kalman smoother  (ℓ₂ / ℓ₂)")
    print("=" * 70)
    y_ls = solve(H_mat, meas, proc=l2(),
                 K_proc=G_mat, k_proc=w, silent=False)
    x_ls = y_ls.reshape(N, n)

    # ── 2. Robust Kalman smoother (Huber / Huber) ──────────────────────
    print("\n" + "=" * 70)
    print("2. Robust Kalman smoother  (Huber / Huber)")
    print("=" * 70)
    y_rob = solve(H_mat, meas, meas=huber(), proc=huber(),
                  K_proc=G_mat, k_proc=w, silent=False)
    x_rob = y_rob.reshape(N, n)

    # ── 3. Constrained Kalman smoother (ℓ₂ / ℓ₂ + box) ───────────────
    print("\n" + "=" * 70)
    print("3. Constrained Kalman smoother  (ℓ₂ / ℓ₂ + box)")
    print("=" * 70)
    y_con = solve(H_mat, meas, proc=l2(),
                  K_proc=G_mat, k_proc=w,
                  A_ineq=A_con, a_ineq=a_con, silent=False)
    x_con = y_con.reshape(N, n)

    # ── 4. Constrained + Robust (Huber / Huber + box) ─────────────────
    print("\n" + "=" * 70)
    print("4. Constrained Robust Kalman  (Huber / Huber + box)")
    print("=" * 70)
    y_con_rob = solve(H_mat, meas, meas=huber(), proc=huber(),
                      K_proc=G_mat, k_proc=w,
                      A_ineq=A_con, a_ineq=a_con, silent=False)
    x_con_rob = y_con_rob.reshape(N, n)

    # ── Report ─────────────────────────────────────────────────────────
    err = lambda xhat: np.sqrt(np.mean((xhat[:, 1] - x_true) ** 2))
    print("\n" + "=" * 70)
    print("RMSE of position estimates:")
    print(f"  Standard LS:          {err(x_ls):.4f}")
    print(f"  Robust Huber:         {err(x_rob):.4f}")
    print(f"  Constrained LS:       {err(x_con):.4f}")
    print(f"  Constrained Robust:   {err(x_con_rob):.4f}")
    print("=" * 70)

    # ── Plot (optional) ───────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

        for ax, title, x_est1, x_est2, lab1, lab2 in [
            (axes[0], "Unconstrained", x_ls, x_rob, "Least Squares", "Robust"),
            (axes[1], "Constrained",   x_con, x_con_rob, "Least Squares", "Robust"),
        ]:
            ax.plot(t, x_true, "r-", lw=2, label="Truth")
            ax.plot(t, z_raw, "ko", ms=2, alpha=0.5, label="Measurements")
            ax.plot(t, x_est1[:, 1], "b--", lw=2, label=lab1)
            ax.plot(t, x_est2[:, 1], "m-.", lw=2, label=lab2)
            ax.set_title(title)
            ax.legend(loc="lower right", fontsize=8)
            ax.set_xlabel("t")

        fig.tight_layout()
        os.makedirs("figures", exist_ok=True)
        fig.savefig("figures/kalman_demo.png", dpi=150)
        print("\nPlot saved to kalman_demo.png")
    except Exception as e:
        print(f"\nSkipping plot: {e}")


if __name__ == "__main__":
    main()
