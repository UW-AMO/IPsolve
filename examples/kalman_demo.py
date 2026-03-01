"""
Kalman Smoothing -- robust and constrained state estimation.

Shows two advantages of PLQ penalties for state estimation:

  1. Robustness: A Huber measurement penalty tracks truth far better than
     the standard L2 smoother when observations contain outliers.
  2. Constraints: Box constraints on the state exploit prior knowledge
     (known amplitude bounds) for further improvement.

Model
-----
State:        x_k = [position_k, velocity_k]
Dynamics:     x_{k+1} = F x_k + w_k,  F = [[1, dt], [0, 1]]  (constant velocity)
Observation:  z_k = position_k + v_k   (position only, corrupted by outliers)

True signal:  position(t) = sin(2t)
"""

import os
import numpy as np
import scipy.sparse as sp
from ipsolve import solve, huber, l2


def blktridiag(main, sub, N):
    """Sparse block lower-bidiagonal: N diagonal blocks + N-1 sub-diagonal."""
    p, q = main.shape
    blocks = [[None] * N for _ in range(N)]
    for i in range(N):
        blocks[i][i] = sp.csc_matrix(main)
        if i > 0:
            blocks[i][i - 1] = sp.csc_matrix(sub)
    return sp.bmat(blocks, format="csc")


def main():
    np.random.seed(42)

    # -- Signal ----------------------------------------------------------
    N = 200
    n = 2                            # state: [position, velocity]
    dt = 4 * np.pi / N              # 2 full sine periods over [0, 4pi]
    t = np.arange(N) * dt

    pos_true = np.sin(2 * t)
    vel_true = 2 * np.cos(2 * t)

    # -- Noisy measurements (position only) ------------------------------
    sigma = 0.50                     # high baseline noise
    noise = sigma * np.random.randn(N)

    # 30% outliers with magnitude 3-8 (harsh — even robust methods overshoot)
    n_out = int(0.30 * N)
    out_idx = np.random.choice(N, n_out, replace=False)
    out_sign = np.sign(np.random.randn(n_out))
    noise[out_idx] += out_sign * (3.0 + 5.0 * np.random.rand(n_out))

    z = pos_true + noise

    # -- Measurement matrix (scaled by 1/sigma) --------------------------
    scale = 1.0 / sigma
    H_blk = scale * np.array([[1.0, 0.0]])
    H = sp.kron(sp.eye(N), sp.csc_matrix(H_blk), format="csc")
    z_sc = scale * z

    # -- Process model (constant-velocity dynamics) ----------------------
    F = np.array([[1.0, dt], [0.0, 1.0]])

    q_noise = 10.0                   # high process noise -> flexible smoother
    Q = q_noise * np.array([[dt**3 / 3, dt**2 / 2],
                             [dt**2 / 2, dt]])
    Qinv_half = np.linalg.inv(np.linalg.cholesky(Q))

    K = blktridiag(Qinv_half, -Qinv_half @ F, N)

    # anchor initial state
    w0 = 50.0
    K[:n, :n] = w0 * sp.eye(n)
    k_proc = np.zeros(n * N)
    k_proc[0] = w0 * pos_true[0]
    k_proc[1] = w0 * vel_true[0]

    # -- Box constraints: |position| <= 1.05 -----------------------------
    # We know sin(2t) in [-1, 1]; tight bound encodes this knowledge.
    # Solver convention: A^T x <= a.  For block k we want
    #   [[1, 0], [-1, 0]] @ [pos_k, vel_k] <= [bound, bound]
    # so the stored block must be the transpose: [[1, -1], [0, 0]].
    bound = 1.05
    con_blk = np.array([[1.0, -1.0],
                         [0.0,  0.0]])
    A_con = sp.kron(sp.eye(N), sp.csc_matrix(con_blk), format="csc")
    a_con = bound * np.ones(2 * N)

    # -- Solve -----------------------------------------------------------
    kappa = 3.0                       # Huber threshold (moderate robustness)
    kw = dict(K_proc=K, k_proc=k_proc, silent=True)

    print("Running 4 Kalman smoother variants ...")

    x_std  = solve(H, z_sc, proc=l2(), **kw).reshape(N, n)
    x_rob  = solve(H, z_sc, meas=huber(kappa=kappa), proc=l2(),
                   **kw).reshape(N, n)
    x_con  = solve(H, z_sc, proc=l2(),
                   A_ineq=A_con, a_ineq=a_con, **kw).reshape(N, n)
    x_best = solve(H, z_sc, meas=huber(kappa=kappa), proc=l2(),
                   A_ineq=A_con, a_ineq=a_con, **kw).reshape(N, n)

    # -- RMSE ------------------------------------------------------------
    def rmse(xhat):
        return np.sqrt(np.mean((xhat[:, 0] - pos_true)**2))

    r_std  = rmse(x_std)
    r_rob  = rmse(x_rob)
    r_con  = rmse(x_con)
    r_best = rmse(x_best)

    print(f"\nPosition RMSE:")
    print(f"  Standard  L2:            {r_std:.4f}")
    print(f"  Robust    Huber:         {r_rob:.4f}")
    print(f"  Constrained L2+box:      {r_con:.4f}")
    print(f"  Robust + Constrained:    {r_best:.4f}")

    # -- Figure ----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 8),
                                 sharex=True, sharey=True)

        panels = [
            (axes[0, 0], x_std,  "Standard L2",          r_std,  "#4477AA"),
            (axes[0, 1], x_rob,  "Robust (Huber)",       r_rob,  "#228833"),
            (axes[1, 0], x_con,  "L2 + Constraints",     r_con,  "#EE6677"),
            (axes[1, 1], x_best, "Robust + Constraints", r_best, "#AA3377"),
        ]

        for ax, xhat, label, err, color in panels:
            ax.plot(t, z, ".", color="0.72", ms=2.5, alpha=0.5,
                    zorder=1, rasterized=True)
            ax.plot(t, pos_true, "k-", lw=1.2, label="Truth", zorder=3)
            ax.plot(t, xhat[:, 0], "-", color=color, lw=2.0,
                    label=f"Estimate (RMSE {err:.3f})", zorder=4)

            # show constraint bounds on bottom row
            if ax.get_subplotspec().rowspan.start == 1:
                ax.axhline(bound, color="0.45", ls="--", lw=0.7)
                ax.axhline(-bound, color="0.45", ls="--", lw=0.7)
                ax.fill_between(t, bound, 4, color="red", alpha=0.04)
                ax.fill_between(t, -bound, -4, color="red", alpha=0.04)

            ax.set_title(label, fontsize=13, fontweight="bold")
            ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

        for ax in axes[1, :]:
            ax.set_xlabel("t", fontsize=11)
        for ax in axes[:, 0]:
            ax.set_ylabel("Position", fontsize=11)

        axes[0, 0].set_ylim(-4.5, 5.5)

        fig.suptitle(
            "Kalman Smoothing: Outlier Robustness & Constraint Benefits",
            fontsize=14, fontweight="bold", y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        os.makedirs("figures", exist_ok=True)
        fig.savefig("figures/kalman_demo.png", dpi=150, bbox_inches="tight")
        print("\nSaved figures/kalman_demo.png")
    except Exception as e:
        print(f"\nSkipping plot: {e}")


if __name__ == "__main__":
    main()
