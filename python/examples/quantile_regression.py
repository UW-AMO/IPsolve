"""
Quantile Regression example.

Solves:  min  Σ [τ·pos(rᵢ) + (1−τ)·pos(−rᵢ)]  where r = z − Hx

Estimates conditional quantiles by using the asymmetric (check / pinball) loss.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ipsolve import solve


def main():
    rng = np.random.default_rng(3)

    # 1D regression
    n_pts = 200
    x_data = np.sort(rng.uniform(0, 5, n_pts))
    # True function + heteroscedastic noise
    y_true = np.sin(x_data)
    noise = (0.1 + 0.3 * x_data) * rng.standard_normal(n_pts)
    z = y_true + noise

    # Design matrix (polynomial basis deg 5)
    deg = 5
    H = np.column_stack([x_data ** p for p in range(deg + 1)])

    taus = [0.10, 0.25, 0.50, 0.75, 0.90]
    x_plot = np.linspace(0, 5, 300)
    H_plot = np.column_stack([x_plot ** p for p in range(deg + 1)])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x_data, z, s=6, alpha=0.3, c="grey", label="data")

    for tau in taus:
        w = solve(H, z, meas="qreg", meas_tau=tau, meas_lambda=1.0, silent=True)
        y_pred = H_plot @ w
        ax.plot(x_plot, y_pred, label=f"τ = {tau}")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Quantile Regression via IPsolve")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig("quantile_regression.png", dpi=120)
    print("Saved quantile_regression.png")


if __name__ == "__main__":
    main()
