"""
CVaR (Conditional Value-at-Risk) portfolio optimisation.

Solves:  min  α + (1 / (q(1−β))) Σ pos(Yx − α)
         s.t.  x ≥ 0,  1ᵀx = 1

Using hinge loss with simplex constraints.
Compares with CVXPY.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ipsolve import solve, hinge

try:
    import cvxpy
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False


def main():
    rng = np.random.default_rng(5)

    n_stocks = 30
    n_samples = 100
    beta = 0.1  # CVaR confidence level

    # Random returns
    Y = rng.standard_normal((n_samples, n_stocks))

    # Augmented system: last variable is α
    # Hinge penalty computes pos(z − Hx), so we set H = [−Y, 1] and z = 0
    # to get pos(Yx − α)
    Y_aug = np.hstack([-Y, np.ones((n_samples, 1))])
    z = np.zeros(n_samples)  # target for the hinge loss
    lin_term = np.zeros(n_stocks + 1)
    lin_term[-1] = 1.0  # picks out α in the objective

    lam = 1.0 / (n_samples * (1 - beta))

    # Constraints: x >= 0, sum(x) = 1  (ignore α)
    # Format:  A @ [x; α] ≤ a
    # -x ≤ 0,  sum(x) ≤ 1, -sum(x) ≤ -1
    A_top = np.hstack([-np.eye(n_stocks), np.zeros((n_stocks, 1))])
    A_eq1 = np.zeros((1, n_stocks + 1))
    A_eq1[0, :n_stocks] = 1
    A_eq2 = np.zeros((1, n_stocks + 1))
    A_eq2[0, :n_stocks] = -1
    A_ineq = np.vstack([A_top, A_eq1, A_eq2])
    a_ineq = np.concatenate([np.zeros(n_stocks), [1.0], [-1.0]])

    # Solve
    x_ip = solve(
        Y_aug, z,
        meas=hinge(lam=lam),
        lin_term=lin_term,
        A_ineq=A_ineq, a_ineq=a_ineq,
        opt_tol=1e-7,
    )

    portfolio = x_ip[:n_stocks]
    alpha = x_ip[-1]
    # The actual CVaR objective uses [Y, -ones] @ x = Yx - alpha
    Y_orig = np.hstack([Y, -np.ones((n_samples, 1))])
    obj_ip = alpha + lam * np.sum(np.maximum(Y_orig @ x_ip, 0))
    print(f"\nPortfolio weights sum: {portfolio.sum():.4f}")
    print(f"α (VaR): {alpha:.4f}")
    print(f"Objective: {obj_ip:.6f}")

    if HAS_CVXPY:
        x_cv = cvxpy.Variable(n_stocks + 1)
        prob = cvxpy.Problem(
            cvxpy.Minimize(x_cv[-1] + lam * cvxpy.sum(cvxpy.pos(Y_orig @ x_cv))),
            [x_cv[:n_stocks] >= 0, cvxpy.sum(x_cv[:n_stocks]) == 1],
        )
        prob.solve(solver=cvxpy.CLARABEL, verbose=False)
        obj_cv = prob.value
        print(f"\nCVXPY objective:   {obj_cv:.6f}")
        print(f"Relative diff:     {abs(obj_cv - obj_ip) / max(abs(obj_cv), 1e-12):.2e}")

    # Plot portfolio weights
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(n_stocks), portfolio, color="steelblue")
    ax.set_xlabel("Stock index")
    ax.set_ylabel("Weight")
    ax.set_title(f"CVaR-optimal portfolio  (β={beta}, α={alpha:.3f})")
    ax.axhline(0, c="k", lw=0.5)
    fig.tight_layout()
    fig.savefig("cvar_portfolio.png", dpi=120)
    print("Saved cvar_portfolio.png")


if __name__ == "__main__":
    main()
