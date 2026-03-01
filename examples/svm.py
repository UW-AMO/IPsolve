"""
SVM (Support Vector Machine) example.

Solves:  min  Σ pos(1 − yᵢ wᵀxᵢ)  +  (λ/2) ||w||²

Using hinge loss (measurement) + L2 regularisation (process).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ipsolve import solve, hinge, l2


def main():
    rng = np.random.default_rng(42)

    n_inst, n_vars = 2000, 50

    # Two-class Gaussian data
    cent = 2.0 * np.ones(n_vars)
    X_train = np.vstack([
        cent + rng.standard_normal((n_inst // 2, n_vars)),
        -cent + rng.standard_normal((n_inst // 2, n_vars)),
    ])
    y_train = np.concatenate([np.ones(n_inst // 2), -np.ones(n_inst // 2)])

    # Pre-multiply by labels
    yX = np.diag(y_train) @ X_train

    # Regularisation parameter
    lam_svm = 0.01

    # Solve SVM
    print("Solving SVM with IPsolve...")
    w = solve(
        yX, np.ones(n_inst),
        meas=hinge(),
        proc=l2(lam=lam_svm),
    )

    # Test on new data
    n_test = 500
    X_test = np.vstack([
        cent + rng.standard_normal((n_test // 2, n_vars)),
        -cent + rng.standard_normal((n_test // 2, n_vars)),
    ])
    y_test = np.concatenate([np.ones(n_test // 2), -np.ones(n_test // 2)])

    predictions = np.sign(X_test @ w)
    accuracy = np.mean(predictions == y_test)
    print(f"\nTest accuracy: {accuracy:.1%}")
    print(f"||w||: {np.linalg.norm(w):.4f}")

    # Objective value
    obj = np.sum(np.maximum(1 - yX @ w, 0)) + 0.5 * lam_svm * np.sum(w ** 2)
    print(f"Objective: {obj:.4f}")

    # Plot: project to first 2 principal components
    from numpy.linalg import svd
    U, s, Vt = svd(X_test, full_matrices=False)
    X2 = X_test @ Vt[:2].T  # project to 2D

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, colour, marker in [(1, "tab:blue", "o"), (-1, "tab:red", "s")]:
        mask = y_test == label
        ax.scatter(X2[mask, 0], X2[mask, 1], c=colour, marker=marker,
                   alpha=0.5, s=20, label=f"class {int(label)}")
    ax.set_title(f"SVM  (test accuracy {accuracy:.1%},  obj = {obj:.2f})")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend()
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/svm_example.png", dpi=120)
    print("Saved svm_example.png")


if __name__ == "__main__":
    main()
