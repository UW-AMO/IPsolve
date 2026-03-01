#!/usr/bin/env python3
"""
Scaling benchmark -- direct (LU) vs inexact (CG) Schur solve.

Generates a Lasso problem at several sizes and compares wall-clock time
and solution quality for both solve paths.

Usage:
    python benchmarks/scaling_demo.py

Output:
    - Console table of timings
    - benchmarks/scaling_benchmark.png  (log-log plot)
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ipsolve import solve


def _make_lasso(m, n, k=None, noise=0.05, seed=42):
    """Random sparse regression problem (Lasso-style)."""
    rng = np.random.default_rng(seed)
    if k is None:
        k = max(5, n // 10)
    H = rng.standard_normal((m, n))
    x_true = np.zeros(n)
    idx = rng.choice(n, k, replace=False)
    x_true[idx] = rng.standard_normal(k) * 3
    z = H @ x_true + noise * rng.standard_normal(m)
    return H, z, x_true


def benchmark_lasso():
    """Benchmark Lasso across problem sizes, direct vs CG."""
    sizes = [
        (50,  100),
        (100, 200),
        (200, 500),
        (400, 1000),
        (600, 1500),
    ]

    results = []
    print("=" * 85)
    header = "{:>6s} {:>6s} | {:>10s} {:>10s} | {:>10s} {:>10s} | {:>10s}"
    print(header.format("m", "n", "Direct(s)", "Obj_D", "CG(s)", "Obj_CG", "Rel_diff"))
    print("=" * 85)

    for m, n in sizes:
        H, z, x_true = _make_lasso(m, n)
        lam = 0.1

        # Direct solve
        t0 = time.perf_counter()
        x_d = solve(H, z, meas="l2", proc="l1", proc_lambda=lam,
                    silent=True, delta=1e-8)
        t_direct = time.perf_counter() - t0

        obj_d = 0.5 * np.sum((z - H @ x_d)**2) + lam * np.sum(np.abs(x_d))

        # CG (inexact) solve
        t0 = time.perf_counter()
        x_cg = solve(H, z, meas="l2", proc="l1", proc_lambda=lam,
                     silent=True, delta=1e-8, inexact=True)
        t_cg = time.perf_counter() - t0

        obj_cg = 0.5 * np.sum((z - H @ x_cg)**2) + lam * np.sum(np.abs(x_cg))
        rel_diff = np.linalg.norm(x_d - x_cg) / max(np.linalg.norm(x_d), 1e-12)

        row = "{:6d} {:6d} | {:10.4f} {:10.4f} | {:10.4f} {:10.4f} | {:10.2e}"
        print(row.format(m, n, t_direct, obj_d, t_cg, obj_cg, rel_diff))
        results.append(dict(m=m, n=n, t_direct=t_direct, t_cg=t_cg,
                            obj_d=obj_d, obj_cg=obj_cg, rel_diff=rel_diff))

    print("=" * 85)
    return results


def benchmark_huber():
    """Benchmark Huber regression at various sizes."""
    sizes = [
        (100,  50),
        (200, 100),
        (500, 200),
        (1000, 400),
        (2000, 600),
    ]

    results = []
    print("\nHuber regression (meas=huber, no proc):")
    print("=" * 65)
    header = "{:>6s} {:>6s} | {:>10s} {:>10s} | {:>10s}"
    print(header.format("m", "n", "Direct(s)", "CG(s)", "Rel_diff"))
    print("=" * 65)

    for m, n in sizes:
        rng = np.random.default_rng(42)
        H = rng.standard_normal((m, n))
        x_true = rng.standard_normal(n)
        z = H @ x_true + 0.1 * rng.standard_normal(m)
        # Add outliers
        outliers = rng.choice(m, m // 10, replace=False)
        z[outliers] += 10 * rng.standard_normal(len(outliers))

        t0 = time.perf_counter()
        x_d = solve(H, z, meas="huber", meas_kappa=1.0,
                    silent=True, delta=1e-8)
        t_direct = time.perf_counter() - t0

        t0 = time.perf_counter()
        x_cg = solve(H, z, meas="huber", meas_kappa=1.0,
                     silent=True, delta=1e-8, inexact=True)
        t_cg = time.perf_counter() - t0

        rel_diff = np.linalg.norm(x_d - x_cg) / max(np.linalg.norm(x_d), 1e-12)
        row = "{:6d} {:6d} | {:10.4f} {:10.4f} | {:10.2e}"
        print(row.format(m, n, t_direct, t_cg, rel_diff))
        results.append(dict(m=m, n=n, t_direct=t_direct, t_cg=t_cg,
                            rel_diff=rel_diff))

    print("=" * 65)
    return results


def _plot_scaling(lasso_res, huber_res):
    """Create log-log scaling plot."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Lasso
    ax = axes[0]
    ns = [r["n"] for r in lasso_res]
    td = [r["t_direct"] for r in lasso_res]
    tc = [r["t_cg"] for r in lasso_res]
    ax.loglog(ns, td, "o-", label="Direct (LU)", linewidth=2)
    ax.loglog(ns, tc, "s--", label="Inexact (CG)", linewidth=2)
    ax.set_xlabel("n (variables)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Lasso (l2 + l1)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Huber
    ax = axes[1]
    ms = [r["m"] for r in huber_res]
    td = [r["t_direct"] for r in huber_res]
    tc = [r["t_cg"] for r in huber_res]
    ax.loglog(ms, td, "o-", label="Direct (LU)", linewidth=2)
    ax.loglog(ms, tc, "s--", label="Inexact (CG)", linewidth=2)
    ax.set_xlabel("m (observations)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Huber regression")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("IPsolve: Direct vs CG Schur Solve Scaling", fontsize=14)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "scaling_benchmark.png"
    fig.savefig(out, dpi=150)
    print(str(out))
    plt.close(fig)


if __name__ == "__main__":
    print("IPsolve Scaling Benchmark")
    print("=" * 85)
    lasso_res = benchmark_lasso()
    huber_res = benchmark_huber()
    _plot_scaling(lasso_res, huber_res)
    print("\nDone.")
