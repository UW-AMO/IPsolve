#!/usr/bin/env python3
"""
Scaling benchmark -- direct (LU) vs inexact (CG) Schur solve.

Generates Lasso and Huber problems at several sizes and compares wall-clock
time, IP iterations, CG iterations, and solution quality.

Part 1: Direct vs CG comparison at moderate sizes.
Part 2: CG-only scaling pushed to large problem sizes.

Usage:
    python benchmarks/scaling_demo.py

Output:
    - Console tables of timings and iteration counts
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

from ipsolve import solve, l1, huber


# ---------------------------------------------------------------------------
# Problem generators
# ---------------------------------------------------------------------------

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


def _make_huber(m, n, seed=42):
    """Random regression problem with outliers (for Huber)."""
    rng = np.random.default_rng(seed)
    H = rng.standard_normal((m, n))
    x_true = rng.standard_normal(n)
    z = H @ x_true + 0.1 * rng.standard_normal(m)
    outliers = rng.choice(m, m // 10, replace=False)
    z[outliers] += 10 * rng.standard_normal(len(outliers))
    return H, z, x_true


# ===================================================================
# Part 1 -- Direct vs CG comparison (moderate sizes)
# ===================================================================

def benchmark_lasso():
    """Benchmark Lasso: direct vs CG at moderate sizes."""
    sizes = [
        (100,   200),
        (200,   500),
        (500,  1000),
    ]

    results = []
    W = 110
    print("Lasso (l2 + l1):  Direct vs CG")
    print("=" * W)
    hdr = ("{:>6s} {:>6s} | {:>8s} {:>4s} {:>6s} {:>10s}"
           " | {:>8s} {:>4s} {:>6s} {:>10s} | {:>10s}")
    print(hdr.format("m", "n",
                      "Dir(s)", "IP", "CG", "Obj",
                      "CG(s)", "IP", "CG", "Obj",
                      "Rel_diff"))
    print("-" * W)

    for m, n in sizes:
        H, z, _ = _make_lasso(m, n)
        lam = 0.1

        # Direct
        t0 = time.perf_counter()
        res_d = solve(H, z, proc=l1(lam=lam),
                      silent=True, delta=1e-8, return_result=True)
        t_direct = time.perf_counter() - t0
        x_d = res_d.y
        obj_d = 0.5 * np.sum((z - H @ x_d)**2) + lam * np.sum(np.abs(x_d))

        # CG
        t0 = time.perf_counter()
        res_cg = solve(H, z, proc=l1(lam=lam),
                       silent=True, delta=1e-8, inexact=True,
                       return_result=True)
        t_cg = time.perf_counter() - t0
        x_cg = res_cg.y
        obj_cg = 0.5 * np.sum((z - H @ x_cg)**2) + lam * np.sum(np.abs(x_cg))

        rel = np.linalg.norm(x_d - x_cg) / max(np.linalg.norm(x_d), 1e-12)

        row = ("{:6d} {:6d} | {:8.3f} {:4d} {:6d} {:10.4f}"
               " | {:8.3f} {:4d} {:6d} {:10.4f} | {:10.2e}")
        print(row.format(m, n,
                          t_direct, res_d.iterations, res_d.cg_iters, obj_d,
                          t_cg, res_cg.iterations, res_cg.cg_iters, obj_cg,
                          rel))
        results.append(dict(m=m, n=n, t_direct=t_direct, t_cg=t_cg,
                            ip_d=res_d.iterations, cg_d=res_d.cg_iters,
                            ip_cg=res_cg.iterations, cg_cg=res_cg.cg_iters,
                            obj_d=obj_d, obj_cg=obj_cg, rel_diff=rel))

    print("=" * W)
    return results


def benchmark_huber():
    """Benchmark Huber regression: direct vs CG at moderate sizes."""
    sizes = [
        (500,   200),
        (1000,  500),
        (2000, 1000),
    ]

    results = []
    W = 85
    print("\nHuber regression:  Direct vs CG")
    print("=" * W)
    hdr = ("{:>6s} {:>6s} | {:>8s} {:>4s} {:>6s}"
           " | {:>8s} {:>4s} {:>6s} | {:>10s}")
    print(hdr.format("m", "n",
                      "Dir(s)", "IP", "CG",
                      "CG(s)", "IP", "CG",
                      "Rel_diff"))
    print("-" * W)

    for m, n in sizes:
        H, z, _ = _make_huber(m, n)

        t0 = time.perf_counter()
        res_d = solve(H, z, meas=huber(kappa=1.0),
                      silent=True, delta=1e-8, return_result=True)
        t_direct = time.perf_counter() - t0

        t0 = time.perf_counter()
        res_cg = solve(H, z, meas=huber(kappa=1.0),
                       silent=True, delta=1e-8, inexact=True,
                       return_result=True)
        t_cg = time.perf_counter() - t0

        rel = np.linalg.norm(res_d.y - res_cg.y) / max(np.linalg.norm(res_d.y), 1e-12)

        row = ("{:6d} {:6d} | {:8.3f} {:4d} {:6d}"
               " | {:8.3f} {:4d} {:6d} | {:10.2e}")
        print(row.format(m, n,
                          t_direct, res_d.iterations, res_d.cg_iters,
                          t_cg, res_cg.iterations, res_cg.cg_iters,
                          rel))
        results.append(dict(m=m, n=n, t_direct=t_direct, t_cg=t_cg,
                            ip_d=res_d.iterations, cg_d=res_d.cg_iters,
                            ip_cg=res_cg.iterations, cg_cg=res_cg.cg_iters,
                            rel_diff=rel))

    print("=" * W)
    return results


# ===================================================================
# Part 2 -- CG-only scaling to large sizes
# ===================================================================

def benchmark_lasso_cg():
    """CG-only Lasso scaling to large problem sizes."""
    sizes = [
        (500,   1000),
        (1000,  2000),
        (2000,  5000),
    ]

    results = []
    W = 65
    print("\nLasso (l2 + l1):  CG-only scaling")
    print("=" * W)
    hdr = "{:>7s} {:>7s} | {:>8s} {:>5s} {:>7s} {:>12s}"
    print(hdr.format("m", "n", "Time(s)", "IP", "CG", "Objective"))
    print("-" * W)

    for m, n in sizes:
        H, z, _ = _make_lasso(m, n)
        lam = 0.1

        t0 = time.perf_counter()
        res = solve(H, z, proc=l1(lam=lam),
                    silent=True, delta=1e-8, inexact=True,
                    return_result=True)
        elapsed = time.perf_counter() - t0
        x = res.y
        obj = 0.5 * np.sum((z - H @ x)**2) + lam * np.sum(np.abs(x))

        row = "{:7d} {:7d} | {:8.2f} {:5d} {:7d} {:12.4f}"
        print(row.format(m, n, elapsed, res.iterations, res.cg_iters, obj))
        results.append(dict(m=m, n=n, t_cg=elapsed,
                            ip=res.iterations, cg=res.cg_iters, obj=obj))

    print("=" * W)
    return results


def benchmark_huber_cg():
    """CG-only Huber scaling to large problem sizes."""
    sizes = [
        (2000,  1000),
        (5000,  2000),
        (10000, 5000),
    ]

    results = []
    W = 55
    print("\nHuber regression:  CG-only scaling")
    print("=" * W)
    hdr = "{:>7s} {:>7s} | {:>8s} {:>5s} {:>7s}"
    print(hdr.format("m", "n", "Time(s)", "IP", "CG"))
    print("-" * W)

    for m, n in sizes:
        H, z, _ = _make_huber(m, n)

        t0 = time.perf_counter()
        res = solve(H, z, meas=huber(kappa=1.0),
                    silent=True, delta=1e-8, inexact=True,
                    return_result=True)
        elapsed = time.perf_counter() - t0

        row = "{:7d} {:7d} | {:8.2f} {:5d} {:7d}"
        print(row.format(m, n, elapsed, res.iterations, res.cg_iters))
        results.append(dict(m=m, n=n, t_cg=elapsed,
                            ip=res.iterations, cg=res.cg_iters))

    print("=" * W)
    return results


# ===================================================================
# Plotting
# ===================================================================

def _plot_scaling(lasso_res, huber_res, lasso_cg_res, huber_cg_res):
    """Create log-log scaling plot (2x2: top = direct vs CG, bottom = CG-only)."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # -- Top row: Direct vs CG comparison --
    ax = axes[0, 0]
    ns = [r["n"] for r in lasso_res]
    td = [r["t_direct"] for r in lasso_res]
    tc = [r["t_cg"] for r in lasso_res]
    ax.loglog(ns, td, "o-", label="Direct (LU)", linewidth=2)
    ax.loglog(ns, tc, "s--", label="Inexact (CG)", linewidth=2)
    ax.set_xlabel("n (variables)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Lasso: Direct vs CG")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ms = [r["m"] for r in huber_res]
    td = [r["t_direct"] for r in huber_res]
    tc = [r["t_cg"] for r in huber_res]
    ax.loglog(ms, td, "o-", label="Direct (LU)", linewidth=2)
    ax.loglog(ms, tc, "s--", label="Inexact (CG)", linewidth=2)
    ax.set_xlabel("m (observations)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Huber: Direct vs CG")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # -- Bottom row: CG-only large-scale --
    ax = axes[1, 0]
    ns = [r["n"] for r in lasso_cg_res]
    tc = [r["t_cg"] for r in lasso_cg_res]
    ax.loglog(ns, tc, "s-", color="C1", label="CG", linewidth=2, markersize=7)
    ax.set_xlabel("n (variables)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Lasso CG-only (large scale)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ms = [r["m"] for r in huber_cg_res]
    tc = [r["t_cg"] for r in huber_cg_res]
    ax.loglog(ms, tc, "s-", color="C1", label="CG", linewidth=2, markersize=7)
    ax.set_xlabel("m (observations)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Huber CG-only (large scale)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("IPsolve Scaling Benchmark", fontsize=14, y=1.01)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "scaling_benchmark.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    print("IPsolve Scaling Benchmark")
    print()

    # Part 1: Direct vs CG
    lasso_res = benchmark_lasso()
    huber_res = benchmark_huber()

    # Part 2: CG-only large-scale
    lasso_cg_res = benchmark_lasso_cg()
    huber_cg_res = benchmark_huber_cg()

    _plot_scaling(lasso_res, huber_res, lasso_cg_res, huber_cg_res)
    print("\nDone.")
