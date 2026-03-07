#!/usr/bin/env python3
"""Compare IPsolve Direct vs CG on 500 x 1000 Lasso."""

import sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ipsolve import solve, l1

# --- Generate 500 x 1000 Lasso problem (same as scaling_demo) ---
rng = np.random.default_rng(42)
m, n, k = 500, 1000, max(5, 1000 // 10)
H = rng.standard_normal((m, n))
x_true = np.zeros(n)
idx = rng.choice(n, k, replace=False)
x_true[idx] = rng.standard_normal(k) * 3
z = H @ x_true + 0.05 * rng.standard_normal(m)
lam = 0.1

print("Lasso 500 x 1000:  Direct vs CG")
print("=" * 80)

# --- Direct ---
t0 = time.perf_counter()
res_d = solve(H, z, proc=l1(lam=lam), silent=True, delta=1e-8, return_result=True)
t_direct = time.perf_counter() - t0
x_d = res_d.y
obj_d = 0.5 * np.sum((z - H @ x_d)**2) + lam * np.sum(np.abs(x_d))

# --- CG ---
t0 = time.perf_counter()
res_cg = solve(H, z, proc=l1(lam=lam), silent=True, delta=1e-8,
               inexact=True, return_result=True)
t_cg = time.perf_counter() - t0
x_cg = res_cg.y
obj_cg = 0.5 * np.sum((z - H @ x_cg)**2) + lam * np.sum(np.abs(x_cg))

rel = np.linalg.norm(x_d - x_cg) / max(np.linalg.norm(x_d), 1e-12)

print(f"  Direct:  time={t_direct:.4f}s   IP iters={res_d.iterations}"
      f"   CG iters={res_d.cg_iters:>5d}   obj={obj_d:.6f}")
print(f"  CG:      time={t_cg:.4f}s   IP iters={res_cg.iterations}"
      f"   CG iters={res_cg.cg_iters:>5d}   obj={obj_cg:.6f}")
print(f"  Speedup: {t_direct/t_cg:.2f}x")
print(f"  Rel solution diff: {rel:.2e}")
print("=" * 80)
