# IPsolve – Python

A Python implementation of the **interior-point solver for piecewise
linear-quadratic (PLQ) composite problems**, based on the JMLR paper:

> Aravkin, Burke, Pillonetto.
> *Sparse/Robust Estimation and Kalman Smoothing with Nonsmooth
> Log-Concave Densities: Modeling, Computation, and Theory.*
> JMLR 14(82):2689–2728, 2013.

## Quick start

```bash
cd python
pip install -e ".[test]"           # editable install with test deps
pytest                             # run 12 tests (verified vs CVXPY)
```

```python
from ipsolve import solve, huber, l1, l2, hinge, qreg

# Least squares
x = solve(H, z)

# Robust regression
x = solve(H, z, meas=huber(kappa=1.0))

# Lasso
x = solve(H, z, proc=l1(lam=0.5))

# Huber + L1 (robust sparse)
x = solve(H, z, meas=huber(kappa=1.0), proc=l1(lam=0.5))

# SVM  (was: proc_mMult=1.0/lam ... now just proc=l2(lam=lam))
w = solve(yX, np.ones(m), meas=hinge(), proc=l2(lam=0.01))

# Quantile regression
x = solve(H, z, meas=qreg(tau=0.25))

# Box constraints
x = solve(H, z, bounds=(-1, 1))
```

## What it solves

$$
\min_{x} \;\; \ell^{\!\top} x \;+\; \rho_{\text{meas}}\!\bigl(z - Hx\bigr)
  \;+\; \rho_{\text{proc}}\!\bigl(Kx - k\bigr)
  \quad\text{s.t.}\quad Ax \le a
$$

where $\rho_{\text{meas}}$ and $\rho_{\text{proc}}$ are **PLQ penalties**
(ℓ₁, ℓ₂, Huber, hinge, quantile, Vapnik, …) represented in their
conjugate/dual form via matrices $(M, B, C, c, b)$.

The solver uses a **barrier interior-point method** with Schur-complement
elimination and optional Mehrotra predictor-corrector.

## Design

| Module | Purpose |
|--------|---------|
| `ipsolve.plq` | PLQ dataclass, `compose()`, `ensure_sparse` helpers |
| `ipsolve.penalties` | Penalty library (10 penalties -- see table below) |
| `ipsolve.kkt` | `SchurOperator` for mat-vec/dense Schur complement; KKT residual & solve |
| `ipsolve.solver` | Interior-point loop with extracted `_max_step` / `_armijo_search` helpers |
| `ipsolve.api` | High-level `solve(H, z, meas, proc, ...)` entry point |

## Penalties

Each penalty is a factory function.  Call without a dimension for `solve()`:

```python
meas=huber(kappa=1.0)   # lazy spec -- dimension filled in by solve()
proc=l1(lam=0.5)
```

Call with a dimension to get the raw PLQ object:

```python
plq = huber(100, kappa=1.0)   # returns PLQ directly
```

| Name | Formula | Key parameter |
|------|---------|---------------|
| `l2` | (lam/2) \\|v\\|^2 | `lam` (or `mMult`) |
| `l1` | lam * \\|v\\|_1 | `lam` |
| `huber` | Huber loss | `kappa` |
| `hinge` | lam * sum pos(v) | `lam` |
| `vapnik` | lam * sum max(\\|v\\| - eps, 0) | `lam`, `eps` |
| `qreg` | check / pinball loss | `tau`, `lam` |
| `qhuber` | quantile Huber (smooth quantile) | `tau`, `kappa` |
| `infnorm` | lam * \\|v\\|_inf | `lam` |
| `logreg` | sum log(1 + exp(v)) | `alpha` |
| `hybrid` | sum(sqrt(1 + v^2/s) - 1) | `scale` |

## Examples

All examples produce figures saved to `figures/`.  Run from the repo root:

```bash
python examples/lasso.py
python examples/robust_regression.py
python examples/svm.py
python examples/huber_l1.py
python examples/quantile_regression.py
python examples/cvar_portfolio.py
python examples/infnorm.py
python examples/kalman_demo.py
```

| Example | Description | MATLAB equivalent |
|---------|-------------|-------------------|
| `lasso.py` | Sparse signal recovery via ℓ₂ + ℓ₁ | `lassoAndEnet.m` |
| `robust_regression.py` | Huber loss vs LS with outliers | `RobustRegression.m` |
| `svm.py` | Binary classification with hinge + ℓ₂ | `SVMdemo.m` |
| `huber_l1.py` | Robust sparse regression, verified vs CVXPY | `HuberL1Example.m` |
| `quantile_regression.py` | Multi-quantile curves with pinball loss | — |
| `cvar_portfolio.py` | CVaR portfolio with simplex constraints | `CvarDemo.m` |
| `infnorm.py` | ℓ∞-norm regularisation (simplex dual) | `infNorm.m` |
| `kalman_demo.py` | Kalman smoothing: LS, robust, constrained | `KalmanDemo.m` |

## Tests

12 tests verify all penalties against CVXPY ground truth:

```
pytest -v
```

Tests cover: ℓ₂, Huber, lasso, Huber+ℓ₁, SVM, ℓ₁ fitting, quantile
regression, ℓ₁+ℓ₁, Vapnik, box constraints, ℓ∞ norm, and Kalman smoothing.

## Future work

- **Nonlinear extension** -- Gauss-Newton outer loop for rho(F(x)) where F is nonlinear
- **Logistic regression / hybrid** examples

## Inexact (CG) Schur solve

The `SchurOperator` class in `kkt.py` encapsulates the Schur complement
Omega = B' T^{-1} B as either a dense matrix (for direct LU) or a
matrix-free `LinearOperator` (for conjugate-gradient).

Pass `inexact=True` to `solve()` to use the CG path.  This avoids forming
and factoring the dense (N x N) Schur complement, reducing memory from
O(N^2) to O(nnz) and scaling better for overdetermined problems (m >> n).

### Scaling benchmark

```
python benchmarks/scaling_demo.py
```

#### Part 1 — Direct (LU) vs CG comparison

**Lasso (ℓ₂ + ℓ₁, underdetermined m < n):**

| m | n | Direct (s) | IP | CG | CG (s) | IP | CG | Rel diff |
|---|---|-----------|---|---|--------|---|---|----------|
| 100 | 200 | 0.08 | 17 | 0 | 0.22 | 17 | 1960 | 3.4e-8 |
| 200 | 500 | 0.66 | 20 | 0 | 1.15 | 20 | 4588 | 6.9e-8 |
| 500 | 1000 | 3.89 | 20 | 0 | 5.26 | 20 | 4499 | 5.1e-5 |

**Huber regression (overdetermined m ≫ n) — CG wins:**

| m | n | Direct (s) | IP | CG | CG (s) | IP | CG | Speedup | Rel diff |
|---|---|-----------|---|---|--------|---|---|---------|----------|
| 500 | 200 | 0.09 | 10 | 0 | 0.08 | 10 | 233 | 1.2× | 4.5e-9 |
| 1000 | 500 | 1.09 | 13 | 0 | 0.55 | 13 | 413 | 2.0× | 1.7e-9 |
| 2000 | 1000 | 10.14 | 16 | 0 | 2.84 | 16 | 522 | 3.6× | 5.4e-10 |

#### Part 2 — CG-only scaling (large problems)

**Lasso (CG-only):**

| m | n | Time (s) | IP iters | CG iters | Objective |
|---|---|---------|---------|---------|-----------|
| 500 | 1000 | 5.4 | 20 | 4499 | 24.59 |
| 1000 | 2000 | 29.4 | 23 | 6209 | 45.15 |
| 2000 | 5000 | 369.5 | 36 | 16221 | 120.27 |

**Huber (CG-only):**

| m | n | Time (s) | IP iters | CG iters |
|---|---|---------|---------|---------|
| 2000 | 1000 | 2.5 | 16 | 522 |
| 5000 | 2000 | 9.0 | 14 | 351 |
| 10000 | 5000 | 70.6 | 16 | 536 |
