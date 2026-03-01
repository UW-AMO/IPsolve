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
| `ipsolve.plq` | PLQ dataclass and `stack` / `ensure_sparse` helpers |
| `ipsolve.penalties` | Penalty library (10 penalties – see table below) |
| `ipsolve.kkt` | KKT residual and Newton solve (barrier formulation) |
| `ipsolve.solver` | Interior-point loop with Mehrotra corrector |
| `ipsolve.api` | High-level `solve(H, z, meas, proc, …)` entry point |

## Penalties

| Name | Formula | Key parameter |
|------|---------|---------------|
| `l2` | $\frac{1}{2\sigma}\|v\|^2$ | `mMult` (= $\sigma$) |
| `l1` | $\lambda\|v\|_1$ | `lam` |
| `huber` | $\sum H_\kappa(v_i)$ | `kappa` |
| `hinge` | $\lambda\sum\max(-v_i,0)$ | `lam` |
| `vapnik` | $\lambda\sum\max(\|v_i\|-\varepsilon,0)$ | `lam`, `eps` |
| `qreg` | $\sum[\tau\cdot\max(v_i,0)+(1{-}\tau)\max(-v_i,0)]$ | `tau`, `lam` |
| `qhuber` | quantile Huber (smooth quantile) | `tau`, `kappa` |
| `infnorm` | $\lambda\|v\|_\infty$ | `lam` |
| `logreg` | $\sum\log(1+e^{v_i})$ (callable M) | `alpha` |
| `hybrid` | $\sum[\sqrt{1+v_i^2/s}-1]$ (callable M) | `scale` |

## Examples

All examples produce figures saved to `figures/`.  Run from `python/figures/`:

```bash
mkdir -p figures && cd figures
python ../examples/lasso.py
python ../examples/robust_regression.py
python ../examples/svm.py
python ../examples/huber_l1.py
python ../examples/quantile_regression.py
python ../examples/cvar_portfolio.py
python ../examples/infnorm.py
python ../examples/kalman_demo.py
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

- **Nonlinear extension** – Gauss-Newton outer loop for $\rho(F(x))$ where $F$ is nonlinear (planned for a separate branch / PR)
- **Inexact (PCG) solve path** – implemented but not yet extensively tested at scale
- **Logistic regression / hybrid** examples
