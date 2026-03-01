"""
Interior-point solver -- barrier formulation with Mehrotra corrector.

Main IP loop maintaining variables (u, q, y, r, w) and driving mu -> 0
while enforcing complementarity and stationarity via damped Newton steps.

Based on:
    Aravkin, Burke, Pillonetto.  "Sparse/Robust Estimation and Kalman
    Smoothing with Nonsmooth Log-Concave Densities."  JMLR 14, 2013.

Refactor highlights (v0.2)
--------------------------
- Extracted _max_step and _armijo_search for readability.
- Renamed ip_solve_barrier -> ip_solve, B2 -> B_proc, M2 -> M_proc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import scipy.sparse as sp

from ipsolve.kkt import kkt_residual, kkt_solve


@dataclass
class SolverResult:
    """Container for solver output."""
    y: np.ndarray                       # primal solution
    u: np.ndarray                       # dual (PLQ) variable
    q: np.ndarray                       # multiplier for dual constraints
    r: Optional[np.ndarray] = None      # constraint slack
    w: Optional[np.ndarray] = None      # constraint multiplier
    obj_history: list = field(default_factory=list)
    kkt_history: list = field(default_factory=list)
    mu_final: float = 0.0
    iterations: int = 0
    converged: bool = False


# ======================================================================
# Step-size helpers
# ======================================================================

def _max_step(d, q, r, w, dd, dq, dr, dw, has_con):
    """Maximum step via ratio test (keep d, q, r, w > 0).

    Returns lam_max in (0, 1].
    """
    if has_con:
        ratio = np.concatenate([dd, dq, dr, dw]) / np.concatenate([d, q, r, w])
    else:
        ratio = np.concatenate([dd, dq]) / np.concatenate([d, q])

    neg = ratio < 0
    if np.any(neg):
        return min(0.99 * np.min(-1.0 / ratio[neg]), 1.0)
    return 1.0


def _armijo_search(
    lin_term, b, B_meas, c, C, M, q, u, r, w, y, mu,
    dq, du, dr, dw, dy, lam_max, G, kkt_kw,
    has_con, gamma=0.01, beta=0.1, max_bt=20,
):
    """Backtracking Armijo line search on the KKT residual.

    Returns (q, u, r, w, y, G_new, lam, kount) or None on failure.
    """
    lam = lam_max / beta          # pre-divide so first trial is lam_max

    for kount in range(1, max_bt + 1):
        lam *= beta
        q_t = q + lam * dq
        u_t = u + lam * du
        y_t = y + lam * dy
        r_t = (r + lam * dr) if has_con else None
        w_t = (w + lam * dw) if has_con else None

        if np.min(q_t) <= 0:
            continue

        F_t = kkt_residual(
            lin_term, b, B_meas, c, C, M,
            q_t, u_t, r_t, w_t, y_t, mu, **kkt_kw,
        )
        G_t = np.max(np.abs(F_t))
        if G_t <= (1 - gamma * lam) * G:
            return q_t, u_t, r_t, w_t, y_t, G_t, lam, kount

    return None


# ======================================================================
# Main IP loop
# ======================================================================

def ip_solve(
    lin_term, b, B_meas, c, C, M,
    q0, u0, r0, w0, y0,
    obj_fun=None,
    *,
    B_proc=None, M_proc=None,
    A=None, a=None,
    rho=0.0, delta=0.0,
    inexact=False, mehrotra=False,
    opt_tol=1e-5, max_iter=100, silent=False,
):
    """Run the barrier interior-point solver.

    Parameters
    ----------
    lin_term : (N,) -- linear objective term.
    b, B_meas, c, C, M : PLQ system data (see ipsolve.kkt).
    q0, u0, r0, w0, y0 : initial iterates.
    obj_fun : callable(y) -> float, primal objective (for logging).
    B_proc, M_proc : process model matrices (two-block structure).
    A, a : linear constraints A'y <= a.
    rho, delta : regularisation parameters.
    inexact : use CG instead of direct solves.
    mehrotra : Mehrotra predictor-corrector step.
    opt_tol : optimality tolerance.
    max_iter : iteration limit.
    silent : suppress printing.
    """
    u, q, y = u0.copy(), q0.copy(), y0.copy()
    r = r0.copy() if r0 is not None else None
    w = w0.copy() if w0 is not None else None
    has_con = A is not None

    mu = 1.0
    pcg_tol = 1e-5
    result = SolverResult(y=y, u=u, q=q, r=r, w=w)

    kkt_kw = dict(B_proc=B_proc, M_proc=M_proc, A=A, a=a)
    solve_kw = dict(**kkt_kw, rho=rho, delta=delta,
                    inexact=inexact, pcg_tol=pcg_tol)

    if not silent:
        print("=" * 80)
        fmt = "{:>5s}  {:>13s}  {:>13s}  {:>13s}  {:>10s}  {:>7s}"
        print(fmt.format("Iter", "Objective", "KKT Norm", "mu", "Step", "Armijo"))
        print("=" * 80)

    converged = False
    for itr in range(max_iter):
        # -- KKT residual --
        F = kkt_residual(lin_term, b, B_meas, c, C, M,
                         q, u, r, w, y, mu, **kkt_kw)
        G = np.max(np.abs(F))

        # -- Newton direction --
        sol = kkt_solve(lin_term, b, B_meas, c, C, M,
                        q, u, r, w, y, mu, **solve_kw)
        dq, du, dr, dw, dy = sol.dq, sol.du, sol.dr, sol.dw, sol.dy

        dirs = [dq, du, dy] + ([dr, dw] if has_con else [])
        if np.any(np.isnan(np.concatenate(dirs))):
            raise RuntimeError("NaNs in Newton direction")

        # -- Step-size --
        d = c - C.T @ u
        dd = -C.T @ du
        lam_max = _max_step(d, q, r, w, dd, dq,
                            dr if has_con else np.array([]),
                            dw if has_con else np.array([]),
                            has_con)

        ls = _armijo_search(
            lin_term, b, B_meas, c, C, M, q, u, r, w, y, mu,
            dq, du, dr, dw, dy, lam_max, G, kkt_kw, has_con,
        )
        if ls is None:
            if not silent:
                print("Line search failed -- returning best iterate.")
            break
        q_new, u_new, r_new, w_new, y_new, G_new, lam, kount = ls

        # -- Mehrotra corrector --
        if mehrotra:
            mu_corr = max(mu * (1 - lam), 0.0)
            sol2 = kkt_solve(lin_term, b, B_meas, c, C, M,
                             q_new, u_new, r_new, w_new, y_new,
                             mu_corr, **solve_kw)
            d_new = c - C.T @ u_new
            dd2 = -C.T @ sol2.du
            lam2 = _max_step(d_new, q_new, r_new, w_new, dd2, sol2.dq,
                             sol2.dr if has_con else np.array([]),
                             sol2.dw if has_con else np.array([]),
                             has_con)
            q_new = q_new + lam2 * sol2.dq
            u_new = u_new + lam2 * sol2.du
            y_new = y_new + lam2 * sol2.dy
            if has_con:
                r_new = r_new + lam2 * sol2.dr
                w_new = w_new + lam2 * sol2.dw
            if np.min(q_new) <= 0:
                raise RuntimeError("Negative entries after Mehrotra step")

        # -- Update --
        q, u, y, r, w = q_new, u_new, y_new, r_new, w_new

        obj_val = obj_fun(y) if obj_fun is not None else 0.0
        result.obj_history.append(obj_val)
        result.kkt_history.append(G_new)

        if not silent:
            print("{:5d}  {:13.6e}  {:13.6e}  {:13.6e}  {:10.3e}  {:7d}".format(
                itr, obj_val, G_new, mu, lam_max, kount))

        # -- Mu update --
        d = c - C.T @ u
        G1 = np.sum(q * d)
        n_slacks = 2 * len(q)
        if has_con:
            G1 += np.sum(r * w)
            n_slacks += 2 * len(r)
        mu = 0.1 * G1 / n_slacks

        # -- Convergence --
        converged = (G1 < opt_tol) or (G_new < opt_tol)
        if converged:
            if not silent:
                print("\nConverged at iteration {}.  mu={:.2e}, KKT={:.2e}".format(
                    itr, mu, G_new))
            break

    result.y, result.u, result.q = y, u, q
    result.r, result.w = r, w
    result.mu_final = mu
    result.iterations = itr + 1
    result.converged = converged
    return result
