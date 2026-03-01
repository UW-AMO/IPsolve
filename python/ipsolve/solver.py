"""
Interior-point solver – barrier formulation with Mehrotra corrector.

This is the main IP loop.  It maintains variables ``(u, q, y, r, w)`` and
drives the barrier parameter μ → 0 while enforcing complementarity and
stationarity via damped Newton steps.

The algorithm follows the structure in:

    Aravkin, Burke, Pillonetto.  "Sparse/Robust Estimation and Kalman
    Smoothing with Nonsmooth Log-Concave Densities."  JMLR 14, 2013.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import scipy.sparse as sp

from ipsolve.kkt import kkt_residual_barrier, kkt_solve_barrier


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


def ip_solve_barrier(
    lin_term: np.ndarray,
    b: np.ndarray,
    Bm: sp.spmatrix,
    c: np.ndarray,
    C: sp.spmatrix,
    M,
    q0: np.ndarray,
    u0: np.ndarray,
    r0: Optional[np.ndarray],
    w0: Optional[np.ndarray],
    y0: np.ndarray,
    obj_fun: Optional[Callable] = None,
    *,
    B2: Optional[sp.spmatrix] = None,
    M2=None,
    A: Optional[sp.spmatrix] = None,
    a: Optional[np.ndarray] = None,
    rho: float = 0.0,
    delta: float = 0.0,
    inexact: bool = False,
    mehrotra: bool = False,
    opt_tol: float = 1e-5,
    max_iter: int = 100,
    silent: bool = False,
) -> SolverResult:
    """Run the barrier interior-point solver.

    Parameters
    ----------
    lin_term : (N,) array – linear objective term.
    b, Bm, c, C, M : PLQ system data (see :mod:`ipsolve.kkt`).
    q0, u0, r0, w0, y0 : initial iterates.
    obj_fun : callable(y) → float, primal objective (for logging).
    B2, M2 : process model matrices (if two-block structure).
    A, a : linear constraints A.T @ y ≤ a.
    rho, delta : regularisation parameters.
    inexact : use PCG instead of direct solves.
    mehrotra : apply Mehrotra predictor-corrector step.
    opt_tol : optimality tolerance.
    max_iter : iteration limit.
    silent : suppress printing.

    Returns
    -------
    SolverResult
    """
    # Copy initial iterates
    u = u0.copy()
    q = q0.copy()
    y = y0.copy()
    r = r0.copy() if r0 is not None else None
    w = w0.copy() if w0 is not None else None
    pCon = A is not None

    mu = 1.0
    gamma_ls = 0.01
    pcg_tol = 1e-5

    result = SolverResult(y=y, u=u, q=q, r=r, w=w)

    if not silent:
        header = f"{'Iter':>5s}  {'Objective':>13s}  {'KKT Norm':>13s}  {'mu':>13s}  {'Step':>10s}  {'Armijo':>7s}"
        print("=" * 80)
        print(header)
        print("=" * 80)

    kkt_kwargs = dict(B2=B2, M2=M2, A=A, a=a)
    solve_kwargs = dict(B2=B2, M2=M2, A=A, a=a, rho=rho, delta=delta,
                        inexact=inexact, pcg_tol=pcg_tol)

    for itr in range(max_iter):
        # ------------------------------------------------------------------
        # KKT residual
        # ------------------------------------------------------------------
        F = kkt_residual_barrier(
            lin_term, b, Bm, c, C, M, q, u, r, w, y, mu, **kkt_kwargs
        )
        G = np.max(np.abs(F))

        # ------------------------------------------------------------------
        # Newton direction
        # ------------------------------------------------------------------
        sol = kkt_solve_barrier(
            lin_term, b, Bm, c, C, M, q, u, r, w, y, mu, **solve_kwargs
        )
        dq, du, dr, dw, dy = sol.dq, sol.du, sol.dr, sol.dw, sol.dy

        if np.any(np.isnan(np.concatenate([dq, du, dy] + ([dr, dw] if pCon else [])))):
            raise RuntimeError("NaNs in Newton direction")

        # ------------------------------------------------------------------
        # Step-size via ratio test (keep d, q, r, w > 0)
        # ------------------------------------------------------------------
        d = c - C.T @ u
        dd = -C.T @ du

        if pCon:
            ratio = np.concatenate([dd, dq, dr, dw]) / np.concatenate([d, q, r, w])
        else:
            ratio = np.concatenate([dd, dq]) / np.concatenate([d, q])

        neg_mask = ratio < 0
        if np.any(neg_mask):
            lam_max = 0.99 * np.min(-1.0 / ratio[neg_mask])
            lam_max = min(lam_max, 1.0)
        else:
            lam_max = 1.0

        # Armijo backtracking
        lam = lam_max
        beta = 0.1
        ok = False
        kount = 0
        max_kount = 20

        # Pre-divide so first trial is lam_max
        lam = lam / beta

        while not ok and kount < max_kount:
            kount += 1
            lam *= beta

            q_new = q + lam * dq
            u_new = u + lam * du
            y_new = y + lam * dy
            r_new = (r + lam * dr) if pCon else None
            w_new = (w + lam * dw) if pCon else None

            if np.min(q_new) <= 0:
                continue

            F_new = kkt_residual_barrier(
                lin_term, b, Bm, c, C, M, q_new, u_new, r_new, w_new, y_new, mu,
                **kkt_kwargs
            )
            G_new = np.max(np.abs(F_new))
            ok = G_new <= (1 - gamma_ls * lam) * G

        if not ok:
            if not silent:
                print("Line search failed – returning best iterate.")
            break

        # ------------------------------------------------------------------
        # Mehrotra predictor-corrector
        # ------------------------------------------------------------------
        if mehrotra:
            mu_corr = mu * (1 - lam)
            if mu_corr < 0:
                mu_corr = 0.0

            sol2 = kkt_solve_barrier(
                lin_term, b, Bm, c, C, M, q_new, u_new, r_new, w_new, y_new,
                mu_corr, **solve_kwargs
            )
            d_new = c - C.T @ u_new
            dd2 = -C.T @ sol2.du

            if pCon:
                ratio2 = (np.concatenate([dd2, sol2.dq, sol2.dr, sol2.dw])
                          / np.concatenate([d_new, q_new, r_new, w_new]))
            else:
                ratio2 = np.concatenate([dd2, sol2.dq]) / np.concatenate([d_new, q_new])

            neg2 = ratio2 < 0
            if np.any(neg2):
                lam2 = 0.99 * min(np.min(-1.0 / ratio2[neg2]), 1.0)
            else:
                lam2 = 1.0

            q_new = q_new + lam2 * sol2.dq
            u_new = u_new + lam2 * sol2.du
            y_new = y_new + lam2 * sol2.dy
            if pCon:
                r_new = r_new + lam2 * sol2.dr
                w_new = w_new + lam2 * sol2.dw

            if np.min(q_new) <= 0:
                raise RuntimeError("Negative entries after Mehrotra step")

        # ------------------------------------------------------------------
        # Update iterates
        # ------------------------------------------------------------------
        q, u, y = q_new, u_new, y_new
        r, w = r_new, w_new

        # Log
        obj_val = obj_fun(y) if obj_fun is not None else 0.0
        result.obj_history.append(obj_val)
        result.kkt_history.append(G_new)

        if not silent:
            print(f"{itr:5d}  {obj_val:13.6e}  {G_new:13.6e}  {mu:13.6e}  {lam_max:10.3e}  {kount:7d}")

        # ------------------------------------------------------------------
        # Complementarity gap and mu update
        # ------------------------------------------------------------------
        d = c - C.T @ u
        G1 = np.sum(q * d)
        if pCon:
            G1 += np.sum(r * w)
            num_slacks = 2 * len(q) + 2 * len(r)
        else:
            num_slacks = 2 * len(q)

        comp_frac = G1 / num_slacks
        mu = 0.1 * comp_frac

        # ------------------------------------------------------------------
        # Convergence check
        # ------------------------------------------------------------------
        converged = (G1 < opt_tol) or (G_new < opt_tol)

        if converged:
            if not silent:
                print(f"\nConverged at iteration {itr}. mu={mu:.2e}, KKT={G_new:.2e}")
            break

    result.y = y
    result.u = u
    result.q = q
    result.r = r
    result.w = w
    result.mu_final = mu
    result.iterations = itr + 1
    result.converged = converged if 'converged' in dir() else False
    return result
