"""
High-level interface for solving PLQ problems.

    x = solve(H, z, meas="huber", proc="l1", proc_lambda=0.1)

This is the Python equivalent of the MATLAB ``run_example.m``.

Refactor highlights (v0.2)
--------------------------
- compose() imported from plq (was duplicated here as _compose).
- _PARAM_MAP replaces ad-hoc renaming logic.
- return_result flag to get the full SolverResult.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
import scipy.sparse as sp

from ipsolve.penalties import load_penalty
from ipsolve.plq import PLQ, compose, ensure_sparse
from ipsolve.solver import ip_solve, SolverResult


# Map from user-facing kwarg suffix -> penalty factory arg name
_PARAM_MAP = {
    "lambda": "lam",
    "kappa": "kappa",
    "mMult": "mMult",
    "tau": "tau",
    "scale": "scale",
    "eps": "eps",
    "alpha": "alpha",
}


def solve(
    H: Union[np.ndarray, sp.spmatrix, Callable],
    z: np.ndarray,
    meas: str = "l2",
    proc: Optional[str] = None,
    lin_term: Optional[np.ndarray] = None,
    *,
    # Measurement penalty parameters
    meas_lambda: float = 1.0,
    meas_kappa: float = 1.0,
    meas_mMult: float = 1.0,
    meas_tau: float = 0.5,
    meas_scale: float = 1.0,
    meas_eps: float = 0.2,
    meas_alpha: float = 0.5,
    # Process (regulariser) penalty parameters
    proc_lambda: float = 1.0,
    proc_kappa: float = 1.0,
    proc_mMult: float = 1.0,
    proc_tau: float = 0.5,
    proc_scale: float = 1.0,
    proc_eps: float = 0.2,
    # Process linear model
    K_proc: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    k_proc: Optional[np.ndarray] = None,
    # Linear constraints:  A @ x <= a
    A_ineq: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    a_ineq: Optional[np.ndarray] = None,
    # Solver options
    rho: float = 0.0,
    delta: float = 0.0,
    inexact: bool = False,
    mehrotra: bool = False,
    opt_tol: float = 1e-5,
    max_iter: int = 100,
    silent: bool = False,
    u_init: float = 1e-6,
    return_result: bool = False,
) -> Union[np.ndarray, SolverResult]:
    """Solve a PLQ composite problem.

    Parameters
    ----------
    H : (m, n) matrix or callable
        Linear model (matrix) or nonlinear forward model (callable).
    z : (m,) array
        Observed data / measurements.
    meas : str
        Measurement penalty name (e.g. 'l2', 'huber', 'l1').
    proc : str or None
        Process / regularisation penalty, or None.
    lin_term : (n,) array or None
        Optional linear term l'x in the objective.
    return_result : bool
        If True return the full SolverResult; otherwise return x only.

    Returns
    -------
    x : (n,) array  (default)
    result : SolverResult  (if return_result=True)
    """
    z = np.asarray(z, dtype=float).ravel()

    if callable(H):
        raise NotImplementedError(
            "Nonlinear (callable) H is not yet supported. "
            "Pass a matrix for the linear model."
        )

    # Collect all kwargs into a dict for forwarding
    opts = dict(
        meas_lambda=meas_lambda, meas_kappa=meas_kappa,
        meas_mMult=meas_mMult, meas_tau=meas_tau,
        meas_scale=meas_scale, meas_eps=meas_eps,
        meas_alpha=meas_alpha,
        proc_lambda=proc_lambda, proc_kappa=proc_kappa,
        proc_mMult=proc_mMult, proc_tau=proc_tau,
        proc_scale=proc_scale, proc_eps=proc_eps,
        K_proc=K_proc, k_proc=k_proc,
        A_ineq=A_ineq, a_ineq=a_ineq,
        rho=rho, delta=delta, inexact=inexact, mehrotra=mehrotra,
        opt_tol=opt_tol, max_iter=max_iter, silent=silent, u_init=u_init,
    )
    result = _solve_linear(H, z, meas, proc, lin_term, **opts)

    if return_result:
        return result
    return result.y


# ======================================================================
# Linear (explicit H) path
# ======================================================================

def _solve_linear(H, z, meas, proc, lin_term, **kwargs) -> SolverResult:
    """Solve when H is a matrix.  Returns full SolverResult."""
    H = ensure_sparse(H) if sp.issparse(H) else np.atleast_2d(H)
    m, n = H.shape
    z = z.ravel()

    if lin_term is None:
        lin_term = np.zeros(n)
    else:
        lin_term = np.asarray(lin_term, dtype=float).ravel()

    # ------------------------------------------------------------------
    # Measurement PLQ
    # ------------------------------------------------------------------
    meas_kw = _penalty_kwargs(kwargs, "meas_")
    plq_meas = load_penalty(meas, m, **meas_kw)
    plq_meas = compose(plq_meas, H, z)

    # ------------------------------------------------------------------
    # Process PLQ  (optional)
    # ------------------------------------------------------------------
    has_proc = proc is not None
    if has_proc:
        K_mat = kwargs.get("K_proc")
        k_vec = kwargs.get("k_proc")
        if K_mat is not None:
            K_mat = ensure_sparse(K_mat) if sp.issparse(K_mat) else np.atleast_2d(K_mat)
            if k_vec is None:
                k_vec = np.zeros(K_mat.shape[0])
            p_dim = K_mat.shape[0]
        else:
            K_mat = sp.eye(n, format="csc")
            k_vec = np.zeros(n)
            p_dim = n

        proc_kw = _penalty_kwargs(kwargs, "proc_")
        plq_proc = load_penalty(proc, p_dim, **proc_kw)
        plq_proc = compose(plq_proc, K_mat, k_vec)

    # ------------------------------------------------------------------
    # Assemble the full system
    # ------------------------------------------------------------------
    if has_proc:
        b_full = np.concatenate([plq_meas.b, plq_proc.b])
        c_full = np.concatenate([plq_meas.c, plq_proc.c])
        C_full = sp.block_diag([plq_meas.C, plq_proc.C], format="csc")

        B_meas = ensure_sparse(plq_meas.B)
        B_proc = ensure_sparse(plq_proc.B)
        M_meas = plq_meas.M
        M_proc = plq_proc.M

        K_total = B_meas.shape[0] + B_proc.shape[0]
    else:
        b_full = plq_meas.b
        c_full = plq_meas.c
        C_full = plq_meas.C
        B_meas = ensure_sparse(plq_meas.B)
        M_meas = plq_meas.M
        B_proc = None
        M_proc = None
        K_total = B_meas.shape[0]

    # Store as (K, L) per convention: C' acts on u
    C_full = ensure_sparse(C_full.T)
    L = C_full.shape[1]

    # ------------------------------------------------------------------
    # Objective function for logging
    # ------------------------------------------------------------------
    def obj_fun(x):
        val = lin_term @ x
        if plq_meas.obj is not None:
            val += plq_meas.obj(z - H @ x)
        if has_proc and plq_proc.obj is not None:
            K_mat_local = kwargs.get("K_proc")
            if K_mat_local is not None:
                k_vec_local = kwargs.get("k_proc", np.zeros(K_mat_local.shape[0]))
                val += plq_proc.obj(K_mat_local @ x - k_vec_local)
            else:
                val += plq_proc.obj(x)
        return val

    # ------------------------------------------------------------------
    # Initialise variables
    # ------------------------------------------------------------------
    q0 = 10.0 * np.ones(L)
    u0 = np.zeros(K_total) + kwargs.get("u_init", 1e-6)
    y0 = np.ones(n)

    # Constraints
    A_ineq = kwargs.get("A_ineq")
    a_ineq = kwargs.get("a_ineq")
    if A_ineq is not None:
        A_sp = ensure_sparse(A_ineq)
        if A_sp.shape[0] != n:
            A_sp = A_sp.T
        r0 = 10.0 * np.ones(A_sp.shape[1])
        w0 = 10.0 * np.ones(A_sp.shape[1])
    else:
        A_sp = None
        a_ineq = None
        r0 = None
        w0 = None

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    return ip_solve(
        lin_term, b_full, B_meas, c_full, C_full, M_meas,
        q0, u0, r0, w0, y0,
        obj_fun=obj_fun,
        B_proc=B_proc, M_proc=M_proc,
        A=A_sp, a=a_ineq,
        rho=kwargs.get("rho", 0.0),
        delta=kwargs.get("delta", 0.0),
        inexact=kwargs.get("inexact", False),
        mehrotra=kwargs.get("mehrotra", False),
        opt_tol=kwargs.get("opt_tol", 1e-5),
        max_iter=kwargs.get("max_iter", 100),
        silent=kwargs.get("silent", False),
    )


# ======================================================================
# Helpers
# ======================================================================

def _penalty_kwargs(kwargs: dict, prefix: str) -> dict:
    """Extract penalty-specific kwargs, stripping prefix and renaming.

    E.g. meas_lambda=0.1  ->  lam=0.1.
    """
    out = {}
    for k, v in kwargs.items():
        if k.startswith(prefix):
            short = k[len(prefix):]
            mapped = _PARAM_MAP.get(short, short)
            out[mapped] = v
    return out
