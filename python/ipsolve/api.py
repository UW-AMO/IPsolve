"""
High-level interface for solving PLQ problems.

    x = solve(H, z, meas="huber", proc="l1", proc_lambda=0.1)

This is the Python equivalent of the MATLAB ``run_example.m``.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
import scipy.sparse as sp

from ipsolve.penalties import load_penalty
from ipsolve.plq import PLQ, stack, ensure_sparse
from ipsolve.solver import ip_solve_barrier, SolverResult


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
    # Linear constraints:  A @ x ≤ a
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
) -> np.ndarray:
    """Solve a PLQ composite problem.

    Parameters
    ----------
    H : (m, n) matrix or callable
        Linear model (matrix) or nonlinear forward model (callable).
        If callable: ``residual, Jacobian = H(x)`` where residual is (m,)
        and Jacobian is (m, n).
    z : (m,) array
        Observed data / measurements.
    meas : str
        Name of the measurement penalty (e.g. ``'l2'``, ``'huber'``, ``'l1'``).
    proc : str or None
        Name of the process / regularisation penalty, or ``None`` for no
        regularisation.
    lin_term : (n,) array or None
        Optional linear term ℓᵀx in the objective.

    Returns
    -------
    x : (n,) array
        The optimal solution.
    """
    z = np.asarray(z, dtype=float).ravel()

    if callable(H):
        raise NotImplementedError(
            "Nonlinear (callable) H is not yet supported. "
            "Pass a matrix for the linear model."
        )

    return _solve_linear(
        H, z, meas, proc, lin_term,
        meas_lambda=meas_lambda, meas_kappa=meas_kappa, meas_mMult=meas_mMult,
        meas_tau=meas_tau, meas_scale=meas_scale, meas_eps=meas_eps,
        meas_alpha=meas_alpha,
        proc_lambda=proc_lambda, proc_kappa=proc_kappa, proc_mMult=proc_mMult,
        proc_tau=proc_tau, proc_scale=proc_scale, proc_eps=proc_eps,
        K_proc=K_proc, k_proc=k_proc,
        A_ineq=A_ineq, a_ineq=a_ineq,
        rho=rho, delta=delta, inexact=inexact, mehrotra=mehrotra,
        opt_tol=opt_tol, max_iter=max_iter, silent=silent, u_init=u_init,
    )


# ======================================================================
# Linear (explicit H) path
# ======================================================================

def _solve_linear(
    H, z, meas, proc, lin_term, **kwargs
) -> np.ndarray:
    """Solve when H is a matrix."""
    H = ensure_sparse(H) if sp.issparse(H) else np.atleast_2d(H)
    m, n = H.shape
    z = z.ravel()

    if lin_term is None:
        lin_term = np.zeros(n)
    else:
        lin_term = np.asarray(lin_term, dtype=float).ravel()

    # ------------------------------------------------------------------
    # Build measurement PLQ
    # ------------------------------------------------------------------
    meas_kw = _filter_penalty_kwargs(meas, kwargs, prefix="meas_")
    plq_meas = load_penalty(meas, m, **meas_kw)
    # Compose with linear model:  b ← −b − B·z;  B ← B·H
    plq_meas = _compose(plq_meas, H, z)

    # ------------------------------------------------------------------
    # Build process PLQ  (optional)
    # ------------------------------------------------------------------
    pFlag = proc is not None
    if pFlag:
        # Custom linear operator for regulariser
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

        proc_kw = _filter_penalty_kwargs(proc, kwargs, prefix="proc_")
        plq_proc = load_penalty(proc, p_dim, **proc_kw)
        plq_proc = _compose(plq_proc, K_mat, k_vec)

    # ------------------------------------------------------------------
    # Assemble the full system
    # ------------------------------------------------------------------
    if pFlag:
        # Two-block structure: keep Bm and B2 separate for efficiency
        b_full = np.concatenate([plq_meas.b, plq_proc.b])
        c_full = np.concatenate([plq_meas.c, plq_proc.c])
        C_full = sp.block_diag([plq_meas.C, plq_proc.C], format="csc")

        Bm = ensure_sparse(plq_meas.B)
        B2 = ensure_sparse(plq_proc.B)
        M_meas = plq_meas.M
        M2 = plq_proc.M

        K_total = Bm.shape[0] + B2.shape[0]
    else:
        b_full = plq_meas.b
        c_full = plq_meas.c
        C_full = plq_meas.C
        Bm = ensure_sparse(plq_meas.B)
        M_meas = plq_meas.M
        B2 = None
        M2 = None
        K_total = Bm.shape[0]

    C_full = ensure_sparse(C_full.T)  # Store as (K, L) per MATLAB convention: C is K×L

    L = C_full.shape[1]

    # ------------------------------------------------------------------
    # Build objective function for logging
    # ------------------------------------------------------------------
    def obj_fun(x):
        val = lin_term @ x
        if plq_meas.obj is not None:
            val += plq_meas.obj(z - H @ x)
        if pFlag and plq_proc.obj is not None:
            if kwargs.get("K_proc") is not None:
                val += plq_proc.obj(kwargs["K_proc"] @ x - kwargs.get("k_proc", np.zeros(kwargs["K_proc"].shape[0])))
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
        P = A_sp.shape[1] if A_sp.shape[0] == n else A_sp.shape[0]
        # Convention: A is (N, P), A.T @ y ≤ a
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
    result = ip_solve_barrier(
        lin_term, b_full, Bm, c_full, C_full, M_meas,
        q0, u0, r0, w0, y0,
        obj_fun=obj_fun,
        B2=B2, M2=M2,
        A=A_sp, a=a_ineq,
        rho=kwargs.get("rho", 0.0),
        delta=kwargs.get("delta", 0.0),
        inexact=kwargs.get("inexact", False),
        mehrotra=kwargs.get("mehrotra", False),
        opt_tol=kwargs.get("opt_tol", 1e-5),
        max_iter=kwargs.get("max_iter", 100),
        silent=kwargs.get("silent", False),
    )

    return result.y


# ======================================================================
# Helpers
# ======================================================================

def _compose(plq: PLQ, H, z: np.ndarray) -> PLQ:
    """Compose PLQ penalty with linear model: v = B·(Hx) − (b + B·z).

    Transforms:  b ← −b − B·z;  B ← B·H.
    """
    B_new = ensure_sparse(plq.B @ H)
    b_new = -plq.b - plq.B @ z
    return PLQ(
        M=plq.M,
        B=B_new,
        C=plq.C,
        c=plq.c,
        b=b_new,
        obj=plq.obj,
    )


def _filter_penalty_kwargs(penalty_name: str, kwargs: dict, prefix: str) -> dict:
    """Extract penalty-specific kwargs, stripping the prefix.

    E.g. ``meas_lambda=0.1`` → ``lam=0.1`` for the penalty factory.
    """
    # Map from our keyword names (with prefix stripped) to penalty factory arg names
    name_map = {
        "lambda": "lam",
        "kappa": "kappa",
        "mMult": "mMult",
        "tau": "tau",
        "scale": "scale",
        "eps": "eps",
        "alpha": "alpha",
    }

    result = {}
    for k, v in kwargs.items():
        if k.startswith(prefix):
            short = k[len(prefix):]
            mapped = name_map.get(short, short)
            result[mapped] = v
    return result
