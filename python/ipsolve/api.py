"""
High-level interface for solving PLQ problems.

Simple examples::

    from ipsolve import solve, huber, l1, l2, hinge

    x = solve(H, z)                                       # least squares
    x = solve(H, z, meas=huber(kappa=1.0))                # robust regression
    x = solve(H, z, proc=l1(lam=0.5))                     # lasso
    x = solve(H, z, meas=huber(), proc=l1(lam=0.5))       # huber + l1
    w = solve(yX, np.ones(m), meas=hinge(), proc=l2(lam=0.01))  # SVM
    x = solve(H, z, bounds=(-1, 1))                        # box constraints

Backward compatibility: string names with prefixed kwargs still work::

    x = solve(H, z, meas="huber", proc="l1", meas_kappa=1.0, proc_lambda=0.5)
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import scipy.sparse as sp

from ipsolve.penalties import Penalty, load_penalty
from ipsolve.plq import PLQ, compose, ensure_sparse
from ipsolve.solver import ip_solve, SolverResult


# Map from old prefix-style kwarg suffix -> penalty factory arg name
_PARAM_MAP = {
    "lambda": "lam",
    "kappa": "kappa",
    "mMult": "mMult",
    "tau": "tau",
    "scale": "scale",
    "eps": "eps",
    "alpha": "alpha",
}


def _resolve_penalty(spec, m):
    """Resolve a penalty specifier into a PLQ of dimension m."""
    if isinstance(spec, str):
        return load_penalty(spec, m)
    if isinstance(spec, Penalty):
        return spec.build(m)
    if isinstance(spec, PLQ):
        return spec
    raise TypeError(
        "Expected str, Penalty, or PLQ for meas/proc; got {0}".format(type(spec))
    )


def _extract_prefix(kwargs, prefix):
    """Extract and rename kwargs with a given prefix (backward compat).

    E.g. meas_kappa=1.0 -> kappa=1.0;  proc_lambda=0.5 -> lam=0.5.
    """
    out = {}
    for k, v in kwargs.items():
        if k.startswith(prefix):
            short = k[len(prefix):]
            mapped = _PARAM_MAP.get(short, short)
            out[mapped] = v
    return out


def solve(
    H: Union[np.ndarray, sp.spmatrix],
    z: np.ndarray,
    meas="l2",
    proc=None,
    *,
    lin_term: Optional[np.ndarray] = None,
    K_proc=None,
    k_proc=None,
    A_ineq=None,
    a_ineq=None,
    bounds=None,
    rho: float = 0.0,
    delta: float = 0.0,
    inexact: bool = False,
    mehrotra: bool = False,
    opt_tol: float = 1e-5,
    max_iter: int = 100,
    silent: bool = False,
    u_init: float = 1e-6,
    return_result: bool = False,
    **kwargs,
) -> Union[np.ndarray, SolverResult]:
    """Solve a PLQ composite problem.

    Minimises::

        min_x  l'x  +  rho_meas(z - Hx)  +  rho_proc(Kx - k)
        s.t.   Ax <= a

    Parameters
    ----------
    H : (m, n) array
        Forward / measurement model.
    z : (m,) array
        Observations.
    meas : str or Penalty or PLQ
        Measurement loss.  Default ``"l2"`` (least squares).
        Use penalty objects for clarity::

            meas=huber(kappa=1.0)

    proc : str or Penalty or PLQ or None
        Regularizer / process penalty.  ``None`` means no regularizer.
    lin_term : (n,) array, optional
        Linear objective term l'x.
    K_proc, k_proc : matrix and vector, optional
        Process operator: penalty applied to ``K_proc @ x - k_proc``.
        If omitted, defaults to identity (regularize x directly).
    A_ineq, a_ineq : matrix and vector, optional
        Linear inequality constraints: ``A_ineq.T @ x <= a_ineq``.
    bounds : (lb, ub), optional
        Box constraints ``lb <= x <= ub``.  Scalar or per-variable arrays.
    rho, delta : float
        Regularization parameters for the KKT system.
    inexact : bool
        Use CG Schur solve instead of direct LU.
    opt_tol : float
        Optimality tolerance.
    max_iter : int
        Iteration limit.
    silent : bool
        Suppress solver output.
    return_result : bool
        If True, return full :class:`SolverResult` instead of just x.
    **kwargs
        Backward compatibility: ``meas_kappa=1.0``, ``proc_lambda=0.5``, etc.

    Returns
    -------
    x : (n,) array  (default)
    result : SolverResult  (if ``return_result=True``)
    """
    z = np.asarray(z, dtype=float).ravel()

    if callable(H):
        raise NotImplementedError(
            "Nonlinear (callable) H is not yet supported. "
            "Pass a matrix for the linear model."
        )

    H = ensure_sparse(H) if sp.issparse(H) else np.atleast_2d(H)
    m, n = H.shape

    if lin_term is None:
        lin_term = np.zeros(n)
    else:
        lin_term = np.asarray(lin_term, dtype=float).ravel()

    # ------------------------------------------------------------------
    # Backward compat: convert meas_*/proc_* kwargs to Penalty specs
    # ------------------------------------------------------------------
    if isinstance(meas, str) and any(k.startswith("meas_") for k in kwargs):
        meas = Penalty(meas, _extract_prefix(kwargs, "meas_"))
    if isinstance(proc, str) and any(k.startswith("proc_") for k in kwargs):
        proc = Penalty(proc, _extract_prefix(kwargs, "proc_"))

    # ------------------------------------------------------------------
    # Measurement PLQ
    # ------------------------------------------------------------------
    plq_meas = compose(_resolve_penalty(meas, m), H, z)

    # ------------------------------------------------------------------
    # Process PLQ (optional)
    # ------------------------------------------------------------------
    has_proc = proc is not None
    if has_proc:
        if K_proc is not None:
            K_mat = (
                ensure_sparse(K_proc) if sp.issparse(K_proc)
                else np.atleast_2d(K_proc)
            )
            if k_proc is None:
                k_proc = np.zeros(K_mat.shape[0])
            p_dim = K_mat.shape[0]
        else:
            K_mat = sp.eye(n, format="csc")
            k_proc = np.zeros(n)
            p_dim = n

        plq_proc = compose(_resolve_penalty(proc, p_dim), K_mat, k_proc)

    # ------------------------------------------------------------------
    # Box constraints  (bounds=(lb, ub))
    # ------------------------------------------------------------------
    if bounds is not None:
        lb, ub = bounds
        lb = np.broadcast_to(np.asarray(lb, dtype=float), n).copy()
        ub = np.broadcast_to(np.asarray(ub, dtype=float), n).copy()
        A_box = np.vstack([np.eye(n), -np.eye(n)])
        a_box = np.concatenate([ub, -lb])
        if A_ineq is not None:
            A_ineq_arr = (
                A_ineq.toarray() if sp.issparse(A_ineq)
                else np.asarray(A_ineq)
            )
            A_ineq = np.vstack([A_ineq_arr, A_box])
            a_ineq = np.concatenate([a_ineq, a_box])
        else:
            A_ineq = A_box
            a_ineq = a_box

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
            if K_proc is not None:
                val += plq_proc.obj(
                    K_proc @ x - (k_proc if k_proc is not None else 0)
                )
            else:
                val += plq_proc.obj(x)
        return val

    # ------------------------------------------------------------------
    # Initialize variables
    # ------------------------------------------------------------------
    q0 = 10.0 * np.ones(L)
    u0 = np.zeros(K_total) + u_init
    y0 = np.ones(n)

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
    result = ip_solve(
        lin_term, b_full, B_meas, c_full, C_full, M_meas,
        q0, u0, r0, w0, y0,
        obj_fun=obj_fun,
        B_proc=B_proc, M_proc=M_proc,
        A=A_sp, a=a_ineq,
        rho=rho, delta=delta, inexact=inexact, mehrotra=mehrotra,
        opt_tol=opt_tol, max_iter=max_iter, silent=silent,
    )

    return result if return_result else result.y
