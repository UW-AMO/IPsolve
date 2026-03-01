"""
KKT system – barrier formulation residual and Newton solve.

This module implements the core linear algebra for the interior-point method.
The barrier formulation avoids an explicit slack variable *s* by using
``d = c − Cᵀu`` as the implicit slack.

Notation
--------
- ``u``  : (K,) dual variable for the PLQ penalty
- ``q``  : (L,) multiplier for the dual constraints Cᵀu ≤ c
- ``y``  : (N,) primal decision variable
- ``w``  : (P,) multiplier for linear constraints Ax ≤ a
- ``r``  : (P,) slack for linear constraints
- ``d``  : (L,) implicit slack  d = c − Cᵀu  (must stay > 0)
- ``μ``  : barrier parameter
- ``Bm`` : (m, N) measurement model matrix
- ``M``  : (K, K) or callable – quadratic part of PLQ conjugate
- ``C``  : (K, L) constraint matrix (C.T @ u ≤ c)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ======================================================================
# KKT Residual (barrier formulation)
# ======================================================================

def kkt_residual_barrier(
    lin_term: np.ndarray,
    b: np.ndarray,
    Bm: sp.spmatrix,
    c: np.ndarray,
    C: sp.spmatrix,
    M,
    q: np.ndarray,
    u: np.ndarray,
    r: Optional[np.ndarray],
    w: Optional[np.ndarray],
    y: np.ndarray,
    mu: float,
    *,
    B2: Optional[sp.spmatrix] = None,
    M2=None,
    A: Optional[sp.spmatrix] = None,
    a: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute the KKT residual vector F for the barrier formulation.

    Returns
    -------
    F : (L + K + 2P + N,) array
        Concatenation [r1; r2; r3; r4; r5] of residual blocks.
    """
    m = Bm.shape[0]
    pFlag = B2 is not None
    pCon = A is not None

    # Implicit slack
    d = c - C.T @ u

    # Block 1: complementarity  d .* q − μ
    r1 = d * q - mu

    # Evaluate M @ u  (possibly block-diagonal with process)
    um = u[:m] if pFlag else u
    funM = callable(M)
    if funM:
        Mum, _ = M(um)
    else:
        Mum = M @ um

    if pFlag:
        un = u[m:]
        Mu = np.concatenate([Mum, M2 @ un])
    else:
        Mu = Mum

    # Block 2: stationarity in u
    if pFlag:
        By = np.concatenate([Bm @ y, B2 @ y])
    else:
        By = Bm @ y

    r2 = By - Mu - C @ q + b

    # Blocks 3,4: constraint complementarity and feasibility
    if pCon:
        Aw = A @ w
        Aty = A.T @ y
        r4 = r + Aty - a
        r3 = w * r - mu
    else:
        Aw = 0.0
        r3 = np.array([])
        r4 = np.array([])

    # Block 5: stationarity in y
    if pFlag:
        r5 = Bm.T @ um + B2.T @ un + Aw + lin_term
    else:
        r5 = Bm.T @ um + Aw + lin_term

    return np.concatenate([r1, r2, r3, r4, r5])


# ======================================================================
# KKT Solve (barrier formulation)
# ======================================================================

@dataclass
class KKTSolveResult:
    dq: np.ndarray
    du: np.ndarray
    dr: Optional[np.ndarray]
    dw: Optional[np.ndarray]
    dy: np.ndarray
    pcg_iters: int = 0


def kkt_solve_barrier(
    lin_term: np.ndarray,
    b: np.ndarray,
    Bm: sp.spmatrix,
    c: np.ndarray,
    C: sp.spmatrix,
    M,
    q: np.ndarray,
    u: np.ndarray,
    r: Optional[np.ndarray],
    w: Optional[np.ndarray],
    y: np.ndarray,
    mu: float,
    *,
    B2: Optional[sp.spmatrix] = None,
    M2=None,
    A: Optional[sp.spmatrix] = None,
    a: Optional[np.ndarray] = None,
    rho: float = 0.0,
    delta: float = 0.0,
    inexact: bool = False,
    pcg_tol: float = 1e-8,
) -> KKTSolveResult:
    """Solve the Newton system for the barrier IP method by Schur complement.

    Strategy
    --------
    1. Eliminate dq using  dq = (r1 + Q·Cᵀ·du) / d.
    2. Form  T = MM + C·(Q/D)·Cᵀ  and eliminate du.
    3. Solve the reduced ``(N, N)`` system in dy via Cholesky or PCG.
    4. Back-substitute for du, dq, dw, dr.
    """
    m = Bm.shape[0]
    n = y.shape[0]
    pFlag = B2 is not None
    pCon = A is not None
    funM = callable(M)

    # Diagonal matrices
    d = c - C.T @ u
    Q = sp.diags(q, format="csc")
    QD = sp.diags(q / d, format="csc")

    if pCon:
        WR = sp.diags(w / r, format="csc")
    else:
        WR = None

    # Build MM (full block-diagonal Hessian including regularisation)
    um = u[:m] if pFlag else u
    if funM:
        Mum, MMm = M(um)
    else:
        MMm = M
        Mum = MMm @ um

    if pFlag:
        un = u[m:]
        MMn = M2
        Mu = np.concatenate([Mum, MMn @ un])
        # Build block diagonal MM with regularisation
        sz_m = MMm.shape[0]
        sz_n = MMn.shape[0]
        if rho > 0:
            MMm_reg = MMm + rho * sp.eye(sz_m, format="csc")
            MMn_reg = MMn + rho * sp.eye(sz_n, format="csc")
        else:
            MMm_reg = MMm
            MMn_reg = MMn
        MM = sp.block_diag([MMm_reg, MMn_reg], format="csc")
    else:
        Mu = Mum
        if rho > 0:
            MM = MMm + rho * sp.eye(MMm.shape[0], format="csc")
        else:
            MM = MMm

    # Build T = MM + C · QD · Cᵀ
    T = MM + C @ QD @ C.T
    T = sp.csc_matrix(T)

    # Factor T  (direct solve for T⁻¹)
    try:
        T_factored = spla.splu(T)
        T_solve = T_factored.solve
    except Exception:
        # Fallback: dense solve
        T_dense = T.toarray()
        T_solve = lambda rhs: np.linalg.solve(T_dense, rhs)  # noqa: E731

    # RHS construction
    # r1 = -d.*q + mu
    r1 = -d * q + mu

    # r2: stationarity residual (modified for barrier)
    if pFlag:
        By = np.concatenate([Bm @ y, B2 @ y])
    else:
        By = Bm @ y
    r2 = -(By - Mu + b) + mu * (C @ (1.0 / d))

    # Constraint terms
    if pCon:
        r3 = -r * w + mu
        r4 = -A.T @ y + a - mu / w
        Awr4r = A @ (w - WR @ r4)
    else:
        Awr4r = 0.0

    # u_tilde = u - T⁻¹ r2
    utr = u - T_solve(r2)

    # r5: reduced RHS for dy
    if pFlag:
        r5 = -Bm.T @ utr[:m] - B2.T @ utr[m:] - Awr4r - lin_term
    else:
        r5 = -Bm.T @ utr - Awr4r - lin_term

    # ------------------------------------------------------------------
    # Solve for dy
    # ------------------------------------------------------------------
    pcg_iters = 0

    if pFlag:
        Tm = T[:m, :m]
        Tn = T[m:, m:]
        try:
            Tm_factored = spla.splu(sp.csc_matrix(Tm))
            Tm_solve = Tm_factored.solve
        except Exception:
            Tm_dense = Tm.toarray() if sp.issparse(Tm) else np.asarray(Tm)
            Tm_solve = lambda rhs: np.linalg.solve(Tm_dense, rhs)  # noqa: E731
        try:
            Tn_factored = spla.splu(sp.csc_matrix(Tn))
            Tn_solve = Tn_factored.solve
        except Exception:
            Tn_dense = Tn.toarray() if sp.issparse(Tn) else np.asarray(Tn)
            Tn_solve = lambda rhs: np.linalg.solve(Tn_dense, rhs)  # noqa: E731

    if inexact:
        # PCG on the Schur complement
        if pFlag:
            def omega_mv(x):
                v = Bm.T @ Tm_solve(Bm @ x) + B2.T @ Tn_solve(B2 @ x)
                if pCon:
                    v += A @ (WR @ (A.T @ x))
                return v + delta * x

            Omega_op = spla.LinearOperator((n, n), matvec=omega_mv)
            dy, info_cg = spla.cg(Omega_op, r5, tol=pcg_tol, maxiter=10000)
            pcg_iters = 0  # cg doesn't easily report iters; relying on convergence
        else:
            def omega_mv(x):
                v = Bm.T @ T_solve(Bm @ x)
                if pCon:
                    v += A @ (WR @ (A.T @ x))
                return v + delta * x

            Omega_op = spla.LinearOperator((n, n), matvec=omega_mv)
            dy, info_cg = spla.cg(Omega_op, r5, tol=pcg_tol, maxiter=10000)
    else:
        # Direct Cholesky / LU on Omega
        if pFlag:
            Omega = (Bm.T @ Tm_solve(Bm.toarray())
                     + B2.T @ Tn_solve(B2.toarray()))
            Omega = sp.csc_matrix(Omega)
        else:
            Omega = sp.csc_matrix(Bm.T @ T_solve(Bm.toarray()))

        if pCon:
            Omega = Omega + A @ WR @ A.T
        if delta > 0:
            Omega = Omega + delta * sp.eye(n, format="csc")

        try:
            # Try Cholesky-like factorisation
            Omega_lu = spla.splu(sp.csc_matrix(Omega))
            dy = Omega_lu.solve(r5)
        except Exception:
            dy = np.linalg.solve(Omega.toarray(), r5)

    # ------------------------------------------------------------------
    # Back-substitution
    # ------------------------------------------------------------------

    # dw, dr
    if pCon:
        dw = -WR @ (r4 - A.T @ dy)
        dr = (r3 - r * dw) / w
    else:
        dw = None
        dr = None

    # du
    if pFlag:
        du = T_solve(-r2 + np.concatenate([Bm @ dy, B2 @ dy]))
    else:
        du = T_solve(-r2 + Bm @ dy)

    # dq
    dq = (r1 + q * (C.T @ du)) / d

    return KKTSolveResult(dq=dq, du=du, dr=dr, dw=dw, dy=dy, pcg_iters=pcg_iters)
