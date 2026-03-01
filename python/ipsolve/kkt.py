"""
KKT system -- barrier formulation residual and Newton solve.

Core linear algebra for the interior-point method.  The barrier formulation
uses ``d = c - C'u`` as the implicit slack (avoiding an explicit variable).

Notation
--------
- u      : (K,) dual variable for PLQ penalty
- q      : (L,) multiplier for dual constraints C'u <= c
- y      : (N,) primal decision variable
- w, r   : (P,) constraint multiplier and slack
- d      : (L,) implicit slack  d = c - C'u  (must stay > 0)
- mu     : barrier parameter
- B_meas : (m, N) measurement model matrix
- B_proc : (p, N) process/regulariser model matrix (optional)
- M      : (K, K) or callable -- quadratic part of PLQ conjugate
- M_proc : (p, p) -- quadratic part for process block
- C      : (K, L) constraint matrix (C'u <= c)

Refactor highlights (v0.2)
--------------------------
- SchurOperator encapsulates Omega = B'T^{-1}B as either a dense matrix
  (for direct LU solve) or a matrix-free LinearOperator (for CG).
- _sparse_factor: unified SuperLU -> dense fallback helper.
- Cleaner back-substitution with named variables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ======================================================================
# Helpers
# ======================================================================

def _sparse_factor(A):
    """Factor a sparse matrix; return a callable solve(rhs).

    Tries SuperLU first, falls back to dense LU.
    """
    try:
        lu = spla.splu(sp.csc_matrix(A))
        return lu.solve
    except Exception:
        Ad = A.toarray() if sp.issparse(A) else np.asarray(A)
        return lambda rhs: np.linalg.solve(Ad, rhs)


class _CGCounter:
    """Callback for scipy.sparse.linalg.cg to count iterations."""

    def __init__(self):
        self.n = 0

    def __call__(self, _xk):
        self.n += 1


# ======================================================================
# SchurOperator
# ======================================================================

class SchurOperator:
    """Schur complement  Omega = B' T^{-1} B  (+ constraint / reg terms).

    Can be used either as a dense matrix (for direct LU) or as a
    matrix-free LinearOperator (for CG).

    Parameters
    ----------
    B_meas : (m, N) sparse -- measurement model.
    T_solve_m : callable(rhs) -- solves T_m @ x = rhs for measurement block.
    B_proc : (p, N) sparse or None -- process model.
    T_solve_p : callable(rhs) or None -- solves T_p @ x = rhs for process block.
    WR : (P, P) diag sparse or None -- W/R for constraints.
    A : (N, P) sparse or None -- constraint matrix.
    delta : float -- Tikhonov regularisation.
    """

    def __init__(self, B_meas, T_solve_m, B_proc=None, T_solve_p=None,
                 WR=None, A=None, delta=0.0):
        self.B_meas = B_meas
        self.T_solve_m = T_solve_m
        self.B_proc = B_proc
        self.T_solve_p = T_solve_p
        self.WR = WR
        self.A = A
        self.delta = delta
        self.n = B_meas.shape[1]

    # -- mat-vec (for CG) ------------------------------------------------
    def matvec(self, x):
        v = self.B_meas.T @ self.T_solve_m(self.B_meas @ x)
        if self.B_proc is not None:
            v += self.B_proc.T @ self.T_solve_p(self.B_proc @ x)
        if self.WR is not None:
            v += self.A @ (self.WR @ (self.A.T @ x))
        if self.delta > 0:
            v += self.delta * x
        return v

    def as_linop(self):
        """Return a scipy LinearOperator for CG."""
        return spla.LinearOperator((self.n, self.n), matvec=self.matvec)

    # -- dense (for direct solve) ----------------------------------------
    def to_dense(self):
        """Materialise Omega as a dense/sparse matrix."""
        Omega = self.B_meas.T @ self.T_solve_m(self.B_meas.toarray())
        if self.B_proc is not None:
            Omega += self.B_proc.T @ self.T_solve_p(self.B_proc.toarray())
        Omega = sp.csc_matrix(Omega)
        if self.WR is not None:
            Omega = Omega + self.A @ self.WR @ self.A.T
        if self.delta > 0:
            Omega = Omega + self.delta * sp.eye(self.n, format="csc")
        return Omega


# ======================================================================
# Schur solve dispatcher
# ======================================================================

def _solve_schur(omega, rhs, inexact, pcg_tol):
    """Solve Omega @ dy = rhs, using CG or direct LU.

    Returns (dy, cg_iters).
    """
    if inexact:
        counter = _CGCounter()
        dy, info = spla.cg(omega.as_linop(), rhs,
                           atol=pcg_tol, maxiter=10000,
                           callback=counter)
        return dy, counter.n
    else:
        Om = omega.to_dense()
        try:
            dy = spla.splu(sp.csc_matrix(Om)).solve(rhs)
        except Exception:
            dy = np.linalg.solve(Om.toarray() if sp.issparse(Om) else Om, rhs)
        return dy, 0


# ======================================================================
# KKT Residual
# ======================================================================

def kkt_residual(
    lin_term: np.ndarray,
    b: np.ndarray,
    B_meas: sp.spmatrix,
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
    B_proc: Optional[sp.spmatrix] = None,
    M_proc=None,
    A: Optional[sp.spmatrix] = None,
    a: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute KKT residual F for the barrier formulation.

    Returns
    -------
    F : (L + K + 2P + N,) residual vector [r1; r2; r3; r4; r5].
    """
    m = B_meas.shape[0]
    has_proc = B_proc is not None
    has_con = A is not None

    d = c - C.T @ u

    # Block 1: complementarity  d * q - mu
    r1 = d * q - mu

    # Evaluate M @ u  (possibly block-diagonal with process)
    um = u[:m] if has_proc else u
    if callable(M):
        Mu_m, _ = M(um)
    else:
        Mu_m = M @ um

    if has_proc:
        up = u[m:]
        Mu = np.concatenate([Mu_m, M_proc @ up])
        By = np.concatenate([B_meas @ y, B_proc @ y])
    else:
        Mu = Mu_m
        By = B_meas @ y

    # Block 2: stationarity in u
    r2 = By - Mu - C @ q + b

    # Blocks 3, 4: constraint complementarity and feasibility
    if has_con:
        r3 = w * r - mu
        r4 = r + A.T @ y - a
        Aw = A @ w
    else:
        r3 = np.array([])
        r4 = np.array([])
        Aw = 0.0

    # Block 5: stationarity in y
    if has_proc:
        r5 = B_meas.T @ um + B_proc.T @ up + Aw + lin_term
    else:
        r5 = B_meas.T @ um + Aw + lin_term

    return np.concatenate([r1, r2, r3, r4, r5])


# ======================================================================
# KKT Solve
# ======================================================================

@dataclass
class KKTResult:
    """Newton direction from one KKT solve."""
    dq: np.ndarray
    du: np.ndarray
    dr: Optional[np.ndarray]
    dw: Optional[np.ndarray]
    dy: np.ndarray
    cg_iters: int = 0


def kkt_solve(
    lin_term: np.ndarray,
    b: np.ndarray,
    B_meas: sp.spmatrix,
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
    B_proc: Optional[sp.spmatrix] = None,
    M_proc=None,
    A: Optional[sp.spmatrix] = None,
    a: Optional[np.ndarray] = None,
    rho: float = 0.0,
    delta: float = 0.0,
    inexact: bool = False,
    pcg_tol: float = 1e-8,
) -> KKTResult:
    """Solve the Newton system by Schur-complement elimination.

    Strategy
    --------
    1. Eliminate dq  =>  dq = (r1 + q * C' du) / d.
    2. Form  T = MM + C (Q/D) C'  and eliminate du.
    3. Build SchurOperator Omega = B' T^{-1} B  (+ constraints + reg).
    4. Solve (N x N) system for dy  (direct LU or CG via SchurOperator).
    5. Back-substitute for du, dq, dw, dr.
    """
    m = B_meas.shape[0]
    n = y.shape[0]
    has_proc = B_proc is not None
    has_con = A is not None

    # -- Diagonal quantities --
    d = c - C.T @ u
    QD = sp.diags(q / d, format="csc")

    if has_con:
        WR = sp.diags(w / r, format="csc")
    else:
        WR = None

    # -- Hessian MM (block-diagonal, with optional regularisation) --
    um = u[:m] if has_proc else u
    if callable(M):
        Mu_m, MM_m = M(um)
    else:
        MM_m = M
        Mu_m = MM_m @ um

    if has_proc:
        up = u[m:]
        MM_p = M_proc
        Mu = np.concatenate([Mu_m, MM_p @ up])
        if rho > 0:
            MM_m = MM_m + rho * sp.eye(MM_m.shape[0], format="csc")
            MM_p = MM_p + rho * sp.eye(MM_p.shape[0], format="csc")
        MM = sp.block_diag([MM_m, MM_p], format="csc")
    else:
        Mu = Mu_m
        if rho > 0:
            MM = MM_m + rho * sp.eye(MM_m.shape[0], format="csc")
        else:
            MM = MM_m

    # -- T = MM + C QD C' --
    T = sp.csc_matrix(MM + C @ QD @ C.T)

    # Factor T (full or blocks)
    if has_proc:
        T_solve_m = _sparse_factor(T[:m, :m])
        T_solve_p = _sparse_factor(T[m:, m:])
        T_solve = _sparse_factor(T)           # for back-substitution
    else:
        T_solve = _sparse_factor(T)
        T_solve_m = T_solve
        T_solve_p = None

    # -- RHS construction --
    rhs1 = -d * q + mu

    if has_proc:
        By = np.concatenate([B_meas @ y, B_proc @ y])
    else:
        By = B_meas @ y

    rhs2 = -(By - Mu + b) + mu * (C @ (1.0 / d))

    if has_con:
        rhs3 = -r * w + mu
        rhs4 = -A.T @ y + a - mu / w
        Awr = A @ (w - WR @ rhs4)
    else:
        Awr = 0.0

    u_tilde = u - T_solve(rhs2)

    if has_proc:
        rhs5 = -B_meas.T @ u_tilde[:m] - B_proc.T @ u_tilde[m:] - Awr - lin_term
    else:
        rhs5 = -B_meas.T @ u_tilde - Awr - lin_term

    # -- Solve for dy via SchurOperator --
    omega = SchurOperator(
        B_meas, T_solve_m,
        B_proc=B_proc, T_solve_p=T_solve_p,
        WR=WR, A=A, delta=delta,
    )
    dy, cg_iters = _solve_schur(omega, rhs5, inexact, pcg_tol)

    # -- Back-substitution --
    if has_con:
        dw = -WR @ (rhs4 - A.T @ dy)
        dr = (rhs3 - r * dw) / w
    else:
        dw = None
        dr = None

    if has_proc:
        du = T_solve(-rhs2 + np.concatenate([B_meas @ dy, B_proc @ dy]))
    else:
        du = T_solve(-rhs2 + B_meas @ dy)

    dq = (rhs1 + q * (C.T @ du)) / d

    return KKTResult(dq=dq, du=du, dr=dr, dw=dw, dy=dy, cg_iters=cg_iters)
