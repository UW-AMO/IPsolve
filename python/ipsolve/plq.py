"""
PLQ dataclass – core representation for piecewise linear-quadratic penalties.

A PLQ penalty ρ is represented **via its conjugate** in the form used by the
interior-point solver::

    ρ(v) = max_u  uᵀv − ½ uᵀMu   s.t.  Cᵀu ≤ c

After composing with a linear model v = Bx − b the solver works with:

    (M, B, C, c, b)

where
  M : (K, K) matrix or callable  – quadratic term of conjugate
  B : (K, N) matrix              – linear mapping from x to v
  C : (K, L) matrix              – constraint normals (stored as K×L, i.e. Cᵀ acts on u)
  c : (L,)   vector              – constraint RHS
  b : (K,)   vector              – constant offset (after composing with linear model)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
import scipy.sparse as sp

# Type alias: M can be a matrix or a function handle returning (grad, Hess)
MatrixOrCallable = Union[np.ndarray, sp.spmatrix, Callable]


@dataclass
class PLQ:
    """Representation of a PLQ penalty in conjugate form.

    Attributes
    ----------
    M : matrix or callable
        If matrix: quadratic part of the conjugate, ½ uᵀMu.
        If callable: ``grad, Hess = M(u)`` returns gradient and Hessian of
        the smooth part of the conjugate evaluated at *u*.
    B : sparse or dense (K, N) matrix
        Maps decision variable *x* into the penalty argument: v = Bx − b.
    C : sparse or dense (K, L) matrix
        Constraint normals for the dual variable *u*.
        The constraints are  Cᵀu ≤ c  (i.e. C.T @ u ≤ c).
    c : (L,) array
        Right-hand side of the dual constraints.
    b : (K,) array
        Offset (typically −b − B·z after composition with linear model).
    obj : callable or None
        Optional function to evaluate the primal penalty value ρ(v).
    """

    M: MatrixOrCallable
    B: Union[np.ndarray, sp.spmatrix]
    C: Union[np.ndarray, sp.spmatrix]
    c: np.ndarray
    b: np.ndarray
    obj: Optional[Callable] = None

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def K(self) -> int:
        """Dimension of dual variable u."""
        if sp.issparse(self.B):
            return self.B.shape[0]
        return np.atleast_2d(self.B).shape[0]

    @property
    def L(self) -> int:
        """Number of dual constraints."""
        return self.c.shape[0]

    @property
    def is_func(self) -> bool:
        """True when M is a callable (function handle) rather than a matrix."""
        return callable(self.M)

    # ------------------------------------------------------------------
    # Evaluate the M-part
    # ------------------------------------------------------------------
    def eval_M(self, u: np.ndarray):
        """Return (Mu, MM) – gradient and Hessian-like quantities.

        If M is a matrix:  Mu = M @ u,  MM = M.
        If M is callable:  Mu, MM = M(u).
        """
        if self.is_func:
            return self.M(u)
        else:
            Mu = self.M @ u
            return Mu, self.M


def stack(plq1: PLQ, plq2: PLQ) -> PLQ:
    """Stack two PLQ penalties (e.g. measurement + process).

    Returns a combined PLQ where all matrices are block-diagonal / stacked.
    Note: B matrices are *not* combined here – that is handled by the solver
    which keeps B_meas and B_proc separate for efficiency.
    """
    b_out = np.concatenate([plq1.b, plq2.b])
    c_out = np.concatenate([plq1.c, plq2.c])
    C_out = sp.block_diag([plq1.C, plq2.C], format="csc")
    # B is stacked vertically
    B_out = sp.vstack([plq1.B, plq2.B], format="csc")

    # M: block-diagonal (only works for matrix M)
    if plq1.is_func or plq2.is_func:
        raise ValueError("Cannot stack PLQs with callable M – use the solver's pFlag path.")
    M_out = sp.block_diag([plq1.M, plq2.M], format="csc")

    def obj_out(v):
        k1 = plq1.K
        v1, v2 = v[:k1], v[k1:]
        val = 0.0
        if plq1.obj is not None:
            val += plq1.obj(v1)
        if plq2.obj is not None:
            val += plq2.obj(v2)
        return val

    return PLQ(M=M_out, B=B_out, C=C_out, c=c_out, b=b_out, obj=obj_out)


def ensure_sparse(X):
    """Convert to sparse CSC if not already sparse."""
    if sp.issparse(X):
        return sp.csc_matrix(X)
    return sp.csc_matrix(np.atleast_2d(X))
