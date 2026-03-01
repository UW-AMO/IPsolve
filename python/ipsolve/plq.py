"""
PLQ dataclass -- core representation for piecewise linear-quadratic penalties.

A PLQ penalty rho is represented **via its conjugate** in the form::

    rho(v) = max_u  u'v - 0.5 u'Mu   s.t.  C'u <= c

After composing with a linear model v = Bx - b the solver works with
(M, B, C, c, b) where:

  M : (K, K) matrix or callable  -- quadratic term of conjugate
  B : (K, N) matrix              -- linear mapping from x to v
  C : (K, L) matrix              -- constraint normals (C' acts on u)
  c : (L,)   vector              -- constraint RHS
  b : (K,)   vector              -- constant offset
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
import scipy.sparse as sp

MatrixOrCallable = Union[np.ndarray, sp.spmatrix, Callable]


@dataclass
class PLQ:
    """Representation of a PLQ penalty in conjugate form.

    Attributes
    ----------
    M : matrix or callable
        If matrix: quadratic part of the conjugate, 0.5 u'Mu.
        If callable: grad, Hess = M(u).
    B : (K, N) sparse or dense matrix
        Maps x into penalty argument: v = Bx - b.
    C : (K, L) sparse or dense matrix
        Constraint normals: C'u <= c.
    c : (L,) array -- constraint RHS.
    b : (K,) array -- offset vector.
    obj : callable or None -- primal penalty evaluation for logging.
    """

    M: MatrixOrCallable
    B: Union[np.ndarray, sp.spmatrix]
    C: Union[np.ndarray, sp.spmatrix]
    c: np.ndarray
    b: np.ndarray
    obj: Optional[Callable] = None

    @property
    def K(self) -> int:
        """Dimension of dual variable u."""
        return _nrows(self.B)

    @property
    def L(self) -> int:
        """Number of dual constraints."""
        return self.c.shape[0]

    @property
    def is_func(self) -> bool:
        """True when M is a callable rather than a matrix."""
        return callable(self.M)

    def eval_M(self, u: np.ndarray):
        """Return (Mu, MM) -- gradient and Hessian-like quantities."""
        if self.is_func:
            return self.M(u)
        return self.M @ u, self.M

    def __repr__(self) -> str:
        kind = "func" if self.is_func else f"{self.K}x{self.K}"
        return f"PLQ(K={self.K}, L={self.L}, M={kind})"


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _nrows(X) -> int:
    """Number of rows for sparse or dense matrix."""
    if sp.issparse(X):
        return X.shape[0]
    return np.atleast_2d(X).shape[0]


def ensure_sparse(X):
    """Convert to sparse CSC if not already sparse."""
    if sp.issparse(X):
        return sp.csc_matrix(X)
    return sp.csc_matrix(np.atleast_2d(X))


def compose(plq: PLQ, H, z: np.ndarray) -> PLQ:
    """Compose PLQ penalty with linear model: v = B(Hx) - (b + Bz).

    Transforms:  b <- -b - Bz;  B <- BH.
    """
    B_new = ensure_sparse(plq.B @ H)
    b_new = -plq.b - plq.B @ z
    return PLQ(M=plq.M, B=B_new, C=plq.C, c=plq.c, b=b_new, obj=plq.obj)
