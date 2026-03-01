"""
Penalty library -- build PLQ objects for common penalties.

Each factory function returns a :class:`PLQ` object *before* composition
with a linear model.  The caller (typically :func:`ipsolve.api.solve`)
handles the composition  ``b <- -b - Bz;  B <- BH``.

Supported penalties
-------------------
===========  =============================================================
Name         Description
===========  =============================================================
``l1``       l1 norm
``l2``       Squared l2
``huber``    Huber loss with threshold kappa
``hinge``    Hinge loss (SVM)
``vapnik``   Vapnik epsilon-insensitive loss
``qreg``     Quantile (check / pinball) loss
``qhuber``   Quantile Huber (smooth quantile)
``infnorm``  l_inf norm (via simplex constraint on dual)
``logreg``   Logistic regression (smooth conjugate via function handle)
``hybrid``   Hybrid / robust loss sqrt(1 + v^2/s) - 1
===========  =============================================================
"""

from __future__ import annotations

import inspect

import numpy as np
import scipy.sparse as sp

from ipsolve.plq import PLQ


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _eye(m):
    return sp.eye(m, format="csc")


def _zeros(m):
    return sp.csc_matrix((m, m))


def pos(x):
    """Element-wise positive part."""
    return np.maximum(x, 0.0)


# ---------------------------------------------------------------------------
# Penalty factories
# ---------------------------------------------------------------------------

def l1(m: int, lam: float = 1.0) -> PLQ:
    """l1 penalty: rho(v) = lam * ||v||_1."""
    return PLQ(
        M=_zeros(m), B=_eye(m),
        C=sp.vstack([_eye(m), -_eye(m)], format="csc"),
        c=lam * np.ones(2 * m), b=np.zeros(m),
        obj=lambda v: lam * np.sum(np.abs(v)),
    )


def l2(m: int, mMult: float = 1.0) -> PLQ:
    """Squared l2 penalty: rho(v) = ||v||^2 / (2*mMult)."""
    return PLQ(
        M=mMult * _eye(m), B=_eye(m),
        C=sp.csc_matrix((1, m)), c=np.array([1.0]), b=np.zeros(m),
        obj=lambda v: 0.5 * np.dot(v, v) / mMult,
    )


def huber(m: int, kappa: float = 1.0, mMult: float = 1.0) -> PLQ:
    """Huber loss with threshold kappa*mMult."""
    threshold = mMult * kappa

    def obj(v):
        return np.sum(np.where(
            np.abs(v) > threshold,
            np.abs(v) - 0.5 * threshold,
            0.5 * v ** 2 / threshold,
        ))

    return PLQ(
        M=mMult * kappa * _eye(m), B=_eye(m),
        C=sp.vstack([_eye(m), -_eye(m)], format="csc"),
        c=np.ones(2 * m), b=np.zeros(m), obj=obj,
    )


def hinge(m: int, lam: float = 1.0) -> PLQ:
    """Hinge loss: rho(v) = lam * sum(pos(v)).

    B = -I so after composition the solver evaluates at z - Hx.
    """
    return PLQ(
        M=_zeros(m), B=-_eye(m),
        C=sp.vstack([_eye(m), -_eye(m)], format="csc"),
        c=np.concatenate([lam * np.ones(m), np.zeros(m)]),
        b=np.zeros(m),
        obj=lambda v: lam * np.sum(pos(v)),
    )


def vapnik(m: int, lam: float = 1.0, eps: float = 0.2) -> PLQ:
    """Vapnik epsilon-insensitive loss: lam*max(|v| - eps, 0)."""
    K = 2 * m
    return PLQ(
        M=_zeros(K),
        B=sp.vstack([_eye(m), -_eye(m)], format="csc"),
        C=sp.vstack([sp.eye(K), -sp.eye(K)], format="csc"),
        c=np.concatenate([lam * np.ones(K), np.zeros(K)]),
        b=-eps * np.ones(K),
        obj=lambda v: lam * np.sum(pos(np.abs(v) - eps)),
    )


def qreg(m: int, lam: float = 1.0, tau: float = 0.5) -> PLQ:
    """Quantile (check / pinball) loss."""
    return PLQ(
        M=_zeros(m), B=_eye(m),
        C=sp.vstack([_eye(m), -_eye(m)], format="csc"),
        c=lam * np.concatenate([(1 - tau) * np.ones(m), tau * np.ones(m)]),
        b=np.zeros(m),
        obj=lambda v: lam * (tau * np.sum(pos(v)) + (1 - tau) * np.sum(pos(-v))),
    )


def qhuber(m: int, kappa: float = 1.0, tau: float = 0.5) -> PLQ:
    """Quantile Huber (smoothed quantile regression)."""
    def obj(v):
        left = v <= -2 * tau * kappa
        right = v >= 2 * (1 - tau) * kappa
        mid = ~left & ~right
        return np.sum(
            left * (-2 * tau * v - kappa * tau ** 2)
            + mid * (v ** 2 / (2 * kappa))
            + right * (2 * (1 - tau) * v - kappa * (1 - tau) ** 2)
        )

    return PLQ(
        M=kappa * _eye(m), B=_eye(m),
        C=0.5 * sp.vstack([_eye(m), -_eye(m)], format="csc"),
        c=np.concatenate([(1 - tau) * np.ones(m), tau * np.ones(m)]),
        b=np.zeros(m), obj=obj,
    )


def infnorm(m: int, lam: float = 1.0) -> PLQ:
    """l_inf norm: rho(v) = lam * ||v||_inf."""
    K = 2 * m
    top_row = sp.csc_matrix(np.ones((1, K)))
    neg_eye = -sp.eye(K, format="csc")
    return PLQ(
        M=_zeros(K),
        B=sp.vstack([_eye(m), -_eye(m)], format="csc"),
        C=sp.vstack([top_row, neg_eye], format="csc"),
        c=np.concatenate([[lam], np.zeros(K)]),
        b=np.zeros(K),
        obj=lambda v: lam * np.max(np.abs(v)),
    )


# ---------------------------------------------------------------------------
# Smooth conjugates via function handles
# ---------------------------------------------------------------------------

def _logreg_conjugate(u, alpha=0.5):
    """Logistic regression conjugate (negative entropy)."""
    u = np.clip(u, alpha + 1e-12, 1.0 - alpha - 1e-12)
    grad = np.log(u) - np.log(1.0 - u)
    hess = sp.diags(1.0 / (u * (1.0 - u)), format="csc")
    return grad, hess


def logreg(m: int, alpha: float = 0.5) -> PLQ:
    """Logistic regression loss: rho(v) = sum(log(1 + exp(v)))."""
    return PLQ(
        M=lambda u: _logreg_conjugate(u, alpha),
        B=-_eye(m),
        C=sp.vstack([_eye(m), -_eye(m)], format="csc"),
        c=np.concatenate([np.ones(m), alpha * np.ones(m)]),
        b=np.zeros(m),
        obj=lambda v: np.sum(np.log(1.0 + np.exp(v))),
    )


def _hybrid_conjugate(u, scale=1.0):
    """Hybrid penalty conjugate."""
    m = scale ** 2
    t = (u / m) ** 2
    t = np.clip(t, 0, 1.0 - 1e-12)
    sq = np.sqrt(1.0 - t)
    grad = (u / m) / (m * sq)
    hess = sp.diags(1.0 / (m ** 2 * (1.0 - t) ** 1.5), format="csc")
    return grad, hess


def hybrid(m: int, scale: float = 1.0) -> PLQ:
    """Hybrid penalty: rho(v) = sum(sqrt(1 + v^2/s) - 1)."""
    s2 = scale ** 2
    return PLQ(
        M=lambda u: _hybrid_conjugate(u, scale),
        B=_eye(m),
        C=sp.vstack([_eye(m), -_eye(m)], format="csc"),
        c=s2 * np.ones(2 * m),
        b=np.zeros(m),
        obj=lambda v: np.sum(np.sqrt(1.0 + v ** 2 / scale) - 1.0),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PENALTY_REGISTRY = {
    "l1": l1, "l2": l2, "huber": huber, "hinge": hinge,
    "vapnik": vapnik, "qreg": qreg, "qhuber": qhuber,
    "infnorm": infnorm, "logreg": logreg, "hybrid": hybrid,
}


def load_penalty(name: str, m: int, **kwargs) -> PLQ:
    """Load a named PLQ penalty.

    Parameters
    ----------
    name : str -- registered penalty name (see PENALTY_REGISTRY).
    m : int -- dimension of the penalty argument.
    **kwargs -- penalty-specific parameters (lam, kappa, tau, etc.).
    """
    if name not in PENALTY_REGISTRY:
        raise ValueError(
            f"Unknown penalty \'{name}\'. "
            f"Choose from: {list(PENALTY_REGISTRY.keys())}"
        )
    factory = PENALTY_REGISTRY[name]
    sig = inspect.signature(factory)
    valid = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in valid}
    return factory(m, **filtered)
