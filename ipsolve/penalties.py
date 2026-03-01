"""
Penalty library -- PLQ objects for common penalties.

Basic usage::

    from ipsolve import solve, huber, l1, l2

    x = solve(H, z, meas=huber(kappa=1.0), proc=l1(lam=0.5))

Each factory can be called two ways:

- **Without dimension** (for use with ``solve()``): returns a :class:`Penalty`
  spec that gets materialized when the dimension is known.
- **With dimension** (advanced): returns a built :class:`PLQ` object directly.

Supported penalties
-------------------
===========  ============================================================
Name         Formula
===========  ============================================================
``l1``       lam * ||v||_1
``l2``       (lam/2) ||v||^2
``huber``    Huber loss with threshold kappa
``hinge``    lam * sum pos(v)
``vapnik``   lam * sum max(|v| - eps, 0)
``qreg``     Quantile / pinball loss
``qhuber``   Quantile Huber (smooth quantile)
``infnorm``  lam * ||v||_inf
``logreg``   sum log(1 + exp(v))
``hybrid``   sum(sqrt(1 + v^2/s) - 1)
===========  ============================================================
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from ipsolve.plq import PLQ


# ---------------------------------------------------------------------------
# Penalty specification (lazy -- materialized by solve())
# ---------------------------------------------------------------------------

@dataclass
class Penalty:
    """Lazy penalty spec -- materialized into a PLQ when the dimension is known.

    Created by calling a penalty factory without a dimension::

        >>> huber(kappa=1.0)
        huber(kappa=1.0)
        >>> l1()
        l1()
    """

    name: str
    kwargs: dict = field(default_factory=dict)

    def build(self, m: int) -> PLQ:
        """Build the PLQ object for dimension *m*."""
        return load_penalty(self.name, m, **self.kwargs)

    def __repr__(self):
        parts = ["{0}={1!r}".format(k, v) for k, v in self.kwargs.items()]
        return "{0}({1})".format(self.name, ", ".join(parts))


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

def l1(m=None, *, lam=1.0):
    """l1 norm:  rho(v) = lam * ||v||_1.

    Parameters
    ----------
    m : int or None
        Dimension.  Omit to get a :class:`Penalty` spec for ``solve()``.
    lam : float
        Regularization weight (default 1).
    """
    if m is None:
        kw = {} if lam == 1.0 else {"lam": lam}
        return Penalty("l1", kw)
    return PLQ(
        M=_zeros(m), B=_eye(m),
        C=sp.vstack([_eye(m), -_eye(m)], format="csc"),
        c=lam * np.ones(2 * m), b=np.zeros(m),
        obj=lambda v, _l=lam: _l * np.sum(np.abs(v)),
    )


def l2(m=None, *, lam=None, mMult=1.0):
    """Squared l2:  rho(v) = (lam/2) ||v||^2.

    Parameters
    ----------
    m : int or None
        Dimension.  Omit to get a :class:`Penalty` spec for ``solve()``.
    lam : float or None
        Regularization weight.  Internally sets ``mMult = 1/lam``.
    mMult : float
        Direct scaling: ``rho(v) = ||v||^2 / (2 * mMult)``.
        Ignored when *lam* is given.

    Examples
    --------
    >>> l2(lam=0.01)          # (0.01/2)||v||^2
    l2(lam=0.01)
    >>> l2()                  # default: 0.5||v||^2
    l2()
    """
    if lam is not None:
        mMult = 1.0 / lam
    if m is None:
        kw = {}
        if lam is not None:
            kw["lam"] = lam
        elif mMult != 1.0:
            kw["mMult"] = mMult
        return Penalty("l2", kw)
    return PLQ(
        M=mMult * _eye(m), B=_eye(m),
        C=sp.csc_matrix((1, m)), c=np.array([1.0]), b=np.zeros(m),
        obj=lambda v, _mm=mMult: 0.5 * np.dot(v, v) / _mm,
    )


def huber(m=None, *, kappa=1.0, mMult=1.0):
    """Huber loss with threshold *kappa*.

    ::

        rho(v) = |v| - kappa/2      if |v| > kappa
               = v^2 / (2*kappa)    if |v| <= kappa

    Parameters
    ----------
    m : int or None
    kappa : float
        Huber threshold.
    """
    if m is None:
        kw = {}
        if kappa != 1.0:
            kw["kappa"] = kappa
        if mMult != 1.0:
            kw["mMult"] = mMult
        return Penalty("huber", kw)

    threshold = mMult * kappa

    def obj(v, _t=threshold):
        return np.sum(np.where(
            np.abs(v) > _t, np.abs(v) - 0.5 * _t, 0.5 * v ** 2 / _t,
        ))

    return PLQ(
        M=mMult * kappa * _eye(m), B=_eye(m),
        C=sp.vstack([_eye(m), -_eye(m)], format="csc"),
        c=np.ones(2 * m), b=np.zeros(m), obj=obj,
    )


def hinge(m=None, *, lam=1.0):
    """Hinge loss:  rho(v) = lam * sum pos(v).

    Used for SVM classification.

    Parameters
    ----------
    m : int or None
    lam : float
        Weight (default 1).
    """
    if m is None:
        kw = {} if lam == 1.0 else {"lam": lam}
        return Penalty("hinge", kw)
    return PLQ(
        M=_zeros(m), B=-_eye(m),
        C=sp.vstack([_eye(m), -_eye(m)], format="csc"),
        c=np.concatenate([lam * np.ones(m), np.zeros(m)]),
        b=np.zeros(m),
        obj=lambda v, _l=lam: _l * np.sum(pos(v)),
    )


def vapnik(m=None, *, lam=1.0, eps=0.2):
    """Vapnik epsilon-insensitive loss:  lam * sum max(|v| - eps, 0).

    Parameters
    ----------
    m : int or None
    lam : float
    eps : float
        Insensitivity zone width.
    """
    if m is None:
        kw = {}
        if lam != 1.0:
            kw["lam"] = lam
        if eps != 0.2:
            kw["eps"] = eps
        return Penalty("vapnik", kw)
    K = 2 * m
    return PLQ(
        M=_zeros(K),
        B=sp.vstack([_eye(m), -_eye(m)], format="csc"),
        C=sp.vstack([sp.eye(K), -sp.eye(K)], format="csc"),
        c=np.concatenate([lam * np.ones(K), np.zeros(K)]),
        b=-eps * np.ones(K),
        obj=lambda v, _l=lam, _e=eps: _l * np.sum(pos(np.abs(v) - _e)),
    )


def qreg(m=None, *, lam=1.0, tau=0.5):
    """Quantile (check / pinball) loss.

    Parameters
    ----------
    m : int or None
    lam : float
    tau : float
        Quantile level in (0, 1).
    """
    if m is None:
        kw = {}
        if lam != 1.0:
            kw["lam"] = lam
        if tau != 0.5:
            kw["tau"] = tau
        return Penalty("qreg", kw)
    return PLQ(
        M=_zeros(m), B=_eye(m),
        C=sp.vstack([_eye(m), -_eye(m)], format="csc"),
        c=lam * np.concatenate([(1 - tau) * np.ones(m), tau * np.ones(m)]),
        b=np.zeros(m),
        obj=lambda v, _l=lam, _t=tau: _l * (
            _t * np.sum(pos(v)) + (1 - _t) * np.sum(pos(-v))
        ),
    )


def qhuber(m=None, *, kappa=1.0, tau=0.5):
    """Quantile Huber (smoothed quantile regression).

    Parameters
    ----------
    m : int or None
    kappa : float
    tau : float
        Quantile level in (0, 1).
    """
    if m is None:
        kw = {}
        if kappa != 1.0:
            kw["kappa"] = kappa
        if tau != 0.5:
            kw["tau"] = tau
        return Penalty("qhuber", kw)

    def obj(v, _k=kappa, _t=tau):
        left = v <= -2 * _t * _k
        right = v >= 2 * (1 - _t) * _k
        mid = ~left & ~right
        return np.sum(
            left * (-2 * _t * v - _k * _t ** 2)
            + mid * (v ** 2 / (2 * _k))
            + right * (2 * (1 - _t) * v - _k * (1 - _t) ** 2)
        )

    return PLQ(
        M=kappa * _eye(m), B=_eye(m),
        C=0.5 * sp.vstack([_eye(m), -_eye(m)], format="csc"),
        c=np.concatenate([(1 - tau) * np.ones(m), tau * np.ones(m)]),
        b=np.zeros(m), obj=obj,
    )


def infnorm(m=None, *, lam=1.0):
    """l_inf norm:  rho(v) = lam * ||v||_inf.

    Parameters
    ----------
    m : int or None
    lam : float
    """
    if m is None:
        kw = {} if lam == 1.0 else {"lam": lam}
        return Penalty("infnorm", kw)
    K = 2 * m
    top_row = sp.csc_matrix(np.ones((1, K)))
    neg_eye = -sp.eye(K, format="csc")
    return PLQ(
        M=_zeros(K),
        B=sp.vstack([_eye(m), -_eye(m)], format="csc"),
        C=sp.vstack([top_row, neg_eye], format="csc"),
        c=np.concatenate([[lam], np.zeros(K)]),
        b=np.zeros(K),
        obj=lambda v, _l=lam: _l * np.max(np.abs(v)),
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


def logreg(m=None, *, alpha=0.5):
    """Logistic regression loss:  rho(v) = sum log(1 + exp(v)).

    Parameters
    ----------
    m : int or None
    alpha : float
        Constraint margin.
    """
    if m is None:
        kw = {} if alpha == 0.5 else {"alpha": alpha}
        return Penalty("logreg", kw)
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


def hybrid(m=None, *, scale=1.0):
    """Hybrid penalty:  rho(v) = sum(sqrt(1 + v^2/s) - 1).

    Parameters
    ----------
    m : int or None
    scale : float
    """
    if m is None:
        kw = {} if scale == 1.0 else {"scale": scale}
        return Penalty("hybrid", kw)
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
    """Load a named PLQ penalty of dimension *m*.

    Parameters
    ----------
    name : registered penalty name.
    m : dimension.
    **kwargs : penalty-specific parameters (lam, kappa, tau, ...).
    """
    if name not in PENALTY_REGISTRY:
        raise ValueError(
            "Unknown penalty '{0}'. Choose from: {1}".format(
                name, list(PENALTY_REGISTRY.keys())
            )
        )
    factory = PENALTY_REGISTRY[name]
    sig = inspect.signature(factory)
    valid = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in valid}
    return factory(m, **filtered)
