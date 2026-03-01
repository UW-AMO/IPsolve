"""
Penalty library – build PLQ objects for common penalties.

Each factory function returns a :class:`PLQ` object *before* composition
with a linear model.  The caller (typically :func:`ipsolve.api.solve`)
handles the composition  ``b ← −b − B·z;  B ← B·H``.

Supported penalties
-------------------
===========  =============================================================
Name         Description
===========  =============================================================
``l1``       ℓ₁ norm, λ‖v‖₁
``l2``       Squared ℓ₂, (2σ)⁻¹‖v‖²
``huber``    Huber loss with threshold κ
``hinge``    Hinge loss λ·pos(−v)  (SVM)
``vapnik``   Vapnik ε-insensitive loss
``qreg``     Quantile (check / pinball) loss
``qhuber``   Quantile Huber (smooth quantile)
``infnorm``  ℓ∞ norm (via simplex constraint on dual)
``logreg``   Logistic regression (smooth conjugate via function handle)
``hybrid``   Hybrid / robust loss √(1 + v²/s) − 1
===========  =============================================================
"""

from __future__ import annotations

from typing import Optional

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
    r"""ℓ₁ penalty:  ρ(v) = λ ‖v‖₁.

    Conjugate domain: u ∈ [−λ, λ]ᵐ, so C = [I; −I], c = λ·1.
    """
    M = _zeros(m)
    C = sp.vstack([_eye(m), -_eye(m)], format="csc")
    c = lam * np.ones(2 * m)
    B = _eye(m)
    b = np.zeros(m)
    obj = lambda v: lam * np.sum(np.abs(v))  # noqa: E731
    return PLQ(M=M, B=B, C=C, c=c, b=b, obj=obj)


def l2(m: int, mMult: float = 1.0) -> PLQ:
    r"""Squared ℓ₂ penalty:  ρ(v) = ‖v‖² / (2·mMult).

    Conjugate: ½ uᵀ(mMult·I)u,  unconstrained ⇒ trivial constraint C=0, c=1.
    """
    M = mMult * _eye(m)
    # Trivial constraint: 0ᵀu ≤ 1  (always satisfied)
    C = sp.csc_matrix((1, m))  # (L=1, K=m) – one row of zeros
    c = np.array([1.0])
    B = _eye(m)
    b = np.zeros(m)
    obj = lambda v: 0.5 * np.dot(v, v) / mMult  # noqa: E731
    return PLQ(M=M, B=B, C=C, c=c, b=b, obj=obj)


def huber(m: int, kappa: float = 1.0, mMult: float = 1.0) -> PLQ:
    r"""Huber loss with threshold κ·mMult.

    Conjugate: M = κ·mMult·I, C = [I; −I], c = 1.
    """
    M = mMult * kappa * _eye(m)
    C = sp.vstack([_eye(m), -_eye(m)], format="csc")
    c = np.ones(2 * m)
    B = _eye(m)
    b = np.zeros(m)
    threshold = mMult * kappa

    def obj(v):
        return np.sum(
            np.where(
                np.abs(v) > threshold,
                np.abs(v) - 0.5 * threshold,
                0.5 * v ** 2 / threshold,
            )
        )

    return PLQ(M=M, B=B, C=C, c=c, b=b, obj=obj)


def hinge(m: int, lam: float = 1.0) -> PLQ:
    r"""Hinge loss:  ρ(v) = λ · Σ pos(vᵢ).

    Used for SVM:  ρ(1 − yAx) = λ·Σ pos(1 − yAx).
    B = −I so that after composition the solver evaluates at z − Hx.
    Conjugate domain: u ∈ [0, λ]ᵐ, so C = [I; −I], c = [λ; 0].
    """
    M = _zeros(m)
    C = sp.vstack([_eye(m), -_eye(m)], format="csc")
    c = np.concatenate([lam * np.ones(m), np.zeros(m)])
    B = -_eye(m)  # negative so penalty evaluates at z − Hx
    b = np.zeros(m)
    obj = lambda v: lam * np.sum(pos(v))  # noqa: E731
    return PLQ(M=M, B=B, C=C, c=c, b=b, obj=obj)


def vapnik(m: int, lam: float = 1.0, eps: float = 0.2) -> PLQ:
    r"""Vapnik ε-insensitive loss: λ·max(|v| − ε, 0).

    Conjugate: M=0, C=[I;−I], c=[λ;0]  (same as hinge on [v;−v]),
    with offset b = ε·1.
    """
    M = _zeros(2 * m)
    C = sp.vstack([sp.eye(2 * m), -sp.eye(2 * m)], format="csc")
    c = np.concatenate([lam * np.ones(2 * m), np.zeros(2 * m)])
    B = sp.vstack([_eye(m), -_eye(m)], format="csc")
    b = -eps * np.ones(2 * m)
    obj = lambda v: lam * np.sum(pos(np.abs(v) - eps))  # noqa: E731
    return PLQ(M=M, B=B, C=C, c=c, b=b, obj=obj)


def qreg(m: int, lam: float = 1.0, tau: float = 0.5) -> PLQ:
    r"""Quantile (check / pinball) loss:  λ·[τ·pos(−v) + (1−τ)·pos(v)].

    Conjugate domain: u ∈ [−λτ, λ(1−τ)]ᵐ.
    """
    M = _zeros(m)
    C = sp.vstack([_eye(m), -_eye(m)], format="csc")
    c = lam * np.concatenate([(1 - tau) * np.ones(m), tau * np.ones(m)])
    B = _eye(m)
    b = np.zeros(m)
    obj = lambda v: lam * (tau * np.sum(pos(v)) + (1 - tau) * np.sum(pos(-v)))  # noqa: E731
    return PLQ(M=M, B=B, C=C, c=c, b=b, obj=obj)


def qhuber(m: int, kappa: float = 1.0, tau: float = 0.5) -> PLQ:
    r"""Quantile Huber (smoothed quantile regression).

    Conjugate: M = κI, C = ½[I; −I], c = [(1−τ); τ].
    """
    M = kappa * _eye(m)
    C = 0.5 * sp.vstack([_eye(m), -_eye(m)], format="csc")
    c = np.concatenate([(1 - tau) * np.ones(m), tau * np.ones(m)])
    B = _eye(m)
    b = np.zeros(m)

    def obj(v):
        # Quantile Huber evaluation
        left = v <= -2 * tau * kappa
        right = v >= 2 * (1 - tau) * kappa
        mid = ~left & ~right
        return np.sum(
            left * (-2 * tau * v - kappa * tau ** 2)
            + mid * (v ** 2 / (2 * kappa))  # should actually be v**2 / kappa with the 0.5 factor
            + right * (2 * (1 - tau) * v - kappa * (1 - tau) ** 2)
        )

    return PLQ(M=M, B=B, C=C, c=c, b=b, obj=obj)


def infnorm(m: int, lam: float = 1.0) -> PLQ:
    r"""ℓ∞ norm:  ρ(v) = λ ‖v‖∞.

    Dual: max uᵀv s.t. ‖u‖₁ ≤ λ, i.e. u split as (u⁺, u⁻) with sum ≤ λ.
    C has a simplex row [1ᵀ] on top.
    """
    K = 2 * m
    M = _zeros(K)
    # Row 0: simplex constraint sum(u) ≤ lam,  rows 1..2m: -u_i ≤ 0
    top_row = sp.csc_matrix(np.ones((1, K)))   # (1, 2m)
    neg_eye = -sp.eye(K, format="csc")          # (2m, 2m)
    C = sp.vstack([top_row, neg_eye], format="csc")  # (1+2m, K)
    c = np.concatenate([[lam], np.zeros(K)])
    B = sp.vstack([_eye(m), -_eye(m)], format="csc")
    b = np.zeros(K)
    obj = lambda v: lam * np.max(np.abs(v))  # noqa: E731
    return PLQ(M=M, B=B, C=C, c=c, b=b, obj=obj)


# ---------------------------------------------------------------------------
# Smooth conjugates via function handles
# ---------------------------------------------------------------------------

def _logreg_conjugate(u, alpha=0.5):
    """Logistic regression conjugate: gradient and Hessian.

    Domain: u ∈ (alpha, 1−alpha).
    f*(u) = u·log(u) + (1−u)·log(1−u)   (negative entropy).
    """
    u = np.clip(u, alpha + 1e-12, 1.0 - alpha - 1e-12)
    grad = np.log(u) - np.log(1.0 - u)
    hess = sp.diags(1.0 / (u * (1.0 - u)), format="csc")
    return grad, hess


def logreg(m: int, alpha: float = 0.5) -> PLQ:
    r"""Logistic regression loss: ρ(v) = Σ log(1 + exp(vᵢ)).

    Uses a callable M for the smooth conjugate.
    """
    M_func = lambda u: _logreg_conjugate(u, alpha)  # noqa: E731
    C = sp.vstack([_eye(m), -_eye(m)], format="csc")
    c = np.concatenate([np.ones(m), alpha * np.ones(m)])
    B = -_eye(m)  # Note: negative sign matches MATLAB convention
    b = np.zeros(m)
    obj = lambda v: np.sum(np.log(1.0 + np.exp(v)))  # noqa: E731
    return PLQ(M=M_func, B=B, C=C, c=c, b=b, obj=obj)


def _hybrid_conjugate(u, scale=1.0):
    """Hybrid penalty conjugate: f*(u) = 1 − √(1 − (u/s²)²).

    Gradient: g_i = (u_i / s⁴) / √(1 − (u_i/s²)²)
    Hessian (diagonal): h_i = 1/(s⁴ · (1 − (u_i/s²)²)^{3/2})
    """
    m = scale ** 2
    t = (u / m) ** 2
    t = np.clip(t, 0, 1.0 - 1e-12)
    sq = np.sqrt(1.0 - t)
    grad = (u / m) / (m * sq)
    hess_diag = 1.0 / (m ** 2 * (1.0 - t) ** 1.5)
    hess = sp.diags(hess_diag, format="csc")
    return grad, hess


def hybrid(m: int, scale: float = 1.0) -> PLQ:
    r"""Hybrid penalty: ρ(v) = Σ (√(1 + vᵢ²/s) − 1).

    Uses a callable M for the smooth conjugate.
    """
    s2 = scale ** 2
    M_func = lambda u: _hybrid_conjugate(u, scale)  # noqa: E731
    C = sp.vstack([_eye(m), -_eye(m)], format="csc")
    c = s2 * np.ones(2 * m)
    B = _eye(m)
    b = np.zeros(m)
    obj = lambda v: np.sum(np.sqrt(1.0 + v ** 2 / scale) - 1.0)  # noqa: E731
    return PLQ(M=M_func, B=B, C=C, c=c, b=b, obj=obj)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PENALTY_REGISTRY = {
    "l1": l1,
    "l2": l2,
    "huber": huber,
    "hinge": hinge,
    "vapnik": vapnik,
    "qreg": qreg,
    "qhuber": qhuber,
    "infnorm": infnorm,
    "logreg": logreg,
    "hybrid": hybrid,
}


def load_penalty(name: str, m: int, **kwargs) -> PLQ:
    """Load a named PLQ penalty.

    Parameters
    ----------
    name : str
        One of the registered penalty names (see ``PENALTY_REGISTRY``).
    m : int
        Dimension of the penalty argument.
    **kwargs
        Penalty-specific parameters (``lam``, ``kappa``, ``tau``, etc.).

    Returns
    -------
    PLQ
        The penalty object (before composition with a linear model).
    """
    if name not in PENALTY_REGISTRY:
        raise ValueError(f"Unknown penalty '{name}'. Choose from: {list(PENALTY_REGISTRY.keys())}")

    factory = PENALTY_REGISTRY[name]

    # Map common parameter names
    import inspect
    sig = inspect.signature(factory)
    valid = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in valid}
    return factory(m, **filtered)
