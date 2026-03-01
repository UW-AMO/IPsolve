"""
IPsolve -- Interior-point solver for PLQ composite optimisation.

Minimises problems of the form::

    min_x  l'x + rho_meas(z - Hx) + rho_proc(Kx - k)   s.t. Ax <= a

where rho_meas and rho_proc are **piecewise linear-quadratic** (PLQ) penalties.

Quick start::

    from ipsolve import solve, huber, l1, l2

    x = solve(H, z)                                  # least squares
    x = solve(H, z, meas=huber(kappa=1.0))           # robust regression
    x = solve(H, z, proc=l1(lam=0.5))                # lasso
    x = solve(H, z, bounds=(-1, 1))                  # box constraints
"""

from ipsolve.api import solve  # noqa: F401
from ipsolve.solver import SolverResult  # noqa: F401
from ipsolve.penalties import (  # noqa: F401
    Penalty,
    l1, l2, huber, hinge, vapnik,
    qreg, qhuber, infnorm, logreg, hybrid,
)

__version__ = "0.2.0"
