"""
IPsolve -- Interior-point solver for PLQ composite optimisation.

Minimises problems of the form::

    min_x  l'x + rho_meas(z - Hx) + rho_proc(Kx - k)   s.t. Ax <= a

where rho_meas and rho_proc are **piecewise linear-quadratic** (PLQ) penalties.

Quick example::

    from ipsolve import solve
    x = solve(H, z, meas="l2", proc="l1", proc_lambda=0.1)
"""

from ipsolve.api import solve  # noqa: F401
from ipsolve.solver import SolverResult  # noqa: F401

__version__ = "0.2.0"
