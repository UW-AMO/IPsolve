"""
IPsolve – Interior-point solver for PLQ composite optimisation.

Minimises problems of the form::

    min_x  ℓᵀx + ρ_meas(z − Hx) + ρ_proc(Kx − k)   s.t. Ax ≤ a

where ρ_meas and ρ_proc are **piecewise linear-quadratic** (PLQ) penalties.

Quick example::

    from ipsolve import solve
    x = solve(H, z, meas="l2", proc="l1", proc_lambda=0.1)
"""

from ipsolve.api import solve  # noqa: F401

__version__ = "0.1.0"
