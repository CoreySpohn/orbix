"""Differentiable Kepler solver via custom VJP.

Wraps :func:`orbix.kepler.core.solve_trig` with exact analytical gradients
derived from the Implicit Function Theorem on Kepler's equation
``M = E - e·sin(E)``.

The backward pass computes gradients from ``(sinE, cosE, e)`` alone — no
extra trig calls, no iterative re-solves.
"""

import jax
import jax.numpy as jnp

from orbix.kepler.core import solve_trig


@jax.custom_vjp
def diff_solve_trig(M, e):
    """Solve Kepler's equation, returning (sinE, cosE) with exact gradients.

    This is a drop-in replacement for :func:`orbix.kepler.core.solve_trig`
    that supports reverse-mode autodiff (``jax.grad``, ``jax.vjp``).

    Args:
        M: Mean anomaly array. Shape: ``(n,)``.
        e: Eccentricity (scalar float).

    Returns:
        Tuple of ``(sinE, cosE)``, each with shape ``(n,)``.
    """
    return solve_trig(M, e)


def _fwd(M, e):
    """Forward pass: solve and save residuals for backward."""
    sinE, cosE = solve_trig(M, e)
    return (sinE, cosE), (sinE, cosE, e)


def _bwd(res, g):
    """Backward pass: exact gradients via Implicit Function Theorem.

    From Kepler's equation ``M = E - e·sinE``:
        dE/dM = 1 / (1 - e·cosE)
        dE/de = sinE / (1 - e·cosE)

    Chain rule for outputs ``(sinE, cosE)``:
        d(sinE)/dE = cosE
        d(cosE)/dE = -sinE

    So the combined upstream gradient w.r.t. E is:
        dL/dE = g_sinE·cosE + g_cosE·(-sinE) = g_sinE·cosE - g_cosE·sinE

    And therefore:
        dL/dM = dL/dE · dE/dM = dL/dE / (1 - e·cosE)
        dL/de = dL/dE · dE/de = dL/dE · sinE / (1 - e·cosE)
    """
    sinE, cosE, e = res
    g_sinE, g_cosE = g

    inv_denom = 1.0 / (1.0 - e * cosE)
    dL_dE = g_sinE * cosE - g_cosE * sinE

    dL_dM = dL_dE * inv_denom
    # Sum over the array dimension for the scalar eccentricity
    dL_de = jnp.sum(dL_dE * sinE * inv_denom)

    return dL_dM, dL_de


diff_solve_trig.defvjp(_fwd, _bwd)
