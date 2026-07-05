"""Finite-difference checks on the hand-derived IFT backward pass."""

import jax.numpy as jnp
from jax.test_util import check_grads

from orbix.kepler.core import diff_solve_trig


def _loss(M, e):
    """Scalar reduction of the (sinE, cosE) outputs for gradient checking."""
    sinE, cosE = diff_solve_trig(M, e)
    return jnp.sum(sinE * 1.3 + cosE * 0.7)


def test_diff_solve_trig_gradients_match_finite_differences():
    """The custom VJP for diff_solve_trig matches finite-difference gradients."""
    M = jnp.linspace(0.1, 2 * jnp.pi - 0.1, 9)
    for e in [0.0, 0.05, 0.3, 0.7, 0.95]:
        check_grads(
            _loss, (M, jnp.float64(e)), order=1, modes=["rev"], atol=1e-4, rtol=1e-4
        )
