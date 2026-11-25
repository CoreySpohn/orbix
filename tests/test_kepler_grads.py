"""Finite-difference checks on the hand-derived IFT backward pass."""

import jax.numpy as jnp
import pytest
from jax.test_util import check_grads

from orbix.kepler import kepler


def _loss(M, e):
    """Scalar reduction of the (sinE, cosE) outputs for gradient checking."""
    sinE, cosE = kepler(M, e, 0)
    return jnp.sum(sinE * 1.3 + cosE * 0.7)


@pytest.mark.float64_only(reason="finite-difference derivative reference")
def test_broadcast_reduction_gradients_match_finite_differences():
    """Fails if the implicit rule reduces a shared eccentricity gradient incorrectly."""
    M = jnp.linspace(0.1, 2 * jnp.pi - 0.1, 9)
    for e in [0.0, 0.05, 0.3, 0.7, 0.95]:
        check_grads(
            _loss, (M, jnp.float64(e)), order=1, modes=["rev"], atol=1e-4, rtol=1e-4
        )
