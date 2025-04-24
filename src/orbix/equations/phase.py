"""Exoplanet phase functions."""

import jax.numpy as jnp


def lambert_phase_exact(c):
    """Exact Lambert phase function using an arccos and sqrt call."""
    beta = jnp.arccos(c)
    # sin = (1-cos^2)^{1/2}
    sinb = jnp.sqrt(1.0 - c * c)
    return (sinb + (jnp.pi - beta) * c) / jnp.pi


def lambert_phase_poly(c):
    """Approximate the lambert phase function based on just the cos(beta) value."""
    return 0.318603699 + c * (0.5 + c * (0.153806030 + c * c * 0.0256386115))
