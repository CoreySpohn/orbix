"""Exoplanet phase functions."""

import jax.numpy as jnp


def lambert_phase_exact(cosbeta, sinbeta):
    """Exact Lambert phase function using an arccos and sqrt call.

    Args:
        cosbeta:
            The cosine of the phase angle.
        sinbeta:
            The sine of the phase angle.

    Returns:
         The Lambert phase function value, clipped to be non-negative.
    """
    beta = jnp.arccos(cosbeta)
    # beta = jnp.arctan2(sinbeta, cosbeta)
    phase_raw = (sinbeta + (jnp.pi - beta) * cosbeta) / jnp.pi
    # Clip result to ensure phase is physically non-negative
    return jnp.maximum(phase_raw, 0.0)


def lambert_phase_poly(c):
    """Approximate the lambert phase function based on just the cos(beta) value."""
    return 0.318603699 + c * (0.5 + c * (0.153806030 + c * c * 0.0256386115))
