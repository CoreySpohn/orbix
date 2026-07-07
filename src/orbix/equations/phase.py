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
    phase_raw = (sinbeta + (jnp.pi - beta) * cosbeta) / jnp.pi
    # Clip result to ensure phase is physically non-negative
    return jnp.maximum(phase_raw, 0.0)


def lambert_phase_poly(c):
    """Approximate the lambert phase function based on just the cos(beta) value."""
    # Cubic fit to the Lambert phase over beta in [0, pi]; max abs error
    # ~1.96e-3 vs lambert_phase_exact (10001-point grid over [0, pi]).
    return 0.318603699 + c * (0.5 + c * (0.153806030 + c * c * 0.0256386115))
