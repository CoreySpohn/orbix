"""Fixed eccentricity shortcuts for Kepler's equation.

These are useful if you are solving Kepler's equation many times with the same
eccentricity, since they pre-compute the eccentric anomaly for all mean
anomalies and create fast, JIT-compiled functions that either return the
nearest value (`E_lookup`) or interpolate between them (`E_linear_interp` and
`E_hermite_interp`).

The functions return a JIT-compiled function that takes a mean anomaly
and returns the eccentric anomaly. The eccentric anomaly is computed using
the `E_solve` function from the `kepler.core` module. Each method is optimized
to do as few operations as possible and pre-calculate any constants that are
not dependent on the mean anomaly. Those constants are included in the returned
function by closure.
"""

import jax
import jax.numpy as jnp

from orbix.constants import two_pi
from orbix.kepler.core import E_solve


def E_lookup(e, n=2048):
    """Create a JIT'd function to lookup E for a fixed eccentricity.

    Args:
        e: The eccentricity of the orbit.
        n: The number of mean anomaly values to use in the lookup table.

    Returns:
        A JIT-compiled function that takes M (as either a scalar or an array) and
        returns E (as a scalar or an array).
    """
    M_vals = jnp.linspace(0, two_pi, n, endpoint=False)
    E_vals = E_solve(M_vals, e)
    dM = M_vals[1] - M_vals[0]
    inv_dM = 1.0 / dM
    n_int = jnp.int32(n)

    @jax.jit
    def lookup(M):
        return E_vals[(M * inv_dM).astype(jnp.int32) % n_int]

    return lookup


def E_linear_interp(e, n=2048):
    """Create a JIT'd function to interpolate E for a fixed eccentricity.

    This method uses linear interpolation to find the eccentric anomaly. The
    basic idea is to turn most operations into integer operations by scaling
    the mean anomaly into the same index space as the lookup table, then using
    the integer part of the index to index into the table and the fractional
    part to linearly interpolate.

    Args:
        e: The eccentricity of the orbit.
        n: The number of mean anomaly values to use in the lookup table.

    Returns:
        A JIT-compiled function that takes M (as either a scalar or an array) and
        returns E (as a scalar or an array).
    """
    # Linearly spaced mean anomaly values
    M_vals = jnp.linspace(0, two_pi, n, endpoint=False)
    dM = M_vals[1] - M_vals[0]
    # Use the E_solve function to get the real eccentric anomaly values
    # and the derivative of E (with respect to the array indices)
    E_vals = E_solve(M_vals, e)
    dE_dM = 1 / (1 - e * jnp.cos(E_vals))
    dE_dind = dE_dM * dM

    inv_dM = 1.0 / dM
    n_int = jnp.int32(n)

    @jax.jit
    def linear_interp(M):
        # Scale M to index space
        ind_M = M * inv_dM
        # Get integer part of the index
        ind_M_int = ind_M.astype(jnp.int32)
        # Get the fractional part of the index
        dind = ind_M - ind_M_int
        # Wrap to [0, 2pi) indices with modulo n
        final_inds = ind_M_int % n_int
        # Linear interpolation
        return E_vals[final_inds] + dE_dind[final_inds] * dind

    return linear_interp


def E_hermite_interp(e, n=2048):
    """Create a JIT'd function to interpolate E for a fixed eccentricity.

    This method uses Hermite interpolation to find the eccentric anomaly. Like
    in the linear interpolation method, we turn most operations into integer
    operations by scaling the mean anomaly into the same index space as the
    lookup table, then using the integer part of the index to index into the
    table and the fractional part to do Hermite interpolation.

    Args:
        e: The eccentricity of the orbit.
        n: The number of mean anomaly values to use in the lookup table.

    Returns:
        A JIT-compiled function that takes M (as either a scalar or an array) and
        returns E (as a scalar or an array).
    """
    # Generate linearly spaced mean anomaly values
    M_vals = jnp.linspace(0, two_pi, n, endpoint=False)
    # Compute the step size in mean anomaly per index
    dM = M_vals[1] - M_vals[0]
    # Solve for eccentric anomaly for each mean anomaly
    E_vals = E_solve(M_vals, e)
    # Compute the derivative of E with respect to the index
    dE_dM = 1 / (1 - e * jnp.cos(E_vals))
    dE_dind = dE_dM * dM

    inv_dM = 1.0 / dM
    n_int = jnp.int32(n)

    @jax.jit
    def hermite_interp(M):
        # Scale M to index space
        ind_M = M * inv_dM
        # Get integer part of the index
        ind_M_int = ind_M.astype(jnp.int32)
        # Get the fractional part of the index
        dind = ind_M - ind_M_int
        # Wrap to [0, 2pi) indices with modulo n
        final_inds = ind_M_int % n_int
        final_inds_next = (final_inds + 1) % n_int

        # Retrieve function values and derivatives at the surrounding indices
        E_i = E_vals[final_inds]
        E_ip1 = E_vals[final_inds_next]
        dE_i = dE_dind[final_inds]
        dE_ip1 = dE_dind[final_inds_next]

        # Hermite basis functions
        dind2 = dind * dind
        dind3 = dind2 * dind
        h00 = 2 * dind3 - 3 * dind2 + 1
        h10 = dind3 - 2 * dind2 + dind
        h01 = -2 * dind3 + 3 * dind2
        h11 = dind3 - dind2

        # Perform Hermite interpolation
        E_interp = h00 * E_i + h10 * dE_i + h01 * E_ip1 + h11 * dE_ip1

        return E_interp

    return hermite_interp
