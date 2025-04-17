"""Core fitting functions."""

import jax
import jax.numpy as jnp

from orbix.constants import eps, two_pi
from orbix.kepler.core import E_solve
from orbix.kepler.shortcuts.grid import jaxify_scalar_func


def fitting_grid(n_e=1024, n_M=4096):
    """Creates vectorized JIT func for orbit fitting quantities via 2D grid.

    This factory function precomputes several quantities relevant for orbit fitting
    (X, Y, RV1, RV2) and their derivatives with respect to the mean anomaly grid
    index over a 2D grid of eccentricity (e) and mean anomaly (M).

    Definitions:
        X = cosE - e
        Y = sinE * sqrt(1 - e**2)
        g = sqrt((1+e)/(1-e)) * tan(E/2)
        RV1 = (1-g**2)/(1+g^2) + e  (equivalent to cos(f) + e)
        RV2 = 2*g/(1+g^2)          (equivalent to sin(f))
        Note: RV1 and RV2 are the coefficients of cos(omega) and -sin(omega)
        respectively in the efficient radial velocity formula RV = k *
        (RV1*cos(omega) - RV2*sin(omega)), derived from trigonometric
        identities (see Eq. 21 in the Orvara paper).

    The returned function is JIT-compiled and vectorized using `jax.vmap`. It accepts
    an array of mean anomalies `M_vals` (shape (num_planets, num_times)) and an
    array of eccentricities `e_array` (shape (num_planets,)) and returns arrays for
    X, Y, RV1, and RV2, each with shape (num_planets, num_times), calculated via
    linear interpolation on the precomputed grids. This is optimized for fitting
    procedures involving multiple orbits.

    Args:
        n_e:
            Number of eccentricity steps in the grid (0 <= e < 1). Default is 1000.
        n_M:
            Number of mean anomaly steps in the grid (0 <= M < 2pi). Default is 3600.

    Returns:
        A JIT-compiled, vectorized function `vectorized_lookup(M_vals,
        e_array)` that takes array inputs and returns a tuple `(X_array,
        Y_array, RV1_array, RV2_array)` containing arrays of interpolated
        fitting quantities.
    """
    # --- Grid Definition ---
    # Note: e_grid goes up to, but does not include 1
    e_grid_vec = jnp.linspace(0, 1, n_e, endpoint=False)
    M_grid_vec = jnp.linspace(0, two_pi, n_M, endpoint=False)

    # Reshape e_grid for broadcasting with (n_e, n_M) shapes
    e_grid = e_grid_vec[:, None]  # Shape (n_e, 1)

    dM = M_grid_vec[1] - M_grid_vec[0]
    inv_dM = 1.0 / dM
    dM_int = jnp.int32(n_M)

    dE_grid_step = e_grid_vec[1] - e_grid_vec[0]
    inv_dE_grid_step = 1.0 / dE_grid_step

    # --- Compute E, sinE, cosE and their derivatives w.r.t M_index ---
    # Solve Kepler's equation on the grid
    # vmap over e (rows), solving E for all M (columns) for each e
    E_grid = jax.jit(jax.vmap(lambda e_val: E_solve(M_grid_vec, e_val)))(e_grid_vec)

    cosE_grid = jnp.cos(E_grid)
    sinE_grid = jnp.sin(E_grid)

    # dE/dM = 1 / (1 - e*cosE)
    dE_dM_grid = 1.0 / (1.0 - e_grid * cosE_grid + eps)  # Add eps for safety
    # dE/dind = dE/dM * dM/dind = dE/dM * dM
    dE_dind_grid = dE_dM_grid * dM

    # d(sinE)/dind = d(sinE)/dE * dE/dind = cosE * dE_dind
    dsinE_dind = cosE_grid * dE_dind_grid
    # d(cosE)/dind = d(cosE)/dE * dE/dind = -sinE * dE_dind
    dcosE_dind = -sinE_grid * dE_dind_grid

    # --- Compute X, Y grids and their derivatives w.r.t M_index ---
    sqrt_one_minus_e2_grid = jnp.sqrt(1.0 - e_grid**2 + eps)  # Shape (n_e, 1)

    # X = cosE - e
    X_grid = cosE_grid - e_grid  # Broadcasting works: (n_e, n_M) - (n_e, 1)
    # dX/dind = d(cosE)/dind - d(e)/dind = dcosE_dind - 0
    dX_dind = dcosE_dind

    # Y = sinE * sqrt(1 - e**2)
    Y_grid = (
        sinE_grid * sqrt_one_minus_e2_grid
    )  # Broadcasting works: (n_e, n_M) * (n_e, 1)
    # dY/dind = d(sinE)/dind * sqrt(1-e^2) + sinE * d(sqrt(1-e^2))/dind
    # Since e does not depend on M_index, second term is 0
    dY_dind = dsinE_dind * sqrt_one_minus_e2_grid  # Broadcasting works

    # --- Compute g, RV1, RV2 grids and their derivatives w.r.t M_index ---

    # Calculate g = sqrt((1+e)/(1-e)) * tan(E/2) with stability
    # Using tan(E/2) = (1-cosE)/sinE formulation
    sqrt_term = jnp.sqrt((1.0 + e_grid + eps) / (1.0 - e_grid + eps))  # Shape (n_e, 1)

    tan_half_E_num = 1.0 - cosE_grid
    # Handle sinE near zero carefully
    safe_sinE = jnp.where(
        jnp.abs(sinE_grid) < eps, jnp.sign(sinE_grid) * eps + eps, sinE_grid
    )
    tan_half_E_grid = tan_half_E_num / safe_sinE
    # Explicitly set tan(E/2) to 0 where E=0 (where num and den are both zero)
    is_E_zero = (jnp.abs(tan_half_E_num) < eps) & (jnp.abs(sinE_grid) < eps)
    tan_half_E_grid = jnp.where(is_E_zero, 0.0, tan_half_E_grid)

    g_grid = sqrt_term * tan_half_E_grid  # Broadcasting works: (n_e, 1) * (n_e, n_M)
    # Handle potential NaN/inf
    g_grid = jnp.nan_to_num(g_grid, nan=0.0, posinf=jnp.inf, neginf=-jnp.inf)

    # Calculate RV1, RV2 grids with stability
    g_squared = g_grid**2
    one_plus_gsq = 1.0 + g_squared

    # Handle g -> inf limits using jnp.where
    is_g_inf = jnp.isinf(g_grid) | (jnp.abs(one_plus_gsq) < eps)

    rv1_term_grid = jnp.where(is_g_inf, -1.0, (1.0 - g_squared) / (one_plus_gsq + eps))
    RV1_grid = rv1_term_grid + e_grid  # Broadcasting works

    RV2_grid = jnp.where(is_g_inf, 0.0, (2.0 * g_grid) / (one_plus_gsq + eps))

    # Calculate derivatives dRV1/dind, dRV2/dind using chain rule
    # Need dg/dE = sqrt((1+e)/(1-e)) * d(tan(E/2))/dE
    # d(tan(E/2))/dE = 1 / (1 + cosE)
    dg_dE_grid = sqrt_term / (1.0 + cosE_grid + eps)  # Broadcasting works

    # Need dRV1/dg = -4g / (1+g^2)^2
    dRV1_dg_grid = jnp.where(is_g_inf, 0.0, -4.0 * g_grid / ((one_plus_gsq) ** 2 + eps))

    # Need dRV2/dg = (2 - 2g^2) / (1+g^2)^2
    dRV2_dg_grid = jnp.where(
        is_g_inf, 0.0, (2.0 - 2.0 * g_squared) / ((one_plus_gsq) ** 2 + eps)
    )

    # dRV/dE = dRV/dg * dg/dE
    dRV1_dE_grid = dRV1_dg_grid * dg_dE_grid
    dRV2_dE_grid = dRV2_dg_grid * dg_dE_grid

    # dRV/dind = dRV/dE * dE/dind
    dRV1_dind = dRV1_dE_grid * dE_dind_grid
    dRV2_dind = dRV2_dE_grid * dE_dind_grid

    # Clean up potential NaN/inf in final derivatives
    dRV1_dind = jnp.nan_to_num(dRV1_dind)
    dRV2_dind = jnp.nan_to_num(dRV2_dind)

    # --- Lookup Function Definition ---
    def _lookup_scalar(M_scalar, e_scalar):
        """Performs lookup and interpolation for a single M and e."""
        # Scale M to index space
        ind_M = M_scalar * inv_dM
        ind_M_int = ind_M.astype(jnp.int32)
        # Fractional part of M index for interpolation
        dind_M = ind_M - ind_M_int
        # Integer part of M index (wrap around)
        M_ind = ind_M_int % dM_int

        # Get the e index
        e_ind = (e_scalar * inv_dE_grid_step).astype(jnp.int32)

        # Linear interpolation: Value = Value_grid[ind] + dValue_dind[ind] * dind_M
        X = X_grid[e_ind, M_ind] + dX_dind[e_ind, M_ind] * dind_M
        Y = Y_grid[e_ind, M_ind] + dY_dind[e_ind, M_ind] * dind_M
        RV1 = RV1_grid[e_ind, M_ind] + dRV1_dind[e_ind, M_ind] * dind_M
        RV2 = RV2_grid[e_ind, M_ind] + dRV2_dind[e_ind, M_ind] * dind_M
        return X, Y, RV1, RV2

    return jaxify_scalar_func(_lookup_scalar)
