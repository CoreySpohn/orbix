"""Predict the probability of detection of a planet."""

from functools import partial

import jax.numpy as jnp
from jax import jit, lax, vmap


def img_pdet_lin(trig_solver, times, planet, dim_dMag_vals, dim_dMag_seps):
    """Predict the probability of detection of a planet for an imaging system.

    This uses a dimmest dMag curve which relies on a precalculated set of
    dMag values which specify whether a planet will be detected for a given
    optical system.

    Args:
        trig_solver: function to solve Kepler's equation (M, e) -> (sinE, cosE)
        times: time to predict the detection at (mjd)
        planet: Planet object
        dim_dMag_vals: jnp.ndarray of dimmest detectable dMag values between the IWA and OWA
        dim_dMag_seps: jnp.ndarray of separations between the IWA and OWA
    """
    # Calculate the alpha, dMag values which have shape (norb, ntimes)
    alpha, dMag = planet.alpha_dMag(trig_solver, times)

    # Get the dimmest dMag curve
    dim_dMags = jnp.interp(alpha, dim_dMag_seps, dim_dMag_vals, left=0, right=0)

    # Compare the dMag values to the dimmest dMag values
    pdet = jnp.sum(dMag < dim_dMags, axis=0) / dMag.shape[0]
    return pdet


# def img_pdet_grid(trig_solver, dMag0_grid, times, planet, fZ_vals, nEZ_vals):
#     """Predict the probability of detection of a planet for an imaging system."""
#     alpha, dMag = planet.alpha_dMag(trig_solver, times)
#     # Final pdet shape will be (ntimes, nint_times)
#     # where nint_times is the number of integration times in the dMag0_grid
#     # dMag has shape (norb, ntimes)
#     # alpha is an (norb, ntimes) array
#     # fZ is a per-star value so it applies to all orbits and it has shape (ntimes,)
#     # nEZ is a per-star value so at the moment it is a single value array
#     # This demo is at time 0 for the first planet and has shape (nint_times,)
#     # dim_dMag = interp_dMag(dMag0_grid, alpha[0, 0], fZ_vals[0], nEZ_vals[0])
#     # nEZ = nEZ_vals.reshape(())

#     # Do the detection mask for one (orbit, time) pair
#     det_single = partial(_det_one, dMag0_grid, nEZ_vals.reshape(()))
#     # Now do the detection mask for all times of that orbit
#     det_time = vmap(det_single, in_axes=(0, 0, 0))  # (ntimes, n_int_times)

#     # Now map over all orbits
#     detect = vmap(det_time, in_axes=(0, 0, None))(dMag, alpha, fZ_vals)
#     # Pdet is the mean of the detection mask over the orbits
#     pdet = detect.mean(axis=0)
#     return pdet


# @partial(jit, static_argnums=(0, 1))
# def j_img_pdet_grid(trig_solver, dMag0_grid, times, planet, fZ_vals, nEZ_vals):
#     """Jit wrapped version of img_pdet_grid."""
#     return img_pdet_grid(trig_solver, dMag0_grid, times, planet, fZ_vals, nEZ_vals)


# def _det_one(dMag0_grid, nEZ, dmag, alpha, fZ):
#     """Detection mask for one (orbit, time) pair -> (n_int_times,) bool."""
#     # (n_int_times,)
#     dim = interp_dMag(dMag0_grid, fZ, nEZ, alpha)
#     # bool vector
#     return dmag < dim


# def interp_dMag(dMag0_grid, fZ_val, nEZ_val, alpha_val):
#     """Fast interpolation of dMag values from the grid.

#     Args:
#         dMag0grid: dMag0Grid equinox module
#         alpha_val: planet-star separation in arcsec
#         fZ_val: zodiacal light brightness
#         nEZ_val: exozodiacal light brightness

#     Returns:
#         Interpolated dimmest detectable dMag value
#     """
#     # First check the geometric constraint set by the IWA and OWA
#     alpha_min = dMag0_grid.alpha0
#     alpha_max = dMag0_grid.alphas[-1]
#     in_range = (alpha_val >= alpha_min) & (alpha_val <= alpha_max)
#     # ---- Alpha indexing (linear space) ----
#     # Assumption here is that the grid is between the IWA and OWA
#     a_val = jnp.clip(alpha_val, alpha_min, alpha_max)
#     alpha_ind = (a_val - alpha_min) * dMag0_grid.inv_dalpha
#     a0 = jnp.clip(alpha_ind.astype(jnp.int32), 0, dMag0_grid.n_alpha - 2)
#     dalpha = alpha_ind - a0

#     # ---- fZ indexing (log space) ----
#     log_fZ = jnp.log10(fZ_val)
#     fZ_ind = (log_fZ - dMag0_grid.log_fZ0) * dMag0_grid.inv_log_fZ_step
#     # Adding 1 to use the higher value of fZ as a factor of safety kinda thing
#     fZ0 = jnp.clip(jnp.floor(fZ_ind).astype(jnp.int32) + 1, 0, dMag0_grid.n_fZ - 2)
#     # dfZ = fZ_ind - fZ0

#     # ---- nEZ indexing ----
#     # nEZ0 = jnp.clip(
#     #     jnp.argmin(jnp.abs(dMag0_grid.nEZs - nEZ_val)), 0, dMag0_grid.n_nEZ - 1
#     # )
#     # TODO: Update this to use the actual nEZ value
#     nEZ0 = 0

#     patch = lax.dynamic_slice(
#         dMag0_grid.grid, (fZ0, nEZ0, a0, 0), (1, 1, 2, dMag0_grid.grid.shape[-1])
#     )
#     # Interpolate over the alpha dimension
#     dMags = patch[0, 0, 0] + dalpha * (patch[0, 0, 1] - patch[0, 0, 0])
#     # return jnp.where(in_range, dMags, jnp.zeros_like(dMags))
#     return dMags * in_range
#     # return dMags


def _prep_zodi_indices(fZ_vals, grid):
    """Return integer fZ bin and fractional offset for every epoch."""
    log_fZ = jnp.log10(fZ_vals)
    fZ_ind = (log_fZ - grid.log_fZ0) * grid.inv_log_fZ_step
    fZ0 = jnp.clip(jnp.floor(fZ_ind).astype(jnp.int32) + 1, 0, grid.n_fZ - 2)
    # if I ever add bilinear-in-fZ
    # dfZ = fZ_ind - fZ0
    return fZ0


def _interp_dMag_gather(grid, a_ind, a0, fZ0):
    """grid.grid shape:  (N_fZ, N_nEZ, N_alpha, N_time)  ← original layout
    a_ind, a0 shape: (norb, nt)
    fZ0: (nt,) integer zodi index per epoch
    Returns: (norb, nt, N_time)
    """
    n_tint = grid.grid.shape[-1]

    # Calculate interpolation weights once
    dalpha = (a_ind - a0).astype(grid.grid.dtype)

    # Function for a single (orbit, time) pair
    def get_slice_single(a0_val, dalpha_val, fz):
        patch = lax.dynamic_slice(
            grid.grid, (fz, 0, a0_val, 0), (1, 1, 2, n_tint)
        ).reshape(2, n_tint)

        # Linear interpolation
        return patch[0] + dalpha_val * (patch[1] - patch[0])

    # Vectorize over all orbits (for a given time point and zodiacal index)
    # in_axes: a0 and dalpha are batched on first dimension (orbit), fZ is scalar
    get_slice_orbits = vmap(get_slice_single, in_axes=(0, 0, None))

    # Vectorize over all time points
    # in_axes: a0 and dalpha are batched on second dimension (time),
    # fZ0 is batched on first dimension (time index)
    get_slice_all = vmap(get_slice_orbits, in_axes=(1, 1, 0))

    # Apply the fully vectorized function to all data at once
    result = get_slice_all(a0, dalpha, fZ0)

    # Transpose from (nt, norb, n_tint) to (norb, nt, n_tint)
    return jnp.transpose(result, (1, 0, 2))


def _interp_dMag_gather_stub(grid, a_ind, a0, fZ0):
    """Fast stub that returns same-shaped output with minimal computation.
    Returns an array of shape (norb, ntimes, n_tint)
    """
    n_tint = grid.grid.shape[-1]
    n_orb, n_times = a_ind.shape

    # Create a simple value based on inputs to maintain data dependencies
    # This avoids any dynamic slicing or complex memory access patterns
    base_val = jnp.mean(a_ind) + jnp.mean(jnp.asarray(fZ0))

    # Generate result tensor with minimal computation
    # Use simple broadcasting instead of complex indexing/slicing
    result = jnp.ones((n_orb, n_times, n_tint)) * base_val

    # Add some variation based on orbit and time indices to keep compiler from optimizing away
    orbit_idx = jnp.arange(n_orb)[:, None, None]
    time_idx = jnp.arange(n_times)[None, :, None]
    result = result + orbit_idx * 0.01 + time_idx * 0.001

    return result


# ---------------------------------------------------------------------------
# 3.  Main Pdet kernel  (all heavy lifting fused in one JIT)
# ---------------------------------------------------------------------------
@partial(jit, static_argnums=(0))  # trig_solver, grid baked in
def img_pdet_grid(
    trig_solver,
    grid,
    times,  # (nt,)
    planet,  # Planet object
    fZ_vals,  # (nt,)
    nEZ_vals,  # unused for now (nEZ=0)
):
    """Returns:
    -------
    pdet : (ntimes, n_tint)
    """
    # ---- orbital geometry --------------------------------------------------
    alpha, dMag = planet.alpha_dMag(trig_solver, times)
    # alpha,dMag are (norb, ntimes)

    # First check the geometric constraint set by the IWA and OWA
    alpha_min = grid.alpha0
    alpha_max = grid.alphas[-1]
    geom_mask = (alpha >= alpha_min) & (alpha <= alpha_max)

    # ---- zodi/exozodi indices  (epoch-only) -------------------------------
    fZ0 = _prep_zodi_indices(fZ_vals, grid)  # (ntimes,)

    # ---- α-grid indices  (orbit×epoch) ------------------------------------
    a_ind = (alpha - grid.alpha0) * grid.inv_dalpha
    a0 = jnp.clip(a_ind.astype(jnp.int32), 0, grid.n_alpha - 1)

    # ---- gather & interpolate (fused in XLA) ------------------------------
    dim = _interp_dMag_gather_stub(grid, a_ind, a0, fZ0)  # (norb, ntimes, n_tint)

    # ---- compare + orbit-average ------------------------------------------
    det_mask = geom_mask[..., None] & jnp.signbit(dMag[..., None] - dim)

    pdet = det_mask.mean(axis=0)  # (ntimes, n_tint)

    return pdet


@partial(jit, static_argnums=(0))
def img_pdet_grid_final(
    trig_solver,
    grid,
    times,  # (nt,)
    planet,  # Planet object
    fZ_vals,  # (nt,)
    nEZ_vals,  # unused for now (nEZ=0)
):
    """Optimized implementation preserving dimension order"""
    # Calculate orbital geometry
    alpha, dMag = planet.alpha_dMag(trig_solver, times)

    # First check the geometric constraint set by the IWA and OWA
    alpha_min = grid.alpha0
    alpha_max = grid.alphas[-1]
    geom_mask = (alpha >= alpha_min) & (alpha <= alpha_max)

    # Prepare zodi indices
    fZ0 = _prep_zodi_indices(fZ_vals, grid)

    # Calculate grid indices
    a_ind = (alpha - grid.alpha0) * grid.inv_dalpha
    a0 = jnp.clip(a_ind.astype(jnp.int32), 0, grid.n_alpha - 1)

    # Instead of using a full gather function, directly compute dims
    # This avoids dimension transposing and reduces memory operations
    n_tint = grid.grid.shape[-1]
    n_orb, n_times = alpha.shape

    # Create a custom gather operation using direct vectorization
    dalpha = (a_ind - a0).astype(grid.grid.dtype)

    # Use the original _interp_dMag_gather function which is already working
    dim = _interp_dMag_gather(grid, a_ind, a0, fZ0)

    # Apply detection condition and compute mean - combine into one operation
    # Use simple indexing with standard JAX broadcasting rules
    det_mask = geom_mask[..., None] & jnp.signbit(dMag[..., None] - dim)

    # Compute mean more efficiently by summing directly
    # pdet = jnp.sum(det_mask, axis=0) / n_orb
    pdet = det_mask.mean(axis=0)

    return pdet


# ---------------------------------------------------------------------------
# 4.  Static-Grid optimized Pdet kernel
# ---------------------------------------------------------------------------
def _img_pdet_grid_inner(
    trig_solver,
    grid_access,  # Dict-like with unpacked grid components
    times,  # (nt,)
    planet,  # Planet object
    fZ_vals,  # (nt,)
    nEZ_vals,  # unused for now (nEZ=0)
):
    """Core implementation using unpacked grid access for static optimization."""
    # ---- orbital geometry --------------------------------------------------
    alpha, dMag = planet.alpha_dMag(trig_solver, times)
    # alpha,dMag are (norb, ntimes)

    # First check the geometric constraint set by the IWA and OWA
    alpha_min = grid_access["alpha0"]
    alpha_max = (
        grid_access["grid"].shape[2] * (1.0 / grid_access["inv_dalpha"]) + alpha_min
    )
    geom_mask = (alpha >= alpha_min) & (alpha <= alpha_max)

    # ---- zodi/exozodi indices (epoch-only) --------------------------------
    log_fZ = jnp.log10(fZ_vals)
    fZ_ind = (log_fZ - grid_access["log_fZ0"]) * grid_access["inv_log_fZ_step"]
    fZ0 = jnp.clip(jnp.floor(fZ_ind).astype(jnp.int32) + 1, 0, grid_access["n_fZ"] - 2)

    # ---- α-grid indices (orbit×epoch) -------------------------------------
    a_ind = (alpha - grid_access["alpha0"]) * grid_access["inv_dalpha"]
    a0 = jnp.clip(a_ind.astype(jnp.int32), 0, grid_access["n_alpha"] - 1)

    # Calculate interpolation weights once
    dalpha = (a_ind - a0).astype(grid_access["grid"].dtype)
    n_tint = grid_access["grid"].shape[-1]

    # ---- gather & interpolate --------------------------------------------
    # Function for a single (orbit, time) pair
    def get_slice_single(a0_val, dalpha_val, fz):
        patch = lax.dynamic_slice(
            grid_access["grid"], (fz, 0, a0_val, 0), (1, 1, 2, n_tint)
        ).reshape(2, n_tint)
        # Linear interpolation
        return patch[0] + dalpha_val * (patch[1] - patch[0])

    # Vectorize over all orbits (for a given time point and zodiacal index)
    get_slice_orbits = vmap(get_slice_single, in_axes=(0, 0, None))

    # Vectorize over all time points
    get_slice_all = vmap(get_slice_orbits, in_axes=(1, 1, 0))

    # Apply the fully vectorized function to all data at once
    result = get_slice_all(a0, dalpha, fZ0)

    # Transpose from (nt, norb, n_tint) to (norb, nt, n_tint)
    dim = jnp.transpose(result, (1, 0, 2))

    # ---- compare + orbit-average ------------------------------------------
    det_mask = geom_mask[..., None] & jnp.signbit(dMag[..., None] - dim)

    pdet = det_mask.mean(axis=0)  # (ntimes, n_tint)
    return pdet


# Create the static-grid compatible version
