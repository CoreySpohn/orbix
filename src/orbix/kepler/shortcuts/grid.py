"""Grid-based functions to solve Kepler's equation.

These functions are essentially vectorized versions of the functions in
`fixed_e.py`. They work by precomputing a 2D grid of E, sinE, cosE values and
their derivatives (for linear interpolation). Then, for a given (M, e) pair, a
scalar function is created to do the interpolation using the precomputed grid.
This scalar function is then jit-compiled and vectorized over time and planets.

Generally these are 3-5x faster than a vectorized/compiled form of `E_solve`
when computing thousands of epochs but the difference is less pronounced at the
few per-epoch level. The bilinear interpolation is ~1.5x slower than the linear
interpolation, but is accurate to 32 bit precision, whereas the linear
interpolation has maximum errors around 1e-3 near M=pi.
"""

import jax
import jax.numpy as jnp

from orbix.constants import two_pi
from orbix.kepler.core import E_solve_jit


def _jaxify_scalar_func(func):
    """Vectorize a scalar E function over time, then planets, then compile."""
    _vmapped_time = jax.vmap(func, in_axes=(0, None))
    _lookup_planet = jax.vmap(_vmapped_time, in_axes=(0, 0))
    return jax.jit(_lookup_planet)


def _setup(n_e: int = 1000, n_M: int = 3600):
    """Setup for grid methods."""
    e_grid = jnp.linspace(0.0, 1.0, n_e, endpoint=False)
    M_grid = jnp.linspace(0.0, two_pi, n_M, endpoint=False)

    dM = M_grid[1] - M_grid[0]
    inv_dM = 1.0 / dM
    dM_int = jnp.int32(n_M)

    de = e_grid[1] - e_grid[0]
    inv_de = 1.0 / de

    return e_grid, M_grid, dM, inv_dM, dM_int, de, inv_de


def _E_grids(e_grid, M_grid):
    """Compute E, sinE, cosE grids."""
    E_grid = jax.vmap(lambda e_val: E_solve_jit(M_grid, e_val))(e_grid)
    return E_grid, jnp.sin(E_grid), jnp.cos(E_grid)


def _d_dind_grids(e_grid, E_grid, sinE_grid, cosE_grid, dM):
    """Compute dE/dind, dsinE/dind, dcosE/dind grids."""
    dE_dM_grid = 1 / (1 - e_grid.reshape(-1, 1) * jnp.cos(E_grid))
    dE_dind_grid = dE_dM_grid * dM
    # dsinE/dind = dsinE/dE * dE/dind
    dsinE_dind = cosE_grid * dE_dind_grid
    # dcosE/dind = dcosE/dE * dE/dind
    dcosE_dind = -sinE_grid * dE_dind_grid
    return dE_dind_grid, dsinE_dind, dcosE_dind


def _ind(scalar, inv_d):
    """Returns the indices and fractional difference for linear interpolation."""
    ind = scalar * inv_d
    i0 = ind.astype(jnp.int32)
    di = ind - i0
    return i0, di


# linear interpolation function
def _lin(tab, dtab, e_ind, M_ind, dM):
    """Linear interpolation for a single (M, e) pair."""
    return tab[e_ind, M_ind] + dtab[e_ind, M_ind] * dM


def _grid_lin_params(M_scalar, e_scalar, inv_dM, inv_de, dM_int):
    """Linear lookup for a single (M, e) pair."""
    M0, dM = _ind(M_scalar, inv_dM)
    M0 = M0 % dM_int
    e0 = (e_scalar * inv_de).astype(jnp.int32)
    return e0, M0, dM


def grid_E_trig_lin(n_e=1024, n_M=4096):
    """Creates vectorized JIT func for E, sinE, cosE via linear interp 2D grid.

    This function precomputes the same E, sin(E), cos(E) grids and their derivatives.
    The returned function is designed to efficiently handle arrays of inputs
    using `jax.vmap`. It accepts an array of mean anomalies `M_vals` (shape
    (num_planets, num_times)) and an array of eccentricities `e_array` (shape
    (num_planets,)) and returns arrays for E, sin(E), and cos(E), each with
    shape (num_planets, num_times). This is highly efficient for processing
    multiple orbits or time series simultaneously.

    Args:
        n_e:
            Number of eccentricity steps in the grid (0 <= e < 1). Default is 500.
        n_M:
            Number of mean anomaly steps in the grid (0 <= M < 2pi). Default is 3600.

    Returns:
        A JIT-compiled, vectorized function `vectorized_lookup(M_vals, e_array)`
        that takes array inputs and returns a tuple `(E_array, sinE_array, cosE_array)`
        containing arrays of interpolated eccentric anomalies and their sines/cosines.
    """
    # Setup
    e_grid, M_grid, dM, inv_dM, dM_int, de, inv_de = _setup(n_e, n_M)
    E_grid, sinE_grid, cosE_grid = _E_grids(e_grid, M_grid)
    dE_dind_grid, dsinE_dind, dcosE_dind = _d_dind_grids(
        e_grid, E_grid, sinE_grid, cosE_grid, dM
    )

    def _lookup_scalar(M_scalar, e_scalar):
        """Performs lookup and interpolation for a single M and e."""
        p = _grid_lin_params(M_scalar, e_scalar, inv_dM, inv_de, dM_int)
        return (
            _lin(E_grid, dE_dind_grid, *p),
            _lin(sinE_grid, dsinE_dind, *p),
            _lin(cosE_grid, dcosE_dind, *p),
        )

    return _jaxify_scalar_func(_lookup_scalar)


def grid_E_lin(n_e=1024, n_M=4096):
    """Creates vectorized JIT func for E via linear interp 2D grid."""
    # Setup
    e_grid, M_grid, dM, inv_dM, dM_int, de, inv_de = _setup(n_e, n_M)
    E_grid, sinE_grid, cosE_grid = _E_grids(e_grid, M_grid)
    dE_dind_grid, *_ = _d_dind_grids(e_grid, E_grid, sinE_grid, cosE_grid, dM)

    def _lookup_scalar(M_scalar, e_scalar):
        """Performs lookup and interpolation for a single M and e."""
        p = _grid_lin_params(M_scalar, e_scalar, inv_dM, inv_de, dM_int)
        return _lin(E_grid, dE_dind_grid, *p)

    return _jaxify_scalar_func(_lookup_scalar)


def grid_trig_lin(n_e=1024, n_M=4096):
    """Creates vectorized JIT func for sinE, cosE via linear interp 2D grid."""
    # Setup
    e_grid, M_grid, dM, inv_dM, dM_int, de, inv_de = _setup(n_e, n_M)
    E_grid, sinE_grid, cosE_grid = _E_grids(e_grid, M_grid)
    _, dsinE_dind, dcosE_dind = _d_dind_grids(e_grid, E_grid, sinE_grid, cosE_grid, dM)

    def _lookup_scalar(M_scalar, e_scalar):
        """Performs lookup and interpolation for a single M and e."""
        p = _grid_lin_params(M_scalar, e_scalar, inv_dM, inv_de, dM_int)
        return _lin(sinE_grid, dsinE_dind, *p), _lin(cosE_grid, dcosE_dind, *p)

    return _jaxify_scalar_func(_lookup_scalar)


# Bilinear interpolation functions
def _ind_bilin(scalar, inv_d, n):
    """Returns the indices and fractional difference for bilinear interpolation."""
    i0, di = _ind(scalar, inv_d)
    i0 = i0 % n
    i1 = (i0 + 1) % n
    return i0, i1, di


def _grid_bilin_params(M_scalar, e_scalar, inv_dM, inv_de, dM_int):
    """Bilinear lookup for a single (M, e) pair."""
    M0, M1, dM = _ind_bilin(M_scalar, inv_dM, dM_int)
    e0, de = _ind(e_scalar, inv_de)
    e1 = e0 + 1
    return e0, e1, M0, M1, de, dM


# Extra functions for periodic interpolation on E
def _align_phase(base, arr):
    """Shift each value in `arr` by ±2π s.t. it lies within π of `base`."""
    return jnp.where(
        arr - base > jnp.pi,
        arr - two_pi,
        jnp.where(base - arr > jnp.pi, arr + two_pi, arr),
    )


def _bilin(table, e0, e1, M0, M1, de, dM):
    """Bilinear interpolation for a single (M, e) pair."""
    f00 = table[e0, M0]
    f10 = table[e1, M0]
    f01 = table[e0, M1]
    f11 = table[e1, M1]
    return (
        f00 * (1 - de) * (1 - dM)
        + f10 * de * (1 - dM)
        + f01 * (1 - de) * dM
        + f11 * de * dM
    )


# Bilinear interpolation for periodic tables
def _bilin_per(table, e0, e1, M0, M1, de, dM):
    f00 = table[e0, M0]
    # align all neighbours with f00
    f10, f01, f11 = _align_phase(
        f00, jnp.stack([table[e1, M0], table[e0, M1], table[e1, M1]])
    )

    val = (
        f00 * (1 - de) * (1 - dM)
        + f10 * de * (1 - dM)
        + f01 * (1 - de) * dM
        + f11 * de * dM
    )
    # wrap result back to [0, 2π)
    return jnp.mod(val, two_pi)


def grid_E_trig_bilin(n_e=1024, n_M=4096):
    """Creates vectorized JIT func for E, sinE, cosE via bilinear 2D interp."""
    e_grid, M_grid, _, inv_dM, dM_int, _, inv_de = _setup(n_e, n_M)
    E_grid, sinE_grid, cosE_grid = _E_grids(e_grid, M_grid)

    def _lookup_scalar(M_scalar, e_scalar):
        """Bilinear lookup for a single (M, e) pair."""
        # Get all the necessary indices and fractional differences
        p = _grid_bilin_params(M_scalar, e_scalar, inv_dM, inv_de, dM_int)
        return _bilin_per(E_grid, *p), _bilin(sinE_grid, *p), _bilin(cosE_grid, *p)

    return _jaxify_scalar_func(_lookup_scalar)


def grid_E_bilin(n_e=1024, n_M=4096):
    """Creates vectorized JIT func for E via bilinear 2D interp."""
    e_grid, M_grid, _, inv_dM, dM_int, _, inv_de = _setup(n_e, n_M)
    E_grid, *_ = _E_grids(e_grid, M_grid)

    def _lookup_scalar(M_scalar, e_scalar):
        """Bilinear lookup for a single (M, e) pair."""
        # Get all the necessary indices and fractional differences
        p = _grid_bilin_params(M_scalar, e_scalar, inv_dM, inv_de, dM_int)
        return _bilin(E_grid, *p)

    return _jaxify_scalar_func(_lookup_scalar)


def grid_trig_bilin(n_e=1024, n_M=4096):
    """Creates vectorized JIT func for sinE, cosE via bilinear 2D interp."""
    e_grid, M_grid, _, inv_dM, dM_int, _, inv_de = _setup(n_e, n_M)
    _, sinE_grid, cosE_grid = _E_grids(e_grid, M_grid)

    def _lookup_scalar(M_scalar, e_scalar):
        """Bilinear lookup for a single (M, e) pair."""
        # Get all the necessary indices and fractional differences
        p = _grid_bilin_params(M_scalar, e_scalar, inv_dM, inv_de, dM_int)
        return _bilin(sinE_grid, *p), _bilin(cosE_grid, *p)

    return _jaxify_scalar_func(_lookup_scalar)
