"""Cached trig tables and bilinear indexing for principal mean anomalies."""

from functools import cache

import jax
import jax.numpy as jnp
from hwoutils.constants import two_pi

from orbix.kepler._markley import _markley_pair


def _build_trig_grid(n_e, n_M, *, dtype):
    """Materialize cached arrays outside any active JAX trace."""
    with jax.ensure_compile_time_eval():
        return _cached_trig_grid(n_e, n_M, jnp.dtype(dtype))


@cache
def _cached_trig_grid(n_e, n_M, dtype):
    """Tabulate the positive half-turn, with a right neighbor at pi."""
    eccentricities = jnp.linspace(0.0, 1.0, n_e, dtype=dtype, endpoint=False)
    eccentricities = jnp.append(eccentricities, jnp.asarray(1 - 1e-6, dtype=dtype))
    means = jnp.arange(n_M // 2 + 2, dtype=dtype) * (two_pi / n_M)
    sine, cosine = _markley_pair(means, eccentricities[:, None])
    table = jnp.stack([sine, cosine])
    return jax.device_put(table), 1 / (means[1] - means[0]), n_e


def _indices_frac(signed_mean, e, inv_dM, inv_de):
    """Locate the cell for the magnitude of a signed principal phase."""
    mean_index = jnp.abs(signed_mean) * inv_dM
    e_index = e * inv_de
    mean_int = mean_index.astype(jnp.int32)
    e_int = e_index.astype(jnp.int32)
    return e_int, mean_int, e_index - e_int, mean_index - mean_int
