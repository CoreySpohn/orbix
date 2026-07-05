"""Accuracy tests for the fixed-eccentricity shortcut solvers."""

import jax.numpy as jnp

from orbix.kepler.core import E_solve
from orbix.kepler.shortcuts.fixed_e import E_hermite_interp, E_lookup


def test_hermite_no_wrap_blend_in_last_cell():
    """The last grid cell must not blend E~2pi against the E~0 seam node."""
    e, n = 0.3, 2048
    interp = E_hermite_interp(e, n)
    dM = 2 * jnp.pi / n
    M = jnp.array([2 * jnp.pi - dM / 2])  # middle of the last grid cell
    truth = E_solve(M, e)[0]
    got = interp(M)[0]
    # Pre-fix: |got - truth| ~ pi. Post-fix: sub-cell interpolation error.
    assert abs(float(got - truth)) < 1e-6


def test_lookup_rounds_to_nearest():
    """E_lookup must round to the nearest node, not floor."""
    e, n = 0.2, 512
    lookup = E_lookup(e, n)
    dM = 2 * jnp.pi / n
    # Just under a node: nearest node is the one above, floor picks the one below.
    M = jnp.array([10.4 * dM * 0.999999 + 0.6 * dM])  # 10.6 * dM within fp noise
    truth = E_solve(M, e)[0]
    err = abs(float(lookup(M)[0] - truth))
    # Rounding halves the worst-case error vs flooring: must be within ~0.5 cells
    # of dE (dE/dM <= 1/(1-e)).
    assert err <= 0.55 * dM / (1 - e)
