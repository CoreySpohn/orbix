"""Tests for orbix grid-search utilities."""

import jax.numpy as jnp

from orbix.fitting.grid_search import ParamBounds
from orbix.utils.quasi_random import roberts_sequence


def test_roberts_shape_and_range():
    """Roberts sequence has correct shape and values in [0, 1)."""
    pts = roberts_sequence(1000, 3)
    assert pts.shape == (1000, 3)
    assert jnp.all((pts >= 0.0) & (pts < 1.0))


def test_roberts_low_discrepancy_1d_margin():
    """Max gap in sorted 1D projection is within 10x mean gap."""
    pts = roberts_sequence(2000, 2)
    xs = jnp.sort(pts[:, 0])
    gaps = jnp.diff(xs)
    assert float(jnp.max(gaps)) < 10.0 * float(jnp.mean(gaps))


def test_parambounds_scale_unit_cube():
    """ParamBounds.scale maps unit-cube corners and midpoint correctly."""
    b = ParamBounds(
        names=("logT", "cos_i"),
        low=jnp.array([0.0, -1.0]),
        high=jnp.array([4.0, 1.0]),
    )
    u = jnp.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    phys = b.scale(u)
    assert phys.shape == (3, 2)
    assert jnp.allclose(phys[0], jnp.array([0.0, -1.0]))
    assert jnp.allclose(phys[1], jnp.array([4.0, 1.0]))
    assert jnp.allclose(phys[2], jnp.array([2.0, 0.0]))
