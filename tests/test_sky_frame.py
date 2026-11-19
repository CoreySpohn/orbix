"""SkyFrame optional sky position."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from orbix.system.core import SkyFrame


def test_positional_construction_has_no_position():
    """Positional construction leaves the position unset."""
    frame = SkyFrame(10.0)
    assert frame.ra_deg is None
    assert frame.dec_deg is None


def test_position_is_made_into_arrays():
    """Given coordinates become arrays."""
    frame = SkyFrame(10.0, ra_deg=120.0, dec_deg=-30.0)
    assert float(frame.ra_deg) == 120.0
    assert float(frame.dec_deg) == -30.0


def test_both_or_neither():
    """One coordinate without the other is refused."""
    with pytest.raises(ValueError, match="together"):
        SkyFrame(10.0, ra_deg=1.0)
    with pytest.raises(ValueError, match="together"):
        SkyFrame(10.0, dec_deg=1.0)


def test_shape_must_match_dist_pc():
    """The position must have the shape of dist_pc."""
    with pytest.raises(ValueError, match="shape of dist_pc"):
        SkyFrame(jnp.ones(3), ra_deg=jnp.ones(2), dec_deg=jnp.ones(3))
    with pytest.raises(ValueError, match="shape of dist_pc"):
        SkyFrame(jnp.ones(3), ra_deg=1.0, dec_deg=1.0)
    frame = SkyFrame(jnp.ones(3), ra_deg=jnp.arange(3.0), dec_deg=jnp.zeros(3))
    assert frame.ra_deg.shape == (3,)


def test_pytree_leaves_with_and_without_position():
    """Unset coordinates add no pytree leaves."""
    assert len(jax.tree_util.tree_leaves(SkyFrame(10.0))) == 1
    assert len(jax.tree_util.tree_leaves(SkyFrame(10.0, ra_deg=1.0, dec_deg=2.0))) == 3


def test_vmap_with_and_without_position():
    """A batched frame maps with and without a position."""
    d = jnp.array([5.0, 10.0])
    plain = eqx.filter_vmap(lambda f: f.dist_pc * 2)(SkyFrame(d))
    assert plain.shape == (2,)

    placed = SkyFrame(d, ra_deg=jnp.array([10.0, 20.0]), dec_deg=jnp.array([1.0, 2.0]))
    out = eqx.filter_vmap(lambda f: f.ra_deg + f.dec_deg)(placed)
    assert jnp.allclose(out, jnp.array([11.0, 22.0]))
