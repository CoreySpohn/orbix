"""Gradient sanity for the state-vector -> elements conversion."""

import jax
import jax.numpy as jnp

from orbix.equations.orbit import state_vector_to_keplerian

jax.config.update("jax_enable_x64", True)


def _elements_sum(r, v, mu):
    out = state_vector_to_keplerian(r, v, mu)
    return sum(jnp.sum(o) for o in jax.tree.leaves(out))


def test_grads_finite_for_circular_equatorial_orbit():
    """Gradients are finite at a circular, equatorial orbit (both singular)."""
    mu = 1.0
    r = jnp.array([1.0, 0.0, 0.0])
    v = jnp.array([0.0, 1.0, 0.0])
    g = jax.grad(_elements_sum, argnums=(0, 1))(r, v, mu)
    assert jnp.isfinite(g[0]).all() and jnp.isfinite(g[1]).all()


def test_grads_finite_for_circular_inclined_orbit():
    """Gradients are finite at a circular but inclined (30 deg) orbit."""
    mu = 1.0
    r = jnp.array([1.0, 0.0, 0.0])
    v_mag = 1.0
    v = jnp.array(
        [0.0, v_mag * jnp.cos(jnp.radians(30.0)), v_mag * jnp.sin(jnp.radians(30.0))]
    )
    g = jax.grad(_elements_sum, argnums=(0, 1))(r, v, mu)
    assert jnp.isfinite(g[0]).all() and jnp.isfinite(g[1]).all()


def test_grads_finite_for_eccentric_equatorial_orbit():
    """Gradients are finite at an eccentric but equatorial orbit."""
    mu = 1.0
    r = jnp.array([1.0, 0.0, 0.0])
    v = jnp.array([0.0, 0.6, 0.0])
    g = jax.grad(_elements_sum, argnums=(0, 1))(r, v, mu)
    assert jnp.isfinite(g[0]).all() and jnp.isfinite(g[1]).all()


def test_grads_finite_for_position_on_z_axis():
    """Gradients are finite when r lies on the z-axis (arctan2(0,0) site)."""
    mu = 1.0
    r = jnp.array([0.0, 0.0, 1.0])
    v = jnp.array([1.0, 0.0, 0.0])
    g = jax.grad(_elements_sum, argnums=(0, 1))(r, v, mu)
    assert jnp.isfinite(g[0]).all() and jnp.isfinite(g[1]).all()


def test_values_unchanged_for_generic_orbit():
    """Element values are unchanged for a non-degenerate elliptical inclined orbit."""
    # Expected values captured from the pre-fix function at this state
    # (r in m, v in m/s, mu in m^3/s^2) before the gradient-safety edits.
    mu = 3.986004418e14
    r = jnp.array([7000.0e3, 1000.0e3, 2000.0e3])
    v = jnp.array([-1000.0, 6500.0, 3000.0])
    a, e, inc, W, w, M = state_vector_to_keplerian(r, v, mu)

    expected_a = 7088082.517471771
    expected_e = 0.10980119039762694
    expected_inc = 0.4946314976718742
    expected_W = 5.873057966638095
    expected_w = 4.879990064782114
    expected_M = 1.8084683923634315

    assert jnp.isclose(a, expected_a, rtol=1e-10)
    assert jnp.isclose(e, expected_e, rtol=1e-10)
    assert jnp.isclose(inc, expected_inc, rtol=1e-10)
    assert jnp.isclose(W, expected_W, rtol=1e-10)
    assert jnp.isclose(w, expected_w, rtol=1e-10)
    assert jnp.isclose(M, expected_M, rtol=1e-10)
