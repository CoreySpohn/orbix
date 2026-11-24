"""Gradient sanity for the state-vector -> elements conversion."""

import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import GM_SUN_AU3_PER_D2, AU2m, d2s

from orbix.equations.orbit import state_vector_to_keplerian
from orbix.system import FrameOrbit

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
    # Swap the classical north/east components to preserve the 30 deg orbit.
    r = jnp.array([0.0, 1.0, 0.0])
    v_mag = 1.0
    v = jnp.array(
        [v_mag * jnp.cos(jnp.radians(30.0)), 0.0, v_mag * jnp.sin(jnp.radians(30.0))]
    )
    inc = state_vector_to_keplerian(r, v, mu)[2]
    assert jnp.isclose(inc, jnp.radians(30.0), atol=1e-12, rtol=0)
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
    # Express the same physical state in public east/north/away order;
    # the captured elements used classical north/east/away inputs.
    r = jnp.array([1000.0e3, 7000.0e3, 2000.0e3])
    v = jnp.array([6500.0, -1000.0, 3000.0])
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


def test_circular_elements_independent_of_units():
    """Truth: a constructed solar circle; 1e-12 allows unit-conversion roundoff.

    The circular convention absorbs periapsis into the mean anomaly;
    compare that phase and all orientation angles to 1e-12 radians.
    """
    mu = GM_SUN_AU3_PER_D2
    orbit = FrameOrbit.from_classical(1.0, 0.0, 0.7, 1.1, 0.0, 0.8, mu_AU3_per_d2=mu)
    r, v = orbit.at(0.0)
    for length, time in ((1.0, 1.0), (1.0, d2s), (AU2m, d2s)):
        a, e, *angles = state_vector_to_keplerian(
            r * length, v * length / time, mu * length**3 / time**2
        )
        np.testing.assert_allclose(a / length, 1.0, rtol=1e-12, atol=0)
        assert e < 1e-12
        delta = np.asarray(angles) - [0.7, 1.1, 0.0, 0.8]
        np.testing.assert_allclose(
            np.arctan2(np.sin(delta), np.cos(delta)), 0.0, atol=1e-12, rtol=0
        )


def test_face_on_inclination_endpoints():
    """Truth: classical h along +/-z; exact zero and one ULP at pi suffice.

    A radial state (no angular momentum) keeps the i = pi/2 convention.
    """
    r = jnp.array([0.0, 1.0, 0.0])
    v = jnp.array([1.0, 0.0, 0.0])
    assert state_vector_to_keplerian(r, v, 1.0)[2] == 0.0
    np.testing.assert_allclose(
        state_vector_to_keplerian(r, -v, 1.0)[2],
        np.pi,
        atol=np.spacing(np.pi),
        rtol=0,
    )
    radial = state_vector_to_keplerian(r, jnp.array([0.0, 0.5, 0.0]), 1.0)[2]
    assert radial == np.pi / 2


def test_elements_state_round_trip_degenerate_grid():
    """Truth: FrameOrbit forward states; 1e-10 relative covers x64 inversion.

    Compare vector norms so zero components need no dimensional floor.
    Circular and equatorial angle choices must preserve the physical state.
    The second mean anomaly puts a circular body 1e-6 rad past its
    reference direction (w + M = 1e-6), checking that no clipped cosine
    moves an angle near zero.
    """
    eccentricity, inclination, mean_anomaly = np.meshgrid(
        [0.0, 0.3, 0.9], [0.0, 0.7, np.pi], [0.8, 2 * np.pi - 0.6 + 1e-6]
    )
    orbit = FrameOrbit.from_classical(
        1.7,
        eccentricity.ravel(),
        inclination.ravel(),
        0.4,
        0.6,
        mean_anomaly.ravel(),
        mu_AU3_per_d2=GM_SUN_AU3_PER_D2,
    )
    r, v = orbit.at(0.0)
    elements = jax.jit(jax.vmap(state_vector_to_keplerian, in_axes=(0, 0, None)))(
        r, v, GM_SUN_AU3_PER_D2
    )
    rebuilt = FrameOrbit.from_classical(*elements, mu_AU3_per_d2=GM_SUN_AU3_PER_D2)
    for actual, expected in zip(rebuilt.at(0.0), (r, v), strict=True):
        relative = np.linalg.norm(actual - expected, axis=-1) / np.linalg.norm(
            expected, axis=-1
        )
        assert np.all(relative < 1e-10)
