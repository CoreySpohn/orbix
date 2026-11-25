"""Gradient sanity for the state-vector -> elements conversion."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import GM_SUN_AU3_PER_D2, AU2m, d2s

from orbix.equations.orbit import state_vector_to_keplerian
from orbix.system import FrameOrbit


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
    Equatorial nodes are exactly zero by the documented element convention.
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
    equatorial = (inclination.ravel() == 0.0) | (inclination.ravel() == np.pi)
    np.testing.assert_array_equal(elements[3][equatorial], 0.0)


@pytest.mark.float64_only(
    reason="float64 JAX reference for the float32 precision comparison"
)
@pytest.mark.parametrize("phase", [3 * np.pi / 2, 0.8])
@pytest.mark.parametrize("length", [AU2m, 1.0], ids=["si", "au_seconds"])
def test_float32_circular_state_and_gradients(phase, length):
    """Truth: a float64 solar circle converted to SI or AU/seconds.

    Relative vector and angular tolerances of 1e-5 allow float32 rounding
    through inversion. The circular phase is w + M since rounding can
    produce a small eccentricity and a nonzero periapsis. Every reverse
    derivative must be finite, with no numerical tolerance.
    """
    mu = GM_SUN_AU3_PER_D2
    orbit = FrameOrbit.from_classical(1.0, 0.0, 0.7, 1.1, 0.0, phase, mu_AU3_per_d2=mu)
    r, v = orbit.at(0.0)
    r = jnp.asarray(r * length, dtype=jnp.float32)
    v = jnp.asarray(v * length / d2s, dtype=jnp.float32)
    mu_input = jnp.asarray(mu * length**3 / d2s**2, dtype=jnp.float32)
    elements = state_vector_to_keplerian(r, v, mu_input)
    assert all(x.dtype == jnp.float32 for x in elements)
    a, e, inc, node, periapsis, mean = (x.astype(jnp.float64) for x in elements)
    rebuilt = FrameOrbit.from_classical(
        a / length, e, inc, node, periapsis, mean, mu_AU3_per_d2=mu
    )
    for actual, expected in zip(
        rebuilt.at(0.0),
        (r.astype(jnp.float64) / length, v.astype(jnp.float64) * d2s / length),
        strict=True,
    ):
        relative = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
        assert relative < 1e-5
    delta = periapsis + mean - phase
    assert abs(np.arctan2(np.sin(delta), np.cos(delta))) < 1e-5
    derivative = jax.jacrev(state_vector_to_keplerian, argnums=(0, 1))
    for evaluate in (derivative, jax.jit(derivative)):
        assert all(
            np.isfinite(x).all() for x in jax.tree.leaves(evaluate(r, v, mu_input))
        )


@pytest.mark.float64_only(
    reason="float64 JAX reference for the float32 precision comparison"
)
@pytest.mark.parametrize("inclination", [1e-16, np.pi], ids=["prograde", "retrograde"])
def test_float32_near_equatorial_gradients_finite(inclination):
    """Truth: a float64 solar ellipse in AU/seconds, cast to float32.

    The float64 construction leaves out-of-plane components near 1e-23,
    whose squares underflow float32. The 1e-5 relative state tolerance
    allows float32 rounding through inversion. Every reverse derivative
    must be finite, with no numerical tolerance.
    """
    mu = GM_SUN_AU3_PER_D2
    orbit = FrameOrbit.from_classical(
        1.0, 0.3, inclination, 1.1, 0.4, 0.3, mu_AU3_per_d2=mu
    )
    r, v = orbit.at(0.0)
    r = jnp.asarray(r, dtype=jnp.float32)
    v = jnp.asarray(v / d2s, dtype=jnp.float32)
    mu_input = jnp.asarray(mu / d2s**2, dtype=jnp.float32)
    assert 0.0 < abs(float(v[2])) < 1e-19
    elements = state_vector_to_keplerian(r, v, mu_input)
    elements = [x.astype(jnp.float64) for x in elements]
    rebuilt = FrameOrbit.from_classical(*elements, mu_AU3_per_d2=mu).at(0.0)[0]
    expected = r.astype(jnp.float64)
    assert np.linalg.norm(rebuilt - expected) / np.linalg.norm(expected) < 1e-5
    derivative = jax.jacrev(state_vector_to_keplerian, argnums=(0, 1))
    for evaluate in (derivative, jax.jit(derivative)):
        assert all(
            np.isfinite(x).all() for x in jax.tree.leaves(evaluate(r, v, mu_input))
        )


@pytest.mark.float64_only(
    reason="float64 JAX reference for the float32 precision comparison"
)
@pytest.mark.parametrize("eccentricity", [0.0, 0.3], ids=["circular", "eccentric"])
@pytest.mark.parametrize("phase", [3 * np.pi / 2, 0.8])
def test_float32_si_derivatives_match_float64(eccentricity, phase):
    """Truth: the float64 Jacobian of the same float32 SI state.

    Each derivative is made dimensionless by multiplying by the norm of
    its input (a logarithmic derivative). Per element, the float32 result
    must be within 1e-3 of the float64 one, relative to the float64 norm
    over r, v and mu, which allows float32 rounding amplified by the
    conversion's condition number. A product whose square leaves the
    float32 range would instead drop a derivative term entirely. The
    circular case compares a, i and W only: rounding leaves e near 1e-7,
    where e, w and M have derivatives near 1/e that float32 cannot resolve.
    """
    mu = GM_SUN_AU3_PER_D2
    orbit = FrameOrbit.from_classical(
        1.0, eccentricity, 0.7, 1.1, 0.4, phase, mu_AU3_per_d2=mu
    )
    r, v = orbit.at(0.0)
    inputs = (
        jnp.asarray(r * AU2m, dtype=jnp.float32),
        jnp.asarray(v * AU2m / d2s, dtype=jnp.float32),
        jnp.asarray(mu * AU2m**3 / d2s**2, dtype=jnp.float32),
    )
    derivative = jax.jacrev(state_vector_to_keplerian, argnums=(0, 1, 2))
    single = derivative(*inputs)
    double = derivative(*(x.astype(jnp.float64) for x in inputs))
    if eccentricity == 0.0:
        single, double = (tuple(x[i] for i in (0, 2, 3)) for x in (single, double))
    scales = [np.linalg.norm(x.astype(jnp.float64)) for x in inputs]
    for got, expected in zip(single, double, strict=True):
        assert all(x.dtype == jnp.float32 for x in got)
        error = np.sqrt(
            sum(
                np.sum((np.asarray(g, np.float64) - x) ** 2) * s**2
                for g, x, s in zip(got, expected, scales, strict=True)
            )
        )
        norm = np.sqrt(
            sum(
                np.sum(np.asarray(x) ** 2) * s**2
                for x, s in zip(expected, scales, strict=True)
            )
        )
        assert error <= 1e-3 * norm


def test_float32_si_exact_circle_takes_circular_branch():
    """Truth: an exactly circular SI-scale orbit at argument of latitude 3*pi/2.

    Powers of two make r, v and mu = |r| |v|^2 exact in float32, so the
    eccentricity is exactly zero and the circular inclined branch runs. Its
    magnitudes (|r| = 2**37 m, |h| = 2**52 m^2/s) match a 1 AU solar orbit.
    The phase tolerance of 1e-6 rad is a few float32 ulps of 3*pi/2;
    derivatives must be finite, with no numerical tolerance.
    """
    length, speed = 2.0**37, 2.0**15
    r = jnp.array([0.0, 0.0, -length], dtype=jnp.float32)
    v = jnp.array([speed, 0.0, 0.0], dtype=jnp.float32)
    mu = jnp.asarray(length * speed**2, dtype=jnp.float32)
    a, e, inc, _, periapsis, mean = state_vector_to_keplerian(r, v, mu)
    assert float(e) == 0.0 and float(periapsis) == 0.0
    np.testing.assert_allclose([float(a), float(inc)], [length, np.pi / 2], rtol=1e-6)
    assert abs(float(mean) - 3 * np.pi / 2) < 1e-6
    derivative = jax.jacrev(state_vector_to_keplerian, argnums=(0, 1, 2))
    for evaluate in (derivative, jax.jit(derivative)):
        assert all(np.isfinite(x).all() for x in jax.tree.leaves(evaluate(r, v, mu)))


@pytest.mark.float64_only(
    reason="float64 JAX reference for the float32 precision comparison"
)
def test_float32_periapsis_derivative_near_node_matches_float64():
    """Truth: float64 central differences on the same float32 AU/s states.

    Periapses within 1e-3 rad of the ascending node put the periapsis
    cosine within a few float32 ulps of one, where recovering the angle
    from the cosine alone loses the derivative. The derivative of the
    periapsis with respect to velocity is compared per state, relative to
    its float64 norm. The 1e-4 tolerance covers float32 rounding through
    the conversion (5.3e-7 measured); the float64 differences, with a
    step of 1e-6 of the speed, contribute about 1e-10. Recovering the
    angles from their cosines with arccos fails at 3.5e-2. The rebuilt state
    must match within 1e-5 relative, as in the other float32 tests.
    """
    mu = GM_SUN_AU3_PER_D2
    periapsis, mean_anomaly = np.meshgrid(
        [-1e-3, -1e-4, 0.0, 1e-4, 1e-3], [3 * np.pi / 2, 4.5, 4.85]
    )
    orbit = FrameOrbit.from_classical(
        1.0, 0.3, 0.7, 0.6, periapsis.ravel(), mean_anomaly.ravel(), mu_AU3_per_d2=mu
    )
    r, v = orbit.at(0.0)
    r = jnp.asarray(r, dtype=jnp.float32)
    v = jnp.asarray(v / d2s, dtype=jnp.float32)
    mu_input = jnp.asarray(mu / d2s**2, dtype=jnp.float32)

    def periapsis_of(r, v, mu):
        return state_vector_to_keplerian(r, v, mu)[4]

    batched = jax.jit(jax.vmap(periapsis_of, in_axes=(0, 0, None)))
    gradient = jax.jit(jax.vmap(jax.grad(periapsis_of, argnums=1), (0, 0, None)))
    single = gradient(r, v, mu_input)
    assert single.dtype == jnp.float32
    single = np.asarray(single, dtype=np.float64)

    r64, v64 = r.astype(jnp.float64), v.astype(jnp.float64)
    mu64 = mu_input.astype(jnp.float64)
    step = 1e-6 * np.linalg.norm(np.asarray(v64), axis=-1)
    expected = np.empty_like(single)
    for axis in range(3):
        offset = np.zeros_like(single)
        offset[:, axis] = step
        change = batched(r64, v64 + offset, mu64) - batched(r64, v64 - offset, mu64)
        change = np.arctan2(np.sin(change), np.cos(change))
        expected[:, axis] = change / (2 * step)
    relative = np.linalg.norm(single - expected, axis=-1) / np.linalg.norm(
        expected, axis=-1
    )
    assert np.all(relative < 1e-4)

    elements = jax.jit(jax.vmap(state_vector_to_keplerian, in_axes=(0, 0, None)))(
        r, v, mu_input
    )
    elements = [x.astype(jnp.float64) for x in elements]
    rebuilt = FrameOrbit.from_classical(*elements, mu_AU3_per_d2=mu).at(0.0)[0]
    expected_r = r.astype(jnp.float64)
    error = np.linalg.norm(rebuilt - expected_r, axis=-1) / np.linalg.norm(
        expected_r, axis=-1
    )
    assert np.all(error < 1e-5)
