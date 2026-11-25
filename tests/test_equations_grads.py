"""Gradient sanity for the state-vector -> elements conversion."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import GM_SUN_AU3_PER_D2, AU2m, d2s
from hwoutils.numerics import COARSEST, eps, roundoff

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
    """Compare against recorded float64 elements at the original SI state.

    Floating-point basis: two vector norms, dot and cross products, energy,
    and angle recovery cost up to 32 operations. Finding the periapsis
    direction normalizes an eccentricity vector extracted at unit scale.
    """
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

    tol = float(roundoff(1.0, n_ops=32, dtype=a.dtype))
    np.testing.assert_allclose(a, expected_a, rtol=tol, atol=0)
    np.testing.assert_allclose(
        [e, inc, W], [expected_e, expected_inc, expected_W], atol=tol, rtol=0
    )
    np.testing.assert_allclose(
        [w, M], [expected_w, expected_M], atol=tol / expected_e, rtol=0
    )


def test_circular_elements_independent_of_units():
    """Truth: a constructed solar circle in three unit systems.

    The identifiable phase of a circle is periapsis plus mean anomaly.
    Unit conversion can leave a nonzero eccentricity at roundoff. Allow 32 operations on
    unit-scale quantities for the conversions, norms, energy and rotation.
    """
    mu = GM_SUN_AU3_PER_D2
    orbit = FrameOrbit.from_classical(1.0, 0.0, 0.7, 1.1, 0.0, 0.8, mu_AU3_per_d2=mu)
    r, v = orbit.at(0.0)
    tol = float(roundoff(1.0, n_ops=32, dtype=r.dtype))
    for length, time in ((1.0, 1.0), (1.0, d2s), (AU2m, d2s)):
        a, e, *angles = state_vector_to_keplerian(
            r * length, v * length / time, mu * length**3 / time**2
        )
        np.testing.assert_allclose(a / length, 1.0, rtol=tol, atol=0)
        assert e < tol
        inc, node, periapsis, mean = angles
        delta = np.asarray([inc, node, periapsis + mean]) - [0.7, 1.1, 0.8]
        np.testing.assert_allclose(
            np.arctan2(np.sin(delta), np.cos(delta)), 0.0, atol=tol, rtol=0
        )


@pytest.mark.parametrize(
    ("radius_offset", "speed_offset"),
    [(0, 0), (0, 2), (8, 0), (32, 0)],
    ids=["circle", "below-former-cutoff", "at-former-cutoff", "resolved"],
)
def test_polar_periapsis_and_state_round_trip(radius_offset, speed_offset):
    """A polar orbit at positive z is at periapsis whenever r * v**2 > mu.

    These two-axis inputs are exactly representable in both lanes. The
    circular gauge is w=0; every ellipse has w=pi/2 and M=0. Allow one
    angle rounding for w and 32 unit-scale roundings for a Kepler solve,
    quaternion rotation and the terminal position and velocity.
    """
    unit = eps(COARSEST)
    r = jnp.asarray([0.0, 0.0, 1.0 + radius_offset * unit])
    v = jnp.asarray([0.0, -(1.0 + speed_offset * unit), 0.0])
    for solve in (state_vector_to_keplerian, jax.jit(state_vector_to_keplerian)):
        elements = solve(r, v, 1.0)
        _, e, _, _, periapsis, mean = elements
        if radius_offset or speed_offset:
            assert e > 0.0
            np.testing.assert_allclose(
                periapsis,
                np.pi / 2,
                rtol=0,
                atol=float(roundoff(np.pi / 2, dtype=r.dtype)),
            )
            assert mean == 0.0
        else:
            assert e == 0.0
            assert periapsis == 0.0
        rebuilt = FrameOrbit.from_classical(*elements, mu_AU3_per_d2=1.0)
        for actual, expected in zip(rebuilt.at(0.0), (r, v), strict=True):
            atol = float(roundoff(jnp.linalg.norm(expected), n_ops=32))
            np.testing.assert_allclose(actual, expected, rtol=0, atol=atol)


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
        atol=float(roundoff(np.pi, n_ops=1, dtype=r.dtype)),
        rtol=0,
    )
    radial = state_vector_to_keplerian(r, jnp.array([0.0, 0.5, 0.0]), 1.0)[2]
    assert radial == np.pi / 2


def test_elements_state_round_trip_degenerate_grid():
    """Truth: FrameOrbit forward states across circular and equatorial gauges.

    Allow 64 operations for each forward/inverse chart and the conditioning
    of the Kepler conversion, dE/dM <= 1/(1-e).

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
        assert np.all(
            relative
            < float(roundoff(1.0, n_ops=64, dtype=actual.dtype))
            / (1 - eccentricity.ravel())
        )
    equatorial = (inclination.ravel() == 0.0) | (inclination.ravel() == np.pi)
    np.testing.assert_array_equal(elements[3][equatorial], 0.0)


def _host_periapsis_state(e, inc, argument):
    """One-AU orbit at periapsis, in the host float64 east/north/away basis."""
    node = 1.1
    ascending = np.array([np.sin(node), np.cos(node), 0.0])
    transverse = np.array(
        [np.cos(node) * np.cos(inc), -np.sin(node) * np.cos(inc), np.sin(inc)]
    )
    radius = (1 - e) * (ascending * np.cos(argument) + transverse * np.sin(argument))
    speed = np.sqrt(GM_SUN_AU3_PER_D2 * (1 + e) / (1 - e))
    velocity = speed * (transverse * np.cos(argument) - ascending * np.sin(argument))
    return radius, velocity


@pytest.mark.parametrize("phase", [3 * np.pi / 2, 0.8])
@pytest.mark.parametrize("length", [AU2m, 1.0], ids=["si", "au_seconds"])
def test_float32_circular_state_and_gradients(phase, length):
    """Truth: a float64 solar circle converted to SI or AU/seconds.

    Allow 64 operations for unit conversion, normalized vector inversion,
    wrapped anomaly and Cartesian reconstruction. The circular phase is
    w + M. Every reverse derivative must be finite.
    """
    mu = GM_SUN_AU3_PER_D2
    reference_r, reference_v = _host_periapsis_state(0.0, 0.7, phase)
    r = jnp.asarray(reference_r * length, dtype=jnp.float32)
    v = jnp.asarray(reference_v * length / d2s, dtype=jnp.float32)
    mu_input = jnp.asarray(mu * length**3 / d2s**2, dtype=jnp.float32)
    elements = state_vector_to_keplerian(r, v, mu_input)
    assert all(x.dtype == jnp.float32 for x in elements)
    a, e, inc, node, periapsis, mean = elements
    tol = float(roundoff(1.0, n_ops=64, dtype=a.dtype))
    rebuilt = FrameOrbit.from_classical(
        a / length, e, inc, node, periapsis, mean, mu_AU3_per_d2=mu
    )
    for actual, expected in zip(
        rebuilt.at(0.0),
        (reference_r, reference_v),
        strict=True,
    ):
        relative = np.linalg.norm(np.asarray(actual) - expected) / np.linalg.norm(
            expected
        )
        assert relative < tol
    delta = periapsis + mean - phase
    assert abs(np.arctan2(np.sin(delta), np.cos(delta))) < tol
    derivative = jax.jacrev(state_vector_to_keplerian, argnums=(0, 1))
    for evaluate in (derivative, jax.jit(derivative)):
        assert all(
            np.isfinite(x).all() for x in jax.tree.leaves(evaluate(r, v, mu_input))
        )


@pytest.mark.parametrize("inclination", [1e-16, np.pi], ids=["prograde", "retrograde"])
def test_float32_near_equatorial_gradients_finite(inclination):
    """Truth: a float64 solar ellipse in AU/seconds, cast to float32.

    The float64 construction leaves out-of-plane components near 1e-23,
    whose squares underflow float32. Allow 64 operations through the
    normalized state inverse, Kepler evaluation and frame rotation.
    Every reverse derivative must be finite.
    """
    mu = GM_SUN_AU3_PER_D2
    reference_r, reference_v = _host_periapsis_state(0.3, inclination, 0.4)
    r = jnp.asarray(reference_r, dtype=jnp.float32)
    v = jnp.asarray(reference_v / d2s, dtype=jnp.float32)
    mu_input = jnp.asarray(mu / d2s**2, dtype=jnp.float32)
    assert 0.0 < abs(float(v[2])) < 1e-19
    elements = state_vector_to_keplerian(r, v, mu_input)
    rebuilt = FrameOrbit.from_classical(*elements, mu_AU3_per_d2=mu).at(0.0)[0]
    error = np.linalg.norm(np.asarray(rebuilt) - reference_r) / np.linalg.norm(
        reference_r
    )
    assert error <= float(roundoff(1.0, n_ops=64, dtype=r.dtype))
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
