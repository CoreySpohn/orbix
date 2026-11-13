"""Tests for orbix.orbit (AbstractOrbit, KeplerianOrbit)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import G, Msun2kg, two_pi
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from orbix.equations.orbit import mean_motion, period_a, period_n, period_to_sma
from orbix.kepler.markley import solve_trig
from orbix.orbit import KeplerianOrbit


def test_abstract_orbit_cannot_instantiate():
    """AbstractOrbit is abstract and must not be directly instantiable."""
    from orbix.orbit import AbstractOrbit

    with pytest.raises(TypeError):
        AbstractOrbit()


def _earthlike_orbit_params():
    """Return dict of orbital elements for an Earth-like orbit.

    Each element is a shape (1,) array.
    """
    return dict(
        a_AU=jnp.atleast_1d(1.0),
        e=jnp.atleast_1d(0.0167),
        W_rad=jnp.atleast_1d(0.0),
        i_rad=jnp.atleast_1d(jnp.pi / 3),
        w_rad=jnp.atleast_1d(0.0),
        M0_rad=jnp.atleast_1d(0.0),
        t0_d=jnp.atleast_1d(0.0),
    )


def test_keplerian_orbit_construction():
    """KeplerianOrbit stores orbital elements as unit-suffixed arrays."""
    from orbix.orbit import KeplerianOrbit

    params = _earthlike_orbit_params()
    orbit = KeplerianOrbit(**params)

    np.testing.assert_allclose(orbit.a_AU, params["a_AU"])
    np.testing.assert_allclose(orbit.e, params["e"])
    np.testing.assert_allclose(orbit.W_rad, params["W_rad"])
    np.testing.assert_allclose(orbit.i_rad, params["i_rad"])
    np.testing.assert_allclose(orbit.w_rad, params["w_rad"])
    np.testing.assert_allclose(orbit.M0_rad, params["M0_rad"])
    np.testing.assert_allclose(orbit.t0_d, params["t0_d"])


def test_keplerian_orbit_AB_computes_matrices():
    """KeplerianOrbit._AB() computes A_AU, B_AU (shape (3, K)) on demand."""
    from orbix.orbit import KeplerianOrbit

    orbit = KeplerianOrbit(**_earthlike_orbit_params())
    A_AU, B_AU = orbit._AB()
    assert A_AU.shape == (3, 1)
    assert B_AU.shape == (3, 1)
    assert not hasattr(orbit, "A_AU")
    assert not hasattr(orbit, "B_AU")


def test_keplerian_orbit_AB_matches_equations_helper():
    """AB matrices match the AB_matrices_reduced helper byte-for-byte."""
    from orbix.equations.orbit import AB_matrices_reduced
    from orbix.orbit import KeplerianOrbit

    params = _earthlike_orbit_params()
    orbit = KeplerianOrbit(**params)

    a = params["a_AU"]
    e = params["e"]
    sqrt_1me2 = jnp.sqrt(1 - e**2)
    sini, cosi = jnp.sin(params["i_rad"]), jnp.cos(params["i_rad"])
    sinW, cosW = jnp.sin(params["W_rad"]), jnp.cos(params["W_rad"])
    sinw, cosw = jnp.sin(params["w_rad"]), jnp.cos(params["w_rad"])

    A_expected, B_expected = AB_matrices_reduced(
        a,
        sqrt_1me2,
        sini,
        cosi,
        sinW,
        cosW,
        sinw,
        cosw,
    )

    A_AU, B_AU = orbit._AB()
    np.testing.assert_allclose(A_AU, A_expected)
    np.testing.assert_allclose(B_AU, B_expected)


def test_propagate_returns_three_arrays_with_expected_shapes():
    """Propagate returns (r_AU, phase_angle_rad, dist_AU) with expected shapes.

    Verifies the documented (K, 3, T) / (K, T) / (K, T) shape contract.
    """
    from orbix.kepler.shortcuts.grid import get_grid_solver
    from orbix.orbit import KeplerianOrbit

    orbit = KeplerianOrbit(**_earthlike_orbit_params())
    trig_solver = get_grid_solver(level="scalar", E=False, trig=True, jit=True)
    time_jd = jnp.linspace(0.0, 365.0, 10)

    Ms_kg = jnp.atleast_1d(1.988409870698051e30)  # 1 solar mass
    r_AU, phase_angle_rad, dist_AU = orbit.propagate(
        trig_solver,
        time_jd,
        Ms_kg=Ms_kg,
    )

    assert r_AU.shape == (1, 3, 10)
    assert phase_angle_rad.shape == (1, 10)
    assert dist_AU.shape == (1, 10)


def _bounded(low, high):
    return st.floats(low, high, allow_nan=False, allow_infinity=False)


@jax.jit
def _propagated_invariants(a, e, inc, node, peri, mass, fractions):
    period = period_a(a, G * mass)
    orbit = KeplerianOrbit(
        a_AU=a, e=e, i_rad=inc, W_rad=node, w_rad=peri, M0_rad=0.0, t0_d=0.0
    )
    times = jnp.concatenate([jnp.array([0.0, 0.5, 1.0]), fractions]) * period
    r, v = jax.jvp(
        lambda t: orbit.propagate(solve_trig, t, Ms_kg=mass)[0][0],
        (times,),
        (jnp.ones_like(times),),
    )
    distance = orbit.propagate(solve_trig, times, Ms_kg=mass)[2][0]
    return r.T, v.T, distance


@given(
    a=_bounded(0.1, 30.0),
    e=_bounded(0.0, 0.99),
    inc=_bounded(0.0, np.pi),
    node=_bounded(-two_pi, two_pi),
    peri=_bounded(-two_pi, two_pi),
    mass=_bounded(0.1, 3.0),
    fractions=arrays(np.float64, (6,), elements=_bounded(-2.0, 2.0)),
)
def test_propagation_conserves_energy_and_angular_momentum(
    a, e, inc, node, peri, mass, fractions
):
    """Fails if propagation changes energy/angular momentum, radius, or period.

    Fixed nine-epoch shapes include periapsis, apoapsis, and a full period;
    velocities come from differentiating the public position in AU/day.
    Expected invariants use hwoutils G (AU^3/kg/day^2) and solar mass.
    """
    mu = G * mass * Msun2kg
    r, v, distance = map(
        np.asarray,
        _propagated_invariants(
            jnp.asarray(a),
            jnp.asarray(e),
            jnp.asarray(inc),
            jnp.asarray(node),
            jnp.asarray(peri),
            jnp.asarray(mass * Msun2kg),
            jnp.asarray(fractions),
        ),
    )
    radii = np.linalg.norm(r, axis=1)
    energy = np.sum(v**2, axis=1) / 2 - mu / radii
    np.testing.assert_allclose(energy, -mu / (2 * a), rtol=2e-11, atol=0)
    h = np.cross(r, v)
    expected_h = np.sqrt(mu * a * (1 - e**2)) * np.array(
        [np.sin(inc) * np.cos(node), -np.sin(inc) * np.sin(node), -np.cos(inc)]
    )
    np.testing.assert_allclose(
        h,
        np.broadcast_to(expected_h, h.shape),
        rtol=2e-11,
        atol=np.linalg.norm(expected_h) * 2e-13,
    )
    np.testing.assert_allclose(distance, radii, rtol=1e-12, atol=0)
    np.testing.assert_allclose(
        radii[:2], [a * (1 - e), a * (1 + e)], rtol=2e-12, atol=0
    )
    np.testing.assert_allclose(r[2], r[0], atol=a * 2e-12, rtol=0)
    np.testing.assert_allclose(v[2], v[0], atol=np.sqrt(mu / a) * 2e-10, rtol=0)


def test_propagate_raises_without_Ms_kg_keyword():
    """Ms_kg is keyword-only; positional pass must fail."""
    from orbix.kepler.shortcuts.grid import get_grid_solver
    from orbix.orbit import KeplerianOrbit

    orbit = KeplerianOrbit(**_earthlike_orbit_params())
    trig_solver = get_grid_solver(level="scalar", E=False, trig=True, jit=True)
    time_jd = jnp.array([0.0])

    with pytest.raises(TypeError):
        orbit.propagate(trig_solver, time_jd, 1.988409870698051e30)


def test_position_arcsec_shape_units_and_north_periapsis():
    """Pin (east, north), shape and scale; Green (1985), ch. 17."""
    from orbix.kepler.shortcuts.grid import get_grid_solver
    from orbix.orbit import KeplerianOrbit

    orbit = KeplerianOrbit(**_earthlike_orbit_params())
    trig_solver = get_grid_solver(level="scalar", E=False, trig=True, jit=True)
    time_jd = jnp.linspace(0.0, 365.0, 5)

    Ms_kg = jnp.atleast_1d(1.988409870698051e30)
    dist_pc = jnp.atleast_1d(10.0)

    ra, dec = orbit.position_arcsec(
        trig_solver,
        time_jd,
        Ms_kg=Ms_kg,
        dist_pc=dist_pc,
    )
    assert ra.shape == (1, 5)
    assert dec.shape == (1, 5)
    # Omega=w=M0=0: periapsis is due north, at a*(1-e)/distance.
    np.testing.assert_allclose(ra[0, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(dec[0, 0], 0.09833, atol=1e-12)


def test_tree_at_e_update_recomputes_AB():
    """eqx.tree_at on ``e`` recomputes AB from current elements, not a stale cache."""
    import equinox as eqx

    from orbix.kepler.shortcuts.grid import get_grid_solver
    from orbix.orbit import KeplerianOrbit

    solver = get_grid_solver(level="scalar", E=False, trig=True, jit=True)
    others = dict(
        a_AU=jnp.array([1.0]),
        W_rad=jnp.array([0.3]),
        i_rad=jnp.array([0.5]),
        w_rad=jnp.array([0.7]),
        M0_rad=jnp.array([0.2]),
        t0_d=jnp.array([0.0]),
    )
    orbit = KeplerianOrbit(e=jnp.array([0.1]), **others)
    bumped = eqx.tree_at(lambda o: o.e, orbit, jnp.array([0.4]))
    fresh = KeplerianOrbit(e=jnp.array([0.4]), **others)

    # B scales with sqrt(1 - e**2); a stale cache would keep the e=0.1 B matrix.
    A_bump, B_bump = bumped._AB()
    A_fresh, B_fresh = fresh._AB()
    _, B_old = orbit._AB()
    assert jnp.allclose(A_bump, A_fresh) and jnp.allclose(B_bump, B_fresh)
    assert not jnp.allclose(B_bump, B_old)

    # End-to-end propagation must match a freshly built orbit exactly.
    t = jnp.array([100.0])
    Ms = jnp.array([2.0e30])
    r_bump, _, _ = bumped.propagate(solver, t, Ms_kg=Ms)
    r_fresh, _, _ = fresh.propagate(solver, t, Ms_kg=Ms)
    assert jnp.allclose(r_bump, r_fresh)


def _scalar_diff_solve_trig(M, e):
    """Adapt array-shaped ``diff_solve_trig`` to the scalar-solver contract.

    Round-trips through ``jnp.atleast_1d``; a test-only shim, not a
    KeplerianOrbit change.
    """
    from orbix.kepler.core import diff_solve_trig

    sinE, cosE = diff_solve_trig(jnp.atleast_1d(M), e)
    return sinE[0], cosE[0]


def test_grad_wrt_eccentricity_is_finite_and_nonzero():
    """Gradient of separation w.r.t. eccentricity is finite and nonzero."""
    from orbix.orbit import KeplerianOrbit

    def sep(e_val):
        orbit = KeplerianOrbit(
            a_AU=jnp.array([1.0]),
            e=jnp.array([e_val]),
            W_rad=jnp.array([0.3]),
            i_rad=jnp.array([0.5]),
            w_rad=jnp.array([0.7]),
            M0_rad=jnp.array([0.2]),
            t0_d=jnp.array([0.0]),
        )
        r, _, _ = orbit.propagate(
            _scalar_diff_solve_trig, jnp.array([100.0]), Ms_kg=jnp.array([2.0e30])
        )
        return jnp.sum(r**2)

    g = jax.grad(sep)(0.1)
    assert jnp.isfinite(g) and g != 0.0


def test_phase_angle_grad_finite_at_conjunction():
    """Phase-angle gradient w.r.t. inclination is finite at conjunction."""
    from orbix.orbit import KeplerianOrbit

    def phase_sum(i_val):
        orbit = KeplerianOrbit(
            a_AU=jnp.array([1.0]),
            e=jnp.array([0.0]),
            W_rad=jnp.array([0.0]),
            i_rad=jnp.array([i_val]),
            w_rad=jnp.array([0.0]),
            M0_rad=jnp.array([0.0]),
            t0_d=jnp.array([0.0]),
        )
        _, beta, _ = orbit.propagate(
            _scalar_diff_solve_trig, jnp.array([0.0]), Ms_kg=jnp.array([2.0e30])
        )
        return jnp.sum(beta)

    g = jax.grad(phase_sum)(jnp.pi / 2)  # edge-on at conjunction: cosbeta = +-1
    assert jnp.isfinite(g)


def test_mismatched_leading_axes_raise_at_construction():
    """Mismatched leading-axis shapes raise ValueError at construction."""
    from orbix.orbit import KeplerianOrbit

    with pytest.raises(ValueError):
        KeplerianOrbit(
            a_AU=jnp.ones(2),
            e=jnp.ones(3) * 0.1,
            W_rad=jnp.zeros(2),
            i_rad=jnp.zeros(2),
            w_rad=jnp.zeros(2),
            M0_rad=jnp.zeros(2),
            t0_d=jnp.zeros(2),
        )


def test_separation_and_position_at_circular_quadratures():
    """Pin oriented quadratures and separation; Green (1985), ch. 17."""
    from hwoutils.constants import Msun2kg

    from orbix.kepler.markley import solve_trig
    from orbix.orbit import KeplerianOrbit

    orbit = KeplerianOrbit.from_period(
        T_d=400.0,
        e=0.0,
        cos_i=0.5,
        W_rad=0.0,
        cos_w=1.0,
        sin_w=0.0,
        tp_d=0.0,
        Ms_kg=Msun2kg,
    )
    # Choose distance so that the angular semi-major axis is 0.1 arcsec.
    stellar = dict(Ms_kg=Msun2kg, dist_pc=10.0 * orbit.a_AU)
    time_jd = jnp.array([0.0, 100.0, 200.0, 300.0])
    ra, dec = orbit.position_arcsec(solve_trig, time_jd, **stellar)
    sep = orbit.separation_arcsec(solve_trig, time_jd, **stellar)
    assert ra.shape == dec.shape == sep.shape == (1, 4)
    np.testing.assert_allclose(ra[0], [0.0, 0.05, 0.0, -0.05], atol=1e-12)
    np.testing.assert_allclose(dec[0], [0.1, 0.0, -0.1, 0.0], atol=1e-12)
    np.testing.assert_allclose(sep[0], [0.1, 0.05, 0.1, 0.05], atol=1e-12)


def test_default_trig_solver_matches_explicit_grid_solver():
    """propagate(time_jd=...) with no solver matches the explicit grid solver."""
    from orbix.kepler.shortcuts.grid import get_grid_solver
    from orbix.orbit import KeplerianOrbit

    orbit = KeplerianOrbit(**_earthlike_orbit_params())
    time_jd = jnp.linspace(0.0, 365.0, 7)
    Ms_kg = jnp.atleast_1d(1.988409870698051e30)

    explicit = get_grid_solver(level="scalar", E=False, trig=True, jit=True)
    r_default, beta_default, d_default = orbit.propagate(time_jd=time_jd, Ms_kg=Ms_kg)
    r_explicit, beta_explicit, d_explicit = orbit.propagate(
        explicit, time_jd, Ms_kg=Ms_kg
    )

    np.testing.assert_array_equal(r_default, r_explicit)
    np.testing.assert_array_equal(beta_default, beta_explicit)
    np.testing.assert_array_equal(d_default, d_explicit)


def test_default_solver_on_position_and_separation():
    """position_arcsec/separation_arcsec accept the default-solver path."""
    from orbix.orbit import KeplerianOrbit

    orbit = KeplerianOrbit(**_earthlike_orbit_params())
    time_jd = jnp.linspace(0.0, 365.0, 5)
    Ms_kg = jnp.atleast_1d(1.988409870698051e30)

    ra, dec = orbit.position_arcsec(time_jd=time_jd, Ms_kg=Ms_kg, dist_pc=10.0)
    sep = orbit.separation_arcsec(time_jd=time_jd, Ms_kg=Ms_kg, dist_pc=10.0)
    assert ra.shape == (1, 5)
    np.testing.assert_allclose(sep, jnp.sqrt(ra**2 + dec**2), rtol=1e-10)


def test_propagate_missing_times_raises():
    """Omitting time_jd raises a TypeError that names the argument."""
    from orbix.orbit import KeplerianOrbit

    orbit = KeplerianOrbit(**_earthlike_orbit_params())
    with pytest.raises(TypeError, match="time_jd"):
        orbit.propagate(Ms_kg=jnp.atleast_1d(2.0e30))


def test_propagate_times_passed_as_solver_raises():
    """Passing a time array where the solver goes names the fix."""
    from orbix.orbit import KeplerianOrbit

    orbit = KeplerianOrbit(**_earthlike_orbit_params())
    with pytest.raises(TypeError, match="time_jd"):
        orbit.propagate(jnp.linspace(0.0, 10.0, 3), Ms_kg=jnp.atleast_1d(2.0e30))


@jax.jit
def _period_construction(period, e, inc, node, peri, epoch, mass):
    orbit = KeplerianOrbit.from_period(
        period, e, jnp.cos(inc), node, jnp.cos(peri), jnp.sin(peri), epoch, Ms_kg=mass
    )
    direct = KeplerianOrbit(
        a_AU=period_to_sma(period, mass),
        e=e,
        i_rad=inc,
        W_rad=node,
        w_rad=peri,
        M0_rad=0.0,
        t0_d=epoch,
    )
    times = epoch + jnp.linspace(0.0, 1.0, 9) * period
    r_a, _, d_a = orbit.propagate(solve_trig, times, Ms_kg=mass)
    r_b, _, _ = direct.propagate(solve_trig, times, Ms_kg=mass)
    a = orbit.a_AU[0]
    return (
        jnp.array(
            [
                a,
                orbit.e[0],
                orbit.i_rad[0],
                orbit.W_rad[0],
                orbit.w_rad[0],
                orbit.M0_rad[0],
                orbit.t0_d[0],
            ]
        ),
        r_a,
        r_b,
        d_a,
        period_a(a, G * mass),
        period_n(mean_motion(a, G * mass)),
    )


@given(
    period=_bounded(10.0, 10000.0),
    e=_bounded(0.0, 0.99),
    inc=_bounded(0.05, np.pi - 0.05),
    node=_bounded(-two_pi, two_pi),
    peri=_bounded(-np.pi, np.pi),
    epoch=_bounded(-1000.0, 1000.0),
    mass=_bounded(0.1, 3.0),
)
def test_from_period_obeys_third_law_and_matches_direct_construction(
    period, e, inc, node, peri, epoch, mass
):
    """Fails if period/mass units or from_period's angle/epoch mapping drift."""
    elements, r_a, r_b, distances, from_a, from_n = _period_construction(
        *map(jnp.asarray, (period, e, inc, node, peri, epoch, mass * Msun2kg))
    )
    a = float(elements[0])
    np.testing.assert_allclose(
        a**3, G * mass * Msun2kg * (period / two_pi) ** 2, rtol=3e-15
    )
    np.testing.assert_allclose([from_a, from_n], period, rtol=3e-15)
    np.testing.assert_allclose(
        elements[1:], [e, inc, node, peri, 0.0, epoch], atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(r_a, r_b, atol=a * 1e-12, rtol=1e-12)
    np.testing.assert_allclose(distances[0, 0], a * (1 - e), rtol=1e-12)


def test_from_period_broadcasts_batched_samples():
    """A (K,) batch of posterior-style draws becomes a (K,)-batched orbit."""
    from orbix.orbit import KeplerianOrbit

    K = 4
    orbit = KeplerianOrbit.from_period(
        jnp.linspace(300.0, 500.0, K),
        0.1,
        0.4,
        1.0,
        0.6,
        0.8,
        2460000.0,
        Ms_kg=2.0e30,
    )
    assert orbit.a_AU.shape == (K,)
    assert orbit.t0_d.shape == (K,)
    r_AU, _, _ = orbit.propagate(time_jd=jnp.linspace(0.0, 100.0, 3), Ms_kg=2.0e30)
    assert r_AU.shape == (K, 3, 3)
