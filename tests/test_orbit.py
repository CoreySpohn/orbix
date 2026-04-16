"""Tests for orbix.system.orbit (AbstractOrbit, KeplerianOrbit)."""

import jax.numpy as jnp
import numpy as np
import pytest


def test_abstract_orbit_cannot_instantiate():
    """AbstractOrbit is abstract and must not be directly instantiable."""
    from orbix.system.orbit import AbstractOrbit

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
    from orbix.system.orbit import KeplerianOrbit

    params = _earthlike_orbit_params()
    orbit = KeplerianOrbit(**params)

    np.testing.assert_allclose(orbit.a_AU, params["a_AU"])
    np.testing.assert_allclose(orbit.e, params["e"])
    np.testing.assert_allclose(orbit.W_rad, params["W_rad"])
    np.testing.assert_allclose(orbit.i_rad, params["i_rad"])
    np.testing.assert_allclose(orbit.w_rad, params["w_rad"])
    np.testing.assert_allclose(orbit.M0_rad, params["M0_rad"])
    np.testing.assert_allclose(orbit.t0_d, params["t0_d"])


def test_keplerian_orbit_caches_AB_matrices():
    """KeplerianOrbit computes and caches A_AU, B_AU (shape (3, K))."""
    from orbix.system.orbit import KeplerianOrbit

    orbit = KeplerianOrbit(**_earthlike_orbit_params())
    assert orbit.A_AU.shape == (3, 1)
    assert orbit.B_AU.shape == (3, 1)


def test_keplerian_orbit_AB_matches_equations_helper():
    """AB matrices match the AB_matrices_reduced helper byte-for-byte."""
    from orbix.equations.orbit import AB_matrices_reduced
    from orbix.system.orbit import KeplerianOrbit

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

    np.testing.assert_allclose(orbit.A_AU, A_expected)
    np.testing.assert_allclose(orbit.B_AU, B_expected)


def test_propagate_returns_three_arrays_with_expected_shapes():
    """Propagate returns (r_AU, phase_angle_rad, dist_AU) with expected shapes.

    Verifies the documented (K, 3, T) / (K, T) / (K, T) shape contract.
    """
    from orbix.kepler.shortcuts.grid import get_grid_solver
    from orbix.system.orbit import KeplerianOrbit

    orbit = KeplerianOrbit(**_earthlike_orbit_params())
    trig_solver = get_grid_solver(level="scalar", E=False, trig=True, jit=True)
    t_jd = jnp.linspace(0.0, 365.0, 10)

    Ms_kg = jnp.atleast_1d(1.988409870698051e30)  # 1 solar mass
    r_AU, phase_angle_rad, dist_AU = orbit.propagate(
        trig_solver,
        t_jd,
        Ms_kg=Ms_kg,
    )

    assert r_AU.shape == (1, 3, 10)
    assert phase_angle_rad.shape == (1, 10)
    assert dist_AU.shape == (1, 10)


def test_propagate_distance_matches_kepler_formula():
    """dist_AU at true anomaly 0 (periapsis) is a*(1-e)."""
    from orbix.kepler.shortcuts.grid import get_grid_solver
    from orbix.system.orbit import KeplerianOrbit

    params = _earthlike_orbit_params()
    orbit = KeplerianOrbit(**params)
    trig_solver = get_grid_solver(level="scalar", E=False, trig=True, jit=True)

    # At t == t0_d and M0_rad == 0, planet is at periapsis
    # where r = a*(1-e).
    t_jd = params["t0_d"]
    Ms_kg = jnp.atleast_1d(1.988409870698051e30)

    _, _, dist_AU = orbit.propagate(trig_solver, t_jd, Ms_kg=Ms_kg)

    expected = params["a_AU"] * (1 - params["e"])
    np.testing.assert_allclose(dist_AU.squeeze(), expected.squeeze(), rtol=1e-5)


def test_propagate_raises_without_Ms_kg_keyword():
    """Ms_kg is keyword-only; positional pass must fail."""
    from orbix.kepler.shortcuts.grid import get_grid_solver
    from orbix.system.orbit import KeplerianOrbit

    orbit = KeplerianOrbit(**_earthlike_orbit_params())
    trig_solver = get_grid_solver(level="scalar", E=False, trig=True, jit=True)
    t_jd = jnp.array([0.0])

    with pytest.raises(TypeError):
        orbit.propagate(trig_solver, t_jd, 1.988409870698051e30)
