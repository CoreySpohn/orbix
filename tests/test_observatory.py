"""Tests for orbix.observatory module."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import J2000_JD, two_pi
from hwoutils.epoch import Epoch
from hwoutils.testing import ulps
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from orbix.observatory import (
    ObservatoryL2Halo,
    ecliptic_pointing,
    is_observable,
    sun_ecliptic_longitude_deg,
)
from orbix.observatory.solar_system import (
    OBLIQUITY_J2000_DEG,
    _rot1,
    _rot3,
    earth_position_ecliptic,
    eclip2equat,
    equat2eclip,
    equatorial_to_ecliptic_deg,
    planet_position_ecliptic,
)

_J2000 = Epoch(J2000_JD, scale="tt")
# MJD 60575.25: the bundled halo-orbit equinox
_EQUINOX = Epoch(2400000.5, 60575.25, scale="tt")


@jax.jit
def _frame_round_trips(vectors, angles):
    """Keep scalar/batch transform shapes and matrix shapes constant."""
    forward = equat2eclip(vectors)
    backward = eclip2equat(vectors)
    return (
        eclip2equat(forward),
        equat2eclip(backward),
        eclip2equat(equat2eclip(vectors[0])),
        equat2eclip(jnp.eye(3)),
        eclip2equat(jnp.eye(3)),
        _rot1(angles[0]),
        _rot1(-angles[0]),
        _rot3(angles[1]),
        _rot3(-angles[1]),
        forward,
    )


# ============================================================================
# Solar system ephemerides
# ============================================================================


class TestSolarSystemEphemerides:
    """Test Vallado ephemerides against expected values."""

    def test_obliquity_j2000(self):
        """The J2000 mean obliquity is 23.439 deg."""
        assert jnp.isclose(OBLIQUITY_J2000_DEG, 23.439279, atol=1e-4)

    def test_earth_distance_j2000(self):
        """Earth should be ~1 AU from Sun at J2000."""
        r = earth_position_ecliptic(_J2000, 0.0)
        dist = jnp.linalg.norm(r)
        assert jnp.isclose(dist, 1.0, atol=0.02)

    def test_earth_in_ecliptic_plane(self):
        """Earth z-component in ecliptic should be ~0."""
        r = earth_position_ecliptic(_J2000, 0.0)
        assert jnp.abs(r[2]) < 1e-3

    def test_mars_distance(self):
        """Mars should be 1.3-1.7 AU from Sun."""
        r = planet_position_ecliptic("Mars", _J2000, 0.0)
        dist = jnp.linalg.norm(r)
        assert 1.3 < float(dist) < 1.7

    def test_jupiter_distance(self):
        """Jupiter should be ~5.2 AU from Sun."""
        r = planet_position_ecliptic("Jupiter", _J2000, 0.0)
        dist = jnp.linalg.norm(r)
        assert 4.9 < float(dist) < 5.5

    @example(
        vectors=np.array([[1.0, 0.5, 0.3]] * 4),
        angles=np.array([0.0, np.pi]),
    )
    @given(
        vectors=arrays(
            np.float64,
            (4, 3),
            elements=st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
        ),
        angles=arrays(
            np.float64,
            (2,),
            elements=st.floats(-two_pi, two_pi, allow_nan=False, allow_infinity=False),
        ),
    )
    def test_equat_eclip_roundtrip(self, vectors, angles):
        """Two inverse planar rotations recover the vector and its norm.

        Each rotation rounds its sine/cosine, two products and their sum.
        Twelve operations cover input casting, both rotations and the norm.
        Six unit-scale operations cover the orthogonality dot or the
        determinant of the embedded 2-by-2 rotation. Below the normal range
        the device can flush a vector component to zero.
        """
        first, second, scalar, forward, backward, r1, ir1, r3, ir3, rotated = map(
            np.asarray,
            _frame_round_trips(jnp.asarray(vectors), jnp.asarray(angles)),
        )
        unit = np.ones((), dtype=first.dtype)
        ortho_tol = float(ulps(6, unit))
        vector_tol = float(ulps(12, unit)) * np.hypot.reduce(
            vectors, axis=1
        ).max() + 4 * float(np.finfo(first.dtype).tiny)
        np.testing.assert_allclose(first, vectors, rtol=0, atol=vector_tol)
        np.testing.assert_allclose(second, vectors, rtol=0, atol=vector_tol)
        np.testing.assert_allclose(scalar, vectors[0], rtol=0, atol=vector_tol)
        for matrix, inverse in ((forward, backward), (r1, ir1), (r3, ir3)):
            np.testing.assert_allclose(
                matrix @ inverse, np.eye(3), atol=ortho_tol, rtol=0
            )
            np.testing.assert_allclose(
                matrix.T @ matrix, np.eye(3), atol=ortho_tol, rtol=0
            )
            np.testing.assert_allclose(np.linalg.det(matrix), 1, atol=ortho_tol, rtol=0)
        np.testing.assert_allclose(
            np.hypot.reduce(rotated.astype(np.float64), axis=1),
            np.hypot.reduce(vectors, axis=1),
            rtol=0,
            atol=vector_tol,
        )

    def test_jit_planet_position(self):
        """Planet position should be JIT-compilable."""
        f = jax.jit(lambda t: planet_position_ecliptic("Earth", _J2000, t))
        r = f(0.0)
        assert r.shape == (3,)
        assert jnp.isfinite(r).all()

    def test_planet_position_ecliptic_scalar_time_ok(self):
        """A scalar time_d works."""
        r = planet_position_ecliptic("Earth", _J2000, 0.0)
        assert r.shape == (3,)

    def test_planet_position_ecliptic_array_time_raises(self):
        """Array time_d should raise instead of silently returning wrong results."""
        with pytest.raises(ValueError):
            planet_position_ecliptic("Earth", _J2000, jnp.array([0.0, 1.0]))

    def test_equatorial_to_ecliptic_deg(self):
        """The ecliptic pole is at RA 270 deg, Dec 90 deg minus the obliquity."""
        pole_dec = 90.0 - OBLIQUITY_J2000_DEG
        _, beta = equatorial_to_ecliptic_deg(270.0, pole_dec)
        assert jnp.isclose(beta, 90.0, atol=1e-9)
        lon, beta = equatorial_to_ecliptic_deg(0.0, 0.0)
        assert jnp.isclose(lon, 0.0, atol=1e-9) and jnp.isclose(beta, 0.0, atol=1e-9)
        lon, beta = equatorial_to_ecliptic_deg(90.0, 0.0)
        assert jnp.isclose(lon, 90.0, atol=1e-9)
        assert jnp.isclose(beta, -OBLIQUITY_J2000_DEG, atol=1e-9)


# ============================================================================
# Observatory L2 halo orbit
# ============================================================================


class TestObservatoryL2Halo:
    """Test L2 halo orbit module."""

    @pytest.fixture
    def obs(self):
        """Default L2 halo observatory fixture."""
        return ObservatoryL2Halo.from_default()

    def test_position_distance(self, obs):
        """Observatory should be ~1.01 AU from Sun (near L2)."""
        pos = obs.position_ecliptic(_J2000, 0.0)
        dist = jnp.linalg.norm(pos)
        assert 0.98 < float(dist) < 1.05

    def test_position_3d(self, obs):
        """Position should be a 3-vector."""
        pos = obs.position_ecliptic(_J2000, 0.0)
        assert pos.shape == (3,)

    def test_period(self, obs):
        """Halo period should be ~0.49 years (~180 days)."""
        assert 0.4 < obs.period_yr < 0.6

    def test_solar_elongation_range(self, obs):
        """Solar elongation is in [0, 180] deg."""
        angle = obs.solar_elongation_deg(0.0, 30.0, _J2000, 0.0)
        assert 0.0 <= float(angle) <= 180.0

    def test_jit_position(self, obs):
        """Position should work with eqx.filter_jit."""
        f = eqx.filter_jit(obs.position_ecliptic)
        pos = f(_J2000, 0.0)
        assert jnp.isfinite(pos).all()

    def test_vmap_over_time(self, obs):
        """Vmap over multiple times should work."""
        times = jnp.linspace(0.0, 365.0, 20)
        positions = jax.vmap(lambda t: obs.position_ecliptic(_J2000, t))(times)
        assert positions.shape == (20, 3)
        assert jnp.isfinite(positions).all()

    def test_vmap_over_targets(self, obs):
        """Vmap over multiple targets for solar elongation."""
        ras = jnp.linspace(0, 360.0, 10)
        decs = jnp.ones(10) * 30.0

        def get_angle(ra, dec):
            return obs.solar_elongation_deg(ra, dec, _J2000, 0.0)

        angles = jax.vmap(get_angle)(ras, decs)
        assert angles.shape == (10,)
        assert jnp.isfinite(angles).all()

    def test_gradient_solar_elongation(self, obs):
        """Solar elongation is differentiable with respect to time."""
        grad_fn = jax.grad(lambda t: obs.solar_elongation_deg(57.0, 29.0, _J2000, t))
        assert jnp.isfinite(grad_fn(0.0))


# ============================================================================
# Year-long Sun-target geometry invariants
# ============================================================================


class TestObservatoryGeometryInvariants:
    """Invariants of the L2-halo + RA/Dec geometry over a full year.

    These tests guard against confusing the 3D solar elongation with the
    helioecliptic longitude that indexes the zodiacal-light tables.
    """

    @pytest.fixture
    def obs(self):
        """Default L2 halo observatory at the bundled equinox."""
        return ObservatoryL2Halo.from_default()

    @pytest.fixture
    def time_d(self):
        """Year-long day sweep from the bundled equinox."""
        return jnp.linspace(0.0, 365.25, 73)

    def test_ecliptic_latitude_constant_for_fixed_target(self, time_d):
        """For a sky-fixed target the ecliptic latitude does not change in time."""
        lats, _ = ecliptic_pointing(0.0, 60.0, _EQUINOX, time_d)
        assert float(lats.max() - lats.min()) < 1e-9

    def test_ecliptic_plane_target_sweeps_full_longitude(self, time_d):
        """An ecliptic-plane target visits both anti-Sun and conjunction."""
        _, dlon = ecliptic_pointing(0.0, 0.0, _EQUINOX, time_d)
        assert float(jnp.abs(dlon).max()) > 175.0, "Did not reach anti-Sun."
        assert float(jnp.abs(dlon).min()) < 5.0, "Did not reach conjunction."

    def test_high_latitude_target_never_at_conjunction(self, obs, time_d):
        """A target at |ecliptic latitude| of 65 deg never nears the Sun.

        The minimum 3D elongation is bounded below by the ecliptic latitude.
        """
        elongs = jax.vmap(lambda t: obs.solar_elongation_deg(0.0, 80.0, _EQUINOX, t))(
            time_d
        )
        assert float(elongs.min()) > 50.0

    def test_solar_elongation_matches_helioecliptic_longitude_on_ecliptic(self, obs):
        """On the ecliptic the great-circle angle equals the longitude difference.

        The tolerance covers the halo-orbit shift of the Sun's direction,
        up to about 0.33 deg.
        """
        t = 91.0
        elong = float(obs.solar_elongation_deg(0.0, 0.0, _EQUINOX, t))
        _, dlon = ecliptic_pointing(0.0, 0.0, _EQUINOX, t)
        assert abs(elong - abs(float(dlon))) < 0.5

    def test_solar_elongation_diverges_from_helioecliptic_longitude_off_ecliptic(
        self, obs
    ):
        """At high latitude the elongation and the longitude difference differ.

        Feeding the elongation into a zodiacal-light table lookup in place of
        the helioecliptic longitude would be wrong; this pins the difference.
        """
        elong = float(obs.solar_elongation_deg(0.0, 60.0, _EQUINOX, 0.0))
        _, dlon = ecliptic_pointing(0.0, 60.0, _EQUINOX, 0.0)
        assert abs(elong - abs(float(dlon))) > 10.0

    def test_geocentric_sun_direction_close_to_halo_observer(self, obs, time_d):
        """The Sun's longitude from the telescope differs from the geocentric one.

        The halo orbit moves the telescope off the Sun-Earth line by up to
        about 0.01 AU, shifting the Sun's direction by at most about 0.35 deg.
        """
        position = jax.vmap(lambda t: obs.position_ecliptic(_EQUINOX, t))(time_d)
        halo_lon = jnp.degrees(jnp.arctan2(-position[:, 1], -position[:, 0]))
        geo_lon = sun_ecliptic_longitude_deg(_EQUINOX, time_d)
        diff = (halo_lon - geo_lon + 180.0) % 360.0 - 180.0
        assert float(jnp.abs(diff).max()) < 0.35


# ============================================================================
# Keepout
# ============================================================================


class TestKeepout:
    """Test keepout zone checks."""

    def test_sun_keepout_blocks_sunward_pointing(self):
        """A target lying along the Sun direction from the observatory is blocked."""
        # Observatory at ~(1.01, 0, 0) AU, Sun at origin.
        obs_pos = jnp.array([1.01, 0.0, 0.0])
        # Target at RA=180 deg, Dec=0: looking back toward the Sun.
        blocked = is_observable(obs_pos, 180.0, 0.0, _J2000, 0.0)
        assert not bool(blocked)

    def test_anti_solar_pointing_is_observable(self):
        """A target opposite the Sun direction from the observatory is observable."""
        obs_pos = jnp.array([1.01, 0.0, 0.0])
        # Target at RA=0, Dec=0: anti-solar direction.
        ok = is_observable(obs_pos, 0.0, 0.0, _J2000, 0.0)
        assert bool(ok)

    def test_keepout_jit(self):
        """Keepout should be JIT-compilable."""
        obs_pos = jnp.array([1.01, 0.0, 0.0])
        f = jax.jit(lambda: is_observable(obs_pos, 57.0, 29.0, _J2000, 0.0))
        result = f()
        assert isinstance(result, jax.Array)
