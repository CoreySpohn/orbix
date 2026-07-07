"""Tests for orbix.observatory module."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from orbix.observatory import (
    ObservatoryL2Halo,
    is_observable,
)
from orbix.observatory.solar_system import (
    earth_position_ecliptic,
    eclip2equat,
    equat2eclip,
    obliquity_deg,
    planet_position_ecliptic,
    radec_to_ecliptic,
)

# ============================================================================
# Solar system ephemerides
# ============================================================================


class TestSolarSystemEphemerides:
    """Test Vallado ephemerides against expected values."""

    def test_obliquity_j2000(self):
        """Obliquity at J2000 should be ~23.439 deg."""
        obl = obliquity_deg(51544.5)
        assert jnp.isclose(obl, 23.439279, atol=1e-4)

    def test_earth_distance_j2000(self):
        """Earth should be ~1 AU from Sun at J2000."""
        r = earth_position_ecliptic(51544.5)
        dist = jnp.linalg.norm(r)
        assert jnp.isclose(dist, 1.0, atol=0.02)

    def test_earth_in_ecliptic_plane(self):
        """Earth z-component in ecliptic should be ~0."""
        r = earth_position_ecliptic(51544.5)
        assert jnp.abs(r[2]) < 1e-3

    def test_mars_distance(self):
        """Mars should be 1.3-1.7 AU from Sun."""
        r = planet_position_ecliptic("Mars", 51544.5)
        dist = jnp.linalg.norm(r)
        assert 1.3 < float(dist) < 1.7

    def test_jupiter_distance(self):
        """Jupiter should be ~5.2 AU from Sun."""
        r = planet_position_ecliptic("Jupiter", 51544.5)
        dist = jnp.linalg.norm(r)
        assert 4.9 < float(dist) < 5.5

    def test_equat_eclip_roundtrip(self):
        """Equatorial <-> ecliptic conversion should roundtrip."""
        r_orig = jnp.array([1.0, 0.5, 0.3])
        mjd = 51544.5
        r_eclip = equat2eclip(r_orig, mjd)
        r_back = eclip2equat(r_eclip, mjd)
        assert jnp.allclose(r_orig, r_back, atol=1e-10)

    def test_jit_planet_position(self):
        """Planet position should be JIT-compilable."""
        f = jax.jit(lambda t: planet_position_ecliptic("Earth", t))
        r = f(51544.5)
        assert r.shape == (3,)
        assert jnp.isfinite(r).all()

    def test_planet_position_ecliptic_scalar_mjd_ok(self):
        """Scalar mjd should work as before."""
        r = planet_position_ecliptic("Earth", 51544.5)
        assert r.shape == (3,)

    def test_planet_position_ecliptic_array_mjd_raises(self):
        """Array mjd should raise instead of silently returning wrong results."""
        with pytest.raises(ValueError):
            planet_position_ecliptic("Earth", jnp.array([51544.5, 51545.5]))

    def test_radec_to_ecliptic(self):
        """RA/Dec -> ecliptic at ecliptic pole: lat should be ~90 deg."""
        # North ecliptic pole in J2000 equatorial: RA=270deg, Dec=66.56deg
        ra = jnp.radians(270.0)
        dec = jnp.radians(66.56)
        _, beta = radec_to_ecliptic(ra, dec, 51544.5)
        assert jnp.isclose(jnp.degrees(beta), 90.0, atol=1.0)


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
        pos = obs.position_ecliptic(51544.5)
        dist = jnp.linalg.norm(pos)
        assert 0.98 < float(dist) < 1.05

    def test_position_3d(self, obs):
        """Position should be a 3-vector."""
        pos = obs.position_ecliptic(51544.5)
        assert pos.shape == (3,)

    def test_period(self, obs):
        """Halo period should be ~0.49 years (~180 days)."""
        assert 0.4 < obs.period_yr < 0.6

    def test_sun_angle_range(self, obs):
        """Sun angle should be in [0, pi]."""
        angle = obs.sun_angle(51544.5, 0.0, 0.5)
        assert 0.0 <= float(angle) <= jnp.pi

    def test_jit_position(self, obs):
        """Position should work with eqx.filter_jit."""
        f = eqx.filter_jit(obs.position_ecliptic)
        pos = f(51544.5)
        assert jnp.isfinite(pos).all()

    def test_vmap_over_time(self, obs):
        """Vmap over multiple times should work."""
        mjds = jnp.linspace(51544.5, 51544.5 + 365, 20)
        positions = jax.vmap(obs.position_ecliptic)(mjds)
        assert positions.shape == (20, 3)
        assert jnp.isfinite(positions).all()

    def test_vmap_over_targets(self, obs):
        """Vmap over multiple targets for sun angle."""
        ras = jnp.linspace(0, 2 * jnp.pi, 10)
        decs = jnp.ones(10) * 0.5

        def get_angle(ra, dec):
            return obs.sun_angle(51544.5, ra, dec)

        angles = jax.vmap(get_angle)(ras, decs)
        assert angles.shape == (10,)
        assert jnp.isfinite(angles).all()

    def test_gradient_sun_angle(self, obs):
        """Sun angle should be differentiable w.r.t. time."""
        grad_fn = jax.grad(lambda t: obs.sun_angle(t, 1.0, 0.5))
        g = grad_fn(51544.5)
        assert jnp.isfinite(g)

    def test_solar_elongation_deg_range(self, obs):
        """Solar elongation should be in [0, 180]."""
        sol_elong = obs.solar_elongation_deg(51544.5, 1.0, 0.5)
        assert 0.0 <= float(sol_elong) <= 180.0

    def test_helio_ecliptic_longitude_deg_range(self, obs):
        """|lambda_target - lambda_sun| should be in [0, 180]."""
        dlam = obs.helio_ecliptic_longitude_deg(51544.5, 1.0, 0.5)
        assert 0.0 <= float(dlam) <= 180.0

    def test_ecliptic_latitude_deg_range(self, obs):
        """Ecliptic latitude should be in [-90, 90]."""
        ecl_lat = obs.ecliptic_latitude_deg(51544.5, 1.0, 0.5)
        assert -90.0 <= float(ecl_lat) <= 90.0


# ============================================================================
# Year-long Sun-target geometry invariants
# ============================================================================


class TestObservatoryGeometryInvariants:
    """Invariants of the L2-halo + RA/Dec geometry over a full year.

    These tests guard against argument-order regressions (a stale bug
    that previously caused ``ecliptic_latitude(...)`` to receive ``mjd``
    where it expected ``ra_rad``, and silently returned chaotic
    latitudes) and against confusing the 3D solar elongation with
    Leinert's helio-ecliptic longitude ``Delta_lambda_sun``.
    """

    @pytest.fixture
    def obs(self):
        """Default L2 halo observatory at the bundled equinox."""
        return ObservatoryL2Halo.from_default()

    @pytest.fixture
    def mjds(self):
        """Year-long MJD sweep starting at the bundled equinox."""
        return 60575.25 + jnp.linspace(0.0, 365.25, 73)

    def test_ecliptic_latitude_constant_for_fixed_target(self, obs, mjds):
        """For a sky-fixed target, ecl_lat must barely change over a year.

        Obliquity drift is ~0.5 arcsec/yr -- a fraction of a milli-degree.
        Any larger variation means the function is sampling something it
        shouldn't (e.g. mjd accidentally consumed as ra_rad).
        """
        # Mid-latitude target: ecl_lat ~ +53 deg.
        ra_rad = jnp.deg2rad(0.0)
        dec_rad = jnp.deg2rad(60.0)
        lats = jax.vmap(lambda m: obs.ecliptic_latitude_deg(m, ra_rad, dec_rad))(mjds)
        spread = float(lats.max() - lats.min())
        assert spread < 0.01, (
            "ecl_lat for a fixed RA/Dec target moved {:.3f} deg over a "
            "year; should be effectively constant.".format(spread)
        )

    def test_ecliptic_plane_target_sweeps_full_longitude(self, obs, mjds):
        """Ecliptic-plane target visits both anti-Sun and conjunction.

        Helio-ecliptic longitude difference should reach near 180 deg
        (anti-Sun) and near 0 deg (conjunction) within one year. Tolerate
        a few degrees of sampling slack.
        """
        ra_rad = jnp.deg2rad(0.0)
        dec_rad = jnp.deg2rad(0.0)  # vernal equinox direction -> ecl_lat ~ 0
        dlams = jax.vmap(
            lambda m: obs.helio_ecliptic_longitude_deg(m, ra_rad, dec_rad)
        )(mjds)
        assert float(dlams.max()) > 175.0, "Did not reach anti-Sun within the year."
        assert float(dlams.min()) < 5.0, "Did not reach conjunction within the year."

    def test_high_latitude_target_never_at_conjunction(self, obs, mjds):
        """Ecliptic-pole-ish target stays bounded away from the Sun.

        A target at |ecl_lat| > 60 deg cannot have ``Delta_lambda_sun``
        bring the look vector to the Sun -- the minimum 3D elongation is
        bounded below by ``|ecl_lat|``.
        """
        # (RA=0, Dec=+80) -> ecl_lat ~ +65 deg, well off the ecliptic.
        ra_rad = jnp.deg2rad(0.0)
        dec_rad = jnp.deg2rad(80.0)
        elongs = jax.vmap(lambda m: obs.solar_elongation_deg(m, ra_rad, dec_rad))(mjds)
        assert float(elongs.min()) > 50.0, (
            "High-latitude target should never come closer than its "
            "ecliptic latitude to the Sun."
        )

    def test_solar_elongation_vs_helio_longitude_match_on_ecliptic(self, obs):
        """On the ecliptic plane the two angles must coincide.

        For ``ecl_lat = 0`` the great-circle Sun-target angle and the
        helio-ecliptic longitude difference are mathematically identical.
        """
        # Vernal equinox direction is ecl_lat=0.
        ra_rad = jnp.deg2rad(0.0)
        dec_rad = jnp.deg2rad(0.0)
        mjd = 60575.25 + 91.0  # part-way through the year
        elong = float(obs.solar_elongation_deg(mjd, ra_rad, dec_rad))
        dlam = float(obs.helio_ecliptic_longitude_deg(mjd, ra_rad, dec_rad))
        assert abs(elong - dlam) < 0.5

    def test_solar_elongation_diverges_from_helio_longitude_off_ecliptic(self, obs):
        """High-latitude targets show the bug we are guarding against.

        For a target at large |ecl_lat|, feeding ``solar_elongation_deg``
        into a Leinert lookup (instead of ``helio_ecliptic_longitude_deg``)
        gives a *different* answer. This test pins that difference so a
        future refactor that unifies them gets flagged.
        """
        # (RA=0, Dec=+60) -> ecl_lat ~ +53 deg.
        ra_rad = jnp.deg2rad(0.0)
        dec_rad = jnp.deg2rad(60.0)
        mjd = 60575.25
        elong = float(obs.solar_elongation_deg(mjd, ra_rad, dec_rad))
        dlam = float(obs.helio_ecliptic_longitude_deg(mjd, ra_rad, dec_rad))
        assert abs(elong - dlam) > 10.0

    def test_conjunction_aligns_with_sun_apparent_longitude(self, obs, mjds):
        """Helio-longitude minimum lines up with heliocentric ecliptic longitude.

        Conjunction (``Delta_lambda_sun -> 0``) happens when the Sun's
        apparent ecliptic longitude equals the target's. We can compute
        Sun's apparent longitude from Earth's heliocentric position and
        verify the argmin of the helio-ecliptic-longitude curve matches.
        """
        # Ecliptic-plane target at ecl_lon = 90 deg (RA=90, Dec=+23.44).
        ra_rad = jnp.deg2rad(90.0)
        dec_rad = jnp.deg2rad(23.44)
        dlams = jax.vmap(
            lambda m: obs.helio_ecliptic_longitude_deg(m, ra_rad, dec_rad)
        )(mjds)
        i_min = int(jnp.argmin(dlams))
        # At the conjunction epoch, Earth's heliocentric longitude must
        # be ~target_ecl_lon + 180 (Sun apparent = -Earth_helio).
        r_earth = earth_position_ecliptic(float(mjds[i_min]))
        earth_lon_deg = float(jnp.degrees(jnp.arctan2(r_earth[1], r_earth[0])))
        target_lon_deg = 90.0
        # Sun apparent longitude from observer = earth_helio_lon + 180.
        sun_apparent_lon_deg = (earth_lon_deg + 180.0) % 360.0
        # Difference should be small (within sampling cadence: ~5 deg).
        diff = abs(((sun_apparent_lon_deg - target_lon_deg + 180.0) % 360.0) - 180.0)
        assert diff < 8.0, (
            f"At argmin(Delta_lambda_sun) the Sun's apparent longitude "
            f"({sun_apparent_lon_deg:.2f}) should match the target's "
            f"ecliptic longitude ({target_lon_deg:.2f}); got off by "
            f"{diff:.2f} deg."
        )

    def test_geometry_methods_share_argument_order(self, obs):
        """All four geometry methods take ``(mjd, ra_rad, dec_rad)``.

        Regression: the historical inconsistency
        ``ecliptic_latitude(ra, dec, mjd=...)`` vs
        ``solar_longitude(mjd, ra, dec)`` caused real bugs. This test
        documents the invariant so it cannot drift again silently.
        """
        mjd = 60575.25
        ra = jnp.deg2rad(45.0)
        dec = jnp.deg2rad(30.0)
        # All four methods must accept (mjd, ra, dec) positionally
        # without raising.
        obs.sun_angle(mjd, ra, dec)
        obs.solar_elongation_deg(mjd, ra, dec)
        obs.helio_ecliptic_longitude_deg(mjd, ra, dec)
        obs.ecliptic_latitude_deg(mjd, ra, dec)


# ============================================================================
# Keepout
# ============================================================================


class TestKeepout:
    """Test keepout zone checks."""

    def test_sun_keepout_blocks_sunward_pointing(self):
        """A target lying along the Sun direction from the observatory is blocked."""
        # Observatory at ~(1.01, 0, 0) AU, Sun at origin.
        obs_pos = jnp.array([1.01, 0.0, 0.0])
        mjd = 60000.0
        # Target at RA=180 deg, Dec=0: looking back toward the Sun.
        blocked = is_observable(obs_pos, jnp.pi, 0.0, mjd)
        assert not bool(blocked)

    def test_anti_solar_pointing_is_observable(self):
        """A target opposite the Sun direction from the observatory is observable."""
        obs_pos = jnp.array([1.01, 0.0, 0.0])
        mjd = 60000.0
        # Target at RA=0, Dec=0: anti-solar direction.
        ok = is_observable(obs_pos, 0.0, 0.0, mjd)
        assert bool(ok)

    def test_keepout_jit(self):
        """Keepout should be JIT-compilable."""
        obs_pos = jnp.array([1.01, 0.0, 0.0])
        f = jax.jit(lambda: is_observable(obs_pos, 1.0, 0.5, 51544.5))
        result = f()
        assert isinstance(result, jax.Array)
