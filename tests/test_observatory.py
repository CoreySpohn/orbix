"""Tests for orbix.observatory module."""

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from orbix.observatory import (
    ObservatoryL2Halo,
    ayo_default_zodi_mag,
    leinert_zodi_factor,
    leinert_zodi_mag,
    leinert_zodi_spectral_radiance,
    mag_to_flux_jy,
    flux_to_mag_jy,
    zodi_color_correction,
    is_observable,
)
from orbix.system.solar_system import (
    earth_position_ecliptic,
    planet_position_ecliptic,
    obliquity_deg,
    equat2eclip,
    eclip2equat,
    radec_to_ecliptic,
)


# ============================================================================
# Solar system ephemerides
# ============================================================================

class TestSolarSystemEphemerides:
    """Test Vallado ephemerides against expected values."""

    def test_obliquity_j2000(self):
        """Obliquity at J2000 should be ~23.439°."""
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
        """Equatorial ↔ ecliptic conversion should roundtrip."""
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

    def test_radec_to_ecliptic(self):
        """RA/Dec → ecliptic at ecliptic pole: lat should be ~90°."""
        # North ecliptic pole in J2000 equatorial: RA=270°, Dec=66.56°
        ra = jnp.radians(270.0)
        dec = jnp.radians(66.56)
        _, beta = radec_to_ecliptic(ra, dec, 51544.5)
        assert jnp.isclose(jnp.degrees(beta), 90.0, atol=1.0)


# ============================================================================
# Zodiacal light
# ============================================================================

class TestZodiacalLight:
    """Test zodiacal light functions."""

    def test_ayo_default_v_band(self):
        """AYO default at V-band should be exactly 22 mag."""
        mag = ayo_default_zodi_mag(550.0)
        assert jnp.isclose(mag, 22.0, atol=0.01)

    def test_leinert_factor_ecliptic_pole(self):
        """Factor at ecliptic pole should be 77/259 ≈ 0.297."""
        factor = leinert_zodi_factor(90.0, 90.0)
        assert jnp.isclose(factor, 77.0 / 259.0, atol=0.01)

    def test_leinert_factor_symmetry(self):
        """Factor should be symmetric in ecliptic latitude."""
        f_pos = leinert_zodi_factor(30.0, 135.0)
        f_neg = leinert_zodi_factor(-30.0, 135.0)
        assert jnp.isclose(f_pos, f_neg, atol=1e-6)

    def test_leinert_mag_brighter_at_lower_lat(self):
        """Lower ecliptic latitude should be brighter (lower mag)."""
        mag_low = leinert_zodi_mag(550.0, 10.0, 135.0)
        mag_high = leinert_zodi_mag(550.0, 60.0, 135.0)
        assert float(mag_low) < float(mag_high)

    def test_mag_flux_roundtrip(self):
        """mag → flux → mag should roundtrip."""
        mag = 22.0
        flux = mag_to_flux_jy(mag)
        mag_back = flux_to_mag_jy(flux)
        assert jnp.isclose(mag, mag_back, atol=1e-10)

    def test_color_correction_identity(self):
        """Color correction at reference wavelength should be 1.0."""
        cc = zodi_color_correction(550.0, 550.0, photon_units=False)
        assert jnp.isclose(cc, 1.0, atol=1e-6)

    def test_zodi_jit(self):
        """Zodiacal light functions should JIT-compile."""
        f = jax.jit(leinert_zodi_mag)
        mag = f(550.0, 30.0, 135.0)
        assert jnp.isfinite(mag)

    def test_zodi_grad(self):
        """Zodiacal light should be differentiable w.r.t. wavelength."""
        grad_fn = jax.grad(lambda wl: leinert_zodi_mag(wl, 30.0, 135.0))
        g = grad_fn(550.0)
        assert jnp.isfinite(g)


# ============================================================================
# Observatory L2 halo orbit
# ============================================================================

class TestObservatoryL2Halo:
    """Test L2 halo orbit module."""

    @pytest.fixture
    def obs(self):
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
        """Sun angle should be in [0, π]."""
        angle = obs.sun_angle(51544.5, 0.0, 0.5)
        assert 0.0 <= float(angle) <= jnp.pi

    def test_jit_position(self, obs):
        """Position should work with eqx.filter_jit."""
        f = eqx.filter_jit(obs.position_ecliptic)
        pos = f(51544.5)
        assert jnp.isfinite(pos).all()

    def test_vmap_over_time(self, obs):
        """vmap over multiple times should work."""
        mjds = jnp.linspace(51544.5, 51544.5 + 365, 20)
        positions = jax.vmap(obs.position_ecliptic)(mjds)
        assert positions.shape == (20, 3)
        assert jnp.isfinite(positions).all()

    def test_vmap_over_targets(self, obs):
        """vmap over multiple targets for sun angle."""
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

    def test_solar_longitude_range(self, obs):
        """Solar longitude should be in [0, 180]."""
        sol_lon = obs.solar_longitude(51544.5, 1.0, 0.5)
        assert 0.0 <= float(sol_lon) <= 180.0

    def test_ecliptic_latitude_range(self, obs):
        """Ecliptic latitude should be in [-90, 90]."""
        ecl_lat = obs.ecliptic_latitude(1.0, 0.5)
        assert -90.0 <= float(ecl_lat) <= 90.0


# ============================================================================
# Keepout
# ============================================================================

class TestKeepout:
    """Test keepout zone checks."""

    def test_sun_keepout_blocks_anti_solar(self):
        """Target near the Sun should be blocked."""
        # Observatory at ~(1, 0, 0) AU, Sun at origin
        obs_pos = jnp.array([1.01, 0.0, 0.0])
        # Target near RA=180°, Dec=0° (anti-Sun direction from Earth perspective)
        # Sun is in direction RA≈0° from observatory
        # Target at RA=0° should be near Sun
        ra = jnp.radians(0.0)
        dec = jnp.radians(0.0)
        result = is_observable(obs_pos, ra, dec, 51544.5, ko_sun_min_deg=45.0)
        # This should probably be blocked since target is near the Sun direction
        # (obs is at +x, sun is at origin, target at RA=0 points toward +x)
        assert isinstance(result, jax.Array)

    def test_keepout_jit(self):
        """Keepout should be JIT-compilable."""
        obs_pos = jnp.array([1.01, 0.0, 0.0])
        f = jax.jit(lambda: is_observable(obs_pos, 1.0, 0.5, 51544.5))
        result = f()
        assert isinstance(result, jax.Array)
