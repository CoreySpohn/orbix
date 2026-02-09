"""Cross-validation: orbix observatory vs EXOSIMS.

Compares planet positions, L2 halo orbit, obliquity, and zodiacal light
against the EXOSIMS reference implementation.

Planet positions: orbix uses Vallado (2013) static polynomials while EXOSIMS
uses JPL DE432s.  Agreement is ~0.01 AU for inner planets, worse for outer.

L2 halo orbit: both use the same 6-month halo data; differences arise from
interpolation (interpax vs scipy) and Earth position (Vallado vs JPL).
"""

import numpy as np
import pytest

# ── orbix imports ──
import jax.numpy as jnp
from orbix.system.solar_system import (
    planet_position_ecliptic as orbix_planet,
    earth_position_ecliptic as orbix_earth,
    obliquity_deg as orbix_obliquity,
)
from orbix.observatory import (
    ObservatoryL2Halo,
    zodi_fzodi_ayo,
    zodi_fzodi_leinert,
)

# ── EXOSIMS imports (guarded) ──
try:
    import EXOSIMS
    EXOSIMS.__version__ = getattr(EXOSIMS, "__version__", "3.7.0alpha")
    from astropy.time import Time
    import astropy.units as u
    from EXOSIMS.Observatory.ObservatoryL2Halo import (
        ObservatoryL2Halo as EXO_ObsL2,
    )
    HAS_EXOSIMS = True
except ImportError:
    HAS_EXOSIMS = False

skip_no_exosims = pytest.mark.skipif(
    not HAS_EXOSIMS, reason="EXOSIMS not installed"
)


@pytest.fixture(scope="module")
def exo_obs():
    """Create a bare EXOSIMS ObservatoryL2Halo (module-scoped for speed)."""
    return EXO_ObsL2(equinox=60575.25, haloStartTime=0)


@pytest.fixture(scope="module")
def orbix_obs():
    """Create an orbix ObservatoryL2Halo."""
    return ObservatoryL2Halo.from_default(
        equinox_mjd=60575.25, halo_start_day=0.0
    )


# ============================================================================
# Planet positions (Vallado vs JPL DE432s)
# ============================================================================

@skip_no_exosims
class TestPlanetPositions:
    """Orbix Vallado planets vs EXOSIMS JPL ephemeris.

    Inner planets agree to ~1e-3 AU, outer planets to ~0.05 AU.
    """

    @pytest.mark.parametrize("body,atol", [
        ("Mercury", 5e-3),
        ("Venus", 5e-3),
        ("Earth", 1e-3),
        ("Mars", 0.01),
        ("Jupiter", 0.05),
        ("Saturn", 0.1),
    ])
    @pytest.mark.parametrize("mjd", [51544.5, 55000.0, 58000.0, 60000.0])
    def test_planet_ecliptic(self, exo_obs, body, atol, mjd):
        """Planet position should be within Vallado accuracy vs JPL."""
        r_orbix = np.array(orbix_planet(body, float(mjd)))

        t = Time(np.array([mjd]), format="mjd", scale="tai")
        r_exo = exo_obs.solarSystem_body_position(
            t, body, eclip=True
        ).to_value(u.AU)[0]

        diff = np.linalg.norm(r_orbix - r_exo)
        assert diff < atol, (
            f"{body} at MJD {mjd}: diff={diff:.6e} AU > atol={atol}\n"
            f"  orbix={r_orbix}\n  EXOSIMS={r_exo}"
        )


# ============================================================================
# L2 halo orbit
# ============================================================================

@skip_no_exosims
class TestL2HaloOrbit:
    """Compare orbix vs EXOSIMS L2 halo orbit positions."""

    @pytest.mark.parametrize("mjd", [
        60575.25, 60600.0, 60650.0, 60700.0, 60750.0, 60800.0,
    ])
    def test_orbit_ecliptic(self, orbix_obs, exo_obs, mjd):
        """Halo orbit ecliptic positions should match within 5e-4 AU.

        Differences arise from interpax vs scipy interpolation and
        Vallado vs JPL Earth position.
        """
        r_orbix = np.array(orbix_obs.position_ecliptic(float(mjd)))

        t = Time(np.array([mjd]), format="mjd", scale="tai")
        r_exo = exo_obs.orbit(t, eclip=True).to_value(u.AU)[0]

        np.testing.assert_allclose(
            r_orbix, r_exo, atol=5e-4,
            err_msg=f"L2 orbit at MJD {mjd}: orbix={r_orbix}, EXOSIMS={r_exo}",
        )

    def test_orbit_distance_from_sun(self, orbix_obs):
        """Observatory should be ~1.01 AU from Sun."""
        for mjd in [60575.25, 60600.0, 60700.0, 60800.0]:
            r = np.array(orbix_obs.position_ecliptic(float(mjd)))
            dist = np.linalg.norm(r)
            assert 0.99 < dist < 1.03, f"Distance = {dist} AU at MJD {mjd}"

    def test_halo_period_match(self, orbix_obs, exo_obs):
        """Halo period should match EXOSIMS."""
        np.testing.assert_allclose(
            orbix_obs.period_yr, exo_obs.period_halo, rtol=1e-6
        )


# ============================================================================
# Obliquity
# ============================================================================

@skip_no_exosims
class TestObliquity:
    """Compare obliquity calculation."""

    @pytest.mark.parametrize("mjd", [51544.5, 55000.0, 60000.0])
    def test_obliquity(self, exo_obs, mjd):
        """Obliquity should match EXOSIMS within 1e-4 degrees."""
        obl_orbix = float(orbix_obliquity(float(mjd)))

        t = Time(np.array([mjd]), format="mjd", scale="tai")
        TDB = exo_obs.cent(t)[0]
        obl_exo = float(exo_obs.obe(TDB))

        np.testing.assert_allclose(
            obl_orbix, obl_exo, atol=1e-4,
            err_msg=f"Obliquity at MJD {mjd}",
        )


# ============================================================================
# Zodiacal light (Fzodi)
# ============================================================================

class TestZodiFzodi:
    """Verify zodiacal Fzodi helpers produce sensible values."""

    def test_ayo_v_band(self):
        """AYO Fzodi at V-band should equal 10^(-0.4 * 22.0)."""
        fzodi = float(zodi_fzodi_ayo(550.0))
        expected = 10.0 ** (-0.4 * 22.0)
        np.testing.assert_allclose(fzodi, expected, rtol=1e-4)

    def test_ayo_wavelength_dependence(self):
        """Red wavelengths should have lower Fzodi (fainter zodi)."""
        fzodi_500 = float(zodi_fzodi_ayo(500.0))
        fzodi_800 = float(zodi_fzodi_ayo(800.0))
        # Zodi is fainter at red wavelengths
        assert fzodi_800 < fzodi_500

    def test_leinert_ecliptic_pole_fainter(self):
        """Ecliptic pole should have fainter zodi than ecliptic plane."""
        fzodi_plane = float(zodi_fzodi_leinert(550.0, ecliptic_lat_deg=0.0, solar_lon_deg=90.0))
        fzodi_pole = float(zodi_fzodi_leinert(550.0, ecliptic_lat_deg=90.0, solar_lon_deg=90.0))
        assert fzodi_pole < fzodi_plane

    def test_leinert_antisolar_fainter(self):
        """Anti-solar direction (180°) should be fainter than 90° solar lon."""
        fzodi_90 = float(zodi_fzodi_leinert(550.0, ecliptic_lat_deg=0.0, solar_lon_deg=90.0))
        fzodi_180 = float(zodi_fzodi_leinert(550.0, ecliptic_lat_deg=0.0, solar_lon_deg=180.0))
        assert fzodi_180 < fzodi_90

    def test_fzodi_jit(self):
        """Both Fzodi functions should JIT-compile."""
        import jax
        fzodi_ayo_jit = jax.jit(zodi_fzodi_ayo)(550.0)
        fzodi_leinert_jit = jax.jit(zodi_fzodi_leinert)(550.0, 30.0, 135.0)
        assert float(fzodi_ayo_jit) > 0
        assert float(fzodi_leinert_jit) > 0
