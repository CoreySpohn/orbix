"""Tests for orbix.observatory.pointing: ecliptic latitude and solar longitude."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import J2000_JD
from hwoutils.epoch import Epoch

from orbix.observatory import ecliptic_pointing, sun_longitude_deg

EPOCH = Epoch(J2000_JD, scale="tt")
OBL = 84381.406 / 3600.0


def _ecliptic_to_radec(lon_deg, lat_deg):
    lon, lat, obl = map(np.radians, (lon_deg, lat_deg, OBL))
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon) * np.cos(obl) - np.sin(lat) * np.sin(obl)
    z = np.cos(lat) * np.sin(lon) * np.sin(obl) + np.sin(lat) * np.cos(obl)
    return np.arctan2(y, x), np.arcsin(z)


def _jd(y, m, d):
    a = (14 - m) // 12
    yy, mm = y + 4800 - a, m + 12 * a - 3
    return (
        d
        + (153 * mm + 2) // 5
        + 365 * yy
        + yy // 4
        - yy // 100
        + yy // 400
        - 32045
        - 0.5
    )


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_ecliptic_pole_latitude(sign):
    """Targets at the ecliptic poles have latitude of plus or minus 90 deg."""
    ra, dec = _ecliptic_to_radec(0.0, sign * 90.0)
    beta, _ = ecliptic_pointing(ra, dec, EPOCH, 100.0)
    assert float(beta) == pytest.approx(sign * 90.0, abs=1e-4)


def test_target_at_sun_has_zero_elongation():
    """A target at the Sun has zero longitude difference; opposite gives 180."""
    t = 5000.0
    lon = float(sun_longitude_deg(EPOCH, t))
    ra, dec = _ecliptic_to_radec(lon, 0.0)
    beta, dlon = ecliptic_pointing(ra, dec, EPOCH, t)
    assert float(beta) == pytest.approx(0.0, abs=1e-4)
    assert float(dlon) == pytest.approx(0.0, abs=1e-4)
    ra, dec = _ecliptic_to_radec(lon + 180.0, 0.0)
    _, dlon = ecliptic_pointing(ra, dec, EPOCH, t)
    assert abs(float(dlon)) == pytest.approx(180.0, abs=1e-4)


@pytest.mark.parametrize(
    "date, lon",
    [
        ((2024, 3, 20.1), 0.0),
        ((2024, 6, 20.9), 90.0),
        ((2024, 9, 22.6), 180.0),
        ((2024, 12, 21.4), 270.0),
    ],
)
def test_sun_longitude_at_equinox_and_solstice(date, lon):
    """The Sun's longitude is 0, 90, 180, 270 deg at equinoxes and solstices."""
    # Longitude of date, i.e. add back the precession removed for the J2000 frame.
    t = _jd(*date) - J2000_JD
    of_date = float(sun_longitude_deg(EPOCH, t)) + 5028.796195 / 3600.0 * t / 36525.0
    diff = (of_date - lon + 180.0) % 360.0 - 180.0
    assert abs(diff) < 0.1  # events quoted to 0.1 d is 0.1 deg


def test_dlon_range_and_shape():
    """Outputs broadcast and the longitude difference is wrapped to [-180, 180)."""
    ra = jnp.linspace(0.0, 6.28, 7)
    beta, dlon = ecliptic_pointing(ra, 0.1, EPOCH, jnp.linspace(0.0, 365.0, 7))
    assert beta.shape == dlon.shape == (7,)
    assert bool(jnp.all((dlon >= -180.0) & (dlon < 180.0)))


def test_matches_astropy():
    """Latitude and longitude difference agree with astropy ephemerides."""
    astropy = pytest.importorskip("astropy")  # noqa: F841
    import astropy.units as u
    from astropy.coordinates import (
        BarycentricMeanEcliptic,
        GeocentricMeanEcliptic,
        SkyCoord,
        get_sun,
    )
    from astropy.time import Time

    rng = np.random.default_rng(3)
    ra = rng.uniform(0, 2 * np.pi, 40)
    dec = np.arcsin(rng.uniform(-1, 1, 40))
    t_d = rng.uniform(-9000.0, 9000.0, 40)
    time = Time(J2000_JD + t_d, format="jd", scale="tt")
    sun = get_sun(time).transform_to(GeocentricMeanEcliptic(equinox="J2000"))
    tgt = SkyCoord(ra * u.rad, dec * u.rad, frame="icrs").transform_to(
        BarycentricMeanEcliptic(equinox="J2000")
    )
    want_dlon = (tgt.lon.deg - sun.lon.deg + 180.0) % 360.0 - 180.0
    beta, dlon = ecliptic_pointing(ra, dec, EPOCH, t_d)
    assert np.max(np.abs(np.asarray(beta) - tgt.lat.deg)) < 1e-3
    err = (np.asarray(dlon) - want_dlon + 180.0) % 360.0 - 180.0
    assert np.max(np.abs(err)) < 0.03


def test_gradient_in_time_matches_central_difference():
    """The time derivative matches central differences and the solar rate."""
    ra, dec = 1.0, 0.3
    t0 = 1234.5
    grad = jax.grad(lambda t: ecliptic_pointing(ra, dec, EPOCH, t)[1])(t0)
    h = 0.5
    fd = (
        float(ecliptic_pointing(ra, dec, EPOCH, t0 + h)[1])
        - float(ecliptic_pointing(ra, dec, EPOCH, t0 - h)[1])
    ) / (2 * h)
    assert float(grad) == pytest.approx(fd, rel=1e-4)
    assert float(grad) == pytest.approx(-0.9856, abs=0.06)
