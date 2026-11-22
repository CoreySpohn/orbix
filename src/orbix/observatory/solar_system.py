"""Solar system body positions and the ecliptic frame.

Positions use Vallado (2013) static ephemerides. All positions are
heliocentric ecliptic (J2000 mean ecliptic and equinox), in AU.  Body
positions take MJD; the Sun's longitude takes an ``Epoch`` and ``time_d``.
This module holds the one equatorial-to-ecliptic rotation, a fixed J2000
rotation, shared by the keepout, observatory and pointing geometry.

This module is a pure-JAX port of the ``keplerplanet`` method in
``EXOSIMS.Prototypes.Observatory``, using orbix's own Kepler solver.

Reference:
    Vallado, D. A. (2013). Fundamentals of Astrodynamics and Applications.
    Appendix D.4 -- Planetary ephemerides.
"""

from __future__ import annotations

import jax.numpy as jnp
from hwoutils.constants import J2000_JD
from hwoutils.conversions import jd_to_mjd, years_to_days
from hwoutils.epoch import Epoch
from jaxtyping import Array, ArrayLike

from orbix.kepler import kepler

# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def _mjd_to_julian_centuries(mjd: float) -> float:
    """Convert MJD to Julian centuries since J2000."""
    return (mjd - jd_to_mjd(J2000_JD)) / years_to_days(100.0)


# ---------------------------------------------------------------------------
# Obliquity of the ecliptic
# ---------------------------------------------------------------------------

OBLIQUITY_J2000_DEG = 84381.406 / 3600.0
"""IAU 2006 mean obliquity of the ecliptic at J2000 [deg]."""


# ---------------------------------------------------------------------------
# Coordinate frame rotations
# ---------------------------------------------------------------------------


def _rot1(theta: float) -> jnp.ndarray:
    """Rotation matrix about axis 1 (x)."""
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])


def _rot3(theta: float) -> jnp.ndarray:
    """Rotation matrix about axis 3 (z)."""
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def equat2eclip(r_equat: ArrayLike) -> Array:
    """Rotate equatorial J2000 vectors to the J2000 ecliptic.

    The one equatorial-to-ecliptic rotation: a rotation about the x axis by the
    J2000 mean obliquity. The Vallado ephemerides and the J2000 sky positions
    share this frame, so no obliquity of date enters.

    Args:
        r_equat: Vector(s) in the equatorial frame, shape ``(..., 3)``.

    Returns:
        Vector(s) in the ecliptic frame, same shape as the input.
    """
    rotation = _rot1(jnp.radians(OBLIQUITY_J2000_DEG))
    return jnp.asarray(r_equat) @ rotation.T


def eclip2equat(r_eclip: ArrayLike) -> Array:
    """Rotate J2000 ecliptic vectors to the equatorial J2000 frame.

    Args:
        r_eclip: Vector(s) in the ecliptic frame, shape ``(..., 3)``.

    Returns:
        Vector(s) in the equatorial frame, same shape as the input.
    """
    rotation = _rot1(-jnp.radians(OBLIQUITY_J2000_DEG))
    return jnp.asarray(r_eclip) @ rotation.T


def equatorial_to_ecliptic_deg(ra_deg: ArrayLike, dec_deg: ArrayLike) -> tuple:
    """Convert equatorial J2000 sky positions to ecliptic J2000 longitude and latitude.

    Args:
        ra_deg: Right ascension [deg].
        dec_deg: Declination [deg]. Broadcasts with ``ra_deg``.

    Returns:
        ``(longitude_deg, latitude_deg)`` with longitude in [0, 360) and
        latitude in [-90, 90].
    """
    ra, dec = jnp.broadcast_arrays(jnp.radians(ra_deg), jnp.radians(dec_deg))
    direction = jnp.stack(
        [jnp.cos(dec) * jnp.cos(ra), jnp.cos(dec) * jnp.sin(ra), jnp.sin(dec)],
        axis=-1,
    )
    x, y, z = jnp.moveaxis(equat2eclip(direction), -1, 0)
    longitude = jnp.mod(jnp.degrees(jnp.arctan2(y, x)), 360.0)
    return longitude, jnp.degrees(jnp.arctan2(z, jnp.hypot(x, y)))


# ---------------------------------------------------------------------------
# Vallado Appendix D.4 ephemeris data
# ---------------------------------------------------------------------------
# Each planet has 6 orbital elements: a, e, I, O, w, lM.
# Each element is a polynomial in TDB (Julian centuries since J2000).
# Coefficients are padded to 4 terms and pre-packed into (6, 4) JAX arrays.

_NCOEFF = 4  # max polynomial degree + 1

_EPHEM_RAW = {
    "Mercury": dict(
        a=[0.387098310],
        e=[0.20563175, 0.000020406, -0.0000000284, -0.00000000017],
        I=[7.004986, -0.0059516, 0.00000081, 0.000000041],
        O=[48.330893, -0.1254229, -0.00008833, -0.000000196],
        w=[77.456119, 0.1588643, -0.00001343, 0.000000039],
        lM=[252.250906, 149472.6746358, -0.00000535, 0.000000002],
    ),
    "Venus": dict(
        a=[0.723329820],
        e=[0.00677188, -0.000047766, 0.0000000975, 0.00000000044],
        I=[3.394662, -0.0008568, -0.00003244, 0.000000010],
        O=[76.679920, -0.2780080, -0.00014256, -0.000000198],
        w=[131.563707, 0.0048646, -0.00138232, -0.000005332],
        lM=[181.979801, 58517.8156760, 0.00000165, -0.000000002],
    ),
    "Earth": dict(
        a=[1.000001018],
        e=[0.01670862, -0.000042037, -0.0000001236, 0.00000000004],
        I=[0.0, 0.0130546, -0.00000931, -0.000000034],
        O=[174.873174, -0.2410908, 0.00004067, -0.000001327],
        w=[102.937348, 0.3225557, 0.00015026, 0.000000478],
        lM=[100.466449, 35999.3728519, -0.00000568, 0.0],
    ),
    "Mars": dict(
        a=[1.523679342],
        e=[0.09340062, 0.000090483, -0.0000000806, -0.00000000035],
        I=[1.849726, -0.0081479, -0.00002255, -0.000000027],
        O=[49.558093, -0.2949846, -0.00063993, -0.000002143],
        w=[336.060234, 0.4438898, -0.00017321, 0.000000300],
        lM=[355.433275, 19140.2993313, 0.00000261, -0.000000003],
    ),
    "Jupiter": dict(
        a=[5.202603191, 0.0000001913],
        e=[0.04849485, 0.000163244, -0.0000004719, -0.00000000197],
        I=[1.303270, -0.0019872, 0.00003318, 0.000000092],
        O=[100.464441, 0.1766828, 0.00090387, -0.000007032],
        w=[14.331309, 0.2155525, 0.00072252, -0.000004590],
        lM=[34.351484, 3034.9056746, -0.00008501, 0.000000004],
    ),
    "Saturn": dict(
        a=[9.554909596, -0.0000021389],
        e=[0.05550862, -0.000346818, -0.0000006456, 0.00000000338],
        I=[2.488878, 0.0025515, -0.00004903, 0.000000018],
        O=[113.665524, -0.2566649, -0.00018345, 0.000000357],
        w=[93.056787, 0.5665496, 0.00052809, 0.000004882],
        lM=[50.077471, 1222.1137943, 0.00021004, -0.000000019],
    ),
    "Uranus": dict(
        a=[19.218446062, -0.0000000372, 0.00000000098],
        e=[0.04629590, -0.000027337, 0.0000000790, 0.00000000025],
        I=[0.773196, -0.0016869, 0.00000349, 0.000000016],
        O=[74.005947, 0.0741461, 0.00040540, 0.000000104],
        w=[173.005159, 0.0893206, -0.00009470, 0.000000413],
        lM=[314.055005, 428.4669983, -0.00000486, 0.000000006],
    ),
    "Neptune": dict(
        a=[30.110386869, -0.0000001663, 0.00000000069],
        e=[0.00898809, 0.000006408, -0.0000000008],
        I=[1.769952, 0.0002257, 0.00000023, 0.0],
        O=[131.784057, -0.0061651, -0.00000219, -0.000000078],
        w=[48.123691, 0.0291587, 0.00007051, 0.0],
        lM=[304.348665, 218.4862002, 0.00000059, -0.000000002],
    ),
    "Pluto": dict(
        a=[39.48168677, -0.00076912],
        e=[0.24880766, 0.00006465],
        I=[17.14175, 0.003075],
        O=[110.30347, -0.01036944],
        w=[224.06676, -0.03673611],
        lM=[238.92881, 145.2078],
    ),
}


def _pad(coeffs: list, n: int = _NCOEFF) -> list:
    """Pad coefficient list to length n with zeros."""
    return coeffs + [0.0] * (n - len(coeffs))


def _pack_planet(raw: dict) -> jnp.ndarray:
    """Pack planet ephemeris dict into a (6, NCOEFF) array.

    Row order: a, e, I, O, w, lM.
    """
    return jnp.array(
        [
            _pad(raw["a"]),
            _pad(raw["e"]),
            _pad(raw["I"]),
            _pad(raw["O"]),
            _pad(raw["w"]),
            _pad(raw["lM"]),
        ]
    )


# Pre-pack all planets into JAX arrays at module load time.
# Each entry is a (6, 4) array.
_EPHEM: dict[str, jnp.ndarray] = {
    name: _pack_planet(raw) for name, raw in _EPHEM_RAW.items()
}


def _eval_elements(coeffs: jnp.ndarray, TDB: float) -> jnp.ndarray:
    """Evaluate all 6 orbital element polynomials at once.

    Args:
        coeffs: Shape ``(6, NCOEFF)`` -- rows are [a, e, I, O, w, lM].
        TDB: Julian centuries since J2000.

    Returns:
        Shape ``(6,)`` -- [a, e, I_deg, O_deg, w_deg, lM_deg].
    """
    TDB_powers = jnp.array([1.0, TDB, TDB**2, TDB**3])
    return coeffs @ TDB_powers  # (6, 4) @ (4,) -> (6,)


# ---------------------------------------------------------------------------
# Planet position calculation (Keplerian, matches EXOSIMS keplerplanet)
# ---------------------------------------------------------------------------


def planet_position_ecliptic(body: str, mjd: float) -> jnp.ndarray:
    """Heliocentric ecliptic position of a solar system body.

    Uses Vallado (2013) Algorithms 2 and 10 -- Keplerian elements propagated
    with polynomial time corrections.  All 6 orbital elements are evaluated
    in a single vectorized ``matmul``.

    Args:
        body: Planet name (e.g. ``"Earth"``, ``"Jupiter"``).
        mjd: Modified Julian Date (scalar).

    Returns:
        Position vector in heliocentric ecliptic frame (AU), shape ``(3,)``.
    """
    mjd = jnp.asarray(mjd)
    if mjd.ndim != 0:
        raise ValueError(
            "planet_position_ecliptic takes a scalar mjd; vmap over times instead"
        )
    return _planet_position_ecliptic_tdb(body, _mjd_to_julian_centuries(mjd))


def _planet_position_ecliptic_tdb(body: str, TDB: ArrayLike) -> Array:
    """Heliocentric ecliptic position at ``TDB`` Julian centuries from J2000."""
    coeffs = _EPHEM[body]  # (6, 4) JAX array

    # Evaluate all 6 polynomials at once: [a, e, I_deg, O_deg, w_deg, lM_deg]
    elems = _eval_elements(coeffs, TDB)
    a = elems[0]
    e = elems[1]
    inc = jnp.radians(elems[2])
    W = jnp.radians(elems[3])
    w_tilde = jnp.radians(elems[4])  # longitude of perihelion
    lM = jnp.radians(elems[5])  # mean longitude

    # Argument of perihelion and mean anomaly
    w = w_tilde - W
    M = (lM - w_tilde) % (2.0 * jnp.pi)

    # Solve Kepler's equation using orbix solver
    sinE, cosE = kepler(M, e, 0.0)

    # True anomaly from eccentric anomaly
    nu = jnp.arctan2(jnp.sqrt(1.0 - e**2) * sinE, cosE - e)

    # Distance
    r_mag = a * (1.0 - e * cosE)

    # Rotation to ecliptic frame (Perifocal -> Ecliptic)
    cos_W = jnp.cos(W)
    sin_W = jnp.sin(W)
    cos_inc = jnp.cos(inc)
    sin_inc = jnp.sin(inc)

    cos_wnu = jnp.cos(w + nu)
    sin_wnu = jnp.sin(w + nu)

    x = r_mag * (cos_W * cos_wnu - sin_W * sin_wnu * cos_inc)
    y = r_mag * (sin_W * cos_wnu + cos_W * sin_wnu * cos_inc)
    z = r_mag * (sin_wnu * sin_inc)

    return jnp.array([x, y, z])


def planet_position_equatorial(body: str, mjd: float) -> jnp.ndarray:
    """Heliocentric equatorial position of a solar system body.

    Args:
        body: Planet name (e.g. ``"Earth"``).
        mjd: Modified Julian Date (scalar).

    Returns:
        Position vector in heliocentric equatorial frame (AU), shape ``(3,)``.
    """
    r_eclip = planet_position_ecliptic(body, mjd)
    return eclip2equat(r_eclip)


def earth_position_ecliptic(mjd: float) -> jnp.ndarray:
    """Heliocentric ecliptic position of Earth (AU).

    Args:
        mjd: Modified Julian Date in days, scalar.

    Returns:
        Position vector in AU of shape (3,), heliocentric ecliptic frame.
    """
    return planet_position_ecliptic("Earth", mjd)


def earth_position_equatorial(mjd: float) -> jnp.ndarray:
    """Heliocentric equatorial position of Earth (AU).

    Args:
        mjd: Modified Julian Date in days, scalar.

    Returns:
        Position vector in AU of shape (3,), heliocentric equatorial frame.
    """
    return planet_position_equatorial("Earth", mjd)


# ---------------------------------------------------------------------------
# Sun longitude
# ---------------------------------------------------------------------------


def sun_ecliptic_longitude_deg(epoch: Epoch, time_d: ArrayLike) -> Array:
    """Geocentric ecliptic longitude of the Sun in the J2000 ecliptic frame [deg].

    The direction opposite Earth's heliocentric position from the Vallado
    ephemeris, which shares the J2000 ecliptic with J2000 target coordinates.
    The observatory at Sun-Earth L2 lies near the Sun-Earth line, so the
    geocentric direction is used (the halo orbit shifts the Sun's direction by
    up to about 0.3 deg).

    Args:
        epoch: Static day zero of ``time_d`` (the time scale is not converted;
            the TDB-TT difference of about 1.6 ms is far below the ephemeris
            accuracy).
        time_d: Days from ``epoch``. Differentiable; any shape.

    Returns:
        Longitude in degrees in [0, 360), the shape of ``time_d``.
    """
    days = jnp.asarray(time_d) + epoch.days_between(Epoch(J2000_JD, scale=epoch.scale))

    def longitude(d):
        r_earth = _planet_position_ecliptic_tdb("Earth", d / years_to_days(100.0))
        return jnp.degrees(jnp.arctan2(-r_earth[1], -r_earth[0]))

    return jnp.mod(jnp.vectorize(longitude)(days), 360.0)
