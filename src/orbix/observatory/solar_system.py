"""Solar system body positions and the ecliptic frame.

Positions use Vallado (2013) static ephemerides. All positions are
heliocentric ecliptic (J2000 mean ecliptic and equinox), in AU.  Body
positions and the Sun's longitude take a static ``Epoch`` and ``time_d``, days
from that epoch.  The element polynomials are re-expanded about the epoch on the
host in float64, so traced code only ever sees times of the order of the
observation span and float32 keeps its resolution.
This module holds the one equatorial-to-ecliptic rotation, a fixed J2000
rotation, shared by the keepout, observatory and pointing geometry.

This module is a pure-JAX port of the ``keplerplanet`` method in
``EXOSIMS.Prototypes.Observatory``, using orbix's own Kepler solver.

Reference:
    Vallado, D. A. (2013). Fundamentals of Astrodynamics and Applications.
    Appendix D.4 -- Planetary ephemerides.
"""

from __future__ import annotations

from functools import lru_cache
from math import comb

import jax.numpy as jnp
import numpy as np
from hwoutils.constants import J2000_JD
from hwoutils.conversions import years_to_days
from hwoutils.epoch import Epoch
from hwoutils.precision import like
from jaxtyping import Array, ArrayLike

from orbix.kepler import kepler

_DAYS_PER_CENTURY = years_to_days(100.0)


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
# Coefficients are padded to 4 terms and packed on the host before the epoch shift.

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


def _pack_planet(raw: dict) -> np.ndarray:
    """Pack planet ephemeris dict into a (6, NCOEFF) array.

    Row order: a, e, I, O, w, lM.
    """
    return np.array(
        [
            _pad(raw["a"]),
            _pad(raw["e"]),
            _pad(raw["I"]),
            _pad(raw["O"]),
            _pad(raw["w"]),
            _pad(raw["lM"]),
        ],
        dtype=np.float64,
    )


# Secular rates must retain all their digits until the epoch is subtracted.
_EPHEM: dict[str, np.ndarray] = {
    name: _pack_planet(raw) for name, raw in _EPHEM_RAW.items()
}


@lru_cache(maxsize=64)
def _epoch_coefficients(body: str, epoch: Epoch) -> np.ndarray:
    """Element polynomials of ``body`` re-expanded about ``epoch``, on the host.

    The polynomials in Julian centuries since J2000 are shifted to centuries
    since ``epoch`` in float64 (a cubic shift is exact algebra), and the angle
    constants (O, w, lM) are reduced modulo 360 deg, so the traced evaluation
    sees a small argument and small constants. Host-side NumPy float64 on
    purpose: this is the one place a large absolute date is subtracted, and it
    must not round in the active dtype.

    Args:
        body: Planet name.
        epoch: Static day zero of the traced times.

    Returns:
        Read-only float64 array of shape ``(6, NCOEFF)``, rows a, e, I, O, w, lM.
    """
    t0 = epoch.days_between(Epoch(J2000_JD, scale=epoch.scale)) / _DAYS_PER_CENTURY
    raw = _EPHEM[body]
    shifted = np.zeros_like(raw)
    for k in range(_NCOEFF):
        for j in range(k, _NCOEFF):
            shifted[:, k] += raw[:, j] * comb(j, k) * t0 ** (j - k)
    shifted[3:, 0] = np.mod(shifted[3:, 0], 360.0)
    shifted.setflags(write=False)
    return shifted


def _eval_elements(coeffs: jnp.ndarray, tau: float) -> jnp.ndarray:
    """Evaluate all 6 orbital element polynomials at once.

    Args:
        coeffs: Shape ``(6, NCOEFF)`` -- rows are [a, e, I, O, w, lM].
        tau: Julian centuries since the epoch the coefficients are expanded about.

    Returns:
        Shape ``(6,)`` -- [a, e, I_deg, O_deg, w_deg, lM_deg].
    """
    powers = jnp.stack([jnp.ones_like(tau), tau, tau**2, tau**3])
    return coeffs @ powers  # (6, 4) @ (4,) -> (6,)


# ---------------------------------------------------------------------------
# Planet position calculation (Keplerian, matches EXOSIMS keplerplanet)
# ---------------------------------------------------------------------------


def planet_position_ecliptic(body: str, epoch: Epoch, time_d: ArrayLike) -> jnp.ndarray:
    """Heliocentric ecliptic position of a solar system body.

    Uses Vallado (2013) Algorithms 2 and 10 -- Keplerian elements propagated
    with polynomial time corrections.  All 6 orbital elements are evaluated
    in a single vectorized ``matmul``.

    Args:
        body: Planet name (e.g. ``"Earth"``, ``"Jupiter"``).
        epoch: Static day zero of ``time_d`` (close over it or mark it static
            under ``jax.jit``); the time scale is not converted.
        time_d: Days from ``epoch``, scalar; vmap over times.

    Returns:
        Position vector in heliocentric ecliptic frame (AU), shape ``(3,)``.
    """
    time_d = jnp.asarray(time_d)
    if time_d.ndim != 0:
        raise ValueError(
            "planet_position_ecliptic takes a scalar time_d; vmap over times instead"
        )
    return _planet_position_ecliptic_rel(body, epoch, time_d)


def _planet_position_ecliptic_rel(body: str, epoch: Epoch, time_d: ArrayLike) -> Array:
    """Heliocentric ecliptic position at ``time_d`` days from ``epoch``."""
    coeffs = like(_epoch_coefficients(body, epoch), time_d)
    tau = jnp.asarray(time_d, dtype=coeffs.dtype) / _DAYS_PER_CENTURY

    # Evaluate all 6 polynomials at once: [a, e, I_deg, O_deg, w_deg, lM_deg]
    elems = _eval_elements(coeffs, tau)
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


def planet_position_equatorial(
    body: str, epoch: Epoch, time_d: ArrayLike
) -> jnp.ndarray:
    """Heliocentric equatorial position of a solar system body.

    Args:
        body: Planet name (e.g. ``"Earth"``).
        epoch: Static day zero of ``time_d``.
        time_d: Days from ``epoch``, scalar.

    Returns:
        Position vector in heliocentric equatorial frame (AU), shape ``(3,)``.
    """
    r_eclip = planet_position_ecliptic(body, epoch, time_d)
    return eclip2equat(r_eclip)


def earth_position_ecliptic(epoch: Epoch, time_d: ArrayLike) -> jnp.ndarray:
    """Heliocentric ecliptic position of Earth (AU).

    Args:
        epoch: Static day zero of ``time_d``.
        time_d: Days from ``epoch``, scalar.

    Returns:
        Position vector in AU of shape (3,), heliocentric ecliptic frame.
    """
    return planet_position_ecliptic("Earth", epoch, time_d)


def earth_position_equatorial(epoch: Epoch, time_d: ArrayLike) -> jnp.ndarray:
    """Heliocentric equatorial position of Earth (AU).

    Args:
        epoch: Static day zero of ``time_d``.
        time_d: Days from ``epoch``, scalar.

    Returns:
        Position vector in AU of shape (3,), heliocentric equatorial frame.
    """
    return planet_position_equatorial("Earth", epoch, time_d)


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

    def longitude(d):
        r_earth = _planet_position_ecliptic_rel("Earth", epoch, d)
        return jnp.degrees(jnp.arctan2(-r_earth[1], -r_earth[0]))

    return jnp.mod(jnp.vectorize(longitude)(jnp.asarray(time_d)), 360.0)
