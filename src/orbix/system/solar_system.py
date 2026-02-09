"""Solar system body positions using Vallado (2013) static ephemerides.

All positions are heliocentric ecliptic, in AU.  Time inputs are MJD.

This module is a pure-JAX port of the ``keplerplanet`` method in
``EXOSIMS.Prototypes.Observatory``, using orbix's own Kepler solver.

Reference:
    Vallado, D. A. (2013). Fundamentals of Astrodynamics and Applications.
    Appendix D.4 — Planetary ephemerides.
"""

from __future__ import annotations

import jax.numpy as jnp

from orbix.kepler.core import E_solve

# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

# J2000 epoch in JD and MJD
_J2000_JD = 2451545.0
_J2000_MJD = 51544.5  # = _J2000_JD - 2400000.5

_JULIAN_CENTURY = 36525.0  # days


def _mjd_to_julian_centuries(mjd: float) -> float:
    """Convert MJD to Julian centuries since J2000."""
    return (mjd - _J2000_MJD) / _JULIAN_CENTURY


# ---------------------------------------------------------------------------
# Obliquity of the ecliptic
# ---------------------------------------------------------------------------


def obliquity_deg(mjd: float) -> float:
    """Obliquity of the ecliptic in degrees (Vallado polynomial).

    Args:
        mjd: Modified Julian Date.

    Returns:
        Obliquity in degrees.
    """
    TDB = _mjd_to_julian_centuries(mjd)
    return (
        23.439279
        - 0.0130102 * TDB
        - 5.086e-8 * TDB**2
        + 5.565e-7 * TDB**3
        + 1.6e-10 * TDB**4
        + 1.21e-11 * TDB**5
    )


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


def equat2eclip(r_equat: jnp.ndarray, mjd: float) -> jnp.ndarray:
    """Rotate heliocentric equatorial → ecliptic.

    Args:
        r_equat: Position vector(s) in equatorial frame, shape ``(3,)`` or ``(n, 3)``.
        mjd: MJD (scalar) for obliquity calculation.

    Returns:
        Position vector(s) in ecliptic frame, same shape as input.
    """
    obe = jnp.radians(obliquity_deg(mjd))
    R = _rot1(obe)
    squeeze = r_equat.ndim == 1
    r = jnp.atleast_2d(r_equat)
    r_eclip = (R @ r.T).T
    return r_eclip.squeeze(axis=0) if squeeze else r_eclip


def eclip2equat(r_eclip: jnp.ndarray, mjd: float) -> jnp.ndarray:
    """Rotate heliocentric ecliptic → equatorial.

    Args:
        r_eclip: Position vector(s) in ecliptic frame, shape ``(3,)`` or ``(n, 3)``.
        mjd: MJD (scalar) for obliquity calculation.

    Returns:
        Position vector(s) in equatorial frame, same shape as input.
    """
    obe = jnp.radians(obliquity_deg(mjd))
    R = _rot1(-obe)
    squeeze = r_eclip.ndim == 1
    r = jnp.atleast_2d(r_eclip)
    r_equat = (R @ r.T).T
    return r_equat.squeeze(axis=0) if squeeze else r_equat


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
    return jnp.array([
        _pad(raw["a"]), _pad(raw["e"]), _pad(raw["I"]),
        _pad(raw["O"]), _pad(raw["w"]), _pad(raw["lM"]),
    ])


# Pre-pack all planets into JAX arrays at module load time.
# Each entry is a (6, 4) array.
_EPHEM: dict[str, jnp.ndarray] = {
    name: _pack_planet(raw) for name, raw in _EPHEM_RAW.items()
}


def _eval_elements(coeffs: jnp.ndarray, TDB: float) -> jnp.ndarray:
    """Evaluate all 6 orbital element polynomials at once.

    Args:
        coeffs: Shape ``(6, NCOEFF)`` — rows are [a, e, I, O, w, lM].
        TDB: Julian centuries since J2000.

    Returns:
        Shape ``(6,)`` — [a, e, I_deg, O_deg, w_deg, lM_deg].
    """
    TDB_powers = jnp.array([1.0, TDB, TDB**2, TDB**3])
    return coeffs @ TDB_powers  # (6, 4) @ (4,) → (6,)


# ---------------------------------------------------------------------------
# Planet position calculation (Keplerian, matches EXOSIMS keplerplanet)
# ---------------------------------------------------------------------------


def planet_position_ecliptic(body: str, mjd: float) -> jnp.ndarray:
    """Heliocentric ecliptic position of a solar system body.

    Uses Vallado (2013) Algorithms 2 and 10 — Keplerian elements propagated
    with polynomial time corrections.  All 6 orbital elements are evaluated
    in a single vectorized ``matmul``.

    Args:
        body: Planet name (e.g. ``"Earth"``, ``"Jupiter"``).
        mjd: Modified Julian Date (scalar).

    Returns:
        Position vector in heliocentric ecliptic frame (AU), shape ``(3,)``.
    """
    coeffs = _EPHEM[body]  # (6, 4) JAX array
    TDB = _mjd_to_julian_centuries(mjd)

    # Evaluate all 6 polynomials at once: [a, e, I_deg, O_deg, w_deg, lM_deg]
    elems = _eval_elements(coeffs, TDB)
    a = elems[0]
    e = elems[1]
    I = jnp.radians(elems[2])
    O = jnp.radians(elems[3])
    w_tilde = jnp.radians(elems[4])   # longitude of perihelion
    lM = jnp.radians(elems[5])        # mean longitude

    # Argument of perihelion and mean anomaly
    w = w_tilde - O
    M = (lM - w_tilde) % (2.0 * jnp.pi)

    # Solve Kepler's equation using orbix solver
    E = E_solve(jnp.atleast_1d(M), e)[0]

    # True anomaly from eccentric anomaly
    sinE = jnp.sin(E)
    cosE = jnp.cos(E)
    nu = jnp.arctan2(jnp.sqrt(1.0 - e**2) * sinE, cosE - e)

    # Distance
    r_mag = a * (1.0 - e * cosE)

    # Rotation to ecliptic frame (Perifocal → Ecliptic)
    cos_O = jnp.cos(O)
    sin_O = jnp.sin(O)
    cos_I = jnp.cos(I)
    sin_I = jnp.sin(I)

    cos_wnu = jnp.cos(w + nu)
    sin_wnu = jnp.sin(w + nu)

    x = r_mag * (cos_O * cos_wnu - sin_O * sin_wnu * cos_I)
    y = r_mag * (sin_O * cos_wnu + cos_O * sin_wnu * cos_I)
    z = r_mag * (sin_wnu * sin_I)

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
    return eclip2equat(r_eclip, mjd)


def earth_position_ecliptic(mjd: float) -> jnp.ndarray:
    """Heliocentric ecliptic position of Earth (AU)."""
    return planet_position_ecliptic("Earth", mjd)


def earth_position_equatorial(mjd: float) -> jnp.ndarray:
    """Heliocentric equatorial position of Earth (AU)."""
    return planet_position_equatorial("Earth", mjd)


# ---------------------------------------------------------------------------
# Ecliptic coordinate helpers for targets
# ---------------------------------------------------------------------------


def radec_to_ecliptic(ra_rad: float, dec_rad: float, mjd: float) -> tuple:
    """Convert equatorial RA/Dec to ecliptic longitude/latitude.

    Args:
        ra_rad: Right ascension in radians.
        dec_rad: Declination in radians.
        mjd: MJD for obliquity calculation.

    Returns:
        (ecliptic_lon_rad, ecliptic_lat_rad) tuple.
    """
    obe = jnp.radians(obliquity_deg(mjd))
    cos_obe = jnp.cos(obe)
    sin_obe = jnp.sin(obe)

    sin_ra = jnp.sin(ra_rad)
    cos_ra = jnp.cos(ra_rad)
    sin_dec = jnp.sin(dec_rad)
    cos_dec = jnp.cos(dec_rad)

    # Ecliptic latitude
    sin_beta = sin_dec * cos_obe - cos_dec * sin_obe * sin_ra
    beta = jnp.arcsin(sin_beta)

    # Ecliptic longitude
    y = sin_ra * cos_obe + jnp.tan(dec_rad) * sin_obe
    x = cos_ra
    lam = jnp.arctan2(y, x)

    return lam, beta


def sun_target_angle(
    obs_position_eclip: jnp.ndarray,
    ra_rad: float,
    dec_rad: float,
    mjd: float,
) -> float:
    """Angular separation between the Sun and a target as seen from the observatory.

    Args:
        obs_position_eclip: Observatory position in heliocentric ecliptic (AU), shape ``(3,)``.
        ra_rad: Target right ascension in radians.
        dec_rad: Target declination in radians.
        mjd: MJD for coordinate conversion.

    Returns:
        Angular separation in radians.
    """
    # Sun direction from observatory (heliocentric, so Sun is at origin)
    sun_dir = -obs_position_eclip
    sun_dir = sun_dir / jnp.linalg.norm(sun_dir)

    # Target direction (at infinity, so just the unit vector in ecliptic frame)
    # Convert RA/Dec to ecliptic Cartesian
    lam, beta = radec_to_ecliptic(ra_rad, dec_rad, mjd)
    target_dir = jnp.array([
        jnp.cos(beta) * jnp.cos(lam),
        jnp.cos(beta) * jnp.sin(lam),
        jnp.sin(beta),
    ])

    # Angular separation
    cos_angle = jnp.dot(sun_dir, target_dir)
    cos_angle = jnp.clip(cos_angle, -1.0, 1.0)
    return jnp.arccos(cos_angle)


def solar_elongation_ecliptic(
    obs_position_eclip: jnp.ndarray,
    ecliptic_lon_rad: float,
    ecliptic_lat_rad: float,
) -> float:
    """Solar elongation (angle between Sun and target as seen from observer).

    Args:
        obs_position_eclip: Observatory heliocentric ecliptic position (AU).
        ecliptic_lon_rad: Target ecliptic longitude (rad).
        ecliptic_lat_rad: Target ecliptic latitude (rad).

    Returns:
        Solar elongation in radians.
    """
    sun_dir = -obs_position_eclip
    sun_dir = sun_dir / jnp.linalg.norm(sun_dir)

    target_dir = jnp.array([
        jnp.cos(ecliptic_lat_rad) * jnp.cos(ecliptic_lon_rad),
        jnp.cos(ecliptic_lat_rad) * jnp.sin(ecliptic_lon_rad),
        jnp.sin(ecliptic_lat_rad),
    ])

    cos_angle = jnp.dot(sun_dir, target_dir)
    cos_angle = jnp.clip(cos_angle, -1.0, 1.0)
    return jnp.arccos(cos_angle)
