"""Ecliptic pointing angles of a target for the local zodiacal light model.

The local-zodi tables are indexed by the target's ecliptic latitude and by the
helio-ecliptic longitude difference (target longitude minus the Sun's
longitude). Both follow from the target's equatorial position, an epoch and a
time offset, with no further inputs.

Reference:
    U.S. Naval Observatory and H. M. Nautical Almanac Office, Astronomical
    Almanac, section C, "Low precision formulae for the Sun's coordinates"
    (about 0.01 deg between 1950 and 2050). Mean obliquity of the ecliptic at
    J2000 and the general precession in longitude from the IAU 2006
    precession model (Capitaine, Wallace and Chapront 2003, A&A 412, 567).
"""

from __future__ import annotations

import jax.numpy as jnp
from hwoutils.constants import J2000_JD
from hwoutils.epoch import Epoch
from jaxtyping import Array, ArrayLike

_OBLIQUITY_J2000_DEG = 84381.406 / 3600.0  # IAU 2006 mean obliquity at J2000
_PRECESSION_DEG_PER_DAY = 5028.796195 / 3600.0 / 36525.0  # general precession, J2000
_DAYS_PER_CENTURY = 36525.0


def sun_longitude_deg(epoch: Epoch, time_d: ArrayLike) -> Array:
    """Geocentric ecliptic longitude of the Sun in the J2000 ecliptic frame [deg].

    Astronomical Almanac low-precision formula (mean longitude, mean anomaly
    and the equation of centre), referred to the mean equinox of date, then
    shifted back to the J2000 equinox by the general precession so that it
    shares a frame with J2000 target coordinates. The observatory at Sun-Earth
    L2 lies on the Sun-Earth line, so the geocentric direction is used.

    Args:
        epoch: Static day zero of ``time_d`` (the time scale is not converted;
            the TT-UTC offset of about 70 s is far below the formula's accuracy).
        time_d: Days from ``epoch``. Differentiable.

    Returns:
        Longitude in degrees in [0, 360).
    """
    n = jnp.asarray(time_d) + epoch.days_between(Epoch(J2000_JD, scale=epoch.scale))
    mean_longitude = 280.460 + 0.9856474 * n
    anomaly = jnp.radians(357.528 + 0.9856003 * n)
    longitude = (
        mean_longitude + 1.915 * jnp.sin(anomaly) + 0.020 * jnp.sin(2.0 * anomaly)
    )
    return jnp.mod(longitude - _PRECESSION_DEG_PER_DAY * n, 360.0)


def ecliptic_pointing(
    ra_deg: ArrayLike, dec_deg: ArrayLike, epoch: Epoch, time_d: ArrayLike
) -> tuple[Array, Array]:
    """Ecliptic latitude and Sun-relative longitude of a target for the zodi model.

    The target position is equatorial J2000 (ICRS to well below the accuracy
    here), the frame sky positions carry. It is rotated to the J2000 ecliptic
    with the J2000 mean obliquity; the Sun's longitude comes from
    ``sun_longitude_deg``. The solar longitude returned is the helio-ecliptic
    longitude difference, target longitude minus the Sun's longitude, wrapped to
    [-180, 180) deg: the ``solar_longitude_deg`` the zodi models take (they use
    its absolute value, so the sign carries only east or west of the Sun).
    Elongation from the Sun is ``arccos(cos(beta) cos(dlon))``.

    Args:
        ra_deg: Target right ascension [deg].
        dec_deg: Target declination [deg].
        epoch: Static day zero of ``time_d``.
        time_d: Days from ``epoch``. Differentiable. All array arguments
            broadcast to one output shape.

    Returns:
        ``(ecliptic_latitude_deg, solar_longitude_deg)``. Close over ``epoch``
        or mark it static under ``jax.jit``. Resolution is about 1e-3 deg in
        float32 (the Sun's longitude is about 1e4 deg of mean longitude from
        J2000) and below 1e-6 deg in float64. The derivative in ``time_d`` is
        that of the mean solar motion plus the equation of centre. It is not
        defined at the ecliptic poles, where the longitude is degenerate.
    """
    ra = jnp.radians(ra_deg)
    dec = jnp.radians(dec_deg)
    cos_obl = jnp.cos(jnp.radians(_OBLIQUITY_J2000_DEG))
    sin_obl = jnp.sin(jnp.radians(_OBLIQUITY_J2000_DEG))
    sin_beta = jnp.sin(dec) * cos_obl - jnp.cos(dec) * sin_obl * jnp.sin(ra)
    beta = jnp.arcsin(jnp.clip(sin_beta, -1.0, 1.0))
    lon_y = jnp.sin(ra) * jnp.cos(dec) * cos_obl + jnp.sin(dec) * sin_obl
    lon_x = jnp.cos(ra) * jnp.cos(dec)
    target_lon = jnp.degrees(jnp.arctan2(lon_y, lon_x))
    dlon = target_lon - sun_longitude_deg(epoch, time_d)
    dlon = jnp.mod(dlon + 180.0, 360.0) - 180.0
    return jnp.degrees(beta), dlon
