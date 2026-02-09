"""Sun/Earth/Moon keepout zone checks.

Determines whether a target star is observable based on angular separation
from bright solar system bodies.  Port of the keepout logic from
``EXOSIMS.Prototypes.Observatory.keepout``, simplified to Sun + Earth + Moon.
"""

from __future__ import annotations

import jax.numpy as jnp

from orbix.system.solar_system import (
    earth_position_ecliptic,
    planet_position_ecliptic,
    radec_to_ecliptic,
)


def _unit_vector(v: jnp.ndarray) -> jnp.ndarray:
    """Normalize vector to unit length."""
    return v / jnp.maximum(jnp.linalg.norm(v), 1e-30)


def _angular_sep(v1: jnp.ndarray, v2: jnp.ndarray) -> float:
    """Angular separation between two direction vectors (radians)."""
    u1 = _unit_vector(v1)
    u2 = _unit_vector(v2)
    cos_a = jnp.clip(jnp.dot(u1, u2), -1.0, 1.0)
    return jnp.arccos(cos_a)


def _target_ecliptic_dir(ra_rad: float, dec_rad: float, mjd: float) -> jnp.ndarray:
    """Unit vector toward a target in heliocentric ecliptic frame."""
    lam, beta = radec_to_ecliptic(ra_rad, dec_rad, mjd)
    return jnp.array([
        jnp.cos(beta) * jnp.cos(lam),
        jnp.cos(beta) * jnp.sin(lam),
        jnp.sin(beta),
    ])


def body_angle(
    obs_pos_eclip: jnp.ndarray,
    body_pos_eclip: jnp.ndarray,
    ra_rad: float,
    dec_rad: float,
    mjd: float,
) -> float:
    """Angle between a solar system body and a target as seen from the observatory.

    Args:
        obs_pos_eclip: Observatory heliocentric ecliptic position (AU).
        body_pos_eclip: Body heliocentric ecliptic position (AU).
        ra_rad: Target RA in radians.
        dec_rad: Target Dec in radians.
        mjd: MJD for coordinate conversion.

    Returns:
        Angular separation in radians.
    """
    # Direction from observer to body
    body_dir = body_pos_eclip - obs_pos_eclip
    # Direction from observer to target (at infinity)
    target_dir = _target_ecliptic_dir(ra_rad, dec_rad, mjd)
    return _angular_sep(body_dir, target_dir)


def is_observable(
    obs_pos_eclip: jnp.ndarray,
    ra_rad: float,
    dec_rad: float,
    mjd: float,
    ko_sun_min_deg: float = 45.0,
    ko_sun_max_deg: float = 180.0,
    ko_earth_min_deg: float = 0.0,
    ko_earth_max_deg: float = 180.0,
    ko_moon_min_deg: float = 0.0,
    ko_moon_max_deg: float = 180.0,
) -> bool:
    """Check if a target is observable (outside all keepout zones).

    Args:
        obs_pos_eclip: Observatory heliocentric ecliptic position (AU), shape ``(3,)``.
        ra_rad: Target right ascension in radians.
        dec_rad: Target declination in radians.
        mjd: Modified Julian Date.
        ko_sun_min_deg: Minimum Sun keepout angle (degrees).
        ko_sun_max_deg: Maximum Sun keepout angle (degrees).
        ko_earth_min_deg: Minimum Earth keepout angle (degrees).
        ko_earth_max_deg: Maximum Earth keepout angle (degrees).
        ko_moon_min_deg: Minimum Moon keepout angle (degrees).
        ko_moon_max_deg: Maximum Moon keepout angle (degrees).

    Returns:
        True if the target is observable, False if it falls in a keepout zone.
    """
    # Sun is at origin in heliocentric frame
    sun_pos = jnp.zeros(3)
    sun_angle_rad = body_angle(obs_pos_eclip, sun_pos, ra_rad, dec_rad, mjd)
    sun_angle_deg = jnp.degrees(sun_angle_rad)
    sun_ok = (sun_angle_deg >= ko_sun_min_deg) & (sun_angle_deg <= ko_sun_max_deg)

    # Earth
    earth_pos = earth_position_ecliptic(mjd)
    earth_angle_rad = body_angle(obs_pos_eclip, earth_pos, ra_rad, dec_rad, mjd)
    earth_angle_deg = jnp.degrees(earth_angle_rad)
    earth_ok = (earth_angle_deg >= ko_earth_min_deg) & (earth_angle_deg <= ko_earth_max_deg)

    # Moon (approximate: Earth position + offset)
    # For now, use Earth position as a proxy for the Moon since the Moon
    # is only ~0.0026 AU from Earth, which is small compared to L2 distance.
    # A more accurate model would use a lunar ephemeris.
    moon_ok = earth_ok  # Simplified — refine later with lunar ephemeris

    return sun_ok & earth_ok & moon_ok
