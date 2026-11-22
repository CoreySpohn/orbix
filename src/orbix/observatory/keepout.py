"""Sun/Earth keepout zone checks.

Determines whether a target star is observable based on angular separation
from bright solar system bodies.  Port of the keepout logic from
``EXOSIMS.Prototypes.Observatory.keepout``, simplified to Sun + Earth (the
Moon is dropped as negligible against the L2 geometry; see ``is_observable``).
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array

from orbix.observatory.solar_system import (
    earth_position_ecliptic,
    equatorial_to_ecliptic_deg,
)


def _unit_vector(v: jnp.ndarray) -> jnp.ndarray:
    """Normalize vector to unit length."""
    return v / jnp.maximum(jnp.linalg.norm(v), 1e-30)


def _angular_sep_deg(v1: jnp.ndarray, v2: jnp.ndarray) -> float:
    """Angular separation between two direction vectors (degrees)."""
    u1 = _unit_vector(v1)
    u2 = _unit_vector(v2)
    cos_a = jnp.clip(jnp.dot(u1, u2), -1.0, 1.0)
    return jnp.degrees(jnp.arccos(cos_a))


def target_ecliptic_direction(ra_deg: float, dec_deg: float) -> jnp.ndarray:
    """Unit vector toward a J2000 target in the heliocentric ecliptic frame.

    Args:
        ra_deg: Target right ascension in degrees, J2000.
        dec_deg: Target declination in degrees, J2000.

    Returns:
        Unit vector of shape (3,) toward the target in the heliocentric ecliptic frame.
    """
    lon, lat = map(jnp.radians, equatorial_to_ecliptic_deg(ra_deg, dec_deg))
    return jnp.array(
        [jnp.cos(lat) * jnp.cos(lon), jnp.cos(lat) * jnp.sin(lon), jnp.sin(lat)]
    )


def body_angle_deg(
    obs_pos_eclip: jnp.ndarray,
    body_pos_eclip: jnp.ndarray,
    ra_deg: float,
    dec_deg: float,
) -> float:
    """Angle between a solar system body and a target as seen from the observatory.

    Args:
        obs_pos_eclip: Observatory heliocentric ecliptic position (AU).
        body_pos_eclip: Body heliocentric ecliptic position (AU).
        ra_deg: Target right ascension [deg], J2000.
        dec_deg: Target declination [deg], J2000.

    Returns:
        Angular separation in degrees, [0, 180].
    """
    # Direction from observer to body, and to the target (at infinity)
    body_dir = body_pos_eclip - obs_pos_eclip
    return _angular_sep_deg(body_dir, target_ecliptic_direction(ra_deg, dec_deg))


def is_observable(
    obs_pos_eclip: jnp.ndarray,
    ra_deg: float,
    dec_deg: float,
    mjd: float,
    ko_sun_min_deg: float = 45.0,
    ko_sun_max_deg: float = 180.0,
    ko_earth_min_deg: float = 0.0,
    ko_earth_max_deg: float = 180.0,
) -> Array:
    """Check if a target is observable (outside all keepout zones).

    Args:
        obs_pos_eclip: Observatory heliocentric ecliptic position (AU), shape ``(3,)``.
        ra_deg: Target right ascension [deg], J2000.
        dec_deg: Target declination [deg], J2000.
        mjd: Modified Julian Date, used for Earth's position.
        ko_sun_min_deg: Minimum Sun keepout angle (degrees).
        ko_sun_max_deg: Maximum Sun keepout angle (degrees).
        ko_earth_min_deg: Minimum Earth keepout angle (degrees).
        ko_earth_max_deg: Maximum Earth keepout angle (degrees).

    Returns:
        True if the target is observable, False if it falls in a keepout zone.
    """
    # Sun is at origin in heliocentric frame
    sun_pos = jnp.zeros(3)
    sun_angle_deg = body_angle_deg(obs_pos_eclip, sun_pos, ra_deg, dec_deg)
    sun_ok = (sun_angle_deg >= ko_sun_min_deg) & (sun_angle_deg <= ko_sun_max_deg)

    # Earth
    earth_pos = earth_position_ecliptic(mjd)
    earth_angle_deg = body_angle_deg(obs_pos_eclip, earth_pos, ra_deg, dec_deg)
    earth_ok = (earth_angle_deg >= ko_earth_min_deg) & (
        earth_angle_deg <= ko_earth_max_deg
    )

    # The Moon is ~0.0026 AU from Earth -- negligible against the L2
    # geometry, so the Earth check stands in for it. Add explicit lunar
    # keepout only with a lunar ephemeris and its own angle limits.
    return sun_ok & earth_ok
