"""Observables of last-axis (east, north, away) relative Cartesian vectors.

Positions are AU and velocities AU/day. Distances in parsecs broadcast
against the leading vector shape. No orbit state or Kepler solve is needed.
"""

import jax
import jax.numpy as jnp
from hwoutils.constants import AU2m, d2s, pc2AU, rad2arcsec


def offset_arcsec(r_AU, dist_pc):
    """Return east (Delta alpha cos delta) and north offsets in arcseconds.

    Args:
        r_AU: Relative positions in AU, last axis (east, north, away).
        dist_pc: Distance in parsecs, broadcast against the leading shape of r_AU.

    Returns:
        Tuple of east and north offsets in arcseconds, each with the leading shape of
        r_AU.
    """
    dtype = jnp.result_type(*jax.tree.leaves((r_AU, dist_pc)), 0.0)
    r = jnp.asarray(r_AU, dtype=dtype)
    distance = jnp.asarray(dist_pc, dtype=dtype)
    distance = distance.reshape(distance.shape + (1,) * (r.ndim - 1 - distance.ndim))
    factor = rad2arcsec / (pc2AU * distance)
    return r[..., 0] * factor, r[..., 1] * factor


def separation_arcsec(r_AU, dist_pc):
    """Return projected angular separation in arcseconds.

    Args:
        r_AU: Relative positions in AU, last axis (east, north, away).
        dist_pc: Distance in parsecs, broadcast against the leading shape of r_AU.

    Returns:
        Projected angular separation in arcseconds, with the leading shape of r_AU.
    """
    east, north = offset_arcsec(r_AU, dist_pc)
    return jnp.hypot(east, north)


def position_angle_deg(r_AU):
    """Return position angle east of north in [0, 360) degrees.

    Args:
        r_AU: Relative positions in AU, last axis (east, north, away).

    Returns:
        Position angle east of north in degrees, in [0, 360).
    """
    r = jnp.asarray(r_AU)
    angle = jnp.rad2deg(jnp.arctan2(r[..., 0], r[..., 1])) % 360
    return jnp.where(angle == 360, jnp.zeros_like(angle), angle)


def phase_angle_rad(r_AU):
    """Return illumination phase, zero behind the reference, pi in front.

    The observer is at infinity along negative z. Pole values are finite;
    gradients at the poles and the angle of a zero vector are undefined.

    Args:
        r_AU: Relative positions in AU, last axis (east, north, away).

    Returns:
        Phase angle in radians in [0, pi].
    """
    r = jnp.asarray(r_AU)
    return jnp.arctan2(jnp.hypot(r[..., 0], r[..., 1]), r[..., 2])


def radial_velocity_m_per_s(v_AU_per_d):
    """Return line-of-sight velocity in m/s, positive receding.

    Args:
        v_AU_per_d: Relative velocities in AU/day, last axis (east, north, away).

    Returns:
        Line-of-sight velocity in m/s, positive when receding.
    """
    return jnp.asarray(v_AU_per_d)[..., 2] * (AU2m / d2s)


def distance_AU(r_AU):
    """Return three-dimensional distance in AU.

    Args:
        r_AU: Relative positions in AU, last axis (east, north, away).

    Returns:
        Three-dimensional distance in AU.
    """
    return jnp.linalg.norm(jnp.asarray(r_AU), axis=-1)
