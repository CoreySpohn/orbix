"""Observables of last-axis (east, north, away) relative Cartesian vectors.

Positions are AU and velocities AU/day. Distances in parsecs broadcast
against the leading vector shape. No orbit state or Kepler solve is needed.
"""

import jax.numpy as jnp
from hwoutils.constants import AU2m, d2s, pc2AU, rad2arcsec


def offset_arcsec(r_AU, dist_pc):
    """Return east (Delta alpha cos delta) and north offsets in arcseconds."""
    r = jnp.asarray(r_AU)
    distance = jnp.asarray(dist_pc)
    distance = distance.reshape(distance.shape + (1,) * (r.ndim - 1 - distance.ndim))
    factor = rad2arcsec / (pc2AU * distance)
    return r[..., 0] * factor, r[..., 1] * factor


def separation_arcsec(r_AU, dist_pc):
    """Return projected angular separation in arcseconds."""
    east, north = offset_arcsec(r_AU, dist_pc)
    return jnp.hypot(east, north)


def position_angle_deg(r_AU):
    """Return position angle east of north in [0, 360) degrees."""
    r = jnp.asarray(r_AU)
    angle = jnp.rad2deg(jnp.arctan2(r[..., 0], r[..., 1])) % 360
    return jnp.where(angle == 360, 0.0, angle)


def phase_angle_rad(r_AU):
    """Return illumination phase, zero behind the reference, pi in front.

    The observer is at infinity along negative z. Pole values are finite;
    gradients at the poles and the angle of a zero vector are undefined.
    """
    r = jnp.asarray(r_AU)
    return jnp.arctan2(jnp.hypot(r[..., 0], r[..., 1]), r[..., 2])


def radial_velocity_m_per_s(v_AU_per_d):
    """Return line-of-sight velocity in m/s, positive receding."""
    return jnp.asarray(v_AU_per_d)[..., 2] * (AU2m / d2s)


def distance_AU(r_AU):
    """Return three-dimensional distance in AU."""
    return jnp.linalg.norm(jnp.asarray(r_AU), axis=-1)
