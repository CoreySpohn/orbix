"""Ecliptic pointing angles of a target for the local zodiacal light model.

The local-zodi tables are indexed by the target's ecliptic latitude and by the
helioecliptic longitude (target longitude minus the Sun's longitude, Leinert et
al. 1998). Both follow from the target's equatorial position, an epoch and a
time offset, with no further inputs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from hwoutils.epoch import Epoch
from jaxtyping import Array, ArrayLike

from orbix.observatory.solar_system import (
    equatorial_to_ecliptic_deg,
    sun_ecliptic_longitude_deg,
)


def ecliptic_pointing(
    ra_deg: ArrayLike, dec_deg: ArrayLike, epoch: Epoch, time_d: ArrayLike
) -> tuple[Array, Array]:
    """Ecliptic latitude and Sun-relative longitude of a target for the zodi model.

    The target position is equatorial J2000 (ICRS to well below the accuracy
    here), the frame sky positions carry. It is rotated to the J2000 ecliptic
    with the J2000 mean obliquity; the Sun's longitude comes from
    ``sun_ecliptic_longitude_deg``. The longitude returned is the
    helioecliptic longitude, target longitude minus the Sun's longitude, wrapped
    to [-180, 180) deg: the ``helioecliptic_longitude_deg`` the zodi models take
    (they use its absolute value, so the sign carries only east or west of the
    Sun). Elongation from the Sun is ``arccos(cos(beta) cos(dlon))``.

    Args:
        ra_deg: Target right ascension [deg].
        dec_deg: Target declination [deg].
        epoch: Static day zero of ``time_d``.
        time_d: Days from ``epoch``. Differentiable. All array arguments
            broadcast to one output shape.

    Returns:
        ``(ecliptic_latitude_deg, helioecliptic_longitude_deg)``. Close over
        ``epoch`` or mark it static under ``jax.jit``. Resolution is about
        2e-4 deg in float32 for ``|time_d|`` up to 1000 d (the ephemeris is
        re-expanded about ``epoch`` in float64 on the host) and below 1e-6 deg
        in float64. The Sun's longitude is the
        geometric direction from the Vallado ephemeris (the apparent longitude is
        about 0.006 deg smaller, from aberration).
        The result is not defined at the ecliptic poles, where the longitude is
        degenerate.
    """
    dtype = jnp.result_type(*jax.tree.leaves((ra_deg, dec_deg, time_d)), 0.0)
    ra, dec, time = (
        jnp.asarray(value, dtype=dtype) for value in (ra_deg, dec_deg, time_d)
    )
    target_lon, beta = equatorial_to_ecliptic_deg(ra, dec)
    dlon = target_lon - sun_ecliptic_longitude_deg(epoch, time)
    dlon = (dlon + 180) % 360 - 180
    return beta, dlon
