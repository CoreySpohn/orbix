"""Zodiacal Fzodi helpers for use with pyEDITH's JAX ETC.

The JAX ETC's ``count_rate_zodi`` expects ``Fzodi`` — a dimensionless zodiacal
flux ratio per arcsec² — defined as::

    Fzodi = 10^(-0.4 x mag_per_arcsec2)

so that ``F0 x Fzodi`` gives the zodiacal surface brightness in
physical flux units (ph/s/cm²/nm/arcsec²).

Two approaches are provided:

    1. :func:`zodi_fzodi_ayo` — AYO default: 22 V-mag/arcsec² colour-corrected.
       Position-independent; matches existing yield sims.

    2. :func:`zodi_fzodi_leinert` — Full Leinert position-dependent brightness.
       Takes ecliptic latitude and solar longitude from the observatory module.
"""

from __future__ import annotations

import jax.numpy as jnp

from orbix.observatory.zodiacal import (
    ayo_default_zodi_mag,
    leinert_zodi_mag,
)


def zodi_fzodi_ayo(wavelength_nm: float) -> float:
    """AYO-default Fzodi: ``10^(-0.4 x 22.0)`` colour-corrected.

    Position-independent.  Matches ``calc_zodi_flux`` in pyEDITH when
    observations are near ecliptic pole (sin β ≈ 1) and solar lon ≈ 135°.

    Args:
        wavelength_nm: Observation wavelength in nm.

    Returns:
        Fzodi in arcsec⁻² (dimensionless).
    """
    mag = ayo_default_zodi_mag(wavelength_nm)
    return 10.0 ** (-0.4 * mag)


def zodi_fzodi_leinert(
    wavelength_nm: float,
    ecliptic_lat_deg: float = 0.0,
    solar_lon_deg: float = 135.0,
) -> float:
    """Position-dependent Fzodi from Leinert et al. (1998).

    Uses the full Leinert Table 17 (position) and Table 19 (wavelength)
    to compute the zodiacal surface brightness, then converts to the
    dimensionless Fzodi format expected by pyEDITH.

    Args:
        wavelength_nm: Observation wavelength in nm.
        ecliptic_lat_deg: Ecliptic latitude in degrees.
        solar_lon_deg: Solar longitude in degrees.

    Returns:
        Fzodi in arcsec⁻² (dimensionless).
    """
    mag = leinert_zodi_mag(wavelength_nm, ecliptic_lat_deg, solar_lon_deg)
    return 10.0 ** (-0.4 * mag)
