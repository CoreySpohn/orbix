"""Observatory module for telescope orbit modeling and observing conditions.

Provides:
    - :class:`ObservatoryL2Halo` — L2 halo orbit equinox module
    - Zodiacal light model (Leinert et al. 1998)
    - Keepout zone calculations (Sun/Earth/Moon)
    - ETC-ready zodiacal flux helpers
"""

from orbix.observatory.keepout import body_angle, is_observable
from orbix.observatory.orbit import ObservatoryL2Halo
from orbix.observatory.zodi_fzodi import zodi_fzodi_ayo, zodi_fzodi_leinert
from orbix.observatory.zodiacal import (
    AYO_DEFAULT_ZODI_MAG_V,
    ayo_default_zodi_flux_jy,
    ayo_default_zodi_mag,
    create_zodi_spectrum_jax,
    flux_to_mag_jy,
    leinert_zodi_factor,
    leinert_zodi_mag,
    leinert_zodi_spectral_radiance,
    mag_to_flux_jy,
    zodi_color_correction,
)

__all__ = [
    "ObservatoryL2Halo",
    "is_observable",
    "body_angle",
    "zodi_fzodi_ayo",
    "zodi_fzodi_leinert",
    "AYO_DEFAULT_ZODI_MAG_V",
    "ayo_default_zodi_flux_jy",
    "ayo_default_zodi_mag",
    "create_zodi_spectrum_jax",
    "flux_to_mag_jy",
    "leinert_zodi_factor",
    "leinert_zodi_mag",
    "leinert_zodi_spectral_radiance",
    "mag_to_flux_jy",
    "zodi_color_correction",
]
