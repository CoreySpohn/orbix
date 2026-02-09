"""L2 halo orbit model for space observatories.

Provides an equinox module that interpolates a pre-computed L2 halo orbit
to give telescope position at any time.  Based on the EXOSIMS
``ObservatoryL2Halo`` implementation.

The halo orbit data is stored as an ``.npz`` file (converted from the EXOSIMS
MATLAB data by ``scripts/convert_halo_mat.py``).
"""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import interpax
import jax.numpy as jnp
import numpy as np

from orbix.system.solar_system import (
    earth_position_ecliptic,
    radec_to_ecliptic,
    solar_elongation_ecliptic,
)


def _load_default_halo_data() -> dict:
    """Load the bundled L2 halo orbit .npz data."""
    data_dir = Path(__file__).parent / "data"
    npz_path = data_dir / "L2_halo_orbit_six_month.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Halo orbit data not found at {npz_path}. "
            "Run scripts/convert_halo_mat.py to generate it."
        )
    return dict(np.load(npz_path))


class ObservatoryL2Halo(eqx.Module):
    """Space telescope on an L2 halo orbit.

    This is an equinox module that stores the halo orbit state and provides
    JIT-compatible methods for position and geometry queries.

    The orbit is parameterized as a periodic interpolation of a ~6-month halo
    around the Sun-Earth L2 point.

    Args:
        equinox_mjd: Reference equinox epoch in MJD (default: 60575.25).
        halo_start_day: Offset into halo orbit at mission start (days).

    Example:
        >>> obs = ObservatoryL2Halo.from_default()
        >>> pos = obs.position_ecliptic(60000.0)  # AU, shape (3,)
    """

    # Pre-built interpolators for halo position components
    _interp_x: interpax.Interpolator1D
    _interp_y: interpax.Interpolator1D
    _interp_z: interpax.Interpolator1D
    _period_yr: float          # halo period in years
    _L2_dist_AU: float         # L2 distance from Sun in AU
    _mu: float                 # mass ratio
    _equinox_mjd: float        # reference equinox (MJD)
    _halo_start_yr: float      # offset into halo at mission start (years)
    _d2yr: float = 1.0 / 365.25  # days to years conversion

    @classmethod
    def from_default(
        cls,
        equinox_mjd: float = 60575.25,
        halo_start_day: float = 0.0,
    ) -> "ObservatoryL2Halo":
        """Create from bundled L2 halo orbit data.

        Args:
            equinox_mjd: Reference equinox epoch in MJD.
            halo_start_day: Offset into halo orbit at mission start (days).

        Returns:
            Configured ObservatoryL2Halo instance.
        """
        data = _load_default_halo_data()
        t_yr = jnp.array(data["t_yr"])
        r = jnp.array(data["r_earth_AU"])
        return cls(
            _interp_x=interpax.Interpolator1D(t_yr, r[:, 0], method="linear"),
            _interp_y=interpax.Interpolator1D(t_yr, r[:, 1], method="linear"),
            _interp_z=interpax.Interpolator1D(t_yr, r[:, 2], method="linear"),
            _period_yr=float(data["period_yr"]),
            _L2_dist_AU=float(data["L2_dist_AU"]),
            _mu=float(data["mu"]),
            _equinox_mjd=equinox_mjd,
            _halo_start_yr=halo_start_day / 365.25,
        )

    @classmethod
    def from_npz(
        cls,
        npz_path: str,
        equinox_mjd: float = 60575.25,
        halo_start_day: float = 0.0,
    ) -> "ObservatoryL2Halo":
        """Create from a custom .npz file.

        Args:
            npz_path: Path to .npz file with halo orbit data.
            equinox_mjd: Reference equinox epoch in MJD.
            halo_start_day: Offset into halo orbit at mission start (days).

        Returns:
            Configured ObservatoryL2Halo instance.
        """
        data = dict(np.load(npz_path))
        t_yr = jnp.array(data["t_yr"])
        r = jnp.array(data["r_earth_AU"])
        return cls(
            _interp_x=interpax.Interpolator1D(t_yr, r[:, 0], method="linear"),
            _interp_y=interpax.Interpolator1D(t_yr, r[:, 1], method="linear"),
            _interp_z=interpax.Interpolator1D(t_yr, r[:, 2], method="linear"),
            _period_yr=float(data["period_yr"]),
            _L2_dist_AU=float(data["L2_dist_AU"]),
            _mu=float(data["mu"]),
            _equinox_mjd=equinox_mjd,
            _halo_start_yr=halo_start_day / 365.25,
        )

    @property
    def period_yr(self) -> float:
        """Halo orbital period in years."""
        return self._period_yr

    @property
    def L2_dist_AU(self) -> float:
        """Sun-L2 distance in AU."""
        return self._L2_dist_AU

    def _halo_time(self, mjd: float) -> float:
        """Convert MJD to periodic halo time in years."""
        dt_yr = (mjd - self._equinox_mjd) * self._d2yr + self._halo_start_yr
        return dt_yr % self._period_yr

    def position_ecliptic(self, mjd: float) -> jnp.ndarray:
        """Heliocentric ecliptic position of the telescope at time ``mjd``.

        Args:
            mjd: Modified Julian Date (scalar).

        Returns:
            Position vector in heliocentric ecliptic frame (AU), shape ``(3,)``.
        """
        t_halo = self._halo_time(mjd)

        # Evaluate pre-built interpolators
        r_halo_x = self._interp_x(t_halo)
        r_halo_y = self._interp_y(t_halo)
        r_halo_z = self._interp_z(t_halo)

        # Get Earth position
        r_earth = earth_position_ecliptic(mjd)
        r_earth_norm = jnp.sqrt(r_earth[0] ** 2 + r_earth[1] ** 2)

        # Add radial offset (Earth distance) to halo x component
        r_halo_x = r_halo_x + r_earth_norm

        # Rotate halo position by Earth's ecliptic longitude
        # EXOSIMS applies rot(-lon, 3) @ r_halo, where rot has the
        # convention [[cos, sin], [-sin, cos]], giving the effective
        # rotation: x' = cos(lon)*x - sin(lon)*y
        #           y' = sin(lon)*x + cos(lon)*y
        lon = jnp.sign(r_earth[1]) * jnp.arccos(
            r_earth[0] / jnp.maximum(r_earth_norm, 1e-30)
        )
        cos_lon = jnp.cos(lon)
        sin_lon = jnp.sin(lon)

        # Apply rotation about z-axis (matches EXOSIMS sign convention)
        x = cos_lon * r_halo_x - sin_lon * r_halo_y
        y = sin_lon * r_halo_x + cos_lon * r_halo_y
        z = r_halo_z

        return jnp.array([x, y, z])

    def sun_angle(self, mjd: float, ra_rad: float, dec_rad: float) -> float:
        """Angular separation between Sun and target as seen from the telescope.

        Args:
            mjd: Modified Julian Date.
            ra_rad: Target right ascension in radians.
            dec_rad: Target declination in radians.

        Returns:
            Angular separation in radians.
        """
        obs_pos = self.position_ecliptic(mjd)
        lam, beta = radec_to_ecliptic(ra_rad, dec_rad, mjd)
        return solar_elongation_ecliptic(obs_pos, lam, beta)

    def solar_longitude(
        self, mjd: float, ra_rad: float, dec_rad: float
    ) -> float:
        """Solar longitude at target position (degrees).

        This is the angle between the anti-solar direction and the target,
        projected onto the ecliptic plane. Used for Leinert table lookup.

        Args:
            mjd: Modified Julian Date.
            ra_rad: Target right ascension in radians.
            dec_rad: Target declination in radians.

        Returns:
            Solar longitude in degrees [0, 180].
        """
        angle_rad = self.sun_angle(mjd, ra_rad, dec_rad)
        return jnp.degrees(angle_rad)

    def ecliptic_latitude(
        self, ra_rad: float, dec_rad: float, mjd: float = 51544.5
    ) -> float:
        """Ecliptic latitude of target in degrees.

        Args:
            ra_rad: Target right ascension in radians.
            dec_rad: Target declination in radians.
            mjd: MJD for obliquity (default J2000).

        Returns:
            Ecliptic latitude in degrees.
        """
        _, beta = radec_to_ecliptic(ra_rad, dec_rad, mjd)
        return jnp.degrees(beta)
