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
from hwoutils.constants import MJD_ZERO_JD
from hwoutils.conversions import days_to_years
from hwoutils.epoch import Epoch
from jaxtyping import Array, ArrayLike

from orbix.observatory.keepout import body_angle_deg
from orbix.observatory.solar_system import earth_position_ecliptic


def epoch_mjd(epoch: Epoch, time_d: ArrayLike) -> Array:
    """Modified Julian Date of ``time_d`` days from ``epoch``, on the epoch's scale.

    Args:
        epoch: Static day zero of time_d.
        time_d: Days from epoch, any shape.

    Returns:
        Modified Julian Date in days on the epoch's time scale, with the shape of
        time_d.
    """
    return epoch.days_between(Epoch(MJD_ZERO_JD, scale=epoch.scale)) + jnp.asarray(
        time_d
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
    _period_yr: float  # halo period in years
    _L2_dist_AU: float  # L2 distance from Sun in AU
    _mu: float  # mass ratio
    _equinox_mjd: float  # reference equinox (MJD)
    _halo_start_yr: float  # offset into halo at mission start (years)

    @classmethod
    def from_default(
        cls,
        equinox_mjd: float = 60575.25,
        halo_start_day: float = 0.0,
    ) -> ObservatoryL2Halo:
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
            _halo_start_yr=days_to_years(halo_start_day),
        )

    @classmethod
    def from_npz(
        cls,
        npz_path: str,
        equinox_mjd: float = 60575.25,
        halo_start_day: float = 0.0,
    ) -> ObservatoryL2Halo:
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
            _halo_start_yr=days_to_years(halo_start_day),
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
        dt_yr = days_to_years(mjd - self._equinox_mjd) + self._halo_start_yr
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

    def solar_elongation_deg(
        self, ra_deg: ArrayLike, dec_deg: ArrayLike, epoch: Epoch, time_d: ArrayLike
    ) -> Array:
        """Solar elongation: Sun-target angle as seen from the telescope.

        This is not the helioecliptic longitude that indexes the zodiacal-light
        tables; for that use :func:`orbix.observatory.ecliptic_pointing`.

        Args:
            ra_deg: Target right ascension [deg], J2000.
            dec_deg: Target declination [deg], J2000.
            epoch: Static day zero of ``time_d``.
            time_d: Days from ``epoch``. Scalar; vmap over arrays.

        Returns:
            Solar elongation in degrees, [0, 180].
        """
        obs_pos = self.position_ecliptic(epoch_mjd(epoch, time_d))
        return body_angle_deg(obs_pos, jnp.zeros(3), ra_deg, dec_deg)
