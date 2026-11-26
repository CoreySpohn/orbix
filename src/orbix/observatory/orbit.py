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
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import MJD_ZERO_JD
from hwoutils.conversions import days_to_years, years_to_days
from hwoutils.epoch import Epoch
from hwoutils.precision import like
from jaxtyping import Array, ArrayLike

from orbix.observatory.keepout import body_angle_deg
from orbix.observatory.solar_system import earth_position_ecliptic


def epoch_mjd(epoch: Epoch, time_d: ArrayLike) -> np.ndarray:
    """Modified Julian Date of ``time_d`` days from ``epoch``, for host use only.

    An absolute date does not belong under a trace: in float32 a Julian date
    near 2.5e6 d resolves only a quarter of a day and an MJD near 6e4 d only
    0.004 d. The traced methods take ``(epoch, time_d)`` instead, and
    this helper serves plotting and reporting on concrete values.

    Args:
        epoch: Static day zero of time_d.
        time_d: Days from epoch, any shape, concrete.

    Returns:
        Modified Julian Date in days on the epoch's time scale, NumPy float64,
        with the shape of time_d.

    Raises:
        TypeError: If ``time_d`` is a traced value.
    """
    if isinstance(time_d, jax.core.Tracer):
        raise TypeError(
            "epoch_mjd is host-only: it would round an absolute date in the "
            "active dtype under a trace; pass (epoch, time_d) to the traced "
            "method instead"
        )
    offset = epoch.days_between(Epoch(MJD_ZERO_JD, scale=epoch.scale))
    return offset + np.asarray(time_d, dtype=np.float64)


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

    The reference equinox and the start offset only feed a host-side reduction
    of the halo phase at the query epoch, so they are static. The tabulated
    times and positions are the array leaves.

    Attributes:
        t_yr: Halo sample times in years from the reference equinox.
        r_earth_AU: Earth-relative halo vectors at each sample, in AU.

    Example:
        >>> obs = ObservatoryL2Halo.from_default()
        >>> epoch = Epoch(2400000.5, 60000.0)
        >>> pos = obs.position_ecliptic(epoch, 10.0)  # AU, shape (3,)
    """

    t_yr: Array
    r_earth_AU: Array
    _period_yr: float = eqx.field(static=True)  # halo period in years
    _L2_dist_AU: float = eqx.field(static=True)  # L2 distance from Sun in AU
    _mu: float = eqx.field(static=True)  # mass ratio
    # reference equinox, a date on whatever scale the query epoch carries
    _equinox: Epoch = eqx.field(static=True)
    _halo_start_yr: float = eqx.field(static=True)  # halo offset at day zero (yr)

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
        return cls._from_data(_load_default_halo_data(), equinox_mjd, halo_start_day)

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
        with np.load(npz_path) as data:
            return cls._from_data(data, equinox_mjd, halo_start_day)

    @classmethod
    def _from_data(cls, data, equinox_mjd, halo_start_day):
        """Build from one time axis and the first three position components."""
        time, position = data["t_yr"], data["r_earth_AU"]
        if time.ndim != 1 or position.ndim != 2 or position.shape[0] != time.size:
            raise ValueError("Halo times and positions must share one sample axis")
        if position.shape[1] < 3:
            raise ValueError("Halo positions must have three Cartesian components")
        dtype = jnp.result_type(time, position, 0.0)
        return cls(
            t_yr=jnp.asarray(time, dtype=dtype),
            r_earth_AU=jnp.asarray(position[:, :3], dtype=dtype),
            _period_yr=float(data["period_yr"]),
            _L2_dist_AU=float(data["L2_dist_AU"]),
            _mu=float(data["mu"]),
            _equinox=Epoch(MJD_ZERO_JD, equinox_mjd),
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

    def _phase0_d(self, epoch: Epoch) -> float:
        """Halo phase at ``time_d = 0`` in days, reduced on the host in float64.

        ``(days from the equinox to the epoch + the start offset) mod period``:
        a number smaller than the period, however far the epoch is from the
        equinox.
        """
        equinox = Epoch(self._equinox.jd1, self._equinox.jd2, scale=epoch.scale)
        offset_d = epoch.days_between(equinox) + years_to_days(self._halo_start_yr)
        return offset_d % years_to_days(self._period_yr)

    def _halo_time(self, epoch: Epoch, time_d: ArrayLike) -> Array:
        """Periodic halo time in years at ``time_d`` days from ``epoch``."""
        period_d = like(years_to_days(self._period_yr), time_d)
        # Reduce time_d first so the sum stays below two periods: a sum of
        # phase0 and a time of 1000 d would double the float32 spacing.
        phase_d = (self._phase0_d(epoch) + time_d % period_d) % period_d
        return days_to_years(phase_d)

    def position_ecliptic(self, epoch: Epoch, time_d: ArrayLike) -> jnp.ndarray:
        """Heliocentric ecliptic position of the telescope at ``time_d`` from ``epoch``.

        Args:
            epoch: Static day zero of ``time_d`` (close over it or mark it
                static under ``jax.jit``); the time scale is not converted.
            time_d: Days from ``epoch``, scalar; vmap over times.

        Returns:
            Position vector in heliocentric ecliptic frame (AU), shape ``(3,)``.
        """
        time_d = jnp.asarray(time_d, dtype=jnp.result_type(time_d, self.t_yr))
        t_halo = self._halo_time(epoch, time_d)

        r_halo_x, r_halo_y, r_halo_z = interpax.interp1d(
            t_halo,
            self.t_yr,
            self.r_earth_AU,
            method="linear",
            extrap=like(jnp.nan, self.r_earth_AU),
        )

        # Get Earth position
        r_earth = earth_position_ecliptic(epoch, time_d)
        r_earth_norm = jnp.sqrt(r_earth[0] ** 2 + r_earth[1] ** 2)

        # Add radial offset (Earth distance) to halo x component
        r_halo_x = r_halo_x + r_earth_norm

        # Rotate halo position by Earth's ecliptic longitude
        # EXOSIMS applies rot(-lon, 3) @ r_halo, where rot has the
        # convention [[cos, sin], [-sin, cos]], giving the effective
        # rotation: x' = cos(lon)*x - sin(lon)*y
        #           y' = sin(lon)*x + cos(lon)*y
        # cos(lon) and sin(lon) come straight from the Earth vector: an arccos
        # of the ratio loses half the digits near lon = 0 or pi (3.5e-4 rad in
        # float32).
        safe_norm = jnp.maximum(r_earth_norm, like(1e-30, r_earth_norm))
        cos_lon = r_earth[0] / safe_norm
        sin_lon = r_earth[1] / safe_norm

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
        obs_pos = self.position_ecliptic(epoch, time_d)
        return body_angle_deg(obs_pos, jnp.zeros_like(obs_pos), ra_deg, dec_deg)
