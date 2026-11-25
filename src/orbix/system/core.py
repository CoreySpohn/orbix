"""Jacobi chains and reusable barycentric ephemerides.

See Murray and Dermott, Solar System Dynamics, for hierarchical coordinates.
The hierarchy is a set of independent Kepler orbits, not an N-body integrator.
"""

import dataclasses
from types import MappingProxyType
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import GM_SUN_AU3_PER_D2
from hwoutils.epoch import Epoch
from hwoutils.precision import like
from jaxtyping import Array

from orbix.kepler import Markley
from orbix.system import observables
from orbix.system.frame_orbit import FrameOrbit

_DEFAULT_SOLVER = Markley()


def _ordered_sum(values):
    """Return final sum and prefixes along axis zero, in fixed input order."""

    def add(total, value):
        total = total + value
        return total, total

    return jax.lax.scan(add, jnp.zeros(values.shape[1:], values.dtype), values)


def enclosed_mu(star_mass_Msun, planet_mass_Msun):
    """Return Jacobi mu for star shape B and inner-to-outer masses B + (K,).

    Args:
        star_mass_Msun: Star mass in solar masses, shape B.
        planet_mass_Msun: Planet masses in solar masses, ordered inner to outer, shape B
            + (K,).

    Returns:
        Gravitational parameter GM of the mass enclosed by each Jacobi orbit, in
        AU^3/day^2, shape B + (K,).
    """
    masses = jnp.asarray(planet_mass_Msun)
    _, cumulative = _ordered_sum(jnp.moveaxis(masses, -1, 0))
    return GM_SUN_AU3_PER_D2 * (
        jnp.asarray(star_mass_Msun)[..., None] + jnp.moveaxis(cumulative, 0, -1)
    )


def jacobi_to_barycentric(relative, star_mass_Msun, planet_mass_Msun):
    """Map B + (K,) + time_shape + (3,) to B + (K+1,) + time_shape + (3,).

    Both sums run sequentially inner to outer. Appending zero-mass bodies
    preserves the bits of every preceding body and the stellar reflex.

    Args:
        relative: Jacobi positions or velocities of each planet, shape B + (K,) +
            time_shape + (3,), in AU or AU/day.
        star_mass_Msun: Star mass in solar masses, shape B.
        planet_mass_Msun: Planet masses in solar masses, ordered inner to outer, shape B
            + (K,).

    Returns:
        Barycentric vectors in the units of relative, shape B + (K+1,) + time_shape +
        (3,). Index zero along the body axis is the star.
    """
    relative = jnp.asarray(relative)
    masses = jnp.asarray(planet_mass_Msun)
    axis = masses.ndim - 1
    _, cumulative_mass = _ordered_sum(jnp.moveaxis(masses, -1, 0))
    enclosed = jnp.asarray(star_mass_Msun)[..., None] + jnp.moveaxis(
        cumulative_mass, 0, -1
    )
    weights = masses / enclosed
    weighted = relative * weights.reshape(
        masses.shape + (1,) * (relative.ndim - masses.ndim)
    )
    total, cumulative = _ordered_sum(jnp.moveaxis(weighted, axis, 0))
    star = -total[None]
    preceding = jnp.concatenate([jnp.zeros_like(star), cumulative], axis=0)[:-1]
    relative = jnp.moveaxis(relative, axis, 0)
    return jnp.moveaxis(
        jnp.concatenate([star, star + preceding + relative], axis=0), 0, axis
    )


def barycentric_to_jacobi(values, star_mass_Msun, planet_mass_Msun):
    """Map B + (K+1,) + time_shape + (3,) to B + (K,) + time_shape + (3,).

    Each planet is measured from the center of mass of the star and all
    preceding planets, using ordered inner-to-outer sums. Positions or
    velocities may use any common origin, since only differences enter.
    Supply absolute snapshots as host numbers: their stellar origin is
    subtracted in float64 before transfer. Device arrays and traced
    arguments must already resolve the relative vectors.

    Args:
        values: Positions or velocities, star first and planets inner to outer,
            shape B + (K+1,) + time_shape + (3,), in any consistent units.
        star_mass_Msun: Star mass in solar masses, shape B.
        planet_mass_Msun: Planet masses in solar masses, shape B + (K,).

    Returns:
        Jacobi vectors in the units of values, shape B + (K,) + time_shape + (3,).
    """
    masses = jnp.asarray(planet_mass_Msun)
    axis = masses.ndim - 1
    if any(isinstance(x, jax.Array) for x in jax.tree.leaves(values)):
        rows = jnp.moveaxis(jnp.asarray(values), axis, 0)
        relative = rows[1:] - rows[0]
    else:
        rows = np.moveaxis(np.asarray(values, dtype=np.float64), axis, 0)
        relative = like(rows[1:] - rows[0], masses)
    _, cumulative_mass = _ordered_sum(jnp.moveaxis(masses, -1, 0))
    preceding_mass = jnp.concatenate(
        [jnp.zeros_like(cumulative_mass[:1]), cumulative_mass], axis=0
    )[:-1]
    enclosed = jnp.asarray(star_mass_Msun) + preceding_mass
    weights = jnp.moveaxis(masses, -1, 0).reshape(
        (masses.shape[-1], *masses.shape[:-1]) + (1,) * (relative.ndim - masses.ndim)
    )
    _, cumulative = _ordered_sum(relative * weights)
    preceding = jnp.concatenate([jnp.zeros_like(cumulative[:1]), cumulative], axis=0)[
        :-1
    ]
    center = preceding / enclosed.reshape(
        enclosed.shape + (1,) * (relative.ndim - enclosed.ndim)
    )
    return jnp.moveaxis(relative - center, 0, axis)


class Body(eqx.Module):
    """Body masses in solar masses, with a free batch shape.

    Attributes:
        mass_Msun: Body masses in solar masses, any batch shape.
    """

    mass_Msun: Array = eqx.field(converter=jnp.asarray)


def _optional_array(x):
    """Keep None, make anything else an array."""
    return None if x is None else jnp.asarray(x)


class SkyFrame(eqx.Module):
    """Distance, optional sky position and the reference epoch.

    ra_deg and dec_deg are the equatorial ICRS (J2000) position in degrees
    at the epoch, with no proper motion. Both are given or both are None
    (unplaced); when given they have the shape of dist_pc.

    The epoch is host-only metadata (hwoutils.epoch.Epoch), never a traced
    leaf. time_d converts host dates to offsets before any float32 device
    transfer. No time-scale conversion is applied.

    Attributes:
        dist_pc: Distance to the system in parsecs.
        epoch: Static reference epoch of the sky position and of the time offsets.
        ra_deg: Right ascension in degrees, ICRS J2000, shape of dist_pc, or None.
        dec_deg: Declination in degrees, ICRS J2000, shape of dist_pc, or None.
    """

    dist_pc: Array = eqx.field(converter=jnp.asarray)
    epoch: Epoch = eqx.field(static=True, default=Epoch())
    ra_deg: Array | None = eqx.field(converter=_optional_array, default=None)
    dec_deg: Array | None = eqx.field(converter=_optional_array, default=None)

    def __check_init__(self):
        """Require both or neither sky coordinate, shaped like dist_pc."""
        if (self.ra_deg is None) != (self.dec_deg is None):
            raise ValueError("ra_deg and dec_deg must be given together or not at all")
        if self.ra_deg is not None:
            shape = jnp.shape(self.dist_pc)
            if self.ra_deg.shape != shape or self.dec_deg.shape != shape:
                raise ValueError(
                    f"ra_deg and dec_deg must have the shape of dist_pc {shape}; "
                    f"got {self.ra_deg.shape} and {self.dec_deg.shape}"
                )

    def time_d(self, time=None, *, jd=None, mjd=None):
        """Days from the epoch, in NumPy float64 on the host.

        Exactly one of ``time``, ``jd`` and ``mjd`` is given; see
        ``hwoutils.epoch.Epoch.days``. Dates must use the epoch's time scale.

        Args:
            time: Object exposing jd1 and jd2 (for example astropy Time).
            jd: Julian date [d] as a number or array.
            mjd: Modified Julian date [d] (JD - 2400000.5) as a number or array.

        Returns:
            Days from the epoch as a NumPy float64 array.
        """
        return self.epoch.days(time, jd=jd, mjd=mjd)


class Ephemeris(eqx.Module):
    """Barycentric vectors of shape B + (K+1,) + time_shape + (3,).

    Index zero is the star. reference is a body index or "barycenter".
    All observables reuse these stored states and perform no further
    orbital solve. There is one distance per system: dist_pc has exactly
    the batch shape B. Its rank batch_ndim is derived from dist_pc and only
    declared (and checked) at construction (zero for one system). A distance
    history is not supported. Construction checks run in __post_init__, so
    a subclass must not define its own __init__.

    Attributes:
        position_AU: Barycentric positions in AU, shape B + (K+1,) + time_shape + (3,),
            in the (east, north, away) frame.
        velocity_AU_per_d: Barycentric velocities in AU/day, same shape as position_AU.
        dist_pc: Distance to each system in parsecs, shape exactly B.
        batch_ndim: Rank of B, derived from dist_pc. The construction keyword
            declares this rank and defaults to 0, one system.
    """

    position_AU: Array = eqx.field(converter=jnp.asarray)
    velocity_AU_per_d: Array = eqx.field(converter=jnp.asarray)
    dist_pc: Array = eqx.field(converter=jnp.asarray)
    batch_ndim: dataclasses.InitVar[int] = dataclasses.field(default=0, kw_only=True)

    def __post_init__(self, batch_ndim):
        """Validate the declared batch rank and one distance per system.

        Args:
            batch_ndim: Declared rank of the leading batch shape B.

        Raises:
            ValueError: The rank is not an integer in [0, vector ndim - 2], or
                dist_pc does not have exactly the batch shape B.
        """
        position_shape = jnp.shape(jnp.asarray(self.position_AU))
        if len(position_shape) < 2:
            raise ValueError("Ephemeris vectors must share shape (bodies, ..., 3)")
        if (
            not isinstance(batch_ndim, int)
            or isinstance(batch_ndim, bool)
            or not 0 <= batch_ndim <= len(position_shape) - 2
        ):
            raise ValueError(
                "batch_ndim must be an integer between 0 and vector ndim - 2"
            )
        batch = position_shape[:batch_ndim]
        distance_shape = jnp.shape(jnp.asarray(self.dist_pc))
        if distance_shape != batch:
            raise ValueError(
                "A distance history is not supported; dist_pc must have exactly "
                f"the batch shape B = {batch} (batch_ndim={batch_ndim}); "
                f"got {distance_shape}"
            )

    def __check_init__(self):
        """Validate the batch, body and vector shape contract."""
        if (
            self.position_AU.ndim < 2
            or self.position_AU.shape[-1] != 3
            or self.position_AU.shape != self.velocity_AU_per_d.shape
        ):
            raise ValueError("Ephemeris vectors must share shape (bodies, ..., 3)")

    def _relative(self, values, target, reference):
        if isinstance(reference, str):
            if reference != "barycenter":
                raise ValueError('reference must be a body index or "barycenter"')
            return self._body(values, target)
        target_values = self._body(values, target)
        reference_values = self._body(values, reference)
        axis = self.dist_pc.ndim
        while reference_values.ndim < target_values.ndim:
            reference_values = jnp.expand_dims(reference_values, axis)
        while target_values.ndim < reference_values.ndim:
            target_values = jnp.expand_dims(target_values, axis)
        return target_values - reference_values

    def _body(self, values, target):
        axis = self.dist_pc.ndim
        if jnp.ndim(target) > 1:
            indices = jnp.asarray(target).reshape(
                (*jnp.shape(target), *(1,) * (values.ndim - axis - 1))
            )
            return jnp.take_along_axis(values, indices, axis=axis)
        return jnp.take(values, target, axis=axis)

    def offset_arcsec(self, target, reference=0):
        """Return east and north angular offsets from the reference.

        Args:
            target: Index of the target body, where index 0 is the star.
            reference: Reference body index. Defaults to 0, the star, and may be the
                string "barycenter".

        Returns:
            Tuple of east and north angular offsets of the target from the reference in
            arcseconds.
        """
        return observables.offset_arcsec(
            self._relative(self.position_AU, target, reference), self.dist_pc
        )

    def separation_arcsec(self, target, reference=0):
        """Return projected separation from the reference in arcseconds.

        Args:
            target: Index of the target body, where index 0 is the star.
            reference: Reference body index. Defaults to 0, the star, and may be the
                string "barycenter".

        Returns:
            Projected angular separation in arcseconds.
        """
        return observables.separation_arcsec(
            self._relative(self.position_AU, target, reference), self.dist_pc
        )

    def position_angle_deg(self, target, reference=0):
        """Return position angle east of north in [0, 360) degrees.

        Args:
            target: Index of the target body, where index 0 is the star.
            reference: Reference body index. Defaults to 0, the star, and may be the
                string "barycenter".

        Returns:
            Position angle of the target east of north, in degrees in [0, 360).
        """
        return observables.position_angle_deg(
            self._relative(self.position_AU, target, reference)
        )

    def phase_angle_rad(self, target, reference=0):
        """Return reference-target-observer phase; zero behind the reference.

        Args:
            target: Index of the target body, where index 0 is the star.
            reference: Reference body index. Defaults to 0, the star, and may be the
                string "barycenter".

        Returns:
            Reference-target-observer phase angle in radians, zero when the target is
            behind the reference.
        """
        return observables.phase_angle_rad(
            self._relative(self.position_AU, target, reference)
        )

    def radial_velocity_m_per_s(self, target, reference="barycenter"):
        """Return receding-positive velocity relative to the reference, m/s.

        Args:
            target: Index of the target body, where index 0 is the star.
            reference: Reference body index or the string "barycenter". Defaults to
                "barycenter".

        Returns:
            Line-of-sight velocity of the target relative to the reference in m/s,
            positive when receding.
        """
        return observables.radial_velocity_m_per_s(
            self._relative(self.velocity_AU_per_d, target, reference)
        )

    def distance_AU(self, target, reference=0):
        """Return three-dimensional distance to the reference in AU.

        Args:
            target: Index of the target body, where index 0 is the star.
            reference: Reference body index. Defaults to 0, the star, and may be the
                string "barycenter".

        Returns:
            Three-dimensional distance between target and reference in AU.
        """
        return observables.distance_AU(
            self._relative(self.position_AU, target, reference)
        )


# Install after dataclass construction so the property is not the InitVar default.
Ephemeris.batch_ndim = property(
    lambda self: self.dist_pc.ndim,
    doc="Rank of the system batch, derived from dist_pc.",
)


class System(eqx.Module):
    """A Jacobi chain ordered inner to outer by the caller.

    Orbit j describes planet j relative to the center of mass of the star
    and all preceding planets. The star and distance have shape B; planet
    masses and orbit batch have shape B + (K,). Editing masses changes the
    barycentric map only: rebuild to impose dynamically consistent rates.

    The stored orbits are rated by the enclosed masses and their fields are tied
    together (size and mean motion by Kepler's third law, the eccentricity pair inside
    the unit disk, a unit quaternion), so ``tied_fields`` declares ``orbits`` derived
    from the star and planet masses (see ``hwoutils.blocks.check_ties``): a fit frees
    an orbit through a block, and a fit that frees a mass holds a block that re-rates
    the orbits.

    Attributes:
        tied_fields: Derived field to the fields it is derived from, relative to the
            system: the orbits from the star and planet masses.
        star: Star mass, shape B.
        planets: Planet masses ordered inner to outer, shape B + (K,).
        orbits: Jacobi orbit of each planet, batch shape B + (K,).
        frame: Distance, sky position and reference epoch, shape B.
    """

    tied_fields: ClassVar[MappingProxyType] = MappingProxyType(
        {"orbits": ("star.mass_Msun", "planets.mass_Msun")}
    )
    star: Body
    planets: Body
    orbits: FrameOrbit
    frame: SkyFrame

    def __check_init__(self):
        """Validate the body and vector shape contract."""
        if (
            self.planets.mass_Msun.shape[:-1] != self.star.mass_Msun.shape
            or self.planets.mass_Msun.ndim < 1
            or self.frame.dist_pc.shape != self.star.mass_Msun.shape
            or self.orbits.a_AU.shape != self.planets.mass_Msun.shape
        ):
            raise ValueError(
                "System requires star/distance shape B "
                "and planets/orbits shape B + (K,)"
            )

    @classmethod
    def from_classical(
        cls, star, planets, frame, *, a_AU, e, i_rad, W_rad, w_rad, M0_rad, epoch_d=0.0
    ):
        """Set orbit rates from the mass enclosed by each Jacobi orbit.

        Args:
            star: Star mass.
            planets: Planet masses ordered inner to outer.
            frame: Distance, sky position and reference epoch.
            a_AU: Semi-major axes in AU, shape B + (K,).
            e: Eccentricities, dimensionless.
            i_rad: Inclinations in radians.
            W_rad: Longitudes of the ascending node in radians, measured north through
                east.
            w_rad: Arguments of periapsis in radians.
            M0_rad: Mean anomalies at epoch_d in radians.
            epoch_d: Epoch in days on the time_d axis of the solve methods. Defaults to
                0.

        Returns:
            System whose orbit rates follow from the mass enclosed by each Jacobi orbit.
        """
        orbits = FrameOrbit.from_classical(
            a_AU,
            e,
            i_rad,
            W_rad,
            w_rad,
            M0_rad,
            mu_AU3_per_d2=enclosed_mu(star.mass_Msun, planets.mass_Msun),
            epoch_d=epoch_d,
        )
        return cls(star, planets, orbits, frame)

    @classmethod
    def from_cartesian(
        cls, star, planets, frame, *, position_AU, velocity_AU_per_d, epoch_d=0.0
    ):
        """Build a Jacobi chain from a Cartesian snapshot in any common origin.

        The chain is built from the bodies given only; a body left out is
        absent from the dynamics. Each orbit is relative to the barycenter
        of its interior bodies and rated by its enclosed mass.

        Args:
            star: Body with stellar mass, shape B.
            planets: Body with planet masses, shape B + (K,), inner to outer.
            frame: SkyFrame with distance, sky position and reference epoch.
            position_AU: Positions in AU in the (east, north, away) frame,
                shape B + (K+1, 3), star first and planets inner to outer.
                Each Jacobi state (planet relative to the barycenter of its
                interior bodies) must be bound and nonradial, as required by
                FrameOrbit.from_cartesian.
            velocity_AU_per_d: Velocities in AU/day, same shape and frame.
            epoch_d: One snapshot epoch per system in days, either a scalar
                or an array of shape exactly B. Defaults to 0.

        Returns:
            System reproducing the snapshot relative to the star at epoch_d.

        Raises:
            ValueError: Mass, Cartesian or epoch shapes do not match B and K.
        """
        batch = star.mass_Msun.shape
        if planets.mass_Msun.ndim < 1 or planets.mass_Msun.shape[:-1] != batch:
            raise ValueError("Planet masses must have shape B + (K,) for star shape B")
        expected = (*batch, planets.mass_Msun.shape[-1] + 1, 3)
        pos_shape, vel_shape = jnp.shape(position_AU), jnp.shape(velocity_AU_per_d)
        if pos_shape != expected or vel_shape != expected:
            raise ValueError(
                "Cartesian positions and velocities must have shape B + (K+1, 3) "
                f"= {expected}; got {pos_shape} and {vel_shape}"
            )
        epoch = jnp.asarray(epoch_d)
        if epoch.shape not in ((), batch):
            raise ValueError(
                "epoch_d must be a scalar or have exactly the batch shape B"
            )
        epoch = jnp.broadcast_to(epoch[..., None], planets.mass_Msun.shape)
        orbits = FrameOrbit.from_cartesian(
            barycentric_to_jacobi(position_AU, star.mass_Msun, planets.mass_Msun),
            barycentric_to_jacobi(velocity_AU_per_d, star.mass_Msun, planets.mass_Msun),
            mu_AU3_per_d2=enclosed_mu(star.mass_Msun, planets.mass_Msun),
            epoch_d=epoch,
        )
        return cls(star, planets, orbits, frame)

    def solve(self, time_d, *, solver=_DEFAULT_SOLVER):
        """Solve once at all times; return (K+1,) + time_shape + (3,) states.

        Args:
            time_d: Times in days, shape (T,) or B + (T,).
            solver: Value-only Kepler strategy. Defaults to Markley.

        Returns:
            Ephemeris with the barycentric positions and velocities of the star and all
            planets at the requested times.
        """
        batch = self.star.mass_Msun.shape
        if batch:
            time = jnp.asarray(time_d)
            paired = time.ndim != 1
            if paired and time.shape[:-1] != batch:
                raise ValueError("time_d must have shape (T,) or B + (T,)")

            def solve(system, time):
                return system.solve(time, solver=solver)

            for _ in batch:
                solve = jax.vmap(solve, in_axes=(0, 0 if paired else None))
            solved = solve(self, time)
            return Ephemeris(
                solved.position_AU,
                solved.velocity_AU_per_d,
                solved.dist_pc,
                batch_ndim=len(batch),
            )
        r, v = self.orbits.at(time_d, solver=solver)
        return Ephemeris(
            jacobi_to_barycentric(r, self.star.mass_Msun, self.planets.mass_Msun),
            jacobi_to_barycentric(v, self.star.mass_Msun, self.planets.mass_Msun),
            self.frame.dist_pc,
        )
