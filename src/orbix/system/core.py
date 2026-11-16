"""Jacobi chains and reusable barycentric ephemerides.

See Murray and Dermott, Solar System Dynamics, for hierarchical coordinates.
The hierarchy is a set of independent Kepler orbits, not an N-body integrator.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.constants import G, Msun2kg
from hwoutils.epoch import Epoch
from jaxtyping import Array

from orbix.kepler import Markley
from orbix.system import observables
from orbix.system.frame_orbit import FrameOrbit

GM_SUN_AU3_PER_D2 = G * Msun2kg
_DEFAULT_SOLVER = Markley()


def _ordered_sum(values):
    """Return final sum and prefixes along axis zero, in fixed input order."""

    def add(total, value):
        total = total + value
        return total, total

    return jax.lax.scan(add, jnp.zeros(values.shape[1:], values.dtype), values)


def enclosed_mu(star_mass_Msun, planet_mass_Msun):
    """Return Jacobi mu for star shape B and inner-to-outer masses B + (K,)."""
    masses = jnp.asarray(planet_mass_Msun)
    _, cumulative = _ordered_sum(jnp.moveaxis(masses, -1, 0))
    return GM_SUN_AU3_PER_D2 * (
        jnp.asarray(star_mass_Msun)[..., None] + jnp.moveaxis(cumulative, 0, -1)
    )


def jacobi_to_barycentric(relative, star_mass_Msun, planet_mass_Msun):
    """Map B + (K,) + time_shape + (3,) to B + (K+1,) + time_shape + (3,).

    Both sums run sequentially inner to outer. Appending zero-mass bodies
    preserves the bits of every preceding body and the stellar reflex.
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


class Body(eqx.Module):
    """Body masses in solar masses, with a free batch shape."""

    mass_Msun: Array = eqx.field(converter=jnp.asarray)


class SkyFrame(eqx.Module):
    """Distance and the reference epoch that time offsets are measured from.

    The epoch is host-only metadata (hwoutils.epoch.Epoch), never a traced
    leaf. time_d converts host dates to offsets before any float32 device
    transfer. No time-scale conversion is applied.
    """

    dist_pc: Array = eqx.field(converter=jnp.asarray)
    epoch: Epoch = eqx.field(static=True, default=Epoch())

    def time_d(self, time_jd):
        """Days from the epoch, in NumPy float64 on the host.

        time_jd may be a numeric JD array or an object exposing jd1 and jd2
        (for example astropy Time). Dates must use the epoch's time scale.
        """
        return self.epoch.days(time_jd)


class Ephemeris(eqx.Module):
    """Barycentric vectors of shape (K+1,) + time_shape + (3,).

    Index zero is the star. reference is a body index or "barycenter".
    Distance broadcasts against time_shape. All observables reuse these
    stored states and perform no further orbital solve.
    """

    position_AU: Array = eqx.field(converter=jnp.asarray)
    velocity_AU_per_d: Array = eqx.field(converter=jnp.asarray)
    dist_pc: Array = eqx.field(converter=jnp.asarray)

    def __check_init__(self):
        """Validate the body and vector shape contract."""
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
        """Return east and north angular offsets from the reference."""
        return observables.offset_arcsec(
            self._relative(self.position_AU, target, reference), self.dist_pc
        )

    def separation_arcsec(self, target, reference=0):
        """Return projected separation from the reference in arcseconds."""
        return observables.separation_arcsec(
            self._relative(self.position_AU, target, reference), self.dist_pc
        )

    def position_angle_deg(self, target, reference=0):
        """Return position angle east of north in [0, 360) degrees."""
        return observables.position_angle_deg(
            self._relative(self.position_AU, target, reference)
        )

    def phase_angle_rad(self, target, reference=0):
        """Return reference-target-observer phase; zero behind the reference."""
        return observables.phase_angle_rad(
            self._relative(self.position_AU, target, reference)
        )

    def radial_velocity_m_per_s(self, target, reference="barycenter"):
        """Return receding-positive velocity relative to the reference, m/s."""
        return observables.radial_velocity_m_per_s(
            self._relative(self.velocity_AU_per_d, target, reference)
        )

    def distance_AU(self, target, reference=0):
        """Return three-dimensional distance to the reference in AU."""
        return observables.distance_AU(
            self._relative(self.position_AU, target, reference)
        )


class System(eqx.Module):
    """A Jacobi chain ordered inner to outer by the caller.

    Orbit j describes planet j relative to the center of mass of the star
    and all preceding planets. The star and distance have shape B; planet
    masses and orbit batch have shape B + (K,). Editing masses changes the
    barycentric map only: rebuild to impose dynamically consistent rates.
    """

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
        """Set orbit rates from the mass enclosed by each Jacobi orbit."""
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

    def solve(self, time_d, *, solver=_DEFAULT_SOLVER):
        """Solve once at all times; return (K+1,) + time_shape + (3,) states."""
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
            return solve(self, time)
        r, v = self.orbits.at(time_d, solver=solver)
        return Ephemeris(
            jacobi_to_barycentric(r, self.star.mass_Msun, self.planets.mass_Msun),
            jacobi_to_barycentric(v, self.star.mass_Msun, self.planets.mass_Msun),
            self.frame.dist_pc,
        )
