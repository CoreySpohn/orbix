"""Jacobi chains and reusable barycentric ephemerides.

See Murray and Dermott, Solar System Dynamics, for hierarchical coordinates.
The hierarchy is a set of independent Kepler orbits, not an N-body integrator.
"""

import equinox as eqx
import jax.numpy as jnp
from hwoutils.constants import G, Msun2kg
from hwoutils.epoch import Epoch
from jaxtyping import Array

from orbix.kepler import Markley
from orbix.system import observables
from orbix.system.frame_orbit import FrameOrbit

GM_SUN_AU3_PER_D2 = G * Msun2kg
_DEFAULT_SOLVER = Markley()


def enclosed_mu(star_mass_Msun, planet_mass_Msun):
    """Return Jacobi gravitational parameters, including each planet's mass.

    The star mass is scalar; planet masses have shape (K,), ordered inner
    to outer by the caller. This is the sole mass-to-orbit coupling.
    """
    return GM_SUN_AU3_PER_D2 * (
        jnp.asarray(star_mass_Msun) + jnp.cumsum(jnp.asarray(planet_mass_Msun), axis=0)
    )


def jacobi_to_barycentric(relative, star_mass_Msun, planet_mass_Msun):
    """Map (K,) + time_shape + (3,) relative vectors to K+1 body vectors.

    The same linear mass-weighted map applies to positions and velocities.
    Body zero is the star; a cumulative sum avoids an unrolled planet loop.
    """
    relative = jnp.asarray(relative)
    masses = jnp.asarray(planet_mass_Msun)
    weights = masses / (jnp.asarray(star_mass_Msun) + jnp.cumsum(masses))
    weighted = relative * weights.reshape(masses.shape + (1,) * (relative.ndim - 1))
    cumulative = jnp.cumsum(weighted, axis=0)
    star = -jnp.sum(weighted, axis=0, keepdims=True)
    preceding = jnp.concatenate([jnp.zeros_like(star), cumulative], axis=0)[:-1]
    return jnp.concatenate([star, star + preceding + relative], axis=0)


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
            return values[target]
        return values[target] - values[reference]

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
    and all preceding planets. The star is scalar; masses and orbit batch
    have shape (K,). Changing masses on a constructed System changes the
    barycentric map only: rebuild to impose dynamically consistent rates.
    """

    star: Body
    planets: Body
    orbits: FrameOrbit
    frame: SkyFrame

    def __check_init__(self):
        """Validate the body and vector shape contract."""
        if (
            self.star.mass_Msun.shape != ()
            or self.planets.mass_Msun.ndim != 1
            or self.orbits.a_AU.shape != self.planets.mass_Msun.shape
        ):
            raise ValueError(
                "System requires a scalar star and (K,) planets and orbits"
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
        r, v = self.orbits.at(time_d, solver=solver)
        return Ephemeris(
            jacobi_to_barycentric(r, self.star.mass_Msun, self.planets.mass_Msun),
            jacobi_to_barycentric(v, self.star.mass_Msun, self.planets.mass_Msun),
            self.frame.dist_pc,
        )
