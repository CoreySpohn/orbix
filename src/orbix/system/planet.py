"""Base planet model."""

from __future__ import annotations

from functools import partial

import equinox as eqx
import jax.numpy as jnp
from jax import Array, jit, vmap

import orbix.equations.orbit as oe
from orbix.constants import G, Mearth2kg, Rearth2AU, pc2AU, rad2arcsec, two_pi
from orbix.equations.phase import lambert_phase_poly
from orbix.equations.propagation import single_r


@jit
class Planet(eqx.Module):
    """A JAX‑friendly, fully‑jit‑table planet model.

    This object treats all orbital parameters as arrays, which lets us use one
    model to represent a single planet as, well, one planet OR a cloud of n
    orbits that are consistent with observed data. The later case is useful for
    propagating an orbital fit into the future which can be used to calculate
    the probability of detection.

    Inputs:
        star_mass: mass of the star in kg
        dist: distance to the star in pc
        a: semi-major axis in AU
        e: eccentricity
        W: longitude of the ascending node (degrees)
        i: inclination (degrees)
        w: argument of periapsis (degrees)
        M0: mean anomaly at t0 (degrees)
        t0: epoch of M0 (days since J2000)
        Mp: planet mass in Earth masses
        Rp: planet radius in Earth radii
        p: geometric albedo
    """

    # Input parameters
    dist: Array
    a: Array
    e: Array
    W: Array
    i: Array
    w: Array
    M0: Array
    t0: Array
    Mp: Array
    Rp: Array
    p: Array

    # Derived parameters
    # Cosines and sines of the angles
    cosW: Array
    sinW: Array
    cosi: Array
    sini: Array
    cosw: Array
    sinw: Array

    # Common orbital quantities
    mu: Array  # Standard gravitational parameter
    n: Array  # Mean motion
    T: Array  # Orbital period

    # A and B matrices for orbit propagation
    A_AU: Array  # A matrix in AU
    B_AU: Array  # B matrix in AU

    # RV related quantities
    w_p: Array  # Planet's argument of periapsis (same as w)
    w_s: Array  # Star's argument of periapsis
    secosw: Array  # sqrt(e) * cos(w_p)
    sesinw: Array  # sqrt(e) * sin(w_p)
    tp: Array  # Time of periapsis passage
    K: Array  # Radial velocity amplitude
    Mp_min: Array  # Minimum mass of the planet

    # A and B matrices for on-sky angular separation
    A_as: Array  # A matrix in arcsec
    B_as: Array  # B matrix in arcsec
    a_as: Array  # semi-major axis in arcsec
    pRp2: Array  # geometric albedo * planet radius squared

    def __init__(self, Ms, dist, a, e, W, i, w, M0, t0, Mp, Rp, p):
        """Initialize a planet with orbital elements as JAX arrays.

        Args:
            Ms: Mass of the star in kg
            dist: Distance to the star in pc
            a: Semi-major axis in AU
            e: Eccentricity
            W: Longitude of the ascending node in radians
            i: Inclination in radians
            w: Argument of periapsis in radians
            M0: Mean anomaly at t0 in radians
            t0: Epoch of M0 in days since J2000
            Mp: Planet mass in Earth masses
            Rp: Planet radius in Earth radii
            p: Geometric albedo

        """
        ##### Orbital elements
        self.a, self.e, self.t0 = a, e, t0
        # Angles should be in radians already
        self.W, self.i, self.w, self.M0 = W, i, w, M0

        self.Mp, self.Rp, self.p = Mp * Mearth2kg, Rp * Rearth2AU, p

        # Standard gravitational parameter
        self.mu = G * Ms

        # Mean angular motion
        self.n = oe.mean_motion(self.a, self.mu)

        # Orbital period
        self.T = oe.period_n(self.n)

        # Angle cosines and sines
        self.sinW, self.cosW = jnp.sin(self.W), jnp.cos(self.W)
        self.sini, self.cosi = jnp.sin(self.i), jnp.cos(self.i)
        self.sinw, self.cosw = jnp.sin(self.w), jnp.cos(self.w)

        sqrt_one_minus_e2 = jnp.sqrt(1 - self.e**2)

        ##### Propagation information
        # A and B matrices for orbit position/velocity vectors
        # These have shape (3, n) where n is the number of orbits
        self.A_AU, self.B_AU = oe.AB_matrices_reduced(
            self.a,
            sqrt_one_minus_e2,
            self.sini,
            self.cosi,
            self.sinW,
            self.cosW,
            self.sinw,
            self.cosw,
        )

        ##### RV quantities
        # Calculate the time of periapsis passage
        T_e = self.T * self.M0 / two_pi
        self.tp = self.t0 - T_e
        self.w_p = self.w
        self.w_s = (self.w + jnp.pi) % two_pi
        se = jnp.sqrt(self.e)
        self.secosw = se * self.cosw
        self.sesinw = se * self.sinw
        self.Mp_min = self.Mp * self.sini
        self.K = oe.semi_amplitude_reduced(self.T, Ms, self.Mp_min, sqrt_one_minus_e2)

        ##### Direct imaging quantities
        # since dist >> a, and a_ang = arctan(a/dist), the small angle approximation
        # says a_ang = a / dist, so to get the approximate angular/on-sky angular
        # separation, we can just divide the A/B matrices by dist
        self.dist = dist * pc2AU
        # projected semi-major axis in radians, used to get projected DEC and RA
        _rad2arcsec_dist = rad2arcsec / self.dist
        self.a_as = self.a * _rad2arcsec_dist
        # A and B matrices for on-sky angular separation
        self.A_as = self.A_AU * _rad2arcsec_dist
        self.B_as = self.B_AU * _rad2arcsec_dist
        # Used for the dMag calculations
        self.pRp2 = self.p * self.Rp**2

    def _prop(self, trig_solver, t, A, B):
        """Propagate the orbits to times t returning positions in AU.

        Args:
            trig_solver: function to solve Kepler's equation (M, e) -> (sinE, cosE)
            t: Times to propagate the planet to. Shape (ntimes,)
            A: A matrix
            B: B matrix

        Returns:
            r: jnp.ndarray shape (norb, 3, ntimes)
            sinE: jnp.ndarray shape (norb, ntimes)
            cosE: jnp.ndarray shape (norb, ntimes)
        """
        M = vmap(oe.mean_anomaly_tp, (None, 0, 0))(t, self.n, self.tp)
        sinE, cosE = trig_solver(M, self.e)
        r = vmap(single_r, (1, 1, 0, 0, 0))(A, B, self.e, sinE, cosE)
        return r, sinE, cosE

    @partial(jit, static_argnums=(1,))
    def prop_AU(self, trig_solver, t):
        """Public jitted wrapper around _prop that returns positions in AU.

        Args:
            trig_solver: function to solve Kepler's equation (M, e) -> (sinE, cosE)
            t: Times to propagate the planet to. Shape (ntimes,)

        Returns:
            r: jnp.ndarray shape (norb, 3, ntimes)
        """
        return self._prop(trig_solver, t, self.A_AU, self.B_AU)[0]

    @partial(jit, static_argnums=(1,))
    def prop_as(self, trig_solver, t):
        """Propagate the orbits to times t returning positions in arcsec.

        Args:
            trig_solver: function to solve Kepler's equation (M, e) -> (sinE, cosE)
            t: Times to propagate the planet to. Shape (ntimes,)

        Returns:
            r: jnp.ndarray shape (norb, 3, ntimes)
        """
        return self._prop(trig_solver, t, self.A_as, self.B_as)[0]

    @partial(jit, static_argnums=(1,))
    def s_dMag(self, trig_solver, t):
        """Propagate the orbits to times t returning angular separation and dMag.

        Args:
            trig_solver: function to solve Kepler's equation (M, e) -> (sinE, cosE)
            t: Times to propagate the planet to. Shape (ntimes,)

        Returns:
            s: jnp.ndarray shape (norb, ntimes)
            dMag: jnp.ndarray shape (norb, ntimes)
        """
        r_as, _, cosE = self._prop(trig_solver, t, self.A_as, self.B_as)

        # Get planet's radial distance
        r = self.a_as[:, None] * (1 - self.e[:, None] * cosE)

        # Get cos(beta) from the z-component of the position vector
        cosbeta = -r_as[:, 2] / r

        # Calculate the angular separation
        s = r * jnp.sqrt(1 - cosbeta**2)

        # # Approximate the Lambert phase function with cosbeta
        phase = lambert_phase_poly(cosbeta)
        # # Calculate dMag
        dMag = -2.5 * jnp.log10(self.pRp2[:, None] * phase / r**2)
        return s, dMag
