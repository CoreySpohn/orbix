"""Base planet model."""

from __future__ import annotations

from functools import partial

import equinox as eqx
import jax.numpy as jnp
from jax import Array, jit, vmap

import orbix.equations.orbit as oe
from orbix.constants import G, Mearth2kg, Rearth2AU, pc2AU, rad2arcsec, two_pi
from orbix.equations.phase import (
    lambert_phase_exact,
)
from orbix.equations.propagation import single_r, single_r_v


@jit
class Planets(eqx.Module):
    """A JAX-friendly, fully-jit-table planet model.

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
    Ms: Array
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
    # pRp2: Array  # geometric albedo * planet radius squared
    _rad2arcsec_dist: Array  # rad2arcsec / dist

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
        self.Ms, self.dist = Ms, dist
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
        _dist = dist * pc2AU
        # projected semi-major axis in radians, used to get projected DEC and RA
        self._rad2arcsec_dist = rad2arcsec / _dist
        self.a_as = self.a * self._rad2arcsec_dist
        # A and B matrices for on-sky angular separation
        self.A_as = self.A_AU * self._rad2arcsec_dist
        self.B_as = self.B_AU * self._rad2arcsec_dist

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

    def _prop_v(self, trig_solver, t, A, B):
        """Propagate the orbits to times t returning positions and velocities in AU.

        Args:
            trig_solver: function to solve Kepler's equation (M, e) -> (sinE, cosE)
            t: Times to propagate the planet to. Shape (ntimes,)
            A: A matrix
            B: B matrix

        Returns:
            r: jnp.ndarray shape (norb, 3, ntimes)
            v: jnp.ndarray shape (norb, 3, ntimes)
        """
        M = vmap(oe.mean_anomaly_tp, (None, 0, 0))(t, self.n, self.tp)
        sinE, cosE = trig_solver(M, self.e)
        r, v = vmap(single_r_v, (1, 1, 0, 0, 0, 0))(A, B, self.e, sinE, cosE, self.n)
        return r, v

    def prop_AU(self, trig_solver, t):
        """Public jitted wrapper around _prop that returns positions in AU.

        Args:
            trig_solver: function to solve Kepler's equation (M, e) -> (sinE, cosE)
            t: Times to propagate the planet to. Shape (ntimes,)

        Returns:
            r: jnp.ndarray shape (norb, 3, ntimes)
        """
        return self._prop(trig_solver, t, self.A_AU, self.B_AU)[0]

    def prop_as(self, trig_solver, t):
        """Propagate the orbits to times t returning positions in arcsec.

        Args:
            trig_solver: function to solve Kepler's equation (M, e) -> (sinE, cosE)
            t: Times to propagate the planet to. Shape (ntimes,)

        Returns:
            r: jnp.ndarray shape (norb, 3, ntimes)
        """
        return self._prop(trig_solver, t, self.A_as, self.B_as)[0]

    def s_dMag(self, trig_solver, t):
        """Propagate to times t and return the apparent separation and dMag.

        Args:
            trig_solver: function to solve Kepler's equation (M, e) -> (sinE, cosE)
            t: Times to propagate the planet to. Shape (ntimes,)

        Returns:
            s: jnp.ndarray shape (norb, ntimes) in AU
            dMag: jnp.ndarray shape (norb, ntimes)
        """
        r_AU, _, cosE = self._prop(trig_solver, t, self.A_AU, self.B_AU)

        # Get planet's radial distance in AU (d = a * (1 - e*cosE))
        d_AU = self.a[:, None] * (1 - self.e[:, None] * cosE)

        # Get cos(beta) using the z-component (line-of-sight) from AU position
        # Use r_AU[:, 2] / d_AU convention consistent with EXOSIMS
        cosbeta = r_AU[:, 2] / d_AU

        # Clip cosbeta to handle floating point issue near +/- 1
        # that can cause sqrt(1 - cosbeta**2) to be nan
        cosbeta = jnp.clip(cosbeta, -1.0, 1.0)
        sinbeta = jnp.sqrt(1 - cosbeta**2)
        # Calculate the apparent separation alpha in AU
        s = d_AU * sinbeta

        # Calculate Lambert phase function using cosbeta and sinbeta
        phase = lambert_phase_exact(cosbeta, sinbeta)

        # Calculate dMag using AU units
        # dMag = -2.5 * log10( p * (Rp_AU / d_AU)**2 * Phi )
        # This should be non-negative. Add epsilon for log10 stability.
        log_arg = self.p[:, None] * (self.Rp[:, None] / d_AU) ** 2 * phase
        epsilon = jnp.finfo(log_arg.dtype).tiny
        dMag = -2.5 * jnp.log10(log_arg + epsilon)
        return s, dMag

    def alpha_dMag(self, trig_solver, t):
        """Propagate to times t and return the apparent angular separation and dMag.

        Args:
            trig_solver: function to solve Kepler's equation (M, e) -> (sinE, cosE)
            t: Times to propagate the planet to. Shape (ntimes,)

        Returns:
            alpha: jnp.ndarray shape (norb, ntimes) in arcsec
            dMag: jnp.ndarray shape (norb, ntimes)
        """
        s, dMag = self.s_dMag(trig_solver, t)

        # Convert the apparent separation s to arcsec
        alpha = s * self._rad2arcsec_dist[:, None]

        return alpha, dMag

    def prop_vAU(self, trig_solver, t):
        """Public wrapper around _prop_v that returns positions and velocities in AU.

        Args:
            trig_solver: function to solve Kepler's equation (M, e) -> (sinE, cosE)
            t: Times to propagate the planet to. Shape (ntimes,)

        Returns:
            r: jnp.ndarray shape (norb, 3, ntimes)
            v: jnp.ndarray shape (norb, 3, ntimes)
        """
        return self._prop_v(trig_solver, t, self.A_AU, self.B_AU)

    @partial(jit, static_argnums=(1,))
    def j_prop_AU(self, trig_solver, t):
        """Jit wrapped version of prop_AU."""
        return self.prop_AU(trig_solver, t)

    @partial(jit, static_argnums=(1,))
    def j_prop_as(self, trig_solver, t):
        """Jit wrapped version of prop_as."""
        return self.prop_as(trig_solver, t)

    @partial(jit, static_argnums=(1,))
    def j_s_dMag(self, trig_solver, t):
        """Jit wrapped version of s_dMag."""
        return self.s_dMag(trig_solver, t)

    @partial(jit, static_argnums=(1,))
    def j_alpha_dMag(self, trig_solver, t):
        """Jit wrapped version of alpha_dMag."""
        return self.alpha_dMag(trig_solver, t)

    @partial(jit, static_argnums=(1,))
    def j_prop_vAU(self, trig_solver, t):
        """Jit wrapped version of prop_vAU."""
        return self.prop_vAU(trig_solver, t)
