"""Base planet model."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array, jit, vmap

import orbix.equations.orbit as orbit_eqs
from orbix.constants import G, Mearth2kg, Rearth2AU, pc2AU, two_pi
from orbix.equations.propagation import single_r

# Pre-compile necessary orbit equations for performance
mean_motion_jit = jit(orbit_eqs.mean_motion)
period_n_jit = jit(orbit_eqs.period_n)
AB_matrices_reduced_jit = jit(orbit_eqs.AB_matrices_reduced)
semi_amplitude_reduced_jit = jit(orbit_eqs.semi_amplitude_reduced)
mean_anomaly_tp_jit = jit(orbit_eqs.mean_anomaly_tp)


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
    A: Array  # A matrix
    B: Array  # B matrix

    # RV related quantities
    w_p: Array  # Planet's argument of periapsis (same as w)
    w_s: Array  # Star's argument of periapsis
    secosw: Array  # sqrt(e) * cos(w_p)
    sesinw: Array  # sqrt(e) * sin(w_p)
    tp: Array  # Time of periapsis passage
    # tc: Array  # Time of conjunction
    K: Array  # Radial velocity amplitude
    Mp_min: Array  # Minimum mass of the planet

    # Kepler's equation solver
    trig_solver: callable = eqx.static_field()

    def __init__(self, trig_solver, Ms, dist, a, e, W, i, w, M0, t0, Mp=1, Rp=1, p=0.2):
        """Initialize a planet with orbital elements.

        Args:
            trig_solver: function to solve Kepler's equation (M, e) -> (sinE, cosE)
            Ms: Mass of the star in kg
            dist: Distance to the star in pc
            a: Semi-major axis in AU
            e: Eccentricity
            W: Longitude of the ascending node in degrees
            i: Inclination in degrees
            w: Argument of periapsis in degrees
            M0: Mean anomaly at t0 in degrees
            t0: Epoch of M0 in days since J2000
            Mp: Planet mass in Earth masses (default=1.0)
            Rp: Planet radius in Earth radii (default=1.0)
            p: Geometric albedo (default=0.2)

        """
        # Eccentric anomaly solver that returns sin(E) and cos(E)
        self.trig_solver = trig_solver

        ##### Orbital elements
        # Cast everything to arrays
        a, e, W, i, w, M0, t0, p = map(jnp.asarray, (a, e, W, i, w, M0, t0, p))
        self.a, self.e, self.t0, self.p = a, e, t0, p
        # Convert angles from degrees to radians
        self.W, self.i, self.w, self.M0 = map(jnp.deg2rad, (W, i, w, M0))

        Mp, Rp = map(jnp.asarray, (Mp, Rp))
        self.Mp = Mp * Mearth2kg
        self.Rp = Rp * Rearth2AU

        # Standard gravitational parameter
        self.mu = G * Ms

        # Mean angular motion
        self.n = mean_motion_jit(self.a, self.mu)

        # Orbital period
        self.T = period_n_jit(self.n)

        # Angle cosines and sines
        self.sinW, self.cosW = jnp.sin(self.W), jnp.cos(self.W)
        self.sini, self.cosi = jnp.sin(self.i), jnp.cos(self.i)
        self.sinw, self.cosw = jnp.sin(self.w), jnp.cos(self.w)

        sqrt_one_minus_e2 = jnp.sqrt(1 - self.e**2)

        ##### Propagation information
        # A and B matrices for orbit position/velocity vectors
        # These have shape (3, n) where n is the number of orbits
        self.A, self.B = AB_matrices_reduced_jit(
            self.a,
            sqrt_one_minus_e2,
            self.sini,
            self.cosi,
            self.sinW,
            self.cosW,
            self.sinw,
            self.cosw,
        )
        # since dist >> a, and a_ang = arctan(a/dist), the small angle approximation
        # says a_ang = a / dist, so to get the approximate angular/on-sky angular
        # separation, we can just divide the A/B matrices by dist
        self.dist = jnp.asarray(dist * pc2AU)
        # A and B matrices for on-sky angular separation
        self.A_ang = self.A / self.dist
        self.B_ang = self.B / self.dist

        ##### RV quantities
        # Calculate the time of periapsis passage
        T_e = self.T * self.M0 / two_pi
        self.tp = self.t0 - T_e
        # Calculate the time of conjunction
        # nu_transit = pi_over_2 - self.w_s
        # E_transit = 2 * jnp.arctan(
        #     jnp.tan(nu_transit / 2) * jnp.sqrt((1 - self.e) / (1 + self.e))
        # )
        # self.tc = self.tp + self.T / two_pi * (E_transit - self.e * jnp.sin(E_transit))
        self.w_p = self.w
        self.w_s = (self.w + jnp.pi) % two_pi
        se = jnp.sqrt(self.e)
        self.secosw = se * self.cosw
        self.sesinw = se * self.sinw
        self.Mp_min = self.Mp * self.sini
        self.K = semi_amplitude_reduced_jit(self.T, Ms, self.Mp_min, sqrt_one_minus_e2)

        ##### Direct imaging quantities
        # projected semi-major axis in radians, used to get projected DEC and RA
        # self.a_rad = jnp.arctan(self.a / dist)

    def _prop(self, t, A, B):
        """Propagate the orbits to times t returning positions in AU.

        Args:
            t: Times to propagate the planet to. Shape (ntimes,)
            A: A matrix
            B: B matrix

        Returns:
            r: jnp.ndarray shape (norb, 3, ntimes)
        """
        M = vmap(mean_anomaly_tp_jit, (None, 0, 0))(t, self.n, self.tp)
        sinE, cosE = self.trig_solver(M, self.e)
        r = vmap(single_r, (1, 1, 0, 0, 0))(A, B, self.e, sinE, cosE)
        return r

    @jit
    def propagate(self, t):
        """Public jitted wrapper around _prop that returns positions in AU.

        Args:
            t: Times to propagate the planet to. Shape (ntimes,)

        Returns:
            r: jnp.ndarray shape (norb, 3, ntimes)
        """
        return self._prop(t, self.A, self.B)
