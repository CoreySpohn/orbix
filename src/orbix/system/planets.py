"""Base planet model.

Composes :class:`orbix.system.orbit.KeplerianOrbit` for geometric
propagation; RV-specific derived quantities (semi-amplitude, minimum
mass, ``secosw`` / ``sesinw``) live on ``Planets`` for the orbit-only
fitting pipeline.
"""

from __future__ import annotations

from functools import partial

import equinox as eqx
import jax.numpy as jnp
from jax import jit
from jaxtyping import Array

import orbix.equations.orbit as oe
from orbix.constants import (
    G,
    Mearth2kg,
    Rearth2AU,
    pc2AU,
    rad2arcsec,
    two_pi,
)
from orbix.equations.phase import lambert_phase_exact

from .orbit import AbstractOrbit, KeplerianOrbit


@jit
class Planets(eqx.Module):
    """JAX-friendly planet model that composes an orbit.

    Geometry lives on ``self.orbit`` (an ``AbstractOrbit``, concretely
    ``KeplerianOrbit`` today). RV and photometric derived quantities
    stay on ``Planets`` for the orbit-only fitting pipeline.

    Inputs (unchanged API for backwards compatibility; unit-suffix
    rename is Phase 0c of the migration):
        Ms: stellar mass in kg
        dist: distance to star in pc
        a, e, W, i, w, M0, t0: orbital elements (same units /
            conventions as before)
        Mp: planet mass in Earth masses
        Rp: planet radius in Earth radii
        p: geometric albedo
    """

    # Composed geometry
    orbit: AbstractOrbit

    # Retained public fields (unchanged names this phase)
    Ms: Array
    dist: Array
    a: Array
    e: Array
    W: Array
    i: Array
    w: Array
    M0: Array
    t0: Array
    Mp: Array  # stored in kg (converted from Mearth)
    Rp: Array  # stored in AU (converted from Rearth)
    p: Array

    # Derived
    mu: Array
    n: Array
    T: Array
    tp: Array
    w_p: Array
    w_s: Array
    secosw: Array
    sesinw: Array
    K: Array
    Mp_min: Array
    a_as: Array
    _rad2arcsec_dist: Array

    def __init__(self, Ms, dist, a, e, W, i, w, M0, t0, Mp, Rp, p):
        """Build a Planets instance with a composed KeplerianOrbit.

        See the class docstring for parameter units.
        """
        self.Ms = Ms
        self.dist = dist
        self.a = a
        self.e = e
        self.W = W
        self.i = i
        self.w = w
        self.M0 = M0
        self.t0 = t0
        self.Mp = Mp * Mearth2kg
        self.Rp = Rp * Rearth2AU
        self.p = p

        # Geometry via KeplerianOrbit
        self.orbit = KeplerianOrbit(
            a_AU=a,
            e=e,
            W_rad=W,
            i_rad=i,
            w_rad=w,
            M0_rad=M0,
            t0_d=t0,
        )

        # Derived quantities that need stellar context
        self.mu = G * Ms
        self.n = oe.mean_motion(a, self.mu)
        self.T = oe.period_n(self.n)
        T_e = self.T * self.M0 / two_pi
        self.tp = self.t0 - T_e
        self.w_p = self.w
        self.w_s = (self.w + jnp.pi) % two_pi

        sqrt_1me2 = jnp.sqrt(1 - e**2)
        se = jnp.sqrt(e)
        self.secosw = se * jnp.cos(w)
        self.sesinw = se * jnp.sin(w)
        self.Mp_min = self.Mp * jnp.sin(i)
        self.K = oe.semi_amplitude_reduced(self.T, Ms, self.Mp_min, sqrt_1me2)

        # On-sky projection scale
        _dist_AU = dist * pc2AU
        self._rad2arcsec_dist = rad2arcsec / _dist_AU
        self.a_as = self.a * self._rad2arcsec_dist

    # --- Propagation (delegates to self.orbit) ---

    def prop_AU(self, trig_solver, t):
        """Position vectors in AU, shape ``(K, 3, T)``."""
        r_AU, _, _ = self.orbit.propagate(trig_solver, t, Ms_kg=self.Ms)
        return r_AU

    def prop_as(self, trig_solver, t):
        """Position vectors in arcsec, shape ``(K, 3, T)``."""
        r_AU = self.prop_AU(trig_solver, t)
        return r_AU * self._rad2arcsec_dist[:, None, None]

    def prop_ra_dec(self, trig_solver, t):
        """(RA_arcsec, Dec_arcsec) each ``(K, T)``."""
        r_as = self.prop_as(trig_solver, t)
        return r_as[:, 0], r_as[:, 1]

    def s_dMag(self, trig_solver, t):
        """Return (projected separation AU, dMag) each ``(K, T)``."""
        r_AU, phase_angle_rad, dist_AU = self.orbit.propagate(
            trig_solver,
            t,
            Ms_kg=self.Ms,
        )
        # Projected separation from x, y components
        s = jnp.sqrt(r_AU[:, 0] ** 2 + r_AU[:, 1] ** 2)

        cosbeta = jnp.cos(phase_angle_rad)
        sinbeta = jnp.sin(phase_angle_rad)
        phase = lambert_phase_exact(cosbeta, sinbeta)

        log_arg = self.p[:, None] * (self.Rp[:, None] / dist_AU) ** 2 * phase
        eps = jnp.finfo(log_arg.dtype).tiny
        dMag = -2.5 * jnp.log10(log_arg + eps)
        return s, dMag

    def alpha_dMag(self, trig_solver, t):
        """Return (projected separation arcsec, dMag) each ``(K, T)``."""
        s, dMag = self.s_dMag(trig_solver, t)
        alpha = s * self._rad2arcsec_dist[:, None]
        return alpha, dMag

    # --- jitted variants (retained API) ---

    @partial(jit, static_argnums=(1,))
    def j_prop_AU(self, trig_solver, t):
        """JITted ``prop_AU``."""
        return self.prop_AU(trig_solver, t)

    @partial(jit, static_argnums=(1,))
    def j_prop_as(self, trig_solver, t):
        """JITted ``prop_as``."""
        return self.prop_as(trig_solver, t)

    @partial(jit, static_argnums=(1,))
    def j_prop_ra_dec(self, trig_solver, t):
        """JITted ``prop_ra_dec``."""
        return self.prop_ra_dec(trig_solver, t)

    @partial(jit, static_argnums=(1,))
    def j_s_dMag(self, trig_solver, t):
        """JITted ``s_dMag``."""
        return self.s_dMag(trig_solver, t)

    @partial(jit, static_argnums=(1,))
    def j_alpha_dMag(self, trig_solver, t):
        """JITted ``alpha_dMag``."""
        return self.alpha_dMag(trig_solver, t)
