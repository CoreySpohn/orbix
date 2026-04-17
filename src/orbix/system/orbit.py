"""Orbit models: AbstractOrbit interface + KeplerianOrbit concrete class.

The orbit owns orbital-element parameters only. Stellar context
(``Ms_kg``, ``dist_pc``) is passed keyword-only into ``propagate``
and the fast-path helpers. This keeps the orbit self-describing,
avoids duplicating stellar state, and leaves ``Star`` as the single
source of truth for that context.
"""

from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.constants import G, pc2AU, rad2arcsec, two_pi
from jaxtyping import Array

from orbix.equations.orbit import (
    AB_matrices_reduced,
    mean_anomaly_tp,
    mean_motion,
    period_n,
)
from orbix.equations.propagation import single_r


class AbstractOrbit(eqx.Module):
    """Abstract orbital-motion model.

    Subclasses own whatever parameters describe "this kind of
    orbital motion" (Keplerian, TTV, interpolated ephemeris).
    Stellar context is threaded in at call time.
    """

    @abstractmethod
    def propagate(
        self,
        trig_solver,
        t_jd: Array,
        *,
        Ms_kg: Array,
    ) -> tuple[Array, Array, Array]:
        """Propagate to times ``t_jd``.

        Args:
            trig_solver: Scalar solver for Kepler's equation,
                signature ``(M, e) -> (sinE, cosE)``.
            t_jd: Times in Julian Days, shape ``(T,)``.
            Ms_kg: Stellar mass in kg, shape ``(K,)`` or scalar.

        Returns:
            r_AU: Position vectors, shape ``(K, 3, T)``.
            phase_angle_rad: Phase angle beta, shape ``(K, T)``.
            dist_AU: Star-planet distance, shape ``(K, T)``.
        """


class KeplerianOrbit(AbstractOrbit):
    """Seven-element Keplerian orbit.

    Owns the orbital elements (``a_AU``, ``e``, ``W_rad``,
    ``i_rad``, ``w_rad``, ``M0_rad``, ``t0_d``) and caches the
    AB propagation matrices. Stellar context (``Ms_kg``) is
    passed keyword-only into ``propagate``.

    All parameter arrays share a leading axis ``(K,)`` -- ``K=1``
    for simulation, ``K=N`` for a posterior cloud of N orbits.
    """

    a_AU: Array
    e: Array
    W_rad: Array
    i_rad: Array
    w_rad: Array
    M0_rad: Array
    t0_d: Array

    # Cached shape-only derived state
    A_AU: Array
    B_AU: Array

    def __init__(
        self,
        a_AU: Array,
        e: Array,
        W_rad: Array,
        i_rad: Array,
        w_rad: Array,
        M0_rad: Array,
        t0_d: Array,
    ):
        """Store orbital elements and cache the AB propagation matrices."""
        self.a_AU = a_AU
        self.e = e
        self.W_rad = W_rad
        self.i_rad = i_rad
        self.w_rad = w_rad
        self.M0_rad = M0_rad
        self.t0_d = t0_d

        sqrt_1me2 = jnp.sqrt(1 - e**2)
        sini, cosi = jnp.sin(i_rad), jnp.cos(i_rad)
        sinW, cosW = jnp.sin(W_rad), jnp.cos(W_rad)
        sinw, cosw = jnp.sin(w_rad), jnp.cos(w_rad)
        self.A_AU, self.B_AU = AB_matrices_reduced(
            a_AU,
            sqrt_1me2,
            sini,
            cosi,
            sinW,
            cosW,
            sinw,
            cosw,
        )

    def propagate(
        self,
        trig_solver,
        t_jd: Array,
        *,
        Ms_kg: Array,
    ) -> tuple[Array, Array, Array]:
        """Propagate Keplerian orbit to times ``t_jd``.

        Returns:
            r_AU: (K, 3, T) position vectors.
            phase_angle_rad: (K, T) phase angle beta = arccos(r_z / |r|).
            dist_AU: (K, T) star-planet distance.
        """
        t_jd = jnp.atleast_1d(t_jd)

        # Derived quantities that depend on stellar context
        mu = G * Ms_kg
        n = mean_motion(self.a_AU, mu)
        T_d = period_n(n)
        tp_d = self.t0_d - T_d * self.M0_rad / two_pi

        # Mean anomaly at each time, shape (K, T)
        M = jax.vmap(mean_anomaly_tp, (None, 0, 0))(t_jd, n, tp_d)

        # Kepler solve -> sinE, cosE each shape (K, T)
        solver_t = jax.vmap(trig_solver, in_axes=(0, None))
        solver_kt = jax.vmap(solver_t, in_axes=(0, 0))
        sinE, cosE = solver_kt(M, self.e)

        # Position shape (K, 3, T)
        r_AU = jax.vmap(single_r, (1, 1, 0, 0, 0))(
            self.A_AU,
            self.B_AU,
            self.e,
            sinE,
            cosE,
        )

        # Star-planet distance from Kepler, shape (K, T).
        # d = a * (1 - e * cosE)
        dist_AU = self.a_AU[:, None] * (1.0 - self.e[:, None] * cosE)

        # Phase angle: cos(beta) = r_z / d, clipped for fp safety.
        cosbeta = jnp.clip(r_AU[:, 2] / dist_AU, -1.0, 1.0)
        phase_angle_rad = jnp.arccos(cosbeta)

        return r_AU, phase_angle_rad, dist_AU

    def position_arcsec(
        self,
        trig_solver,
        t_jd: Array,
        *,
        Ms_kg: Array,
        dist_pc: Array,
    ) -> tuple[Array, Array]:
        """On-sky (RA, Dec) in arcsec, each shape ``(K, T)``.

        Thin wrapper around ``propagate`` for callers that only
        need projected position.
        """
        r_AU, _, _ = self.propagate(trig_solver, t_jd, Ms_kg=Ms_kg)
        dist_AU = dist_pc * pc2AU
        scale = rad2arcsec / dist_AU
        ra_arcsec = r_AU[:, 0] * scale[:, None]
        dec_arcsec = r_AU[:, 1] * scale[:, None]
        return ra_arcsec, dec_arcsec

    def separation_arcsec(
        self,
        trig_solver,
        t_jd: Array,
        *,
        Ms_kg: Array,
        dist_pc: Array,
    ) -> Array:
        """Projected angular separation in arcsec, shape ``(K, T)``."""
        ra, dec = self.position_arcsec(
            trig_solver,
            t_jd,
            Ms_kg=Ms_kg,
            dist_pc=dist_pc,
        )
        return jnp.sqrt(ra**2 + dec**2)
