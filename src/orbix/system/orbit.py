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
import jax.numpy as jnp
from jaxtyping import Array

from orbix.equations.orbit import AB_matrices_reduced


class AbstractOrbit(eqx.Module):
    """Abstract orbital-motion model.

    Subclasses own whatever parameters describe "this kind of
    orbital motion" (Keplerian, Kozai, TTV, interpolated ephemeris).
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
        """Not yet implemented; will propagate orbit to times ``t_jd``."""
        raise NotImplementedError
