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
from jaxtyping import Array


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
