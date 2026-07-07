"""Orbit models: AbstractOrbit interface + KeplerianOrbit concrete class.

The orbit owns orbital-element parameters only. Stellar context
(``Ms_kg``, ``dist_pc``) is passed keyword-only into ``propagate``
and the fast-path helpers. This keeps the orbit self-describing and
avoids duplicating stellar state; callers supply the stellar context
per call.
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

    Owns the orbital elements only; everything derived (AB matrices,
    mean motion, period) is recomputed per call so that ``eqx.tree_at``
    updates and gradients through any element are always consistent.
    All parameter arrays share a leading axis ``(K,)``.
    """

    a_AU: Array = eqx.field(converter=jnp.atleast_1d)
    e: Array = eqx.field(converter=jnp.atleast_1d)
    W_rad: Array = eqx.field(converter=jnp.atleast_1d)
    i_rad: Array = eqx.field(converter=jnp.atleast_1d)
    w_rad: Array = eqx.field(converter=jnp.atleast_1d)
    M0_rad: Array = eqx.field(converter=jnp.atleast_1d)
    t0_d: Array = eqx.field(converter=jnp.atleast_1d)

    def __check_init__(self):
        """Validate that all seven elements share one leading (K,) shape."""
        shapes = {
            self.a_AU.shape,
            self.e.shape,
            self.W_rad.shape,
            self.i_rad.shape,
            self.w_rad.shape,
            self.M0_rad.shape,
            self.t0_d.shape,
        }
        if len(shapes) != 1:
            raise ValueError(
                f"KeplerianOrbit elements must share one (K,) shape, got {shapes}"
            )

    def _AB(self) -> tuple[Array, Array]:
        """Compute the AB propagation matrices from the current elements."""
        sqrt_1me2 = jnp.sqrt(1 - self.e**2)
        return AB_matrices_reduced(
            self.a_AU,
            sqrt_1me2,
            jnp.sin(self.i_rad),
            jnp.cos(self.i_rad),
            jnp.sin(self.W_rad),
            jnp.cos(self.W_rad),
            jnp.sin(self.w_rad),
            jnp.cos(self.w_rad),
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
            phase_angle_rad: (K, T) phase angle beta = arctan2(rho, r_z),
                rho = sqrt(r_x**2 + r_y**2); gradient-safe at conjunction.
            dist_AU: (K, T) star-planet distance.
        """
        t_jd = jnp.atleast_1d(t_jd)

        A_AU, B_AU = self._AB()

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
            A_AU,
            B_AU,
            self.e,
            sinE,
            cosE,
        )

        # Star-planet distance from Kepler, shape (K, T).
        # d = a * (1 - e * cosE)
        dist_AU = self.a_AU[:, None] * (1.0 - self.e[:, None] * cosE)

        # Phase angle beta = angle from the +z (observer) axis.
        # arctan2 avoids the arccos(clip(...)) NaN-gradient at conjunction.
        rho = jnp.sqrt(r_AU[:, 0] ** 2 + r_AU[:, 1] ** 2)
        phase_angle_rad = jnp.arctan2(rho, r_AU[:, 2])

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

    def __repr__(self) -> str:
        """Compact summary of the seven Keplerian elements.

        Angles are converted from radians to degrees for readability.
        Arrays are summarized inline; if the leading axis K > 3, only
        the first few entries are shown.
        """
        K = int(self.a_AU.shape[0]) if self.a_AU.ndim else 1
        a = _fmt(self.a_AU)
        e = _fmt(self.e)
        i_deg = _fmt(jnp.rad2deg(self.i_rad))
        w_deg = _fmt(jnp.rad2deg(self.w_rad))
        W_deg = _fmt(jnp.rad2deg(self.W_rad))
        M0_deg = _fmt(jnp.rad2deg(self.M0_rad))
        t0 = _fmt(self.t0_d)
        return (
            f"KeplerianOrbit(K={K}, a={a} AU, e={e}, i={i_deg} deg, "
            f"w={w_deg} deg, W={W_deg} deg, M0={M0_deg} deg, t0={t0} JD)"
        )


def _fmt(x: Array, fmt: str = ".3g", max_items: int = 3) -> str:
    """Format a scalar/array compactly for KeplerianOrbit's repr."""
    a = jnp.asarray(x)
    if isinstance(a, jax.core.Tracer):
        return "<traced>"
    if a.shape == () or a.shape == (1,):
        return f"{float(a.reshape(-1)[0]):{fmt}}"
    if a.size <= max_items:
        return "[" + ", ".join(f"{float(v):{fmt}}" for v in a) + "]"
    head = ", ".join(f"{float(v):{fmt}}" for v in a[:max_items])
    return f"[{head}, ...]"
