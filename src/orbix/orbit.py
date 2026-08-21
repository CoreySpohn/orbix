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
    period_to_sma,
)
from orbix.equations.propagation import single_r
from orbix.kepler.shortcuts.grid import get_grid_solver


def _resolve_trig_solver(trig_solver):
    """Return ``trig_solver``, or the cached default grid solver when None.

    The default is the scalar bilinear grid solver (trig outputs only),
    which ``get_grid_solver`` lru-caches, so repeated resolution is free.
    Passing a non-callable is almost always a time array that was meant
    for ``t_jd``, so that mistake is named at the call site.
    """
    if trig_solver is None:
        return get_grid_solver(level="scalar", E=False, trig=True, jit=True)
    if not callable(trig_solver):
        raise TypeError(
            "trig_solver must be callable with signature (M, e) -> (sinE, cosE); "
            f"got {type(trig_solver).__name__}. If you meant to pass times, "
            "use the keyword: propagate(t_jd=..., Ms_kg=...)."
        )
    return trig_solver


class AbstractOrbit(eqx.Module):
    """Abstract orbital-motion model.

    Subclasses own whatever parameters describe "this kind of
    orbital motion" (Keplerian, TTV, interpolated ephemeris).
    Stellar context is threaded in at call time.
    """

    @abstractmethod
    def propagate(
        self,
        trig_solver=None,
        t_jd: Array = None,
        *,
        Ms_kg: Array,
    ) -> tuple[Array, Array, Array]:
        """Propagate to times ``t_jd``.

        Args:
            trig_solver: Scalar solver for Kepler's equation,
                signature ``(M, e) -> (sinE, cosE)``. None selects the
                cached default grid solver.
            t_jd: Times in Julian Days, shape ``(T,)``. Required; it is
                keyword-friendly (``propagate(t_jd=..., Ms_kg=...)``) so
                callers relying on the default solver need not pass a
                positional None.
            Ms_kg: Stellar mass in kg, shape ``(K,)`` or scalar.

        Returns:
            r_AU: Position vectors, shape ``(K, 3, T)``.
            phase_angle_rad: Phase angle beta, shape ``(K, T)``, measured
                from the observer (+z) axis to the position vector. The
                standard planetary star-planet-observer phase angle is
                ``pi`` minus this; convert before any Lambert phase
                function.
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

    @classmethod
    def from_period(
        cls,
        T_d: Array,
        e: Array,
        cos_i: Array,
        W_rad: Array,
        cos_w: Array,
        sin_w: Array,
        tp_d: Array,
        *,
        Ms_kg: Array,
    ) -> "KeplerianOrbit":
        """Construct from the period parameterization used by orbit-fitting code.

        Fitting code samples ``(T, e, cos i, W, cos w, sin w, tp)`` rather
        than the seven fields this class stores, so posterior draws reach
        the class through this constructor: period converts to semi-major
        axis via Kepler's third law (which is why ``Ms_kg`` is required
        here, unlike ``__init__``), and periapsis passage maps exactly to
        ``(M0_rad=0, t0_d=tp_d)``.

        Args:
            T_d: Orbital period in days.
            e: Eccentricity.
            cos_i: Cosine of the inclination (the fitting basis; the
                gradient of ``arccos`` diverges at ``|cos_i| = 1``, so
                keep exactly face-on/edge-on samples out of gradients).
            W_rad: Longitude of the ascending node in radians.
            cos_w: Cosine of the argument of periapsis.
            sin_w: Sine of the argument of periapsis.
            tp_d: Time of periapsis passage in days (JD in practice).
            Ms_kg: Stellar mass in kg.

        Returns:
            A ``KeplerianOrbit`` whose leading axis is the common
            broadcast shape of the seven inputs, so a batch of posterior
            draws becomes a ``(K,)``-batched orbit in one call.
        """
        T_d, e, cos_i, W_rad, cos_w, sin_w, tp_d = jnp.broadcast_arrays(
            *(
                jnp.atleast_1d(jnp.asarray(x))
                for x in (T_d, e, cos_i, W_rad, cos_w, sin_w, tp_d)
            )
        )
        return cls(
            a_AU=period_to_sma(T_d, Ms_kg),
            e=e,
            W_rad=W_rad,
            i_rad=jnp.arccos(cos_i),
            w_rad=jnp.arctan2(sin_w, cos_w),
            M0_rad=jnp.zeros_like(T_d),
            t0_d=tp_d,
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
        trig_solver=None,
        t_jd: Array = None,
        *,
        Ms_kg: Array,
    ) -> tuple[Array, Array, Array]:
        """Propagate Keplerian orbit to times ``t_jd``.

        Returns:
            r_AU: (K, 3, T) position vectors.
            phase_angle_rad: (K, T) phase angle beta = arctan2(rho, r_z),
                rho = sqrt(r_x**2 + r_y**2); gradient-safe at conjunction.
                Measured from the observer (+z) axis, so the standard
                star-planet-observer phase angle (beta = 0 at full phase)
                is pi minus this; convert before any Lambert phase
                function.
            dist_AU: (K, T) star-planet distance.
        """
        if t_jd is None:
            raise TypeError("propagate() missing required argument: 't_jd'")
        trig_solver = _resolve_trig_solver(trig_solver)
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
        trig_solver=None,
        t_jd: Array = None,
        *,
        Ms_kg: Array,
        dist_pc: Array,
    ) -> tuple[Array, Array]:
        """On-sky (RA, Dec) in arcsec, each shape ``(K, T)``.

        Thin wrapper around ``propagate`` for callers that only
        need projected position.
        """
        r_AU, _, _ = self.propagate(trig_solver, t_jd, Ms_kg=Ms_kg)
        dist_AU = jnp.atleast_1d(dist_pc) * pc2AU
        scale = rad2arcsec / dist_AU
        ra_arcsec = r_AU[:, 0] * scale[:, None]
        dec_arcsec = r_AU[:, 1] * scale[:, None]
        return ra_arcsec, dec_arcsec

    def separation_arcsec(
        self,
        trig_solver=None,
        t_jd: Array = None,
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
