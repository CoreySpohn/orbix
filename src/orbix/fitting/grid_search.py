"""GPU-parallel orbit grid-search (adaptive importance sampling).

Discovery tier of orbix.fitting: enumerate orbits consistent with sparse data
and return weighted particles. See the design spec for the configurable seams.
"""

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp

from orbix.fitting.priors import period_to_sma


class ParamBounds(eqx.Module):
    """Box bounds over an ordered parameter set, on the unit cube."""

    low: jnp.ndarray
    high: jnp.ndarray
    names: tuple = eqx.field(static=True)

    def __check_init__(self):
        """Validate that low/high shapes agree with the number of names."""
        if self.low.shape != self.high.shape:
            raise ValueError("low and high must share a shape")
        if self.low.shape[-1] != len(self.names):
            raise ValueError("bounds width must match number of names")

    def scale(self, u):
        """Map unit-cube points ``(n, d)`` to physical box values ``(n, d)``."""
        return self.low + u * (self.high - self.low)


class AbstractShapeParam(eqx.Module):
    """Maps an ordered unit-cube sample to a physical orbit-shape dict."""

    @abstractmethod
    def default_bounds(self, log_T_range, e_max) -> ParamBounds:
        """Return default ParamBounds for this parameterization."""

    @abstractmethod
    def to_physical(self, u, bounds, Ms) -> dict:
        """Convert unit-cube samples to a physical orbit-parameter dict."""


class EccVectorShape(AbstractShapeParam):
    """Eccentricity-vector coordinate ``(logT, ex, ey, cos_i, W, tp_frac)``.

    Sampled names in order: ``logT, ex, ey, cos_i, W, tp_frac``.
    Derived quantities: ``T = 10**logT``, ``a = period_to_sma(T, Ms)``,
    ``e = hypot(ex, ey)``, ``cos_w = ex/e`` (1 when e=0), ``sin_w = ey/e``
    (0 when e=0), ``W`` in radians, ``tp = tp_frac * T``.
    """

    def default_bounds(self, log_T_range=(0.0, 4.0), e_max=0.9):
        """Return default ParamBounds for EccVectorShape.

        Args:
            log_T_range: ``(log10_T_min, log10_T_max)`` in days.
            e_max: Maximum eccentricity vector component magnitude.

        Returns:
            ParamBounds with six named parameters.
        """
        names = ("logT", "ex", "ey", "cos_i", "W", "tp_frac")
        low = jnp.array([log_T_range[0], -e_max, -e_max, -1.0, 0.0, 0.0])
        high = jnp.array([log_T_range[1], e_max, e_max, 1.0, 2.0 * jnp.pi, 1.0])
        return ParamBounds(low=low, high=high, names=names)

    def to_physical(self, u, bounds, Ms):
        """Convert unit-cube samples to physical orbit parameters.

        Args:
            u: Unit-cube samples of shape ``(n, 6)``.
            bounds: ParamBounds returned by ``default_bounds``.
            Ms: Stellar mass in kg.

        Returns:
            Dict of physical parameter arrays, each of shape ``(n,)``.
            Keys: ``T, a, e, cos_i, W, cos_w, sin_w, tp``.
        """
        p = bounds.scale(u)
        logT, ex, ey, cos_i, W, tp_frac = (p[:, i] for i in range(6))
        T = 10.0**logT
        a = period_to_sma(T, Ms)
        e = jnp.hypot(ex, ey)
        safe_e = jnp.where(e > 0.0, e, 1.0)
        cos_w = jnp.where(e > 0.0, ex / safe_e, 1.0)
        sin_w = jnp.where(e > 0.0, ey / safe_e, 0.0)
        return {
            "T": T,
            "a": a,
            "e": e,
            "cos_i": cos_i,
            "W": W,
            "cos_w": cos_w,
            "sin_w": sin_w,
            "tp": tp_frac * T,
        }
