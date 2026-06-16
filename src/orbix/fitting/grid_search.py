"""GPU-parallel orbit grid-search (adaptive importance sampling).

Discovery tier of orbix.fitting: enumerate orbits consistent with sparse data
and return weighted particles. See the design spec for the configurable seams.
"""

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp

from orbix.fitting.data import AstromData
from orbix.fitting.forward import predict_astrometry
from orbix.fitting.likelihoods import loglike_astrom
from orbix.fitting.priors import period_to_sma
from orbix.utils.quasi_random import roberts_sequence


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


class AbstractGridStrategy(eqx.Module):
    """Produces Stage-1 global samples and a Stage-2 refined proposal."""

    @abstractmethod
    def stage1(self, key, ndim, n):
        """Return ``(n, ndim)`` unit-cube samples for the global exploration stage."""

    @abstractmethod
    def stage2(self, key, survivors, n):
        """Return ``(samples, log_q)`` from a refined proposal around survivors."""


class AdaptiveImportanceSampler(AbstractGridStrategy):
    """Roberts global fill, Gaussian-mixture refinement around survivors.

    Attributes:
        n_modes: Number of Gaussian mixture components in Stage 2.
        jitter: Diagonal regularization added to the empirical covariance.
    """

    n_modes: int = eqx.field(static=True, default=5)
    jitter: float = eqx.field(static=True, default=1e-6)

    def stage1(self, key, ndim, n):
        """Return ``(n, ndim)`` Roberts quasi-random points in the unit cube.

        Args:
            key: JAX PRNG key for a Cranley-Patterson rotation.
            ndim: Dimension of the parameter space.
            n: Number of points.

        Returns:
            Array of shape ``(n, ndim)`` with values in ``[0, 1)``.
        """
        return roberts_sequence(n, ndim, key=key)

    def stage2(self, key, survivors, n):
        """Gaussian-mixture proposal around the survivor set.

        Args:
            key: JAX PRNG key.
            survivors: Best unit-cube points from Stage 1, shape ``(m, d)``.
            n: Number of new samples to draw.

        Returns:
            Tuple ``(z, log_q)`` where ``z`` has shape ``(n, d)`` and
            ``log_q`` has shape ``(n,)``.
        """
        # implemented in Task 6
        raise NotImplementedError


def build_evaluator(data, Ms, dist_pc, shape):
    """Return ``single_eval(phys) -> scalar log-likelihood`` over present data.

    ``data`` is a tuple of present data containers. v1 handles AstromData;
    absent observables contribute nothing. The ``if astrom is not None`` check
    runs at build time (static), not under JAX trace, so it is JAX-safe.

    Args:
        data: Tuple of data containers (e.g. AstromData).
        Ms: Stellar mass (kg).
        dist_pc: Distance to system (parsec).
        shape: An AbstractShapeParam instance (unused here, reserved for Plan 2).

    Returns:
        A pure function ``single_eval(phys) -> scalar`` where ``phys`` is a
        dict of scalar orbit parameters (as returned by ``to_physical`` for
        a single particle).
    """
    astrom = next((d for d in data if isinstance(d, AstromData)), None)

    def single_eval(phys):
        ll = 0.0
        if astrom is not None:
            ra, dec = predict_astrometry(
                astrom.times,
                phys["a"],
                phys["e"],
                phys["cos_i"],
                phys["W"],
                phys["cos_w"],
                phys["sin_w"],
                phys["tp"],
                Ms,
                dist_pc,
            )
            ll = ll + loglike_astrom(ra, dec, astrom)
        return ll

    return single_eval
