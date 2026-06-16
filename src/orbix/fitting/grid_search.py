"""GPU-parallel orbit grid-search (adaptive importance sampling).

Discovery tier of orbix.fitting: enumerate orbits consistent with sparse data
and return weighted particles. See the design spec for the configurable seams.
"""

import equinox as eqx
import jax.numpy as jnp

from orbix.fitting.priors import period_to_sma  # noqa: F401


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
