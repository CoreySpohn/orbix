"""The isotropic orientation measure as distribution records.

The isotropic measure on orientations is the Haar measure on rotations. In the
classical angles it is ``sin(i)/2 di`` for the inclination and uniform for the
node, argument of periapsis and mean anomaly, which is the reference measure of
`ClassicalChart` normalized to a probability.
"""

import math
from typing import ClassVar

import jax
import jax.numpy as jnp
import priorities
from jax import Array


class IsotropicInclination(priorities.AbstractDistribution):
    """Inclination of an isotropic orbit: density ``sin(i)/2`` on ``[0, pi]``.

    The density vanishes at both ends, which are poles of the orientation sphere.
    """

    event_dim: ClassVar[int] = 0
    bounds: ClassVar[str] = "interval"
    has_icdf: ClassVar[bool] = True

    def support(self):
        """Return the bounds ``(0, pi)``."""
        return jnp.asarray(0.0), jnp.asarray(math.pi)

    def log_prob(self, value):
        """Return ``log(sin(i)/2)``, ``-inf`` outside the open interval."""
        value = jnp.asarray(value)
        s = jnp.sin(value)
        inside = (value > 0) & (value < math.pi) & (s > 0)
        out = jnp.where(inside, jnp.log(jnp.where(inside, s, 1.0) / 2), -jnp.inf)
        return jnp.where(jnp.isnan(value), jnp.nan, out)

    def sample(self, key: Array, shape: tuple[int, ...] = ()):
        """Draw inclinations of the given shape."""
        return self.icdf(jax.random.uniform(key, shape))

    def cdf(self, value):
        """Return ``sin(i/2)^2``, accurate near the pole."""
        value = jnp.asarray(value)
        out = jnp.sin(jnp.clip(value, 0.0, math.pi) / 2) ** 2
        return jnp.where(jnp.isnan(value), jnp.nan, out)

    def icdf(self, unit):
        """Return ``2 arcsin(sqrt(u))``, accurate near the pole."""
        unit = jnp.asarray(unit)
        out = 2 * jnp.arcsin(jnp.sqrt(jnp.clip(unit, 0.0, 1.0)))
        return jnp.where((unit >= 0) & (unit <= 1), out, jnp.nan)


def isotropic_orientation():
    """Return the isotropic orientation as ``(inc, node, argperi, mean_anomaly)``.

    The inclination is `IsotropicInclination`; the other three angles are
    independent `priorities.Uniform` on ``[0, 2 pi]``. All are in radians.
    """
    angle = priorities.Uniform(0.0, 2 * math.pi)
    return IsotropicInclination(), angle, angle, angle
