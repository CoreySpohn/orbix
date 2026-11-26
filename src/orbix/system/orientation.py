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
from hwoutils.precision import float_dtype, like
from jax import Array


class IsotropicInclination(priorities.AbstractDistribution):
    """Inclination of an isotropic orbit: density ``sin(i)/2`` on ``[0, pi]``.

    The density vanishes at both ends, which are poles of the orientation sphere.

    Attributes:
        event_dim: Number of event dimensions, zero for a scalar.
        bounds: Kind of support, "interval".
        has_icdf: True: the inverse CDF is available.
    """

    event_dim: ClassVar[int] = 0
    bounds: ClassVar[str] = "interval"
    has_icdf: ClassVar[bool] = True

    def support(self):
        """Return the bounds ``(0, pi)``.

        Returns:
            Tuple (0, pi) of the inclination bounds in radians.
        """
        return jnp.asarray(0.0), jnp.asarray(math.pi)

    def log_prob(self, value):
        """Return ``log(sin(i)/2)``, ``-inf`` outside the open interval.

        Args:
            value: Inclination in radians.

        Returns:
            Log density, with the shape of value.
        """
        value = jnp.asarray(value)
        s = jnp.sin(value)
        inside = (value > 0) & (value < math.pi) & (s > 0)
        out = jnp.where(
            inside,
            jnp.log(jnp.where(inside, s, jnp.ones_like(s)) / 2),
            like(-jnp.inf, s),
        )
        return jnp.where(jnp.isnan(value), like(jnp.nan, value), out)

    def sample(self, key: Array, sample_shape: tuple[int, ...] = (), *, dtype=None):
        """Draw inclinations of shape ``sample_shape``.

        Args:
            key: JAX PRNG key.
            sample_shape: Shape of the draw. Defaults to a scalar.
            dtype: Draw precision. Defaults to the process construction precision.

        Returns:
            Inclinations in radians, with shape sample_shape.
        """
        if dtype is None:
            dtype = float_dtype()
        return self.icdf(
            jax.random.uniform(
                key,
                sample_shape,
                dtype=dtype,
                minval=jnp.zeros((), dtype),
                maxval=jnp.ones((), dtype),
            )
        )

    def cdf(self, value):
        """Return ``sin(i/2)^2``, accurate near the pole.

        Args:
            value: Inclination in radians.

        Returns:
            Cumulative probability, with the shape of value.
        """
        value = jnp.asarray(value)
        out = jnp.sin(jnp.clip(value, 0, like(math.pi, value)) / 2) ** 2
        return jnp.where(jnp.isnan(value), like(jnp.nan, value), out)

    def icdf(self, q):
        """Return ``2 arcsin(sqrt(q))``, accurate near the pole.

        Args:
            q: Cumulative probability in [0, 1].

        Returns:
            Inclination in radians, with the shape of q; NaN outside [0, 1].
        """
        q = jnp.asarray(q)
        out = 2 * jnp.arcsin(jnp.sqrt(jnp.clip(q, 0, 1)))
        return jnp.where((q >= 0) & (q <= 1), out, like(jnp.nan, out))
