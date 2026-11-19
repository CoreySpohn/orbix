"""Laws on bound orbits: a density against the orbit reference measure.

The reference measure of every chart is normalized Haar orientation measure times
``dex dey`` times ``d(ln a)``, per orbit. An orbit law is a normalized density against
that measure, independent of any chart, so the same law is a population model in a draw
and a prior in a fit. A law is not a ``priorities.AbstractDistribution``: its value
is an orbit, and its measure is not ``d(value)``.

The scalar factors are ``priorities`` records. orbix does not import skyscapes, so a
named eccentricity law such as ``skyscapes.population.kipping_eccentricity`` is passed
in as the ``eccentricity`` argument.
"""

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
import priorities
from hwoutils.constants import two_pi
from jax import Array

from orbix.equations.orbit import mean_motion
from orbix.system.frame_orbit import FrameOrbit


class AbstractOrbitLaw(eqx.Module):
    """A normalized density on bound orbits against the orbit reference measure."""

    @abstractmethod
    def log_prob(self, orbit: FrameOrbit) -> Array:
        """Return the log density of each orbit, with the orbit batch shape."""

    @abstractmethod
    def sample(self, key, *, mu_AU3_per_d2, epoch_d=0.0) -> FrameOrbit:
        """Draw orbits at the given rate parameter and epoch.

        The batch shape is the broadcast of ``mu_AU3_per_d2`` and the law's parameters.
        """


def _batch_shape(*trees) -> tuple[int, ...]:
    leaves = jax.tree_util.tree_leaves(trees)
    return jnp.broadcast_shapes(*(jnp.shape(leaf) for leaf in leaves))


def _draw(dist: priorities.AbstractDistribution, key, batch) -> Array:
    """Draw a scalar distribution at the batch shape, parameters broadcast."""
    own = _batch_shape(dist)
    lead = batch[: len(batch) - len(own)] if own else tuple(batch)
    return jnp.broadcast_to(dist.sample(key, lead), batch)


class OrbitLaw(AbstractOrbitLaw):
    """Independent size and eccentricity, isotropic orientation, uniform phase.

    Exactly one of ``a_AU`` and ``period_d`` is given. Orientation and phase are
    uniform, which is density one against the reference measure. A period law is
    conditional on the masses: the period is read from the orbit's own rate, and a draw
    at a given ``mu_AU3_per_d2`` has the period from the law, so its support in ``a``
    moves with the mass and stays normalized.

    Parameters are array leaves, so one law serves several rows through a trailing
    rows axis.

    Attributes:
        eccentricity: Scalar law on the eccentricity, supported on ``[0, 1)``.
        a_AU: Scalar law on the semi-major axis in AU, or ``None``.
        period_d: Scalar law on the period in days, or ``None``.
    """

    eccentricity: priorities.AbstractDistribution
    a_AU: priorities.AbstractDistribution | None = None
    period_d: priorities.AbstractDistribution | None = None

    def __check_init__(self):
        """Require exactly one size law."""
        if (self.a_AU is None) == (self.period_d is None):
            raise ValueError("give exactly one of a_AU and period_d")

    def log_prob(self, orbit: FrameOrbit) -> Array:
        """Return ``log p_e(e) - log(2 pi e)`` plus the size term, per orbit.

        The size term is ``log p_a(a) + log a`` for ``a_AU`` and
        ``log p_P(P) + log P + log(3/2)`` for ``period_d``, with ``P`` from the
        orbit's own rate.
        """
        e = jnp.hypot(orbit.ex, orbit.ey)
        out = self.eccentricity.log_prob(e) - jnp.log(two_pi * e)
        if self.a_AU is not None:
            return out + self.a_AU.log_prob(orbit.a_AU) + jnp.log(orbit.a_AU)
        period = two_pi / orbit.n_rad_per_d
        return out + self.period_d.log_prob(period) + jnp.log(1.5 * period)

    def sample(self, key, *, mu_AU3_per_d2, epoch_d=0.0) -> FrameOrbit:
        """Draw orbits; orientation and phase are uniform."""
        mu = jnp.asarray(mu_AU3_per_d2)
        batch = _batch_shape(self, mu)
        k_e, k_size, k_phase, k_quat = jax.random.split(key, 4)
        e = _draw(self.eccentricity, k_e, batch)
        if self.a_AU is not None:
            a = _draw(self.a_AU, k_size, batch)
        else:
            period = _draw(self.period_d, k_size, batch)
            a = (mu * (period / two_pi) ** 2) ** (1.0 / 3.0)
        phase = jax.random.uniform(k_phase, batch, maxval=two_pi)
        quat = jax.random.normal(k_quat, (*batch, 4))
        quat = quat / jnp.linalg.norm(quat, axis=-1, keepdims=True)
        a, mu, epoch = jnp.broadcast_arrays(a, mu, jnp.asarray(epoch_d))
        return FrameOrbit(
            a,
            mean_motion(a, mu),
            e * jnp.cos(phase),
            e * jnp.sin(phase),
            quat,
            epoch,
        )
