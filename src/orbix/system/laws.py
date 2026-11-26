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
from hwoutils.precision import like, require_float64
from jax import Array

from orbix.equations.orbit import mean_motion
from orbix.system.frame_orbit import FrameOrbit, _element_dtype
from orbix.system.orientation import IsotropicInclination


class AbstractOrbitLaw(eqx.Module):
    """A normalized density on bound orbits against the orbit reference measure."""

    @abstractmethod
    def log_prob(self, orbit: FrameOrbit) -> Array:
        """Return the log density of each orbit, with the orbit batch shape.

        Args:
            orbit: Orbits to evaluate, with batch shape B.

        Returns:
            Log density against the orbit reference measure, shape B.
        """

    @abstractmethod
    def sample(self, key, *, mu_AU3_per_d2, epoch_d=0.0) -> FrameOrbit:
        """Draw orbits at the given rate parameter and epoch.

        The batch shape is the broadcast of ``mu_AU3_per_d2`` and the law's parameters.

        Args:
            key: JAX PRNG key.
            mu_AU3_per_d2: Gravitational parameter in AU^3/day^2, broadcast with the
                law's parameters.
            epoch_d: Epoch of the returned orbits in days. Defaults to 0.

        Returns:
            Sampled orbits.
        """

    def support(self) -> dict[str, tuple[Array, Array]]:
        """Bounds of the scalar elements the law confines to a finite interval.

        The keys are ``"e"`` (eccentricity) and ``"a_AU"`` or ``"period_d"`` (the
        size element), present only for an element with a bounded interval; the set of
        keys is fixed by the law's static structure. Each value is ``(low, high)``,
        arrays that broadcast against the law's parameters. A block that frees
        coordinates uses these to give its coordinates no hard wall. The default
        reports no bound.

        Returns:
            Dictionary from "e", "a_AU" or "period_d" to a (low, high) pair of arrays,
            for each bounded element only.
        """
        return {}

    def edges(self) -> dict[str, tuple[Array, Array]]:
        """Edges of the support of each scalar element, bounded on any side or not.

        The keys are ``"e"`` and ``"a_AU"`` or ``"period_d"``; each value is
        ``(low, high)``, infinite on an open side. Unlike :meth:`support`, an element
        bounded on one side only is reported. A block compares these with the bounds
        its coordinates map to find its walls. Edges are the hull of the support: a
        law whose support has an interior gap (a mixture of disjoint intervals) is
        not detected. The default is :meth:`support`, so a law that reports no bound
        has no edge.

        Returns:
            Dictionary from "e" and "a_AU" or "period_d" to a (low, high) pair of
            arrays.
        """
        return self.support()

    @property
    def unit_size(self) -> int | None:
        """Unit-cube length per orbit of :meth:`from_unit`, or ``None``."""
        return None

    def from_unit(self, c, *, mu_AU3_per_d2, epoch_d=0.0) -> FrameOrbit:
        """Map unit-cube points ``c`` (shape ``(..., unit_size)``) to orbits.

        Deterministic, piecewise continuous, and with the pushforward of
        :meth:`sample` for uniform ``c``. A law with no inverse CDF keeps this
        refusal.

        Args:
            c: Unit-cube points in (0, 1), shape (..., unit_size).
            mu_AU3_per_d2: Gravitational parameter in AU^3/day^2, broadcast with the
                law's parameters.
            epoch_d: Epoch of the returned orbits in days. Defaults to 0.

        Returns:
            Orbits with the batch shape of the leading axes of c.

        Raises:
            NotImplementedError: When ``unit_size`` is ``None``.
        """
        raise NotImplementedError(f"{type(self).__name__} has no unit-cube map")


def _uniform_quaternion(c) -> Array:
    """Unit quaternion, uniform on the three-sphere, from ``c`` of shape ``(..., 3)``.

    Shoemake's map: two polar angles and a radius split. It is continuous except where
    an angle wraps. The normalized Gaussian four-vector of :meth:`OrbitLaw.sample` has
    the same law, so the two draws agree in distribution.
    """
    u1, u2, u3 = c[..., 0], c[..., 1], c[..., 2]
    lower, upper = jnp.sqrt(1.0 - u1), jnp.sqrt(u1)
    return jnp.stack(
        [
            lower * jnp.sin(two_pi * u2),
            lower * jnp.cos(two_pi * u2),
            upper * jnp.sin(two_pi * u3),
            upper * jnp.cos(two_pi * u3),
        ],
        axis=-1,
    )


def _batch_shape(*parts) -> tuple[int, ...]:
    """Broadcast shape of arrays and scalar laws.

    A law contributes the batch of its support, which leaves out the component axis
    of a mixture, rather than the shapes of its parameter leaves.
    """
    shapes = [
        _distribution_batch(part)
        if isinstance(part, priorities.AbstractDistribution)
        else jnp.shape(part)
        for part in parts
    ]
    return jnp.broadcast_shapes(*shapes)


def _distribution_batch(dist):
    return () if type(dist) is IsotropicInclination else dist.support()[0].shape


def _draw(dist: priorities.AbstractDistribution, key, batch, *, dtype=None) -> Array:
    """Draw a scalar law at ``batch``, one independent variate per element.

    The law's own batch broadcasts to ``batch``. Sample axes are requested exactly
    where the law has no axis or an axis of length one, so an axis of length one is the
    same law for each element, never one shared draw. Only the shape rule of
    ``sample`` (``sample_shape + batch``) is used; no parameter is touched.
    """
    batch = tuple(batch)
    own = _distribution_batch(dist)
    lead = len(batch) - len(own)
    fresh = tuple(i for i in range(len(batch)) if i < lead or own[i - lead] != batch[i])
    shape = tuple(batch[i] for i in fresh)
    name = type(dist).__name__
    try:
        if type(dist) is IsotropicInclination:
            out = dist.sample(key, shape, dtype=dtype)
        else:
            out = dist.sample(key, shape)
    except (ValueError, TypeError, IndexError) as err:
        raise ValueError(
            f"{name}.sample failed for sample_shape {shape} with batch shape {own}; "
            "a law must return draws of shape sample_shape + batch"
        ) from err
    if jnp.shape(out) != shape + own:
        raise ValueError(
            f"{name}.sample returned shape {jnp.shape(out)} for sample_shape {shape}; "
            f"expected {shape + own} (sample_shape + batch)"
        )
    out = out.squeeze(tuple(len(shape) + i - lead for i in fresh if i >= lead))
    return jnp.moveaxis(out, range(len(shape)), fresh)


def size_term(log_p, log_volume) -> Array:
    """``log_p + log_volume``; a zero of the law wins over an infinite volume.

    Args:
        log_p: Log density of the size law.
        log_volume: Log of the size Jacobian factor.

    Returns:
        Sum log_p + log_volume, or -inf where log_p is -inf.
    """
    return jnp.where(jnp.isneginf(log_p), -jnp.inf, log_p + log_volume)


class OrbitLaw(AbstractOrbitLaw):
    """Independent size and eccentricity, isotropic orientation, uniform phase.

    Exactly one of ``a_AU`` and ``period_d`` is given. Orientation and phase are
    uniform, which is density one against the reference measure. A period law is
    conditional on the masses: the period is read from the orbit's own rate, and a draw
    at a given ``mu_AU3_per_d2`` has the period from the law, so its support in ``a``
    moves with the mass and stays normalized.

    Parameters are array leaves, so one law serves several rows through a trailing
    rows axis. Parameter shapes broadcast against each other and the masses, and each
    element of the broadcast batch gets its own independent draw: a parameter of shape
    ``(1,)`` means the same law for each planet. A custom law must follow the shape
    rule of ``priorities.AbstractDistribution.sample``; a law that does not is refused
    when it is drawn.

    Attributes:
        eccentricity: Scalar law on the eccentricity, supported on ``[0, 1)``.
        a_AU: Scalar law on the semi-major axis in AU, or ``None``.
        period_d: Scalar law on the period in days, or ``None``.
    """

    eccentricity: priorities.AbstractDistribution
    a_AU: priorities.AbstractDistribution | None = None
    period_d: priorities.AbstractDistribution | None = None

    def __check_init__(self):
        """Require exactly one size law, and scalar laws for the size and shape."""
        if (self.a_AU is None) == (self.period_d is None):
            raise ValueError("give exactly one of a_AU and period_d")
        for name in ("eccentricity", "a_AU", "period_d"):
            law = getattr(self, name)
            if law is not None and law.event_dim != 0:
                raise ValueError(f"{name} must be a scalar law (event_dim 0)")

    def log_prob(self, orbit: FrameOrbit) -> Array:
        """Return ``log p_e(e) - log(2 pi e)`` plus the size term, per orbit.

        The size term is ``log p_a(a) + log a`` for ``a_AU`` and
        ``log p_P(P) + log P + log(3/2)`` for ``period_d``, with ``P`` from the
        orbit's own rate. The exactly circular orbit has no density here (``-inf``):
        the eccentricity-vector density ``p_e(e) / (2 pi e)`` is singular there, a set
        of measure zero that no draw lands on. Where a size law gives zero density the
        result is ``-inf`` even at an infinite size. Density evaluation requires
        float64 and widens the supplied orbital coordinates before arithmetic.

        Args:
            orbit: Orbits to evaluate, with batch shape B.

        Returns:
            Log density against the orbit reference measure, shape B.
        """
        require_float64("OrbitLaw.log_prob", self)
        e = jnp.hypot(like(orbit.ex, self), like(orbit.ey, self))
        safe_e = jnp.where(e > 0, e, 1.0)
        out = jnp.where(
            e > 0, self.eccentricity.log_prob(e) - jnp.log(two_pi * safe_e), -jnp.inf
        )
        if self.a_AU is not None:
            a = like(orbit.a_AU, self)
            return out + size_term(self.a_AU.log_prob(a), jnp.log(a))
        period = two_pi / like(orbit.n_rad_per_d, self)
        return out + size_term(self.period_d.log_prob(period), jnp.log(1.5 * period))

    def sample(self, key, *, mu_AU3_per_d2, epoch_d=0.0) -> FrameOrbit:
        """Draw orbits; orientation and phase are uniform.

        Args:
            key: JAX PRNG key.
            mu_AU3_per_d2: Gravitational parameter in AU^3/day^2, broadcast with the
                law's parameters.
            epoch_d: Epoch of the returned orbits in days. Defaults to 0.

        Returns:
            Sampled orbits with the broadcast batch shape of mu_AU3_per_d2 and the law's
            parameters.
        """
        size_law = self.a_AU if self.a_AU is not None else self.period_d
        parameters = jax.tree.leaves((self.eccentricity, size_law))
        dtype = _element_dtype(
            tuple(filter(eqx.is_inexact_array, parameters)), mu_AU3_per_d2, epoch_d
        )
        mu = jnp.asarray(mu_AU3_per_d2, dtype=dtype)
        batch = _batch_shape(self.eccentricity, size_law, mu)
        k_e, k_size, k_phase, k_quat = jax.random.split(key, 4)
        e = _draw(self.eccentricity, k_e, batch, dtype=dtype)
        size = _draw(size_law, k_size, batch, dtype=dtype)
        dtype = _element_dtype(dtype, e, size)
        mu = jnp.asarray(mu, dtype=dtype)
        size = jnp.asarray(size, dtype=dtype)
        a = (
            size
            if self.a_AU is not None
            else (mu * (size / two_pi) ** 2) ** (1.0 / 3.0)
        )
        phase = jax.random.uniform(
            k_phase,
            batch,
            dtype=dtype,
            minval=jnp.zeros((), dtype),
            maxval=jnp.asarray(two_pi, dtype=dtype),
        )
        quat = jax.random.normal(k_quat, (*batch, 4), dtype=dtype)
        quat = quat / jnp.linalg.norm(quat, axis=-1, keepdims=True)
        epoch = jnp.asarray(epoch_d, dtype=dtype)
        a, mu, epoch = jnp.broadcast_arrays(a, mu, epoch)
        return FrameOrbit(
            a,
            mean_motion(a, mu),
            e * jnp.cos(phase),
            e * jnp.sin(phase),
            quat,
            epoch,
        )

    def support(self) -> dict[str, tuple[Array, Array]]:
        """Bounds of the eccentricity and size laws that have a finite interval.

        The eccentricity bounds are clipped to ``[0, 1]``. A half-bounded or
        unbounded scalar law reports nothing.

        Returns:
            Dictionary from "e", "a_AU" or "period_d" to a (low, high) pair of arrays,
            for each bounded element only.
        """
        out = {}
        if self.eccentricity.bounds == "interval":
            low, high = self.eccentricity.support()
            out["e"] = (jnp.clip(low, 0.0, 1.0), jnp.clip(high, 0.0, 1.0))
        key, size = (
            ("a_AU", self.a_AU)
            if self.a_AU is not None
            else ("period_d", self.period_d)
        )
        if size.bounds == "interval":
            out[key] = size.support()
        return out

    def edges(self) -> dict[str, tuple[Array, Array]]:
        """The scalar laws' own supports, the eccentricity clipped to ``[0, 1]``.

        A law bounded on one side, which :meth:`support` leaves out, is reported
        here with an infinite other edge.

        Returns:
            Dictionary from "e" and "a_AU" or "period_d" to a (low, high) pair of
            arrays.
        """
        low, high = self.eccentricity.support()
        key, size = (
            ("a_AU", self.a_AU)
            if self.a_AU is not None
            else ("period_d", self.period_d)
        )
        return {
            "e": (jnp.clip(low, 0.0, 1.0), jnp.clip(high, 0.0, 1.0)),
            key: size.support(),
        }

    @property
    def unit_size(self) -> int | None:
        """Six per orbit if both scalar laws have an inverse CDF, else ``None``.

        The six are the eccentricity, the size, the phase and three for the
        orientation.
        """
        size = self.a_AU if self.a_AU is not None else self.period_d
        return 6 if self.eccentricity.has_icdf and size.has_icdf else None

    def from_unit(self, c, *, mu_AU3_per_d2, epoch_d=0.0) -> FrameOrbit:
        """Inverse-CDF draws of ``e`` and the size, then phase and orientation.

        ``c[..., 0]`` gives the eccentricity, ``c[..., 1]`` the size element,
        ``c[..., 2]`` the phase (times ``2 pi``) and ``c[..., 3:6]`` the orientation
        through :func:`_uniform_quaternion`. The pushforward of uniform ``c`` is that of
        :meth:`sample`.

        Args:
            c: Unit-cube points in (0, 1), shape (..., 6).
            mu_AU3_per_d2: Gravitational parameter in AU^3/day^2, broadcast with the
                law's parameters.
            epoch_d: Epoch of the returned orbits in days. Defaults to 0.

        Returns:
            Orbits with the batch shape of the leading axes of c.
        """
        if self.unit_size is None:
            raise NotImplementedError("a scalar law has no inverse CDF")
        c = jnp.asarray(c)
        mu = jnp.asarray(
            mu_AU3_per_d2, dtype=jnp.result_type(*jax.tree.leaves(mu_AU3_per_d2), c)
        )
        e = self.eccentricity.icdf(c[..., 0])
        if self.a_AU is not None:
            a = self.a_AU.icdf(c[..., 1])
        else:
            a = (mu * (self.period_d.icdf(c[..., 1]) / two_pi) ** 2) ** (1.0 / 3.0)
        phase = two_pi * c[..., 2]
        quat = _uniform_quaternion(c[..., 3:6])
        epoch = jnp.asarray(
            epoch_d, dtype=jnp.result_type(*jax.tree.leaves(epoch_d), c)
        )
        e, phase, a, mu, epoch = jnp.broadcast_arrays(e, phase, a, mu, epoch)
        return FrameOrbit(
            a,
            mean_motion(a, mu),
            e * jnp.cos(phase),
            e * jnp.sin(phase),
            jnp.broadcast_to(quat, (*a.shape, 4)),
            epoch,
        )
