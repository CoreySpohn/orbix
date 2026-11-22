"""Unconstrained coordinates and volumes for bound Keplerian orbit frames.

The common reference measure is normalized Haar orientation measure times
dex dey times d(ln a). For a density p against that measure, the coordinate log
density is log(p(chart.to_orbit(x))) + chart.log_volume(x), and every
log_volume is exact against it: no constant is omitted. The frame chart
also supplies a normalized radial gauge density. Plain real angles have periodic
predictions; densities are interpreted per fundamental cell of 2 pi.

Transforms do not clip. As with exp and sigmoid, floating-point overflow,
underflow and rounding to a boundary limit the representable input range. A
wall-free inverse (``unconstrain``) of a value outside its interval by more than
rounding is NaN in the whole row, never a finite coordinate on the wall.
"""

import math
from abc import abstractmethod
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.constants import two_pi
from jax.nn import log_sigmoid, sigmoid
from jax.scipy.special import gammainc, log_ndtr, ndtr, ndtri
from jaxtyping import Array

from orbix.equations.orbit import mean_motion
from orbix.system.frame_orbit import FrameOrbit
from orbix.system.linear import (
    frame_to_sky_block,
    sky_block_geometry,
    sky_block_to_frame,
)
from orbix.system.rotation import rotate


def _coordinates(x, size):
    """Convert coordinates and validate their static trailing dimension."""
    x = jnp.asarray(x)
    if x.shape[-1:] != (size,):
        raise ValueError(f"Coordinates must have last axis of length {size}")
    return x


def plane_to_disk(u):
    """Map the plane smoothly onto the open unit disk, including its origin.

    ``u`` has trailing axis 2 and the result is the eccentricity vector
    ``(ex, ey) = u / sqrt(1 + |u|^2)`` of the chart coordinates ``u``.

    Args:
        u: Plane coordinates, trailing axis of length 2.

    Returns:
        Eccentricity vector (ex, ey), dimensionless, with the shape of u and norm below
        1.
    """
    return u / jnp.sqrt(1 + jnp.sum(u * u, axis=-1, keepdims=True))


def disk_to_plane(evec):
    """Invert ``plane_to_disk`` without polar coordinates.

    ``evec`` is an eccentricity vector ``(ex, ey)`` with ``|evec| < 1`` on the
    trailing axis; the result is ``u = evec / sqrt(1 - |evec|^2)``.

    Args:
        evec: Eccentricity vector (ex, ey), dimensionless, trailing axis of length 2
            with norm below 1.

    Returns:
        Plane coordinates u with the shape of evec.
    """
    return evec / jnp.sqrt(1 - jnp.sum(evec * evec, axis=-1, keepdims=True))


def periapsis_chart_shape(T_d, e, tp_d, epoch_d=0.0):
    """Chart shape ``(ln n, ex, ey)`` from period, eccentricity and periapsis time.

    ``(ex, ey) = e (cos M, -sin M)`` on the epoch mean-longitude axes, with
    ``M = n (epoch_d - tp_d)`` the epoch mean anomaly and ``n = 2 pi / T_d``
    the mean motion in radians per day. Inputs broadcast.

    Args:
        T_d: Orbital period in days.
        e: Eccentricity, dimensionless.
        tp_d: Time of periapsis in days.
        epoch_d: Epoch in days. Defaults to 0.

    Returns:
        Tuple (ln n, ex, ey) with n the mean motion in radians/day and ex, ey the
        dimensionless eccentricity-vector components, each with the broadcast shape of
        the inputs.
    """
    n = two_pi / jnp.asarray(T_d)
    phase = n * (epoch_d - tp_d)
    return jnp.log(n), e * jnp.cos(phase), -e * jnp.sin(phase)


def _disk_log_volume(u):
    """Log determinant: radial/tangential eigenvalues are (1+r^2)^(-3/2,-1/2)."""
    return -2 * jnp.log1p(jnp.sum(u * u, axis=-1))


def _inclination(z):
    """Invert atanh(cos i) without arccos of a rounded endpoint."""
    return 2 * jnp.arctan(jnp.exp(-z))


def _inclination_coordinate(inc):
    """Return atanh(cos i) using the half-angle identity."""
    return -jnp.log(jnp.tan(inc / 2))


def _inclination_log_volume(z):
    """Return log(sech(z)^2) without cancellation at large absolute z."""
    return 2 * (jnp.log(2.0) - jnp.logaddexp(z, -z))


# Total Haar measure of the orientation angles (cos i, W, u0): 2 * 2 pi * 2 pi.
_LOG_HAAR = math.log(8 * math.pi**2)


def _wrap_angles(coords, columns):
    """Reduce the listed coordinate columns to ``[0, 2 pi)``."""
    cols = jnp.asarray(columns)
    return coords.at[..., cols].set(jnp.mod(coords[..., cols], two_pi))


def _chi_four_icdf(q):
    """Quantile of the norm of a standard normal four-vector, by bisection.

    The squared norm is gamma with shape 2 and scale 2, so ``gammainc(2, t)`` at
    ``t`` half of it is its CDF. Eighty halvings of ``[0, 800]`` leave an error below
    1e-21 in ``t``.
    """
    q = jnp.asarray(q)

    def step(_, bounds):
        low, high = bounds
        mid = 0.5 * (low + high)
        below = gammainc(2.0, mid) < q
        return jnp.where(below, mid, low), jnp.where(below, high, mid)

    low, high = jax.lax.fori_loop(
        0, 80, step, (jnp.zeros_like(q), jnp.full_like(q, 800.0))
    )
    return jnp.sqrt(low + high)


def _interval(y, low, high):
    """Map the real line onto ``(low, high)``: the value and its log derivative."""
    width = high - low
    return low + width * sigmoid(y), jnp.log(width) + log_sigmoid(y) + log_sigmoid(-y)


def _interval_inverse(value, low, high):
    """Invert :func:`_interval`; NaN outside the interval by more than rounding.

    A value within ``sqrt(eps)`` of the width of a bound is held ``eps`` inside it;
    one further out has no coordinate and returns NaN.
    """
    eps = jnp.finfo(jnp.result_type(value, float)).eps
    position = (value - low) / (high - low)
    outside = (position < -jnp.sqrt(eps)) | (position > 1.0 + jnp.sqrt(eps))
    position = jnp.clip(position, eps, 1.0 - eps)
    inverse = jnp.log(position) - jnp.log1p(-position)
    return jnp.where(outside, jnp.nan, inverse)


class AbstractChart(eqx.Module):
    """Static-only orbit coordinates with a free leading batch shape B.

    Coordinates have shape B + (n_coords,), and log_volume returns shape B.
    mu_AU3_per_d2 must be positive; it and epoch_d broadcast against B.
    Inverses use the state's own epoch and cannot recover angle windings.
    ``coordinate_names`` names the trailing axis, in order, with a unit suffix where
    the coordinate has a unit, for hosts that keep coordinates as a dict of named
    arrays. ``periods`` is aligned with ``coordinate_names``: the period of each
    coordinate (the orbit depends on it only modulo the period, and the density is
    that of one cell), or ``None`` for a coordinate with no period.

    Attributes:
        n_coords: Number of coordinates on the trailing axis.
        coordinate_names: Coordinate names in trailing-axis order, with a unit suffix
            where the coordinate has a unit.
        periods: Period of each coordinate in radians, or None for a coordinate without
            one.
        size_coordinate: Kind of size coordinate that a law's bounds can map, "ln_a",
            "ln_n" or None.
        eccentricity_coordinate: Kind of eccentricity coordinate that a law's bounds can
            map, "logit", "disk" or None.
        gauge_unit_size: Number of unit-cube coordinates that unit_coordinates consumes
            for the gauge.
    """

    n_coords: eqx.AbstractClassVar[int]
    coordinate_names: eqx.AbstractClassVar[tuple[str, ...]]
    periods: eqx.AbstractClassVar[tuple[float | None, ...]]
    # Which coordinates a law's bounds can map: the size coordinate (the first,
    # ``"ln_a"`` or ``"ln_n"``) and the eccentricity (``"logit"`` at index 1, or
    # ``"disk"`` at 1, 2).
    size_coordinate: ClassVar[str | None] = None
    eccentricity_coordinate: ClassVar[str | None] = None
    # Unit-cube coordinates a gauge consumes in ``unit_coordinates``.
    gauge_unit_size: ClassVar[int] = 0

    @abstractmethod
    def to_orbit(self, x, *, mu_AU3_per_d2, epoch_d=0.0) -> FrameOrbit:
        """Map coordinates to a bound orbit with the supplied rate and epoch.

        Args:
            x: Chart coordinates, shape B + (n_coords,).
            mu_AU3_per_d2: Gravitational parameter in AU^3/day^2, positive, broadcast
                against B.
            epoch_d: Epoch in days of the built orbit, broadcast against B. Defaults to
                0.

        Returns:
            FrameOrbit of batch shape B.
        """

    @abstractmethod
    def from_orbit(self, orbit: FrameOrbit) -> Array:
        """Return coordinates representing the orbit at its stored epoch.

        Args:
            orbit: Orbit to represent, batch shape B.

        Returns:
            Coordinates at the orbit's stored epoch, shape B + (n_coords,).
        """

    @abstractmethod
    def log_volume(self, x) -> Array:
        """Return the exact log density of the reference measure, one per batch.

        This is the log Jacobian from the coordinates to normalized Haar times
        dex dey times d(ln a), including the normalization of any gauge.

        Args:
            x: Chart coordinates, shape B + (n_coords,).

        Returns:
            Log density of the reference measure, shape B.
        """

    @abstractmethod
    def sample_coordinates(self, key, orbit: FrameOrbit) -> Array:
        """Return coordinates of ``orbit``, with any gauge freedom drawn.

        Equals ``from_orbit`` for a chart without a gauge, with plain angles in
        ``[0, 2 pi)``; for a chart with a gauge it is ``from_orbit`` plus a draw
        from the gauge density, so that the coordinates of an orbit drawn from a
        law have the density ``law.log_prob + log_volume``.

        Args:
            key: JAX PRNG key for the gauge draw.
            orbit: Orbit whose coordinates are wanted, batch shape B.

        Returns:
            Coordinates of the orbit, shape B + (n_coords,).
        """

    def unit_coordinates(self, c, orbit: FrameOrbit) -> Array:
        """Return :meth:`sample_coordinates` with the gauge drawn from the unit cube.

        ``c`` has shape ``B + (gauge_unit_size,)`` with values in ``(0, 1)``; a chart
        without a gauge ignores it. The map is deterministic and its pushforward of
        uniform ``c`` is the gauge law of :meth:`sample_coordinates`.

        Args:
            c: Unit-cube values in (0, 1) for the gauge, shape B + (gauge_unit_size,).
            orbit: Orbit whose coordinates are wanted, batch shape B.

        Returns:
            Coordinates of the orbit, shape B + (n_coords,).
        """
        if self.gauge_unit_size:
            raise NotImplementedError(f"{type(self).__name__} must draw its gauge")
        return self.sample_coordinates(None, orbit)

    def exclude(self, x) -> tuple[Array, Array]:
        """Replace coordinates that have no orbit by a regular point.

        Returns ``(x_safe, excluded)``: ``excluded`` has the batch shape of ``x`` and is
        true where the chart excludes the coordinates (a measure-zero set where
        :meth:`to_orbit` is undefined), and ``x_safe`` equals ``x`` elsewhere and is a
        regular point there, so that densities evaluated at ``x_safe`` are finite and
        a caller can report no density at ``excluded`` without a NaN or its gradient.
        Charts without such a set return ``x`` and all false.

        Args:
            x: Chart coordinates, shape B + (n_coords,).

        Returns:
            Tuple (x_safe, excluded): the coordinates with excluded points replaced by a
            regular point, and a boolean array of shape B marking the excluded points.
        """
        x = _coordinates(x, self.n_coords)
        return x, jnp.zeros(x.shape[:-1], bool)

    def constrain(self, y, support) -> tuple[Array, Array]:
        """Map wall-free coordinates ``y`` into the interior of a law's bounds.

        ``support`` maps ``"ln_size"`` to the ``(low, high)`` interval of the size
        coordinate and ``"e"`` to that of the eccentricity, each optional and an array
        that broadcasts against the batch. A bounded coordinate is reached from the
        real line by a logit on its interval; the disk charts scale the eccentricity
        disk to radius ``high`` and ignore ``low``, leaving a disk of radius ``low`` of
        zero law density around the origin when it is above zero. The density in
        ``y`` then has no wall. An element the chart has no coordinate for is ignored.

        Args:
            y: Wall-free coordinates, shape B + (n_coords,).
            support: Mapping from "ln_size" and "e" to (low, high) intervals that
                broadcast against B. Each key is optional.

        Returns:
            The chart coordinates ``x`` (the argument of :meth:`to_orbit`) and
            ``log |dx/dy|``, one per batch element: ``y`` itself and zero when
            ``support`` is empty.
        """
        y = _coordinates(y, self.n_coords)
        x, log_jac = y, jnp.zeros(y.shape[:-1], y.dtype)
        if self.size_coordinate is not None and "ln_size" in support:
            value, log_d = _interval(y[..., 0], *support["ln_size"])
            x, log_jac = x.at[..., 0].set(value), log_jac + log_d
        if self.eccentricity_coordinate is not None and "e" in support:
            low, high = support["e"]
            if self.eccentricity_coordinate == "logit":
                width, t = high - low, y[..., 1]
                log_e = jnp.logaddexp(jnp.log(low), jnp.log(width) + log_sigmoid(t))
                log_rest = jnp.logaddexp(
                    jnp.log1p(-high), jnp.log(width) + log_sigmoid(-t)
                )
                log_d = (
                    jnp.log(width) + log_sigmoid(t) + log_sigmoid(-t) - log_e - log_rest
                )
                x = x.at[..., 1].set(log_e - log_rest)
            else:
                k = 1 - high**2
                r2 = jnp.sum(y[..., 1:3] ** 2, axis=-1)
                scale = high / jnp.sqrt(1 + k * r2)
                x = x.at[..., 1:3].set(scale[..., None] * y[..., 1:3])
                log_d = 2 * jnp.log(high) - 2 * jnp.log1p(k * r2)
            log_jac = log_jac + log_d
        return x, log_jac

    def unconstrain(self, x, support) -> Array:
        """Invert :meth:`constrain`: wall-free coordinates of chart coordinates ``x``.

        A coordinate within one rounding of its bound is held ``eps`` inside it, so
        the result is finite. One outside by more than ``sqrt(eps)`` of the interval
        width (of ``high`` for the eccentricity disk) has no wall-free coordinate: the
        whole row of the result is NaN.

        Args:
            x: Chart coordinates, shape B + (n_coords,).
            support: Mapping from "ln_size" and "e" to (low, high) intervals that
                broadcast against B, as for constrain.

        Returns:
            Wall-free coordinates, shape B + (n_coords,), NaN across a row whose
            coordinates lie outside the bounds.
        """
        x = _coordinates(x, self.n_coords)
        y = x
        if self.size_coordinate is not None and "ln_size" in support:
            y = y.at[..., 0].set(_interval_inverse(x[..., 0], *support["ln_size"]))
        if self.eccentricity_coordinate is not None and "e" in support:
            low, high = support["e"]
            if self.eccentricity_coordinate == "logit":
                y = y.at[..., 1].set(_interval_inverse(sigmoid(x[..., 1]), low, high))
            else:
                k = 1 - high**2
                u2 = jnp.sum(x[..., 1:3] ** 2, axis=-1)
                eps = jnp.finfo(x.dtype).eps
                e = jnp.sqrt(u2 / (1.0 + u2))
                outside = e > high * (1.0 + jnp.sqrt(eps))
                e2 = jnp.minimum(e, high * (1.0 - eps)) ** 2
                u2_held = e2 / (1.0 - e2)
                u2_held = jnp.where(e > high * (1.0 - eps), u2_held, u2)
                denom = jnp.sqrt(high**2 - k * u2_held)
                hold = jnp.sqrt(u2_held / jnp.where(u2 > 0, u2, 1.0))
                held = jnp.where(e > high * (1.0 - eps), hold, 1.0)
                y = y.at[..., 1:3].set(
                    (x[..., 1:3] * held[..., None]) / denom[..., None]
                )
                y = jnp.where(outside[..., None], jnp.nan, y)
        return jnp.where(jnp.any(jnp.isnan(y), axis=-1, keepdims=True), jnp.nan, y)


class FrameChart(AbstractChart):
    """Coordinates (ln a, ux, uy, vw, vx, vy, vz), with a quaternion gauge.

    The disk map is u/sqrt(1+norm(u)^2); the quaternion is v/norm(v). The inverse
    chooses unit length and makes the largest-magnitude component positive
    (first component wins ties). Thus neither length nor sign is recovered.

    At v=0 the identity quaternion is returned, with zero autodiff tangent
    for the quaternion coordinates. This measure-zero convention is not a
    derivative: radial normalization has no continuous extension at zero.
    Use nonzero v for differentiable inference; a Gaussian draw is nonzero
    almost surely.

    For positive gauge_sd, with r=norm(v), log_volume is
    -2 log(1+norm(u)^2) - 0.5*((r-1)/gauge_sd)^2 - 3 log(r)
    - log(gauge_sd sqrt(2 pi)) - log(Phi(1/gauge_sd)) - log(2 pi^2).
    The gauge is normalized: r is a normal of mean one truncated to r>0
    (Phi(1/gauge_sd) is the mass kept), and the direction is uniform on the
    three-sphere of area 2 pi^2; r=0 has zero density. Since d^4 v=r^3 dr dS3,
    the r^3 factor cancels: length and Haar-uniform direction are independent.
    The default keeps r near one to limit changes in orientation scales during
    inference.

    With gauge_sd=None, log_volume is instead
    -2 log(1+norm(u)^2) - norm(v)^2/2 - 2 log(2*pi), unchanged and normalized.
    Gaussian four-vector draws then give exact isotropic orientations.
    Both quaternion signs are integrated in either gauge.

    Attributes:
        n_coords: Number of coordinates on the trailing axis.
        coordinate_names: Coordinate names in trailing-axis order, with a unit suffix
            where the coordinate has a unit.
        size_coordinate: Kind of size coordinate that a law's bounds can map, "ln_a".
        eccentricity_coordinate: Kind of eccentricity coordinate that a law's bounds can
            map, "disk".
        periods: Period of each coordinate in radians, or None for a coordinate without
            one.
        gauge_unit_size: Number of unit-cube coordinates that unit_coordinates consumes
            for the gauge.
        gauge_sd: Standard deviation of the truncated normal gauge on the quaternion
            length, dimensionless; None selects the chi gauge of four degrees of
            freedom. Static under JIT.
    """

    n_coords: ClassVar[int] = 7
    coordinate_names: ClassVar[tuple[str, ...]] = (
        "ln_a_AU",
        "ux",
        "uy",
        "vw",
        "vx",
        "vy",
        "vz",
    )
    size_coordinate: ClassVar[str | None] = "ln_a"
    eccentricity_coordinate: ClassVar[str | None] = "disk"
    periods: ClassVar[tuple[float | None, ...]] = (None,) * 7
    gauge_unit_size: ClassVar[int] = 2
    gauge_sd: float | None = eqx.field(static=True, default=0.1)

    def to_orbit(self, x, *, mu_AU3_per_d2, epoch_d=0.0) -> FrameOrbit:
        """Build an epoch frame directly, without eccentricity or node angles.

        Args:
            x: Chart coordinates, shape B + (n_coords,).
            mu_AU3_per_d2: Gravitational parameter in AU^3/day^2, positive, broadcast
                against B.
            epoch_d: Epoch in days of the built orbit, broadcast against B. Defaults to
                0.

        Returns:
            FrameOrbit of batch shape B.
        """
        x = _coordinates(x, self.n_coords)
        evec = plane_to_disk(x[..., 1:3])
        v = x[..., 3:]
        squared = jnp.sum(v * v, axis=-1, keepdims=True)
        nonzero = squared > 0
        norm = jnp.sqrt(jnp.where(nonzero, squared, 1.0))
        quat = jnp.where(nonzero, v / norm, jnp.array([1.0, 0.0, 0.0, 0.0]))
        a, ex, ey, mu, epoch = jnp.broadcast_arrays(
            jnp.exp(x[..., 0]),
            evec[..., 0],
            evec[..., 1],
            jnp.asarray(mu_AU3_per_d2),
            jnp.asarray(epoch_d),
        )
        return FrameOrbit(
            a,
            mean_motion(a, mu),
            ex,
            ey,
            jnp.broadcast_to(quat, (*a.shape, 4)),
            epoch,
        )

    def from_orbit(self, orbit: FrameOrbit) -> Array:
        """Return a canonical representative, also at circular or face-on states.

        Args:
            orbit: Orbit to represent, batch shape B.

        Returns:
            Coordinates at the orbit's stored epoch, shape B + (7,), with a unit
            quaternion whose largest component is positive.
        """
        u = disk_to_plane(jnp.stack([orbit.ex, orbit.ey], -1))
        quat = orbit.quat / jnp.linalg.norm(orbit.quat, axis=-1, keepdims=True)
        pivot = jnp.take_along_axis(
            quat, jnp.argmax(jnp.abs(quat), axis=-1)[..., None], axis=-1
        )
        quat = quat * jnp.where(pivot < 0, -1.0, 1.0)
        return jnp.concatenate([jnp.log(orbit.a_AU)[..., None], u, quat], axis=-1)

    def log_volume(self, x) -> Array:
        """Return disk area density times the selected radial gauge lift.

        Args:
            x: Chart coordinates, shape B + (n_coords,).

        Returns:
            Log density of the reference measure, shape B.
        """
        x = _coordinates(x, self.n_coords)
        squared = jnp.sum(x[..., 3:] ** 2, axis=-1)
        if self.gauge_sd is None:
            gauge = -squared / 2 - 2 * jnp.log(2 * jnp.pi)
        else:
            nonzero = squared > 0
            radius = jnp.sqrt(jnp.where(nonzero, squared, 1.0))
            norm = (
                jnp.log(self.gauge_sd * jnp.sqrt(2 * jnp.pi))
                + log_ndtr(1.0 / self.gauge_sd)
                + jnp.log(2 * jnp.pi**2)
            )
            gauge = (
                -0.5 * ((radius - 1) / self.gauge_sd) ** 2 - 3 * jnp.log(radius) - norm
            )
            gauge = jnp.where(nonzero, gauge, -jnp.inf)
        return _disk_log_volume(x[..., 1:3]) + gauge

    def sample_coordinates(self, key, orbit: FrameOrbit) -> Array:
        """Return ``from_orbit`` with the quaternion length and sign drawn.

        Args:
            key: JAX PRNG key for the gauge draw.
            orbit: Orbit whose coordinates are wanted, batch shape B.

        Returns:
            Coordinates of the orbit with the quaternion length and sign drawn, shape B
            + (7,).
        """
        coords = self.from_orbit(orbit)
        k_radius, k_sign = jax.random.split(key)
        shape = coords.shape[:-1]
        if self.gauge_sd is None:
            radius = jnp.sqrt(2 * jax.random.gamma(k_radius, 2.0, shape))
        else:
            sd = self.gauge_sd
            radius = 1.0 + sd * jax.random.truncated_normal(
                k_radius, -1.0 / sd, jnp.inf, shape
            )
        sign = jnp.where(jax.random.bernoulli(k_sign, shape=shape), 1.0, -1.0)
        scale = (radius * sign)[..., None]
        return jnp.concatenate([coords[..., :3], scale * coords[..., 3:]], axis=-1)

    def unit_coordinates(self, c, orbit: FrameOrbit) -> Array:
        """Return ``from_orbit`` with the quaternion length and sign from ``c``.

        ``c[..., 0]`` is the length through the inverse CDF of its gauge law (a normal
        of mean one truncated to positive values, or a chi variable of four degrees of
        freedom without ``gauge_sd``) and ``c[..., 1]`` the sign (negative above one
        half).

        Args:
            c: Unit-cube values in (0, 1), shape B + (2,): the quaternion length and the
                sign.
            orbit: Orbit whose coordinates are wanted, batch shape B.

        Returns:
            Coordinates of the orbit, shape B + (7,).
        """
        c = jnp.asarray(c)
        coords = self.from_orbit(orbit)
        if self.gauge_sd is None:
            radius = _chi_four_icdf(c[..., 0])
        else:
            sd = self.gauge_sd
            below = ndtr(-1.0 / sd)
            radius = 1.0 + sd * ndtri(below + c[..., 0] * (1.0 - below))
            radius = jnp.maximum(radius, jnp.finfo(radius.dtype).tiny)
        sign = jnp.where(c[..., 1] < 0.5, 1.0, -1.0)
        scale = (radius * sign)[..., None]
        return jnp.concatenate([coords[..., :3], scale * coords[..., 3:]], axis=-1)


class EccentricityVectorChart(AbstractChart):
    """Coordinates (ln a, ux, uy, atanh(cos i), W, M0+w).

    u/sqrt(1+norm(u)^2) gives (e cos w, e sin w). Angles are plain real radians;
    predictions are periodic in them. Conventions match from_classical on
    FrameOrbit. The inverse remains regular at e=0 but is undefined at
    either inclination pole, which finite coordinates do not reach in exact
    arithmetic.

    log_volume = -2 log(1+norm(u)^2) + log(sech(z)^2) - log(8*pi^2), z=atanh(cos i).
    The constant is the total Haar measure of (cos i, W, M0+w) over one
    2*pi cell for each angle. Rotation of the eccentricity
    plane by -(M0+w) preserves area, including its coupled phase dependence.

    Attributes:
        n_coords: Number of coordinates on the trailing axis.
        coordinate_names: Coordinate names in trailing-axis order, with a unit suffix
            where the coordinate has a unit.
        size_coordinate: Kind of size coordinate that a law's bounds can map, "ln_a".
        eccentricity_coordinate: Kind of eccentricity coordinate that a law's bounds can
            map, "disk".
        periods: Period of each coordinate in radians, or None for a coordinate without
            one.
    """

    n_coords: ClassVar[int] = 6
    coordinate_names: ClassVar[tuple[str, ...]] = (
        "ln_a_AU",
        "ux",
        "uy",
        "atanh_cos_i",
        "W_rad",
        "M0_plus_w_rad",
    )
    size_coordinate: ClassVar[str | None] = "ln_a"
    eccentricity_coordinate: ClassVar[str | None] = "disk"
    periods: ClassVar[tuple[float | None, ...]] = (
        None,
        None,
        None,
        None,
        two_pi,
        two_pi,
    )

    def to_orbit(self, x, *, mu_AU3_per_d2, epoch_d=0.0) -> FrameOrbit:
        """Build the frame using mean longitude, without a periapsis singularity.

        Args:
            x: Chart coordinates, shape B + (n_coords,).
            mu_AU3_per_d2: Gravitational parameter in AU^3/day^2, positive, broadcast
                against B.
            epoch_d: Epoch in days of the built orbit, broadcast against B. Defaults to
                0.

        Returns:
            FrameOrbit of batch shape B.
        """
        x = _coordinates(x, self.n_coords)
        h, k = jnp.moveaxis(plane_to_disk(x[..., 1:3]), -1, 0)
        longitude = x[..., 5]
        c, s = jnp.cos(longitude), jnp.sin(longitude)
        frame = FrameOrbit.from_classical(
            jnp.exp(x[..., 0]),
            0.0,
            _inclination(x[..., 3]),
            x[..., 4],
            longitude,
            0.0,
            mu_AU3_per_d2=mu_AU3_per_d2,
            epoch_d=epoch_d,
        )
        return eqx.tree_at(
            lambda o: (o.ex, o.ey),
            frame,
            (
                jnp.broadcast_to(h * c + k * s, frame.a_AU.shape),
                jnp.broadcast_to(-h * s + k * c, frame.a_AU.shape),
            ),
        )

    def from_orbit(self, orbit: FrameOrbit) -> Array:
        """Recover node and mean longitude directly, including circular states.

        Args:
            orbit: Orbit to represent, batch shape B.

        Returns:
            Coordinates at the orbit's stored epoch, shape B + (6,).
        """
        f = rotate(orbit.quat, jnp.array([1.0, 0.0, 0.0]))
        normal = rotate(orbit.quat, jnp.array([0.0, 0.0, 1.0]))
        inc = jnp.arctan2(jnp.linalg.norm(normal[..., :2], axis=-1), -normal[..., 2])
        node = jnp.arctan2(-normal[..., 1], normal[..., 0])
        node_axis = jnp.stack([jnp.sin(node), jnp.cos(node), jnp.zeros_like(node)], -1)
        tangent = jnp.cross(normal, node_axis)
        longitude = jnp.arctan2(jnp.sum(f * tangent, -1), jnp.sum(f * node_axis, -1))
        c, s = jnp.cos(longitude), jnp.sin(longitude)
        u = disk_to_plane(
            jnp.stack([orbit.ex * c - orbit.ey * s, orbit.ex * s + orbit.ey * c], -1)
        )
        return jnp.concatenate(
            [
                jnp.log(orbit.a_AU)[..., None],
                u,
                jnp.stack([_inclination_coordinate(inc), node, longitude], -1),
            ],
            axis=-1,
        )

    def log_volume(self, x) -> Array:
        """Return disk area and isotropic inclination Jacobians.

        Args:
            x: Chart coordinates, shape B + (n_coords,).

        Returns:
            Log density of the reference measure, shape B.
        """
        x = _coordinates(x, self.n_coords)
        return (
            _disk_log_volume(x[..., 1:3])
            + _inclination_log_volume(x[..., 3])
            - _LOG_HAAR
        )

    def sample_coordinates(self, key, orbit: FrameOrbit) -> Array:
        """Return ``from_orbit`` with the angles reduced to ``[0, 2 pi)``.

        Args:
            key: JAX PRNG key. Unused: the chart has no gauge.
            orbit: Orbit whose coordinates are wanted, batch shape B.

        Returns:
            Coordinates with the angles reduced to [0, 2 pi), shape B + (6,).
        """
        return _wrap_angles(self.from_orbit(orbit), (4, 5))


class CartesianChart(AbstractChart):
    """Epoch position and escape-speed-scaled velocity coordinates (p, w).

    With positive scale_AU=s, r=s*p is in (east, north, away) AU and
    v=sqrt(2*mu/norm(r))*w/sqrt(1+norm(w)^2) is in AU/day. The velocity map is
    smooth and bijective onto the open escape-speed ball, including w=0,
    with no division by norm(w) and no clipping.

    Nonzero position and nonradial velocity are required by FrameOrbit:
    every finite input has negative energy at r!=0, but exactly radial
    states (including w=0) have e=1 and no orbital-plane orientation.
    This measure-zero set is excluded; circular and face-on orbits are
    regular. Floating-point boundary limitations also apply.

    Attributes:
        n_coords: Number of coordinates on the trailing axis.
        coordinate_names: Coordinate names in trailing-axis order, with a unit suffix
            where the coordinate has a unit.
        periods: Period of each coordinate in radians, or None for a coordinate without
            one.
        scale_AU: Length in AU that multiplies the position coordinates, positive.
            Static under JIT.
    """

    n_coords: ClassVar[int] = 6
    coordinate_names: ClassVar[tuple[str, ...]] = ("px", "py", "pz", "wx", "wy", "wz")
    periods: ClassVar[tuple[float | None, ...]] = (None,) * 6
    scale_AU: float = eqx.field(static=True, default=1.0)

    def to_orbit(self, x, *, mu_AU3_per_d2, epoch_d=0.0) -> FrameOrbit:
        """Build a nonradial bound orbit from its Cartesian epoch state.

        Args:
            x: Chart coordinates, shape B + (n_coords,).
            mu_AU3_per_d2: Gravitational parameter in AU^3/day^2, positive, broadcast
                against B.
            epoch_d: Epoch in days of the built orbit, broadcast against B. Defaults to
                0.

        Returns:
            FrameOrbit of batch shape B.
        """
        x = _coordinates(x, self.n_coords)
        r = self.scale_AU * x[..., :3]
        escape = jnp.sqrt(2 * jnp.asarray(mu_AU3_per_d2) / jnp.linalg.norm(r, axis=-1))
        return FrameOrbit.from_cartesian(
            r,
            escape[..., None] * plane_to_disk(x[..., 3:]),
            mu_AU3_per_d2=mu_AU3_per_d2,
            epoch_d=epoch_d,
        )

    def from_orbit(self, orbit: FrameOrbit) -> Array:
        """Recover position and inverse-ball velocity at the stored epoch.

        Args:
            orbit: Orbit to represent, batch shape B.

        Returns:
            Coordinates at the orbit's stored epoch, shape B + (6,).
        """
        r, v = orbit.to_cartesian()
        escape = jnp.sqrt(2 * orbit.mu_AU3_per_d2 / jnp.linalg.norm(r, axis=-1))
        return jnp.concatenate(
            [r / self.scale_AU, disk_to_plane(v / escape[..., None])], -1
        )

    def exclude(self, x) -> tuple[Array, Array]:
        """Exclude the origin, where the escape speed is infinite.

        Args:
            x: Chart coordinates, shape B + (n_coords,).

        Returns:
            Tuple (x_safe, excluded): the coordinates with a zero position replaced by a
            unit vector, and a boolean array of shape B marking those points.
        """
        x = _coordinates(x, self.n_coords)
        excluded = jnp.all(x[..., :3] == 0, axis=-1)
        unit = jnp.array([1.0, 0.0, 0.0])
        return x.at[..., :3].set(
            jnp.where(excluded[..., None], unit, x[..., :3])
        ), excluded

    def log_volume(self, x) -> Array:
        """Return -3 log(norm(p)) - 4 log(1+norm(w)^2), independent of mu and scale.

        Let s=scale_AU, R=s*norm(p), b=1+norm(w)^2, a=R*b/2, and dnu be the
        reference measure e de d(cos i) dW dw dM d(ln a) with unnormalized
        Haar orientation. Delaunay actions L=sqrt(mu*a), G=L*sqrt(1-e^2),
        H=G*cos(i), and J=s^3*(2*mu/R)^(3/2)*b^(-5/2) give:
        d^3r d^3v = dL dG dH dM dw dW = (mu*a)^(3/2)/2 * dnu.
        dnu/d^6x = 2*J/(mu*a)^(3/2) = 16*norm(p)^(-3)*b^(-4).

        The ball Jacobian b^(-5/2) has one radial and two tangential
        eigenvalues b^(-3/2), b^(-1/2), b^(-1/2). Position dependence of
        escape speed leaves the phase-space Jacobian block triangular.
        Thus escape-speed scaling cancels mu exactly, and R=s*norm(p) cancels
        s. The constant is independent of scale: log(16) against unnormalized
        Haar, so log(16/(8 pi^2)) = log(2/pi^2) against normalized Haar, which
        is what is returned.

        Args:
            x: Chart coordinates, shape B + (n_coords,).

        Returns:
            Log density of the reference measure, shape B.
        """
        x = _coordinates(x, self.n_coords)
        return (
            -3 * jnp.log(jnp.linalg.norm(x[..., :3], axis=-1))
            - 4 * jnp.log1p(jnp.sum(x[..., 3:] ** 2, axis=-1))
            + jnp.log(2 / jnp.pi**2)
        )

    def sample_coordinates(self, key, orbit: FrameOrbit) -> Array:
        """Return ``from_orbit``; the chart has no gauge.

        Args:
            key: JAX PRNG key. Unused: the chart has no gauge.
            orbit: Orbit whose coordinates are wanted, batch shape B.

        Returns:
            Coordinates of the orbit, shape B + (6,).
        """
        return self.from_orbit(orbit)


class ClassicalChart(AbstractChart):
    """Coordinates (ln a, logit(e), atanh(cos i), W, w, M0).

    Angles are plain real radians, with periodic predictions and the
    conventions of FrameOrbit.from_classical. The inverse is undefined at
    e=0 or either inclination pole.

    log_volume = 2 log(e) + log(1-e) + log(sech(z)^2) - log(8*pi^2), z=atanh(cos i).
    Polar eccentricity area contributes e, the logit contributes e*(1-e),
    and Haar orientation contributes d(cos i) dW d(M0+w). The constant is the
    total Haar measure over one 2*pi cell for each angle. There is no size
    Jacobian because size is ln a.

    Attributes:
        n_coords: Number of coordinates on the trailing axis.
        coordinate_names: Coordinate names in trailing-axis order, with a unit suffix
            where the coordinate has a unit.
        size_coordinate: Kind of size coordinate that a law's bounds can map, "ln_a".
        eccentricity_coordinate: Kind of eccentricity coordinate that a law's bounds can
            map, "logit".
        periods: Period of each coordinate in radians, or None for a coordinate without
            one.
    """

    n_coords: ClassVar[int] = 6
    coordinate_names: ClassVar[tuple[str, ...]] = (
        "ln_a_AU",
        "logit_e",
        "atanh_cos_i",
        "W_rad",
        "w_rad",
        "M0_rad",
    )
    size_coordinate: ClassVar[str | None] = "ln_a"
    eccentricity_coordinate: ClassVar[str | None] = "logit"
    periods: ClassVar[tuple[float | None, ...]] = (
        None,
        None,
        None,
        two_pi,
        two_pi,
        two_pi,
    )

    def to_orbit(self, x, *, mu_AU3_per_d2, epoch_d=0.0) -> FrameOrbit:
        """Build from smoothly transformed classical elements.

        Args:
            x: Chart coordinates, shape B + (n_coords,).
            mu_AU3_per_d2: Gravitational parameter in AU^3/day^2, positive, broadcast
                against B.
            epoch_d: Epoch in days of the built orbit, broadcast against B. Defaults to
                0.

        Returns:
            FrameOrbit of batch shape B.
        """
        x = _coordinates(x, self.n_coords)
        return FrameOrbit.from_classical(
            jnp.exp(x[..., 0]),
            sigmoid(x[..., 1]),
            _inclination(x[..., 2]),
            x[..., 3],
            x[..., 4],
            x[..., 5],
            mu_AU3_per_d2=mu_AU3_per_d2,
            epoch_d=epoch_d,
        )

    def from_orbit(self, orbit: FrameOrbit) -> Array:
        """Recover classical coordinates, with each angle reduced modulo 2*pi.

        Args:
            orbit: Orbit to represent, batch shape B.

        Returns:
            Coordinates at the orbit's stored epoch, shape B + (6,), with angles in [0,
            2 pi) radians.
        """
        elements = orbit.to_classical()
        e = elements["e"]
        return jnp.stack(
            [
                jnp.log(elements["a_AU"]),
                jnp.log(e) - jnp.log1p(-e),
                _inclination_coordinate(elements["i_rad"]),
                elements["W_rad"],
                elements["w_rad"],
                elements["M0_rad"],
            ],
            axis=-1,
        )

    def log_volume(self, x) -> Array:
        """Return eccentricity area, logit and isotropic inclination Jacobians.

        Args:
            x: Chart coordinates, shape B + (n_coords,).

        Returns:
            Log density of the reference measure, shape B.
        """
        x = _coordinates(x, self.n_coords)
        return (
            2 * log_sigmoid(x[..., 1])
            + log_sigmoid(-x[..., 1])
            + _inclination_log_volume(x[..., 2])
            - _LOG_HAAR
        )

    def sample_coordinates(self, key, orbit: FrameOrbit) -> Array:
        """Return ``from_orbit`` with the angles reduced to ``[0, 2 pi)``.

        Args:
            key: JAX PRNG key. Unused: the chart has no gauge.
            orbit: Orbit whose coordinates are wanted, batch shape B.

        Returns:
            Coordinates with the angles reduced to [0, 2 pi), shape B + (6,).
        """
        return _wrap_angles(self.from_orbit(orbit), (3, 4, 5))


class ThieleInnesChart(AbstractChart):
    """Coordinates (ln n, ux, uy, A, B, F, G), with free implied mass.

    n is in radians/day and (ex,ey)=u/sqrt(1+norm(u)^2). The four constants
    are in AU on the stored epoch mean-longitude axes f,g, NOT periapsis
    axes: A,F are north, B,G east, so the sky block is [[B,G],[A,F]].
    These are the east/north axes of SkyFrame observables. See linear_map
    on FrameOrbit for the single rotation to classical Thiele-Innes constants.

    There are seven independent coordinates. The phase is already carried
    by the epoch frame and eccentricity pair: for e>0, M0=atan2(-ey,ex).
    At e=0 that angle is undefined but the epoch frame still fixes orbital
    phase. Adding an independent phase to these seven coordinates would
    create an in-plane rotation gauge, not a bijective chart.

    a is the sky block's largest singular value. mu_AU3_per_d2 is accepted
    for AbstractChart compatibility but IGNORED, including its shape;
    mu=n^2*a^3 is implied, available through implied_mu. epoch_d broadcasts
    against the coordinate batch. Thus orbit round trips preserve the
    orbit's own rate and mass, irrespective of the passed mu.

    Static sheet selects the sign of the largest-magnitude away-row entry
    (first wins ties), as in sky_block_to_frame. cos(i)=(A*G-B*F)/a^2 is
    already determined by the sky block; sheet selects the remaining
    line-of-sight reflection, not prograde versus retrograde. On each
    declared sheet this is one-to-one onto bound orbit states, modulo
    quaternion sign, with zero sky block excluded. An orbit on the other
    sheet is reflected on round trip. The sheet has a pivot seam and
    both sheets meet at face-on. Values remain finite there; derivatives
    of the inverse and the volume density are singular.

    Reference measure is the other charts' normalized Haar*dex*dey*
    d(ln a), extended by d(ln mu). Write c=cos(i), l=ln(a), and u0 for
    the angle locating f from the node. The volume derivation is:
    dnu = dex dey dl dc dW du0 d(ln mu).
    ``|d(A,B,F,G)/d(l,c,W,u0)|`` = a^4*(1-c^2).
    d(ln mu) = 2*d(ln n)+3*dl, giving a factor 2 in the wedge product.
    ``|d(ex,ey)/d(ux,uy)|`` = (1+ux^2+uy^2)^(-2).
    log_volume = log(2)-4*log(a)-log(1-c^2)-2*log(1+ux^2+uy^2)-log(8*pi^2).
    The log(2) is the Jacobian of d(ln mu), not a prior factor. The chart
    covers one line-of-sight sheet; a density over its coordinates for a
    law on orbits is the sum of the law over both sheets (ThieleInnesBlock).
    At face-on log_volume=+inf, an integrable coordinate singularity, not a
    finite density approximation.

    Attributes:
        n_coords: Number of coordinates on the trailing axis.
        coordinate_names: Coordinate names in trailing-axis order, with a unit suffix
            where the coordinate has a unit.
        size_coordinate: Kind of size coordinate that a law's bounds can map, "ln_n".
        eccentricity_coordinate: Kind of eccentricity coordinate that a law's bounds can
            map, "disk".
        periods: Period of each coordinate in radians, or None for a coordinate without
            one.
        sheet: Line-of-sight sheet, +1 or -1, selecting the sign of the
            largest-magnitude away-row entry. Static under JIT.
    """

    n_coords: ClassVar[int] = 7
    coordinate_names: ClassVar[tuple[str, ...]] = (
        "ln_n_rad_per_d",
        "ux",
        "uy",
        "A_AU",
        "B_AU",
        "F_AU",
        "G_AU",
    )
    size_coordinate: ClassVar[str | None] = "ln_n"
    eccentricity_coordinate: ClassVar[str | None] = "disk"
    periods: ClassVar[tuple[float | None, ...]] = (None,) * 7
    sheet: int = eqx.field(static=True, default=1)

    def __check_init__(self):
        """Validate the static choice of line-of-sight sheet."""
        if self.sheet not in (-1, 1):
            raise ValueError("sheet must be +1 or -1")

    def to_orbit(self, x, *, mu_AU3_per_d2, epoch_d=0.0) -> FrameOrbit:
        """Build an epoch frame, ignoring mu in favor of the coordinate rate.

        Args:
            x: Chart coordinates, shape B + (n_coords,).
            mu_AU3_per_d2: Accepted for compatibility and ignored; the rate comes from
                the coordinates.
            epoch_d: Epoch in days of the built orbit, broadcast against B. Defaults to
                0.

        Returns:
            FrameOrbit of batch shape B.
        """
        x = _coordinates(x, self.n_coords)
        a, quat = sky_block_to_frame(*jnp.moveaxis(x[..., 3:], -1, 0), sheet=self.sheet)
        evec = plane_to_disk(x[..., 1:3])
        a, n, ex, ey, epoch = jnp.broadcast_arrays(
            a, jnp.exp(x[..., 0]), evec[..., 0], evec[..., 1], jnp.asarray(epoch_d)
        )
        return FrameOrbit(a, n, ex, ey, jnp.broadcast_to(quat, (*a.shape, 4)), epoch)

    def from_orbit(self, orbit: FrameOrbit) -> Array:
        """Project an orbit; inversion reproduces its declared sheet only.

        Args:
            orbit: Orbit to project, batch shape B.

        Returns:
            Coordinates, shape B + (7,): ln n with n in radians/day, the disk-plane
            pair, then A, B, F, G in AU.
        """
        return jnp.concatenate(
            [
                jnp.log(orbit.n_rad_per_d)[..., None],
                disk_to_plane(jnp.stack([orbit.ex, orbit.ey], -1)),
                frame_to_sky_block(orbit.a_AU, orbit.quat),
            ],
            -1,
        )

    def exclude(self, x) -> tuple[Array, Array]:
        """Exclude the zero sky block, which has no size.

        Args:
            x: Chart coordinates, shape B + (n_coords,).

        Returns:
            Tuple (x_safe, excluded): the coordinates with a zero sky block replaced by
            ones, and a boolean array of shape B marking those points.
        """
        x = _coordinates(x, self.n_coords)
        excluded = jnp.all(x[..., 3:] == 0, axis=-1)
        sky = jnp.where(excluded[..., None], 1.0, x[..., 3:])
        return x.at[..., 3:].set(sky), excluded

    def implied_mu(self, x) -> Array:
        """Return n^2*a^3 in AU^3/day^2 with the coordinate batch shape.

        Args:
            x: Chart coordinates, shape B + (n_coords,).

        Returns:
            Implied gravitational parameter in AU^3/day^2, shape B.
        """
        x = _coordinates(x, self.n_coords)
        a, _ = sky_block_geometry(*jnp.moveaxis(x[..., 3:], -1, 0))
        return jnp.exp(2 * x[..., 0]) * a**3

    def log_volume(self, x) -> Array:
        """Return the density against Haar*dex*dey*d(ln a)*d(ln mu).

        Args:
            x: Chart coordinates, shape B + (n_coords,).

        Returns:
            Log density against the reference measure extended by d(ln mu), shape B.
        """
        x = _coordinates(x, self.n_coords)
        a, gap = sky_block_geometry(*jnp.moveaxis(x[..., 3:], -1, 0))
        # gap=a^2*(1-c^2), computed without cancellation near face-on.
        return (
            jnp.log(2.0)
            - 2 * jnp.log(a)
            - jnp.log(gap)
            + _disk_log_volume(x[..., 1:3])
            - _LOG_HAAR
        )

    def sample_coordinates(self, key, orbit: FrameOrbit) -> Array:
        """Return ``from_orbit``; an orbit on either sheet maps to this sheet.

        Args:
            key: JAX PRNG key. Unused: the chart has no gauge.
            orbit: Orbit whose coordinates are wanted, batch shape B.

        Returns:
            Coordinates on this chart's sheet, shape B + (7,).
        """
        return self.from_orbit(orbit)
