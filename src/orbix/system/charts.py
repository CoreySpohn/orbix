"""Unconstrained coordinates and volumes for bound Keplerian orbit frames.

The common reference measure is Haar orientation measure times dex dey
times d(ln a). For a density p against that measure, the coordinate log
density is log(p(chart.to_orbit(x))) + chart.log_volume(x). The frame chart
also supplies a radial gauge density, possibly up to a constant. Plain real
angles have periodic predictions; densities are interpreted per fundamental cell.

Transforms do not clip. As with exp and sigmoid, floating-point overflow,
underflow and rounding to a boundary limit the representable input range.
"""

from abc import abstractmethod
from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
from jax.nn import log_sigmoid, sigmoid
from jaxtyping import Array

from orbix.equations.orbit import mean_motion
from orbix.system.frame_orbit import FrameOrbit
from orbix.system.rotation import rotate


def _coordinates(x, size):
    """Convert coordinates and validate their static trailing dimension."""
    x = jnp.asarray(x)
    if x.shape[-1:] != (size,):
        raise ValueError(f"Coordinates must have last axis of length {size}")
    return x


def _disk(u):
    """Map the plane smoothly onto the open disk, including its origin."""
    return u / jnp.sqrt(1 + jnp.sum(u * u, axis=-1, keepdims=True))


def _plane(evec):
    """Invert the disk map without polar coordinates."""
    return evec / jnp.sqrt(1 - jnp.sum(evec * evec, axis=-1, keepdims=True))


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


class AbstractChart(eqx.Module):
    """Static-only orbit coordinates with a free leading batch shape B.

    Coordinates have shape B + (n_coords,), and log_volume returns shape B.
    mu_AU3_per_d2 must be positive; it and epoch_d broadcast against B.
    Inverses use the state's own epoch and cannot recover angle windings.
    """

    n_coords: eqx.AbstractClassVar[int]

    @abstractmethod
    def to_orbit(self, x, *, mu_AU3_per_d2, epoch_d=0.0) -> FrameOrbit:
        """Map coordinates to a bound orbit with the supplied rate and epoch."""

    @abstractmethod
    def from_orbit(self, orbit: FrameOrbit) -> Array:
        """Return coordinates representing the orbit at its stored epoch."""

    @abstractmethod
    def log_volume(self, x) -> Array:
        """Return the log reference-measure density, one scalar per batch."""


class FrameChart(AbstractChart):
    """Coordinates (ln a, ux, uy, vw, vx, vy, vz), with a quaternion gauge.

    The disk map is u/sqrt(1+|u|^2); the quaternion is v/|v|. The inverse
    chooses unit length and makes the largest-magnitude component positive
    (first component wins ties). Thus neither length nor sign is recovered.

    At v=0 the identity quaternion is returned, with zero autodiff tangent
    for the quaternion coordinates. This measure-zero convention is not a
    derivative: radial normalization has no continuous extension at zero.
    Use nonzero v for differentiable inference; a Gaussian draw is nonzero
    almost surely.

    For positive gauge_sd, with r=|v|, log_volume is
    -2 log(1+|u|^2) - 0.5*((r-1)/gauge_sd)^2 - 3 log(r).
    The additive normalization constant is omitted, including the normal's
    truncation to r>0 (a negligible correction for small gauge_sd); r=0
    has zero density. Since d^4 v=r^3 dr dS3, the r^3 factor cancels:
    length and Haar-uniform direction are independent. The default keeps
    r near one to limit changes in orientation scales during inference.

    With gauge_sd=None, log_volume is instead
    -2 log(1+|u|^2) - |v|^2/2 - 2 log(2*pi), unchanged and normalized.
    Gaussian four-vector draws then give exact isotropic orientations.
    Both quaternion signs are integrated in either gauge.
    """

    n_coords: ClassVar[int] = 7
    gauge_sd: float | None = eqx.field(static=True, default=0.1)

    def to_orbit(self, x, *, mu_AU3_per_d2, epoch_d=0.0) -> FrameOrbit:
        """Build an epoch frame directly, without eccentricity or node angles."""
        x = _coordinates(x, self.n_coords)
        evec = _disk(x[..., 1:3])
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
        """Return a canonical representative, also at circular or face-on states."""
        u = _plane(jnp.stack([orbit.ex, orbit.ey], -1))
        quat = orbit.quat / jnp.linalg.norm(orbit.quat, axis=-1, keepdims=True)
        pivot = jnp.take_along_axis(
            quat, jnp.argmax(jnp.abs(quat), axis=-1)[..., None], axis=-1
        )
        quat = quat * jnp.where(pivot < 0, -1.0, 1.0)
        return jnp.concatenate([jnp.log(orbit.a_AU)[..., None], u, quat], axis=-1)

    def log_volume(self, x) -> Array:
        """Return disk area density times the selected radial gauge lift."""
        x = _coordinates(x, self.n_coords)
        squared = jnp.sum(x[..., 3:] ** 2, axis=-1)
        if self.gauge_sd is None:
            gauge = -squared / 2 - 2 * jnp.log(2 * jnp.pi)
        else:
            nonzero = squared > 0
            radius = jnp.sqrt(jnp.where(nonzero, squared, 1.0))
            gauge = -0.5 * ((radius - 1) / self.gauge_sd) ** 2 - 3 * jnp.log(radius)
            gauge = jnp.where(nonzero, gauge, -jnp.inf)
        return _disk_log_volume(x[..., 1:3]) + gauge


class EccentricityVectorChart(AbstractChart):
    """Coordinates (ln a, ux, uy, atanh(cos i), W, M0+w).

    u/sqrt(1+|u|^2) gives (e cos w, e sin w). Angles are plain real radians;
    predictions are periodic in them. Conventions match from_classical on
    FrameOrbit. The inverse remains regular at e=0 but is undefined at
    either inclination pole, which finite coordinates do not reach in exact
    arithmetic.

    log_volume = -2 log(1+|u|^2) + log(sech(z)^2), z=atanh(cos i).
    The omitted additive constant is -log(8*pi^2) for normalized Haar
    measure and one 2*pi cell for each angle. Rotation of the eccentricity
    plane by -(M0+w) preserves area, including its coupled phase dependence.
    """

    n_coords: ClassVar[int] = 6

    def to_orbit(self, x, *, mu_AU3_per_d2, epoch_d=0.0) -> FrameOrbit:
        """Build the frame using mean longitude, without a periapsis singularity."""
        x = _coordinates(x, self.n_coords)
        h, k = jnp.moveaxis(_disk(x[..., 1:3]), -1, 0)
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
        """Recover node and mean longitude directly, including circular states."""
        f = rotate(orbit.quat, jnp.array([1.0, 0.0, 0.0]))
        normal = rotate(orbit.quat, jnp.array([0.0, 0.0, 1.0]))
        inc = jnp.arctan2(jnp.linalg.norm(normal[..., :2], axis=-1), -normal[..., 2])
        node = jnp.arctan2(-normal[..., 1], normal[..., 0])
        node_axis = jnp.stack([jnp.sin(node), jnp.cos(node), jnp.zeros_like(node)], -1)
        tangent = jnp.cross(normal, node_axis)
        longitude = jnp.arctan2(jnp.sum(f * tangent, -1), jnp.sum(f * node_axis, -1))
        c, s = jnp.cos(longitude), jnp.sin(longitude)
        u = _plane(
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
        """Return disk area and isotropic inclination Jacobians."""
        x = _coordinates(x, self.n_coords)
        return _disk_log_volume(x[..., 1:3]) + _inclination_log_volume(x[..., 3])


class CartesianChart(AbstractChart):
    """Epoch position and escape-speed-scaled velocity coordinates (p, w).

    With positive scale_AU=s, r=s*p is in (east, north, away) AU and
    v=sqrt(2*mu/|r|)*w/sqrt(1+|w|^2) is in AU/day. The velocity map is
    smooth and bijective onto the open escape-speed ball, including w=0,
    with no division by |w| and no clipping.

    Nonzero position and nonradial velocity are required by FrameOrbit:
    every finite input has negative energy at r!=0, but exactly radial
    states (including w=0) have e=1 and no orbital-plane orientation.
    This measure-zero set is excluded; circular and face-on orbits are
    regular. Floating-point boundary limitations also apply.
    """

    n_coords: ClassVar[int] = 6
    scale_AU: float = eqx.field(static=True, default=1.0)

    def to_orbit(self, x, *, mu_AU3_per_d2, epoch_d=0.0) -> FrameOrbit:
        """Build a nonradial bound orbit from its Cartesian epoch state."""
        x = _coordinates(x, self.n_coords)
        r = self.scale_AU * x[..., :3]
        escape = jnp.sqrt(2 * jnp.asarray(mu_AU3_per_d2) / jnp.linalg.norm(r, axis=-1))
        return FrameOrbit.from_cartesian(
            r,
            escape[..., None] * _disk(x[..., 3:]),
            mu_AU3_per_d2=mu_AU3_per_d2,
            epoch_d=epoch_d,
        )

    def from_orbit(self, orbit: FrameOrbit) -> Array:
        """Recover position and inverse-ball velocity at the stored epoch."""
        r, v = orbit.to_cartesian()
        escape = jnp.sqrt(2 * orbit.mu_AU3_per_d2 / jnp.linalg.norm(r, axis=-1))
        return jnp.concatenate([r / self.scale_AU, _plane(v / escape[..., None])], -1)

    def log_volume(self, x) -> Array:
        """Return -3 log(|p|) - 4 log(1+|w|^2), independent of mu and scale.

        Let s=scale_AU, R=s*|p|, b=1+|w|^2, a=R*b/2, and dnu be the
        reference measure e de d(cos i) dW dw dM d(ln a) with unnormalized
        Haar orientation. Delaunay actions L=sqrt(mu*a), G=L*sqrt(1-e^2),
        H=G*cos(i), and J=s^3*(2*mu/R)^(3/2)*b^(-5/2) give:
        d^3r d^3v = dL dG dH dM dw dW = (mu*a)^(3/2)/2 * dnu.
        dnu/d^6x = 2*J/(mu*a)^(3/2) = 16*|p|^(-3)*b^(-4).

        The ball Jacobian b^(-5/2) has one radial and two tangential
        eigenvalues b^(-3/2), b^(-1/2), b^(-1/2). Position dependence of
        escape speed leaves the phase-space Jacobian block triangular.
        Thus escape-speed scaling cancels mu exactly, and R=s*|p| cancels
        s. The omitted additive constant may depend on scale_AU only;
        here it is independent of scale too: log(16) for unnormalized
        Haar, or log(2/pi^2) for normalized Haar.
        """
        x = _coordinates(x, self.n_coords)
        return -3 * jnp.log(jnp.linalg.norm(x[..., :3], axis=-1)) - 4 * jnp.log1p(
            jnp.sum(x[..., 3:] ** 2, axis=-1)
        )


class ClassicalChart(AbstractChart):
    """Coordinates (ln a, logit(e), atanh(cos i), W, w, M0).

    Angles are plain real radians, with periodic predictions and the
    conventions of FrameOrbit.from_classical. The inverse is undefined at
    e=0 or either inclination pole.

    log_volume = 2 log(e) + log(1-e) + log(sech(z)^2), z=atanh(cos i).
    Polar eccentricity area contributes e, the logit contributes e*(1-e),
    and Haar orientation contributes d(cos i) dW d(M0+w). The omitted
    additive constant is -log(8*pi^2) for normalized Haar measure and one
    2*pi cell for each angle. There is no size Jacobian because size is ln a.
    """

    n_coords: ClassVar[int] = 6

    def to_orbit(self, x, *, mu_AU3_per_d2, epoch_d=0.0) -> FrameOrbit:
        """Build from smoothly transformed classical elements."""
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
        """Recover classical coordinates, with each angle reduced modulo 2*pi."""
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
        """Return eccentricity area, logit and isotropic inclination Jacobians."""
        x = _coordinates(x, self.n_coords)
        return (
            2 * log_sigmoid(x[..., 1])
            + log_sigmoid(-x[..., 1])
            + _inclination_log_volume(x[..., 2])
        )
