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
from hwoutils.constants import two_pi
from jax.nn import log_sigmoid, sigmoid
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
    """
    return u / jnp.sqrt(1 + jnp.sum(u * u, axis=-1, keepdims=True))


def disk_to_plane(evec):
    """Invert ``plane_to_disk`` without polar coordinates.

    ``evec`` is an eccentricity vector ``(ex, ey)`` with ``|evec| < 1`` on the
    trailing axis; the result is ``u = evec / sqrt(1 - |evec|^2)``.
    """
    return evec / jnp.sqrt(1 - jnp.sum(evec * evec, axis=-1, keepdims=True))


def periapsis_chart_shape(T_d, e, tp_d, epoch_d=0.0):
    """Chart shape ``(ln n, ex, ey)`` from period, eccentricity and periapsis time.

    ``(ex, ey) = e (cos M, -sin M)`` on the epoch mean-longitude axes, with
    ``M = n (epoch_d - tp_d)`` the epoch mean anomaly and ``n = 2 pi / T_d``
    the mean motion in radians per day. Inputs broadcast.
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


class AbstractChart(eqx.Module):
    """Static-only orbit coordinates with a free leading batch shape B.

    Coordinates have shape B + (n_coords,), and log_volume returns shape B.
    mu_AU3_per_d2 must be positive; it and epoch_d broadcast against B.
    Inverses use the state's own epoch and cannot recover angle windings.
    ``coordinate_names`` names the trailing axis, in order, for hosts that keep
    coordinates as a dict of named arrays.
    """

    n_coords: eqx.AbstractClassVar[int]
    coordinate_names: ClassVar[tuple[str, ...]]

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

    The disk map is u/sqrt(1+norm(u)^2); the quaternion is v/norm(v). The inverse
    chooses unit length and makes the largest-magnitude component positive
    (first component wins ties). Thus neither length nor sign is recovered.

    At v=0 the identity quaternion is returned, with zero autodiff tangent
    for the quaternion coordinates. This measure-zero convention is not a
    derivative: radial normalization has no continuous extension at zero.
    Use nonzero v for differentiable inference; a Gaussian draw is nonzero
    almost surely.

    For positive gauge_sd, with r=norm(v), log_volume is
    -2 log(1+norm(u)^2) - 0.5*((r-1)/gauge_sd)^2 - 3 log(r).
    The additive normalization constant is omitted, including the normal's
    truncation to r>0 (a negligible correction for small gauge_sd); r=0
    has zero density. Since d^4 v=r^3 dr dS3, the r^3 factor cancels:
    length and Haar-uniform direction are independent. The default keeps
    r near one to limit changes in orientation scales during inference.

    With gauge_sd=None, log_volume is instead
    -2 log(1+norm(u)^2) - norm(v)^2/2 - 2 log(2*pi), unchanged and normalized.
    Gaussian four-vector draws then give exact isotropic orientations.
    Both quaternion signs are integrated in either gauge.
    """

    n_coords: ClassVar[int] = 7
    coordinate_names: ClassVar[tuple[str, ...]] = (
        "ln_a",
        "ux",
        "uy",
        "vw",
        "vx",
        "vy",
        "vz",
    )
    gauge_sd: float | None = eqx.field(static=True, default=0.1)

    def to_orbit(self, x, *, mu_AU3_per_d2, epoch_d=0.0) -> FrameOrbit:
        """Build an epoch frame directly, without eccentricity or node angles."""
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
        """Return a canonical representative, also at circular or face-on states."""
        u = disk_to_plane(jnp.stack([orbit.ex, orbit.ey], -1))
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

    u/sqrt(1+norm(u)^2) gives (e cos w, e sin w). Angles are plain real radians;
    predictions are periodic in them. Conventions match from_classical on
    FrameOrbit. The inverse remains regular at e=0 but is undefined at
    either inclination pole, which finite coordinates do not reach in exact
    arithmetic.

    log_volume = -2 log(1+norm(u)^2) + log(sech(z)^2), z=atanh(cos i).
    The omitted additive constant is -log(8*pi^2) for normalized Haar
    measure and one 2*pi cell for each angle. Rotation of the eccentricity
    plane by -(M0+w) preserves area, including its coupled phase dependence.
    """

    n_coords: ClassVar[int] = 6
    coordinate_names: ClassVar[tuple[str, ...]] = (
        "ln_a",
        "ux",
        "uy",
        "atanh_cos_i",
        "W_rad",
        "M0_plus_w_rad",
    )

    def to_orbit(self, x, *, mu_AU3_per_d2, epoch_d=0.0) -> FrameOrbit:
        """Build the frame using mean longitude, without a periapsis singularity."""
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
        """Recover node and mean longitude directly, including circular states."""
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
        """Return disk area and isotropic inclination Jacobians."""
        x = _coordinates(x, self.n_coords)
        return _disk_log_volume(x[..., 1:3]) + _inclination_log_volume(x[..., 3])


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
    """

    n_coords: ClassVar[int] = 6
    coordinate_names: ClassVar[tuple[str, ...]] = ("px", "py", "pz", "wx", "wy", "wz")
    scale_AU: float = eqx.field(static=True, default=1.0)

    def to_orbit(self, x, *, mu_AU3_per_d2, epoch_d=0.0) -> FrameOrbit:
        """Build a nonradial bound orbit from its Cartesian epoch state."""
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
        """Recover position and inverse-ball velocity at the stored epoch."""
        r, v = orbit.to_cartesian()
        escape = jnp.sqrt(2 * orbit.mu_AU3_per_d2 / jnp.linalg.norm(r, axis=-1))
        return jnp.concatenate(
            [r / self.scale_AU, disk_to_plane(v / escape[..., None])], -1
        )

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
    coordinate_names: ClassVar[tuple[str, ...]] = (
        "ln_a",
        "logit_e",
        "atanh_cos_i",
        "W_rad",
        "w_rad",
        "M0_rad",
    )

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

    Static branch selects the sign of the largest-magnitude away-row entry
    (first wins ties), as in sky_block_to_frame. cos(i)=(A*G-B*F)/a^2 is
    already determined by the sky block; branch selects the remaining
    line-of-sight reflection, not prograde versus retrograde. On each
    declared branch this is one-to-one onto bound orbit states, modulo
    quaternion sign, with zero sky block excluded. An orbit on the other
    branch is reflected on round trip. The branch has a pivot seam and
    both sheets meet at face-on. Values remain finite there; derivatives
    of the inverse and the volume density are singular.

    Reference measure is the other charts' unnormalized Haar*dex*dey*
    d(ln a), extended by d(ln mu). Write c=cos(i), l=ln(a), and u0 for
    the angle locating f from the node. The volume derivation is:
    dnu = dex dey dl dc dW du0 d(ln mu).
    |d(A,B,F,G)/d(l,c,W,u0)| = a^4*(1-c^2).
    d(ln mu) = 2*d(ln n)+3*dl, giving a factor 2 in the wedge product.
    |d(ex,ey)/d(ux,uy)| = (1+ux^2+uy^2)^(-2).
    log_volume = log(2)-4*log(a)-log(1-c^2)-2*log(1+ux^2+uy^2).
    No additive constant is omitted relative to ClassicalChart; normalized
    Haar would subtract log(8*pi^2). At face-on log_volume=+inf, an
    integrable coordinate singularity, not a finite density approximation.
    """

    n_coords: ClassVar[int] = 7
    coordinate_names: ClassVar[tuple[str, ...]] = (
        "ln_n",
        "ux",
        "uy",
        "A",
        "B",
        "F",
        "G",
    )
    branch: int = eqx.field(static=True, default=1)

    def __check_init__(self):
        """Validate the static choice of line-of-sight sheet."""
        if self.branch not in (-1, 1):
            raise ValueError("branch must be +1 or -1")

    def to_orbit(self, x, *, mu_AU3_per_d2, epoch_d=0.0) -> FrameOrbit:
        """Build an epoch frame, ignoring mu in favor of the coordinate rate."""
        x = _coordinates(x, self.n_coords)
        a, quat = sky_block_to_frame(
            *jnp.moveaxis(x[..., 3:], -1, 0), branch=self.branch
        )
        evec = plane_to_disk(x[..., 1:3])
        a, n, ex, ey, epoch = jnp.broadcast_arrays(
            a, jnp.exp(x[..., 0]), evec[..., 0], evec[..., 1], jnp.asarray(epoch_d)
        )
        return FrameOrbit(a, n, ex, ey, jnp.broadcast_to(quat, (*a.shape, 4)), epoch)

    def from_orbit(self, orbit: FrameOrbit) -> Array:
        """Project an orbit; inversion reproduces its declared branch only."""
        return jnp.concatenate(
            [
                jnp.log(orbit.n_rad_per_d)[..., None],
                disk_to_plane(jnp.stack([orbit.ex, orbit.ey], -1)),
                frame_to_sky_block(orbit.a_AU, orbit.quat),
            ],
            -1,
        )

    def implied_mu(self, x) -> Array:
        """Return n^2*a^3 in AU^3/day^2 with the coordinate batch shape."""
        x = _coordinates(x, self.n_coords)
        a, _ = sky_block_geometry(*jnp.moveaxis(x[..., 3:], -1, 0))
        return jnp.exp(2 * x[..., 0]) * a**3

    def log_volume(self, x) -> Array:
        """Return the density against Haar*dex*dey*d(ln a)*d(ln mu)."""
        x = _coordinates(x, self.n_coords)
        a, gap = sky_block_geometry(*jnp.moveaxis(x[..., 3:], -1, 0))
        # gap=a^2*(1-c^2), computed without cancellation near face-on.
        return (
            jnp.log(2.0) - 2 * jnp.log(a) - jnp.log(gap) + _disk_log_volume(x[..., 1:3])
        )
