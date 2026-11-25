"""Non-singular bound-orbit state; Broucke and Cefola (1972), Danby (1988)."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from orbix.equations.orbit import mean_motion
from orbix.kepler import Markley, kepler
from orbix.system.observables import offset_arcsec
from orbix.system.rotation import from_basis, rotate

_DEFAULT_SOLVER = Markley()


class FrameOrbit(eqx.Module):
    """Elliptic kinematics in AU and days, with one free batch shape B.

    All scalar fields have shape B; quat has shape B + (4,). The unit
    quaternion is scalar-first (w, x, y, z), rotating the right-handed
    (f, g, angular-momentum) frame into (east, north, away). f is the
    mean-longitude direction at epoch_d, so mean longitude then is zero.
    ex and ey are eccentricity-vector components along f and g.

    Callers supply a_AU > 0, n_rad_per_d > 0, ex**2 + ey**2 < 1 and a
    unit quaternion. Construction validates shapes only, including under JIT.

    Attributes:
        a_AU: Semi-major axis in AU, shape B.
        n_rad_per_d: Mean motion in radians/day, shape B.
        ex: Eccentricity-vector component along the f axis, dimensionless, shape B.
        ey: Eccentricity-vector component along the g axis, dimensionless, shape B.
        quat: Scalar-first unit quaternion (w, x, y, z), shape B + (4,). It rotates
            the plane axes (1, 0, 0), (0, 1, 0), (0, 0, 1) onto the f axis, the g
            axis and the angular-momentum direction, expressed in (east, north,
            away).
        epoch_d: Epoch in days at which the mean longitude is zero, shape B.
    """

    a_AU: Array = eqx.field(converter=jnp.asarray)
    n_rad_per_d: Array = eqx.field(converter=jnp.asarray)
    ex: Array = eqx.field(converter=jnp.asarray)
    ey: Array = eqx.field(converter=jnp.asarray)
    quat: Array = eqx.field(converter=jnp.asarray)
    epoch_d: Array = eqx.field(converter=jnp.asarray)

    def __check_init__(self):
        """Validate only static shapes, never traced numerical values."""
        shape = self.a_AU.shape
        if any(
            x.shape != shape for x in (self.n_rad_per_d, self.ex, self.ey, self.epoch_d)
        ):
            raise ValueError("Scalar orbit fields must share one batch shape")
        if self.quat.shape != (*shape, 4):
            raise ValueError("quat must have batch shape + (4,)")

    @classmethod
    def from_classical(
        cls, a_AU, e, i_rad, W_rad, w_rad, M0_rad, *, mu_AU3_per_d2, epoch_d=0.0
    ):
        """Build from broadcast classical elements in the library sky convention.

        W_rad is measured north through east; the ascending node moves away.
        The first two columns of Rz(W) Rx(i) Rz(w+M0) have east/north swapped;
        their cross product supplies the proper right-handed third column.

        Args:
            a_AU: Semi-major axis in AU.
            e: Eccentricity, dimensionless.
            i_rad: Inclination in radians.
            W_rad: Longitude of the ascending node in radians, measured north through
                east.
            w_rad: Argument of periapsis in radians.
            M0_rad: Mean anomaly at epoch_d in radians.
            mu_AU3_per_d2: Gravitational parameter in AU^3/day^2, broadcast with the
                other inputs.
            epoch_d: Epoch of M0_rad in days. Defaults to 0.

        Returns:
            FrameOrbit with the batch shape of the broadcast inputs.
        """
        inc, node, peri, mean = map(jnp.asarray, (i_rad, W_rad, w_rad, M0_rad))
        return cls.from_classical_trig(
            a_AU=a_AU,
            e=e,
            cos_i=jnp.cos(inc),
            sin_i=jnp.sin(inc),
            cos_W=jnp.cos(node),
            sin_W=jnp.sin(node),
            cos_w=jnp.cos(peri),
            sin_w=jnp.sin(peri),
            cos_M0=jnp.cos(mean),
            sin_M0=jnp.sin(mean),
            mu_AU3_per_d2=mu_AU3_per_d2,
            epoch_d=epoch_d,
        )

    @classmethod
    def from_classical_trig(
        cls,
        *,
        a_AU,
        e,
        cos_i,
        sin_i,
        cos_W,
        sin_W,
        cos_w,
        sin_w,
        cos_M0,
        sin_M0,
        mu_AU3_per_d2,
        epoch_d=0.0,
    ):
        """Build from broadcast classical sine/cosine pairs without angle inversion.

        Each (cos, sin) pair must lie on the unit circle, with sin_i >= 0.
        Inputs are used as given, without renormalization or value validation.
        The sky convention is that of from_classical: W is measured north
        through east and the ascending node moves away.

        Products form cos(w+M0) and sin(w+M0) directly; the stored frame absorbs
        the epoch mean longitude, so no inverse trigonometric function is
        needed. Independent trig inputs keep gradients finite at face-on
        inclination, whereas the derivative of arccos at 1 is infinite.
        For a time of periapsis tp_d, pass cos_M0=1, sin_M0=0, epoch_d=tp_d.

        Args:
            a_AU: Semi-major axis in AU.
            e: Eccentricity, dimensionless.
            cos_i: Cosine of the inclination.
            sin_i: Sine of the inclination, at least zero.
            cos_W: Cosine of the longitude of the ascending node.
            sin_W: Sine of the longitude of the ascending node.
            cos_w: Cosine of the argument of periapsis.
            sin_w: Sine of the argument of periapsis.
            cos_M0: Cosine of the mean anomaly at epoch_d.
            sin_M0: Sine of the mean anomaly at epoch_d.
            mu_AU3_per_d2: Gravitational parameter in AU^3/day^2, broadcast with the
                other inputs.
            epoch_d: Epoch of the mean anomaly in days. Defaults to 0.

        Returns:
            FrameOrbit with the batch shape of the broadcast inputs.
        """
        a, e, ci, si, cO, sO, cw, sw, cm, sm, mu, epoch = jnp.broadcast_arrays(
            *map(
                jnp.asarray,
                (
                    a_AU,
                    e,
                    cos_i,
                    sin_i,
                    cos_W,
                    sin_W,
                    cos_w,
                    sin_w,
                    cos_M0,
                    sin_M0,
                    mu_AU3_per_d2,
                    epoch_d,
                ),
            )
        )
        cu, su = cw * cm - sw * sm, sw * cm + cw * sm
        f = jnp.stack([sO * cu + cO * ci * su, cO * cu - sO * ci * su, si * su], -1)
        g = jnp.stack([-sO * su + cO * ci * cu, -cO * su - sO * ci * cu, si * cu], -1)
        return cls(
            a,
            mean_motion(a, mu),
            e * cm,
            -e * sm,
            from_basis(f, g, jnp.cross(f, g)),
            epoch,
        )

    @classmethod
    def from_periapsis_time(
        cls, a_AU, e, cos_i, W_rad, cos_w, sin_w, tp_d, *, mu_AU3_per_d2
    ):
        """Build from classical elements with periapsis at tp_d (days).

        Inclination enters as cos_i with sin_i = sqrt(1 - cos_i**2), and the
        argument of periapsis as (cos_w, sin_w); only the node W_rad is an
        angle. Equivalent to from_classical_trig with M0 = 0 and epoch tp_d.

        Args:
            a_AU: Semi-major axis in AU.
            e: Eccentricity, dimensionless.
            cos_i: Cosine of the inclination.
            W_rad: Longitude of the ascending node in radians, measured north through
                east.
            cos_w: Cosine of the argument of periapsis.
            sin_w: Sine of the argument of periapsis.
            tp_d: Time of periapsis in days.
            mu_AU3_per_d2: Gravitational parameter in AU^3/day^2, broadcast with the
                other inputs.

        Returns:
            FrameOrbit whose epoch is tp_d and whose mean anomaly there is zero.
        """
        cos_i = jnp.asarray(cos_i)
        return cls.from_classical_trig(
            a_AU=a_AU,
            e=e,
            cos_i=cos_i,
            sin_i=jnp.sqrt(1.0 - cos_i**2),
            cos_W=jnp.cos(W_rad),
            sin_W=jnp.sin(W_rad),
            cos_w=cos_w,
            sin_w=sin_w,
            cos_M0=1.0,
            sin_M0=0.0,
            mu_AU3_per_d2=mu_AU3_per_d2,
            epoch_d=tp_d,
        )

    @classmethod
    def from_unit_plane(cls, e, tp_d, n_rad_per_d):
        """Build a unit-size orbit for dimensionless perifocal coordinates.

        The semi-major axis is one AU and the periapsis lies on the f axis at
        tp_d, so plane_coordinates and perifocal_coordinates give the orbit in
        units of a. Inputs broadcast to one batch shape.

        Args:
            e: Eccentricity, dimensionless.
            tp_d: Time of periapsis in days.
            n_rad_per_d: Mean motion in radians/day.

        Returns:
            FrameOrbit with a semi-major axis of 1 AU, periapsis on the f axis, and the
            broadcast batch shape of the inputs.
        """
        e, tp, n = jnp.broadcast_arrays(e, tp_d, n_rad_per_d)
        return cls(
            jnp.ones_like(e),
            n,
            e,
            jnp.zeros_like(e),
            jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0, 0.0]), (*e.shape, 4)),
            tp,
        )

    @classmethod
    def from_unit_plane_phase(cls, e, ln_period_d, tau, t_ref_d):
        """Build a from_unit_plane orbit with periapsis at t_ref_d + tau periods.

        Args:
            e: Eccentricity, dimensionless.
            ln_period_d: Natural logarithm of the period in days.
            tau: Periapsis offset from t_ref_d in periods.
            t_ref_d: Reference time in days.

        Returns:
            FrameOrbit built by from_unit_plane with periapsis at t_ref_d + tau periods.
        """
        period = jnp.exp(ln_period_d)
        return cls.from_unit_plane(e, t_ref_d + period * tau, 2 * jnp.pi / period)

    def to_classical(self):
        """Return a_AU, e, i_rad, W_rad, w_rad, M0_rad as a dictionary.

        This human-readable view is undefined at e=0 or i=0, pi. It is
        not the differentiable interchange path; use to_cartesian instead.
        Angles are returned modulo 2*pi, except inclination in [0, pi].

        Returns:
            Dictionary with a_AU (AU), e, i_rad, W_rad, w_rad and M0_rad (radians), each
            of the orbit's batch shape.
        """
        f = rotate(self.quat, jnp.array([1.0, 0.0, 0.0]))
        normal = rotate(self.quat, jnp.array([0.0, 0.0, 1.0]))
        inc = jnp.arctan2(jnp.linalg.norm(normal[..., :2], axis=-1), -normal[..., 2])
        node = jnp.arctan2(-normal[..., 1], normal[..., 0])
        node_axis = jnp.stack([jnp.sin(node), jnp.cos(node), jnp.zeros_like(node)], -1)
        tangent = jnp.cross(normal, node_axis)
        u = jnp.arctan2(jnp.sum(f * tangent, -1), jnp.sum(f * node_axis, -1))
        mean = jnp.arctan2(-self.ey, self.ex)
        return dict(
            a_AU=self.a_AU,
            e=self.e,
            i_rad=inc,
            W_rad=node % (2 * jnp.pi),
            w_rad=(u - mean) % (2 * jnp.pi),
            M0_rad=mean % (2 * jnp.pi),
        )

    @classmethod
    def from_cartesian(cls, r_AU, v_AU_per_d, *, mu_AU3_per_d2, epoch_d=0.0):
        """Build from nonradial bound Cartesian states, vectors on the last axis.

        Inputs broadcast to B + (3,); mu and epoch broadcast to B. The
        radius-based temporary frame avoids eccentricity and node angles.
        The eccentric-longitude sine/cosine pair never vanishes, including
        at circular and either face-on orientation (Broucke and Cefola 1972).

        Args:
            r_AU: Position vectors in AU in the (east, north, away) frame, last axis of
                length 3.
            v_AU_per_d: Velocity vectors in AU/day in the same frame, last axis of
                length 3.
            mu_AU3_per_d2: Gravitational parameter in AU^3/day^2, broadcast with the
                other inputs.
            epoch_d: Epoch of the state in days. Defaults to 0.

        Returns:
            FrameOrbit whose state at epoch_d equals the supplied vectors.
        """
        r, v = jnp.asarray(r_AU), jnp.asarray(v_AU_per_d)
        if r.shape[-1:] != (3,) or v.shape[-1:] != (3,):
            raise ValueError("Cartesian vectors must have last axis of length 3")
        shape = jnp.broadcast_shapes(
            r.shape[:-1], v.shape[:-1], jnp.shape(mu_AU3_per_d2), jnp.shape(epoch_d)
        )
        r, v = jnp.broadcast_to(r, (*shape, 3)), jnp.broadcast_to(v, (*shape, 3))
        mu, epoch = (
            jnp.broadcast_to(jnp.asarray(mu_AU3_per_d2), shape),
            jnp.broadcast_to(jnp.asarray(epoch_d), shape),
        )
        radius = jnp.linalg.norm(r, axis=-1)
        h = jnp.cross(r, v)
        normal = h / jnp.linalg.norm(h, axis=-1, keepdims=True)
        a = 1 / (2 / radius - jnp.sum(v * v, axis=-1) / mu)
        evec = jnp.cross(v, h) / mu[..., None] - r / radius[..., None]
        u = r / radius[..., None]
        tangent = jnp.cross(normal, u)
        ex, ey = jnp.sum(evec * u, -1), jnp.sum(evec * tangent, -1)
        root = jnp.sqrt(1 - jnp.sum(evec * evec, -1))
        beta = 1 / (1 + root)
        bxx, bxy, byy = 1 - beta * ey**2, beta * ex * ey, 1 - beta * ex**2
        rhs = radius / a + ex
        cosine = (byy * rhs - bxy * ey) / root
        sine = (bxx * ey - bxy * rhs) / root
        # lambda = F - ex*sin(F) + ey*cos(F); rotate the nonvanishing pair.
        correction = -ex * sine + ey * cosine
        c, s = jnp.cos(correction), jnp.sin(correction)
        cl, sl = cosine * c - sine * s, sine * c + cosine * s
        f = u * cl[..., None] + tangent * sl[..., None]
        g = jnp.cross(normal, f)
        return cls(
            a,
            mean_motion(a, mu),
            jnp.sum(evec * f, -1),
            jnp.sum(evec * g, -1),
            from_basis(f, g, normal),
            epoch,
        )

    def to_cartesian(self):
        """Return the epoch state, each vector of shape B + (3,).

        Returns:
            Tuple of position in AU and velocity in AU/day at the epoch, each of shape B
            + (3,).
        """
        return self.at_each(self.epoch_d)

    def at(self, time_d, *, solver=_DEFAULT_SOLVER):
        """Evaluate the outer product: B + time_shape + (3,), AU and AU/day.

        Args:
            time_d: Times in days on the same axis as epoch_d.
            solver: Value-only Kepler strategy. Defaults to Markley.

        Returns:
            Tuple of position in AU and velocity in AU/day, each of shape B + time_shape
            + (3,), in the (east, north, away) frame.
        """
        xy, xy_rate = self.plane_coordinates(time_d, solver=solver)
        basis = self.plane_basis().reshape(
            (*self.a_AU.shape, *(1,) * jnp.ndim(time_d), 3, 2)
        )

        def apply(coordinates):
            first = jax.lax.optimization_barrier(basis[..., 0] * coordinates[..., :1])
            second = jax.lax.optimization_barrier(basis[..., 1] * coordinates[..., 1:])
            return first + second

        return apply(xy), apply(xy_rate)

    def offsets_arcsec(self, time_d, dist_pc, *, solver=_DEFAULT_SOLVER):
        """Return east and north offsets in arcseconds at time_d (days).

        The orbit's batch axes come first, then the time axes; dist_pc
        broadcasts against the batch axes.

        Args:
            time_d: Times in days on the same axis as epoch_d.
            dist_pc: Distance in parsecs, broadcast against the batch axes.
            solver: Value-only Kepler strategy. Defaults to Markley.

        Returns:
            Tuple of east and north offsets in arcseconds, each of shape B + time_shape.
        """
        position, _ = self.at(time_d, solver=solver)
        return offset_arcsec(position, dist_pc)

    def to_periapsis_elements(self):
        """Return a_AU, e, cos_i, W_rad, cos_w, sin_w, tp_d as a dictionary.

        The keys are the arguments of from_periapsis_time, so
        ``from_periapsis_time(**elements, mu_AU3_per_d2=mu)`` rebuilds the
        orbit. Each value has the orbit's batch shape. The time of periapsis
        is the epoch minus the epoch mean anomaly over the mean motion, so it
        lies within one period before the epoch. Classical angles are
        undefined at e=0 and cos_i=+-1; the elements are for output only.

        Returns:
            Dictionary with a_AU (AU), e, cos_i, W_rad (radians), cos_w, sin_w and tp_d
            (days), each of the orbit's batch shape.
        """
        elements = self.to_classical()
        return {
            "a_AU": elements["a_AU"],
            "e": elements["e"],
            "cos_i": jnp.cos(elements["i_rad"]),
            "W_rad": elements["W_rad"],
            "cos_w": jnp.cos(elements["w_rad"]),
            "sin_w": jnp.sin(elements["w_rad"]),
            "tp_d": self.epoch_d - elements["M0_rad"] / self.n_rad_per_d,
        }

    def plane_basis(self) -> Array:
        """Return the constant B + (3, 2) plane-to-sky matrix in AU.

        Its columns are a_AU times the f and g axes in (east, north, away).
        Multiplying by plane_coordinates gives position and velocity.
        Rows 0 and 1 (east, north) generalize the Thiele-Innes constants
        to these nonsingular axes.

        Let R(phi) = [[cos(phi), -sin(phi)], [sin(phi), cos(phi)]] and
        phi = atan2(ey, ex), the argument of periapsis measured from f.
        Where periapsis is defined, plane_basis()[..., :2, :] @ R(phi)
        equals [[B, G], [A, F]] for the classical Thiele-Innes constants.
        The classical argument w is measured from the ascending node:
        phi = w - u, where u locates f from that node. For from_classical,
        u = w + M0, so phi = -M0. W is north through east and the ascending
        node moves away, as in from_classical.

        Returns:
            Plane-to-sky matrix of shape B + (3, 2) in AU, with rows (east, north, away)
            and columns the f and g axes.
        """
        f = rotate(self.quat, jnp.array([1.0, 0.0, 0.0], dtype=self.quat.dtype))
        g = rotate(self.quat, jnp.array([0.0, 1.0, 0.0], dtype=self.quat.dtype))
        return self.a_AU[..., None, None] * jnp.stack([f, g], -1)

    @property
    def linear_map(self) -> Array:
        """Return a [f g], shape B + (3, 2), in (east, north, away) AU.

        This is plane_basis() unchanged. The stored f axis points along
        epoch mean longitude; g completes the right-handed orbital frame.
        These axes remain defined at e=0 and face-on. Applied to in_plane
        position/rate, this matrix gives at position/velocity.

        For e>0, let phi=atan2(ey,ex) and
        R(phi)=[[cos(phi),-sin(phi)],[sin(phi),cos(phi)]]. Then
        linear_map[..., :2, :] @ R(phi) = [[B,G],[A,F]], where A,F are
        north and B,G east, in the classical perifocal Thiele-Innes
        convention. For from_classical, phi=-M0. No such rotation is
        needed for propagation, including at circular eccentricity.
        """
        return self.plane_basis()

    def in_plane(self, time_d, *, solver=_DEFAULT_SOLVER) -> tuple[Array, Array]:
        """Return dimensionless (X,Y) and rates per day on stored f,g axes.

        Both arrays have shape B + time_shape + (2,), exactly as in at.
        Reuses plane_coordinates, which already has the required units:
        no division by a and no additional Kepler solve is performed.
        See linear_map for the circular-safe axes and perifocal rotation.

        Args:
            time_d: Times in days on the same axis as epoch_d.
            solver: Value-only Kepler strategy. Defaults to Markley.

        Returns:
            Tuple of dimensionless plane position and its rate in 1/day, each of shape B
            + time_shape + (2,).
        """
        return self.plane_coordinates(time_d, solver=solver)

    def in_plane_each(self, time_d, *, solver=_DEFAULT_SOLVER) -> tuple[Array, Array]:
        """Return paired plane position/rate, dimensionless and per day.

        Both arrays have shape broadcast(B,time_shape) + (2,), as in
        at_each. Reuses that method's single eccentric-longitude solve
        and stored f,g axes; see linear_map for the axis convention.

        Args:
            time_d: Times in days, broadcast against the batch shape B.
            solver: Value-only Kepler strategy. Defaults to Markley.

        Returns:
            Tuple of dimensionless plane position and its rate in 1/day, each of shape
            broadcast(B, time_shape) + (2,).
        """
        return self._plane_coordinates_each(time_d, solver=solver)

    def plane_coordinates(
        self, time_d, *, solver=_DEFAULT_SOLVER
    ) -> tuple[Array, Array]:
        """Return dimensionless plane position and its rate in 1/day.

        Both arrays have shape B + time_shape + (2,), with the same outer
        time broadcasting as at. Coordinates use the nonsingular f and g
        axes of ex and ey, in units of the semi-major axis. Position and
        rate are finite and smooth at e=0; one Kepler solve supplies both.

        Args:
            time_d: Times in days on the same axis as epoch_d.
            solver: Value-only Kepler strategy. Defaults to Markley.

        Returns:
            Tuple of dimensionless plane position, in units of the semi-major axis, and
            its rate in 1/day, each of shape B + time_shape + (2,).
        """
        time = jnp.asarray(time_d)
        shape = self.a_AU.shape + (1,) * time.ndim
        expanded = FrameOrbit(
            self.a_AU.reshape(shape),
            self.n_rad_per_d.reshape(shape),
            self.ex.reshape(shape),
            self.ey.reshape(shape),
            self.quat.reshape((*shape, 4)),
            self.epoch_d.reshape(shape),
        )
        return expanded._plane_coordinates_each(time, solver=solver)

    def perifocal_coordinates(
        self, time_d, *, solver=_DEFAULT_SOLVER
    ) -> tuple[Array, Array]:
        """Return dimensionless perifocal position and its rate in 1/day.

        Position is (cos(E)-e, sqrt(1-e**2)*sin(E)); both arrays have shape
        B + time_shape + (2,), with the same outer time broadcasting as at.

        Rotate plane position and rate into the classical periapsis axes
        without another Kepler solve. These axes are undefined at e=0:
        there both results use f and g unchanged. Guarding the angle's inputs
        makes both values and autodiff gradients finite at zero; this arbitrary
        fallback does not make the periapsis view continuous there.

        Args:
            time_d: Times in days on the same axis as epoch_d.
            solver: Value-only Kepler strategy. Defaults to Markley.

        Returns:
            Tuple of dimensionless perifocal position, in units of the semi-major axis,
            and its rate in 1/day, each of shape B + time_shape + (2,).
        """
        xy, xy_rate = self.plane_coordinates(time_d, solver=solver)
        nonzero = (self.ex != 0) | (self.ey != 0)
        phi = jnp.arctan2(
            jnp.where(nonzero, self.ey, 0.0), jnp.where(nonzero, self.ex, 1.0)
        )
        shape = self.a_AU.shape + (1,) * jnp.ndim(time_d)
        c, s = jnp.cos(phi).reshape(shape), jnp.sin(phi).reshape(shape)
        return tuple(
            jnp.stack(
                [
                    c * value[..., 0] + s * value[..., 1],
                    -s * value[..., 0] + c * value[..., 1],
                ],
                -1,
            )
            for value in (xy, xy_rate)
        )

    def at_each(self, time_d, *, solver=_DEFAULT_SOLVER):
        """Evaluate paired times: broadcast(B, time_shape) + (3,), AU and AU/day.

        One eccentric-longitude solve supplies both position and analytic
        velocity; there is no trigonometry outside that solve.

        Args:
            time_d: Times in days, broadcast against the batch shape B.
            solver: Value-only Kepler strategy. Defaults to Markley.

        Returns:
            Tuple of position in AU and velocity in AU/day, each of shape broadcast(B,
            time_shape) + (3,), in the (east, north, away) frame.
        """
        xy, xy_rate = self._plane_coordinates_each(time_d, solver=solver)
        basis = self.plane_basis()

        def apply(coordinates):
            first = jax.lax.optimization_barrier(basis[..., 0] * coordinates[..., :1])
            second = jax.lax.optimization_barrier(basis[..., 1] * coordinates[..., 1:])
            return first + second

        return apply(xy), apply(xy_rate)

    def _plane_coordinates_each(self, time_d, *, solver):
        """Evaluate dimensionless plane position and rate at paired times."""
        sine, cosine = kepler(
            self.n_rad_per_d * (jnp.asarray(time_d) - self.epoch_d),
            self.ex,
            self.ey,
            solver=solver,
        )
        beta = 1 / (1 + jnp.sqrt(1 - self.ex**2 - self.ey**2))
        bxx, bxy, byy = (
            1 - beta * self.ey**2,
            beta * self.ex * self.ey,
            1 - beta * self.ex**2,
        )
        x = bxx * cosine + bxy * sine - self.ex
        y = bxy * cosine + byy * sine - self.ey
        rate = self.n_rad_per_d / (1 - self.ex * cosine - self.ey * sine)
        vx = rate * (-bxx * sine + bxy * cosine)
        vy = rate * (-bxy * sine + byy * cosine)
        return jnp.stack([x, y], -1), jnp.stack([vx, vy], -1)

    @property
    def e(self):
        """Eccentricity (its scalar magnitude is not differentiable at zero)."""
        return jnp.hypot(self.ex, self.ey)

    @property
    def period_d(self):
        """Period in days."""
        return 2 * jnp.pi / self.n_rad_per_d

    @property
    def mu_AU3_per_d2(self):
        """Gravitational parameter implied by the stored size and rate."""
        return self.n_rad_per_d**2 * self.a_AU**3
