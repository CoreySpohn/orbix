"""Non-singular bound-orbit state; Broucke and Cefola (1972), Danby (1988)."""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from orbix.equations.orbit import mean_motion
from orbix.kepler import Markley, kepler
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
        """
        a, e, inc, node, peri, mean, mu, epoch = jnp.broadcast_arrays(
            *map(
                jnp.asarray,
                (a_AU, e, i_rad, W_rad, w_rad, M0_rad, mu_AU3_per_d2, epoch_d),
            )
        )
        cO, sO = jnp.cos(node), jnp.sin(node)
        ci, si = jnp.cos(inc), jnp.sin(inc)
        cu, su = jnp.cos(peri + mean), jnp.sin(peri + mean)
        f = jnp.stack([sO * cu + cO * ci * su, cO * cu - sO * ci * su, si * su], -1)
        g = jnp.stack([-sO * su + cO * ci * cu, -cO * su - sO * ci * cu, si * cu], -1)
        return cls(
            a,
            mean_motion(a, mu),
            e * jnp.cos(mean),
            -e * jnp.sin(mean),
            from_basis(f, g, jnp.cross(f, g)),
            epoch,
        )

    def to_classical(self):
        """Return a_AU, e, i_rad, W_rad, w_rad, M0_rad as a dictionary.

        This human-readable view is undefined at e=0 or i=0, pi. It is
        not the differentiable interchange path; use to_cartesian instead.
        Angles are returned modulo 2*pi, except inclination in [0, pi].
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
        """Return the epoch state, each vector of shape B + (3,)."""
        return self.at_each(self.epoch_d)

    def at(self, time_d, *, solver=_DEFAULT_SOLVER):
        """Evaluate the outer product: B + time_shape + (3,), AU and AU/day."""
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
        return expanded.at_each(time, solver=solver)

    def at_each(self, time_d, *, solver=_DEFAULT_SOLVER):
        """Evaluate paired times: broadcast(B, time_shape) + (3,), AU and AU/day.

        One eccentric-longitude solve supplies both position and analytic
        velocity; there is no trigonometry outside that solve.
        """
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
        x = self.a_AU * (bxx * cosine + bxy * sine - self.ex)
        y = self.a_AU * (bxy * cosine + byy * sine - self.ey)
        rate = self.n_rad_per_d / (1 - self.ex * cosine - self.ey * sine)
        vx = self.a_AU * rate * (-bxx * sine + bxy * cosine)
        vy = self.a_AU * rate * (-bxy * sine + byy * cosine)
        zero = jnp.zeros_like(x)
        return (
            rotate(self.quat, jnp.stack([x, y, zero], -1)),
            rotate(self.quat, jnp.stack([vx, vy, zero], -1)),
        )

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
