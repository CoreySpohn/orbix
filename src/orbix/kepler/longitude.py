"""Non-singular elliptic Kepler equation with interchangeable value solvers.

The eccentric longitude formulation follows Murray and Dermott, Solar System
Dynamics. Its implicit derivatives remain regular at zero eccentricity.
"""

from abc import abstractmethod
from functools import cache

import equinox as eqx
import jax
import jax.numpy as jnp

from orbix.kepler.markley import _solve_trig_value
from orbix.kepler.shortcuts.grid import _build_trig_grid, _indices_frac


class AbstractKeplerSolver(eqx.Module):
    """Value-only strategy for elliptic Kepler's equation, M = E - e*sin(E).

    Inputs broadcast elementwise, with 0 <= e < 1 and any real M in radians.
    Outputs have the floating input result dtype. Strategies use only static
    fields so equal configurations share compiled traces.
    """

    @abstractmethod
    def solve(self, mean_anomaly, e):
        """Return (sin(E), cos(E)) without an attached derivative rule."""
        raise NotImplementedError


class Markley(AbstractKeplerSolver):
    """Cubic starter and fifth-order correction of Markley (1995)."""

    def solve(self, mean_anomaly, e):
        """Evaluate the table-free Markley algorithm elementwise."""
        return _solve_trig_value(mean_anomaly, e)[1:]


class Halley(AbstractKeplerSolver):
    """Fixed Halley iterations from the M + 0.85*e*sign(sin(M)) starter.

    See Danby, Fundamentals of Celestial Mechanics, for iterative elliptic
    Kepler solvers. Six steps reach roundoff for e <= 0.95; more steps may be
    needed very close to e = 1 and periapsis.
    """

    steps: int = eqx.field(static=True, default=6)

    def solve(self, mean_anomaly, e):
        """Apply a fixed number of branch-free Halley updates."""
        dtype = jnp.result_type(mean_anomaly, e, 0.0)
        mean, e = jnp.broadcast_arrays(
            jnp.asarray(mean_anomaly, dtype=dtype), jnp.asarray(e, dtype=dtype)
        )
        mean = jnp.arctan2(jnp.sin(mean), jnp.cos(mean))
        initial = mean + 0.85 * e * jnp.sign(mean)

        def step(_, eccentric):
            sine, cosine = jnp.sin(eccentric), jnp.cos(eccentric)
            residual = eccentric - e * sine - mean
            slope = 1 - e * cosine
            return eccentric - residual / (slope - 0.5 * residual * e * sine / slope)

        eccentric = jax.lax.fori_loop(0, self.steps, step, initial)
        return jnp.sin(eccentric), jnp.cos(eccentric)


class Grid(AbstractKeplerSolver):
    """Bilinear trig lookup, with one table per resolution and default dtype.

    Interpolation error grows near e = 1. The returned pair need not have
    exactly unit norm; derivatives of the longitude function use the implicit
    equation with this approximate value, not derivatives of the interpolant.
    """

    n_e: int = eqx.field(static=True, default=512)
    n_M: int = eqx.field(static=True, default=2048)

    def solve(self, mean_anomaly, e):
        """Gather four table corners for broadcast inputs without internal vmap."""
        dtype = jnp.result_type(mean_anomaly, e, 0.0)
        mean, e = jnp.broadcast_arrays(
            jnp.asarray(mean_anomaly, dtype=dtype), jnp.asarray(e, dtype=dtype)
        )
        table, inv_dM, inv_de = _build_trig_grid(
            self.n_e, self.n_M, dtype=jnp.asarray(0.0).dtype
        )
        e0, m0, de, dm = _indices_frac(mean, e, inv_dM, inv_de, self.n_M, self.n_e)
        pair = (
            table[:, e0, m0] * ((1 - de) * (1 - dm))
            + table[:, e0, m0 + 1] * ((1 - de) * dm)
            + table[:, e0 + 1, m0] * (de * (1 - dm))
            + table[:, e0 + 1, m0 + 1] * (de * dm)
        ).astype(dtype)
        return pair[0], pair[1]


@cache
def _longitude_function(solver):
    """Close over a static strategy and reuse its custom-JVP function."""

    @jax.custom_jvp
    def solve(mean_longitude, ex, ey):
        dtype = jnp.result_type(mean_longitude, ex, ey, 0.0)
        lam, ex, ey = (
            jnp.asarray(value, dtype=dtype) for value in (mean_longitude, ex, ey)
        )
        e = jnp.sqrt(ex * ex + ey * ey)
        circular = e == 0
        divisor = jnp.where(circular, 1.0, e)
        c = jnp.where(circular, 1.0, ex / divisor)
        s = jnp.where(circular, 0.0, ey / divisor)
        sin_lam, cos_lam = jnp.sin(lam), jnp.cos(lam)
        mean = jnp.arctan2(sin_lam * c - cos_lam * s, cos_lam * c + sin_lam * s)
        sin_e, cos_e = solver.solve(mean, e)
        # The exact circular value also avoids interpolation error at e = 0.
        return (
            jnp.where(circular, sin_lam, sin_e * c + cos_e * s),
            jnp.where(circular, cos_lam, cos_e * c - sin_e * s),
        )

    @solve.defjvp
    def solve_jvp(primals, tangents):
        _, ex, ey = primals
        dlam, dex, dey = (
            0.0 if getattr(tangent, "dtype", None) == jax.dtypes.float0 else tangent
            for tangent in tangents
        )
        sine, cosine = solve(*primals)
        dF = (dlam + sine * dex - cosine * dey) / (1 - ex * cosine - ey * sine)
        return (sine, cosine), (cosine * dF, -sine * dF)

    return solve


_DEFAULT_SOLVER = Markley()


def kepler(mean_longitude, ex, ey, *, solver=_DEFAULT_SOLVER):
    """Return (sin(F), cos(F)) solving lam = F - ex*sin(F) + ey*cos(F).

    Inputs broadcast to any shape, with ex**2 + ey**2 < 1 and radians for
    mean_longitude. The output dtype is their floating result dtype. Here
    (ex, ey) = e*(cos(p), sin(p)), where p is longitude of periapsis.

    A custom JVP differentiates the implicit equation, including at circular
    orbits and to higher orders. The value-only solver is closed over and is
    never differentiated. Equal solver instances reuse the same function.
    """
    return _longitude_function(solver)(mean_longitude, ex, ey)
