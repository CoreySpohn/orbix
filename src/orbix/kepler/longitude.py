"""Non-singular elliptic Kepler equation with interchangeable value solvers.

The eccentric longitude formulation follows Murray and Dermott, Solar System
Dynamics. Its implicit derivatives remain regular at zero eccentricity.
"""

from abc import abstractmethod
from functools import cache

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.precision import like

from orbix.kepler._grid import _build_trig_grid, _indices_frac
from orbix.kepler._markley import _markley_pair, _markley_reduced, _principal_angle


def _condition_host_phase(mean, *components):
    """Reduce a wide host phase to a signed angle before it is narrowed."""
    if isinstance(mean, (int, float, np.integer, np.floating, np.ndarray)):
        dtype = jnp.result_type(mean, *components, 0.0)
        if (
            np.issubdtype(dtype, np.floating)
            and np.asarray(mean).dtype.itemsize > dtype.itemsize
        ):
            phase = np.asarray(mean, dtype=np.float64)
            # libm reduces against the mathematical period, including its tail.
            with np.errstate(invalid="ignore"):
                reduced = np.arctan2(np.sin(phase), np.cos(phase))
            mean = float(reduced) if type(mean) is float else reduced.astype(dtype)
    return mean


class AbstractKeplerSolver(eqx.Module):
    """Value-only strategy for elliptic Kepler's equation, M = E - e*sin(E).

    Inputs broadcast elementwise, with 0 <= e < 1 and any real M in radians.
    Outputs have the floating input result dtype. Strategies use only static
    fields so equal configurations share compiled traces.
    """

    @abstractmethod
    def solve(self, mean_anomaly, e):
        """Return (sin(E), cos(E)) without an attached derivative rule.

        Args:
            mean_anomaly: Mean anomaly M in radians, any real value.
            e: Eccentricity, 0 <= e < 1. Broadcasts with mean_anomaly.

        Returns:
            Tuple (sin(E), cos(E)) of the eccentric anomaly E, each with the broadcast
            shape of the inputs.
        """
        raise NotImplementedError


class Markley(AbstractKeplerSolver):
    """Cubic starter and fifth-order correction of Markley (1995)."""

    def solve(self, mean_anomaly, e):
        """Evaluate the table-free Markley algorithm elementwise.

        Args:
            mean_anomaly: Mean anomaly M in radians, any real value.
            e: Eccentricity, 0 <= e < 1. Broadcasts with mean_anomaly.

        Returns:
            Tuple (sin(E), cos(E)) of the eccentric anomaly E, each with the broadcast
            shape of the inputs.
        """
        return _markley_pair(_condition_host_phase(mean_anomaly, e), e)


class Halley(AbstractKeplerSolver):
    """Fixed Halley iterations from the M + 0.85*e*sign(sin(M)) starter.

    See Danby, Fundamentals of Celestial Mechanics, for iterative elliptic
    Kepler solvers. Six steps reach roundoff for e <= 0.95; more steps may be
    needed very close to e = 1 and periapsis.

    Attributes:
        steps: Number of Halley updates applied; static under JIT.
    """

    steps: int = eqx.field(static=True, default=6)

    def solve(self, mean_anomaly, e):
        """Apply a fixed number of branch-free Halley updates.

        Args:
            mean_anomaly: Mean anomaly M in radians, any real value.
            e: Eccentricity, 0 <= e < 1. Broadcasts with mean_anomaly.

        Returns:
            Tuple (sin(E), cos(E)) of the eccentric anomaly E, each with the broadcast
            shape of the inputs.
        """
        mean_anomaly = _condition_host_phase(mean_anomaly, e)
        dtype = jnp.result_type(mean_anomaly, e, 0.0)
        mean, e = jnp.broadcast_arrays(
            jnp.asarray(mean_anomaly, dtype=dtype), jnp.asarray(e, dtype=dtype)
        )
        mean = _principal_angle(jnp.sin(mean), jnp.cos(mean))
        initial = mean + 0.85 * e * jnp.sign(mean)

        def step(_, eccentric):
            sine, cosine = jnp.sin(eccentric), jnp.cos(eccentric)
            residual = eccentric - e * sine - mean
            slope = 1 - e * cosine
            return eccentric - residual / (slope - 0.5 * residual * e * sine / slope)

        eccentric = jax.lax.fori_loop(0, self.steps, step, initial)
        return jnp.sin(eccentric), jnp.cos(eccentric)


class Grid(AbstractKeplerSolver):
    """Bilinear trig lookup, with one table per resolution and input dtype.

    Interpolation error grows near e = 1. The returned pair need not have
    exactly unit norm; derivatives of the longitude function use the implicit
    equation with this approximate value, not derivatives of the interpolant.

    Attributes:
        n_e: Number of eccentricity nodes in the lookup table; static under JIT.
        n_M: Number of mean-anomaly nodes in the lookup table; static under JIT.
    """

    n_e: int = eqx.field(static=True, default=512)
    n_M: int = eqx.field(static=True, default=2048)

    def solve(self, mean_anomaly, e):
        """Gather four table corners for broadcast inputs without internal vmap.

        Args:
            mean_anomaly: Mean anomaly M in radians, any real value.
            e: Eccentricity, 0 <= e < 1. Broadcasts with mean_anomaly.

        Returns:
            Tuple (sin(E), cos(E)) of the eccentric anomaly E, each with the broadcast
            shape of the inputs. The pair is interpolated and need not have unit norm.
        """
        mean_anomaly = _condition_host_phase(mean_anomaly, e)
        dtype = jnp.result_type(mean_anomaly, e, 0.0)
        mean, e = jnp.broadcast_arrays(
            jnp.asarray(mean_anomaly, dtype=dtype), jnp.asarray(e, dtype=dtype)
        )
        return self._solve_reduced(_principal_angle(jnp.sin(mean), jnp.cos(mean)), e)

    def _solve_reduced(self, signed_mean, e):
        """Interpolate the positive phase and restore the odd sine's sign."""
        dtype = signed_mean.dtype
        table, inv_dM, inv_de = _build_trig_grid(self.n_e, self.n_M, dtype=dtype)
        e0, m0, de, dm = _indices_frac(signed_mean, e, inv_dM, inv_de)
        pair = (
            table[:, e0, m0] * ((1 - de) * (1 - dm))
            + table[:, e0, m0 + 1] * ((1 - de) * dm)
            + table[:, e0 + 1, m0] * (de * (1 - dm))
            + table[:, e0 + 1, m0 + 1] * (de * dm)
        ).astype(dtype)
        return jnp.where(signed_mean < 0, -pair[0], pair[0]), pair[1]


@cache
def _longitude_function(solver):
    """Close over a static strategy and reuse its custom-JVP function."""
    # The rotation already supplies a signed principal angle.
    if type(solver) is Markley:
        solve_anomaly = _markley_reduced
    elif type(solver) is Grid:
        solve_anomaly = solver._solve_reduced
    else:
        solve_anomaly = solver.solve

    @jax.custom_jvp
    def solve(lam, ex, ey):
        e = jnp.sqrt(ex * ex + ey * ey)
        circular = e == 0
        one = like(1.0, e)
        divisor = jnp.where(circular, one, e)
        c = jnp.where(circular, one, ex / divisor)
        s = jnp.where(circular, like(0.0, e), ey / divisor)
        sin_lam, cos_lam = jnp.sin(lam), jnp.cos(lam)
        mean = _principal_angle(sin_lam * c - cos_lam * s, cos_lam * c + sin_lam * s)
        sin_e, cos_e = solve_anomaly(mean, e)
        # The exact circular value also avoids interpolation error at e = 0.
        sine = jnp.where(circular, sin_lam, sin_e * c + cos_e * s)
        cosine = jnp.where(circular, cos_lam, cos_e * c - sin_e * s)
        # A CPU sin can copy subnormal bits even when arithmetic reads zero.
        signed_zero = jnp.copysign(jnp.zeros_like(sine), sine)
        return jnp.where(sine == 0, signed_zero, sine), cosine

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
    mean_longitude. Host longitudes retain their signed offset from a full
    turn when narrowed. When passing already narrowed arrays or tracing this
    function, supply reduced longitudes if the original phase needs higher
    precision than the array can represent. The output has the floating result
    dtype of the inputs. Here (ex, ey) = e*(cos(p), sin(p)), where p is
    longitude of periapsis.

    A custom JVP differentiates the implicit equation, including at circular
    orbits and to higher orders. The value-only solver is closed over and is
    never differentiated. Equal solver instances reuse the same function.

    Args:
        mean_longitude: Mean longitude lam in radians.
        ex: Eccentricity-vector component e*cos(p), dimensionless.
        ey: Eccentricity-vector component e*sin(p), dimensionless.
        solver: Value-only strategy used to solve for the eccentric anomaly. Defaults to
            Markley.

    Returns:
        Tuple (sin(F), cos(F)) of the eccentric longitude F, each with the broadcast
        shape of the inputs.
    """
    mean_longitude = _condition_host_phase(mean_longitude, ex, ey)
    dtype = jnp.result_type(mean_longitude, ex, ey, 0.0)
    return _longitude_function(solver)(
        *(jnp.asarray(value, dtype=dtype) for value in (mean_longitude, ex, ey))
    )
