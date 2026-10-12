"""Table-free elliptic Kepler solver with implicit derivatives.

The cubic starter and fifth-order correction follow Markley (1995),
``Kepler Equation Solver``, Celestial Mechanics and Dynamical Astronomy 63,
101-111, https://doi.org/10.1007/BF00691917. An open NASA report is available
at https://ntrs.nasa.gov/citations/19950021346.

This implementation reuses the polynomial sine/cosine helper from the RPP
solver, evaluates the small-angle residual without subtractive cancellation,
and rotates the trig pair through the correction. There is no table setup,
interval search, or convergence loop. All entry points support broadcasting,
JIT, vmap, and forward/reverse automatic differentiation.

Inputs must satisfy 0 <= e < 1 and use radians. Enable JAX float64 for
precision-sensitive work. Phase reduction uses the floating-point value of
2*pi; phase information lost when forming or wrapping M cannot be recovered,
and its error is amplified near periapsis for almost-parabolic orbits.
"""

import jax
import jax.numpy as jnp
from hwoutils.constants import two_pi

from orbix.kepler.core import fast_sinE_cosE


def _denominator(sine, cosine, e):
    """Evaluate 1 - e*cos(E) without cancellation close to periapsis."""
    positive = cosine > 0
    divisor = jnp.where(positive, 1 + cosine, 1.0)
    one_minus_cos = jnp.where(positive, sine * sine / divisor, 1 - cosine)
    return (1 - e) + e * one_minus_cos


@jax.custom_jvp
def E_solve_trig(M, e):
    """Return (E, sin(E), cos(E)) for broadcast-compatible M and e.

    E is wrapped to [0, 2*pi), up to endpoint rounding. The trig pair retains
    small negative phases even when the wrapped E rounds to 2*pi. Derivatives
    are those of the implicit Kepler solution away from the wrap seam; at the
    seam the derivative follows the continuous, unwrapped E convention.
    """
    reduced = jnp.fmod(M, two_pi)
    absolute = jnp.abs(reduced)
    reflected = absolute > jnp.pi
    # Recover the tail of 2*pi when reflecting across the upper half orbit.
    # Without it, a 2.45e-16 phase error becomes a ~1e-6 E error near e = 1.
    # dtype.type executes at trace time and accounts for float32 rounding too.
    period_tail = (two_pi - float(reduced.dtype.type(two_pi))) + 2.4492935982947064e-16
    # Otherwise XLA reassociates the constants and rounds the tail away.
    distance = jax.lax.optimization_barrier(two_pi - absolute)
    mean = jnp.where(reflected, distance + period_tail, absolute)
    negative = (reduced < 0) ^ reflected
    ome = 1 - e

    alpha = (3 * jnp.pi + 1.6 * (jnp.pi - mean) / (1 + e)) / (jnp.pi - 6 / jnp.pi)
    d = 3 * ome + alpha * e
    q = 2 * alpha * d * ome - mean * mean
    r = (3 * alpha * d * (d - ome) + mean * mean) * mean
    w = jnp.cbrt(jnp.abs(r) + jnp.sqrt(q**3 + r * r)) ** 2
    initial = (2 * r * w / (w * w + w * q + q * q) + mean) / d
    sine, cosine = fast_sinE_cosE(initial)

    # For E < 0.3 the omitted E**15/15! term is below 1.1e-20. Evaluating
    # E - sin(E) directly loses the residual when e approaches one.
    x2 = initial * initial
    small_residual = (
        initial
        * x2
        * (
            1 / 6
            - x2
            * (
                1 / 120
                - x2
                * (1 / 5040 - x2 * (1 / 362880 - x2 * (1 / 39916800 - x2 / 6227020800)))
            )
        )
    )
    f = (
        ome * initial
        + e * jnp.where(initial < 0.3, small_residual, initial - sine)
        - mean
    )
    f1 = _denominator(sine, cosine, e)
    f2 = e * sine
    f3 = e * cosine
    d3 = -f / (f1 - 0.5 * f * f2 / f1)
    d4 = -f / (f1 + 0.5 * d3 * f2 + d3 * d3 * f3 / 6)
    correction = -f / (f1 + 0.5 * d4 * f2 + d4 * d4 * f3 / 6 - d4**3 * f2 / 24)

    delta2 = correction * correction
    sin_delta = correction * (1 - delta2 / 6 + delta2 * delta2 / 120)
    cos_delta = 1 - delta2 / 2 + delta2 * delta2 / 24
    sinE = sine * cos_delta + cosine * sin_delta
    cosE = cosine * cos_delta - sine * sin_delta
    eccentric = initial + correction
    E = jnp.where(negative, two_pi - eccentric, eccentric)
    return E, jnp.where(negative, -sinE, sinE), cosE


@E_solve_trig.defjvp
def _solve_jvp(primals, tangents):
    """Differentiate the equation rather than the numerical approximation."""
    M, e = primals
    M_dot, e_dot = tangents
    # A nondifferentiated integer literal (e.g. e=0) has a float0 tangent.
    if getattr(M_dot, "dtype", None) == jax.dtypes.float0:
        M_dot = 0.0
    if getattr(e_dot, "dtype", None) == jax.dtypes.float0:
        e_dot = 0.0
    E, sine, cosine = E_solve_trig(M, e)
    E_dot = (M_dot + sine * e_dot) / _denominator(sine, cosine, e)
    return (E, sine, cosine), (E_dot, cosine * E_dot, -sine * E_dot)


def E_solve(M, e):
    """Return wrapped eccentric anomaly for broadcast-compatible M and e."""
    return E_solve_trig(M, e)[0]


def solve_trig(M, e):
    """Return differentiable (sin(E), cos(E)), suitable as an orbit solver."""
    _, sine, cosine = E_solve_trig(M, e)
    return sine, cosine
