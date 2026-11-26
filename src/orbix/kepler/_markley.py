"""Private Markley (1995) cubic starter and fifth-order correction.

Reference: Celestial Mechanics and Dynamical Astronomy 63, 101-111,
https://doi.org/10.1007/BF00691917. The reduced polynomial trig helper is
adapted from orvara (BSD 3-clause), Copyright (c) 2021 Timothy Brandt,
Trent Dupuy, Yiting Li, G. Mirek Brandt, Yunlin Zeng, Daniel Michalik,
and Virginia Raposo-Pulido. Small-angle residuals avoid cancellation
near parabolic periapsis; phase information lost in input rounding
cannot be recovered.
"""

import jax
import jax.numpy as jnp

_if3 = 1.0 / 6.0
_if5 = 1.0 / (6.0 * 20.0)
_if7 = 1.0 / (6.0 * 20.0 * 42.0)
_if9 = 1.0 / (6.0 * 20.0 * 42.0 * 72.0)
_if11 = 1.0 / (6.0 * 20.0 * 42.0 * 72.0 * 110.0)
_if13 = 1.0 / (6.0 * 20.0 * 42.0 * 72.0 * 110.0 * 156.0)
_if15 = 1.0 / (6.0 * 20.0 * 42.0 * 72.0 * 110.0 * 156.0 * 210.0)


def _shortsin(x):
    """Approximates the sine function using a short polynomial.

    This is only valid between [0, pi].
    """
    x2 = x * x
    return x * (
        1
        - x2
        * (
            _if3
            - x2
            * (
                _if5
                - x2 * (_if7 - x2 * (_if9 - x2 * (_if11 - x2 * (_if13 - x2 * _if15))))
            )
        )
    )


def _sin_cos(E):
    """Compute the sine and cosine of the eccentric anomaly using _shortsin."""
    # Select the reduced angle before evaluating the polynomial. A vmapped
    # switch evaluates all three branches, tripling the polynomial/sqrt work.
    low = E <= (jnp.pi / 4)
    middle = (E > (jnp.pi / 4)) & (E < (3 * jnp.pi / 4))
    x = jnp.where(middle, (jnp.pi / 2) - E, jnp.where(low, E, jnp.pi - E))
    small = _shortsin(x)
    large = jax.lax.sqrt(1.0 - small * small)
    sinE = jnp.where(middle, large, small)
    cosE = jnp.where(middle, small, jnp.where(low, large, -large))
    return sinE, cosE


def _denominator(sine, cosine, e):
    """Evaluate 1 - e*cos(E) without cancellation close to periapsis."""
    positive = cosine > 0
    divisor = jnp.where(positive, 1 + cosine, jnp.ones_like(cosine))
    one_minus_cos = jnp.where(positive, sine * sine / divisor, 1 - cosine)
    return (1 - e) + e * one_minus_cos


def _principal_angle(sine, cosine):
    """Recover the signed phase from a unit pair at working precision."""
    # Here |s/c| <= 2*tiny, so atan(s/c) rounds to s/c. CPU atan2 can
    # lose the smallest normal or misread subnormal bits under flush-to-zero.
    tiny = (jnp.abs(sine) <= jnp.finfo(sine.dtype).tiny) & (cosine >= 0.5)
    divisor = jnp.where(tiny, cosine, 1)
    angle = jnp.arctan2(jnp.where(tiny, 0, sine), cosine)
    return jnp.where(tiny, sine / divisor, angle)


def _markley_pair(M, e):
    """Evaluate Markley's algorithm without a derivative rule."""
    dtype = jnp.result_type(M, e, 0.0)
    M, e = jnp.asarray(M, dtype=dtype), jnp.asarray(e, dtype=dtype)
    reduced = _principal_angle(jnp.sin(M), jnp.cos(M))
    return _markley_reduced(reduced, e)


def _markley_reduced(signed_mean, e):
    """Solve for a signed principal mean that has already been reduced."""
    mean = jnp.abs(signed_mean)
    negative = signed_mean < 0
    ome = 1 - e

    alpha = (3 * jnp.pi + 1.6 * (jnp.pi - mean) / (1 + e)) / (jnp.pi - 6 / jnp.pi)
    d = 3 * ome + alpha * e
    q = 2 * alpha * d * ome - mean * mean
    r = (3 * alpha * d * (d - ome) + mean * mean) * mean
    w = jnp.cbrt(jnp.abs(r) + jnp.sqrt(q**3 + r * r)) ** 2
    initial = (2 * r * w / (w * w + w * q + q * q) + mean) / d
    # A one-ULP starter error at the smallest normal can flush out of the
    # residual. For a circular orbit the exact starter is the mean itself.
    initial = jnp.where((e == 0) & (mean <= jnp.finfo(mean.dtype).tiny), mean, initial)
    sine, cosine = _sin_cos(initial)

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
                * (
                    1 / 5040
                    - x2 * (1 / 362880 - x2 * (1 / 39916800 - x2 / 6227020800.0))
                )
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
    return jnp.where(negative, -sinE, sinE), cosE
