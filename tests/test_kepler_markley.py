"""Verification of the table-free solver against arbitrary-precision roots."""

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
from hwoutils.constants import two_pi
from hwoutils.testing import ulps
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from jax.test_util import check_grads

from orbix.kepler import Markley, kepler


def _pair(mean, e):
    """Exercise the public longitude equation at zero periapsis longitude."""
    return kepler(mean, e, 0, solver=Markley())


def _wrapped_angle(mean, e):
    """Recover an angle only for the equation and monotonicity assertions."""
    sine, cosine = _pair(mean, e)
    return jnp.mod(jnp.arctan2(sine, cosine), two_pi)


def _angle_and_pair(mean, e):
    sine, cosine = _pair(mean, e)
    return jnp.mod(jnp.arctan2(sine, cosine), two_pi), sine, cosine


def _reference(m, e):
    """Bisect the monotonic Kepler equation at 70 decimal digits."""
    with mp.workdps(70):
        mean = mp.mpf(float(m)) % (2 * mp.pi)
        eccentricity = mp.mpf(float(e))
        low, high = mp.mpf(0), 2 * mp.pi
        for _ in range(240):
            mid = (low + high) / 2
            if mid - eccentricity * mp.sin(mid) < mean:
                low = mid
            else:
                high = mid
        eccentric = (low + high) / 2
        return float(eccentric), float(mp.sin(eccentric)), float(mp.cos(eccentric))


@pytest.mark.parametrize(
    "e",
    [
        0.0,
        1e-8,
        0.3,
        0.78,
        0.95,
        0.9999,
        "near-parabolic",
        "last-elliptic",
    ],
)
def test_matches_high_precision_roots(e, precision):
    """Fails if the trig pair disagrees with high-precision periapsis roots.

    The logarithmic periapsis points expose cancellation that a small equation
    residual would hide when dM/dE approaches zero. Use the represented
    eccentricity so near-parabolic input rounding is not solver error. At
    unit trig scale allow 32 roundings for argument reduction, polynomial
    sin/cos, the three root corrections and final rotation. At the seam,
    check the signed sine with the same relative allowance and zero absolute
    slack so that rounding a tiny phase to zero also fails.
    """
    if e == "near-parabolic":
        e = 1 - max(1e-12, 4 * np.finfo(precision).eps)
    elif e == "last-elliptic":
        e = np.nextafter(precision.type(1), precision.type(0))
    wraps = np.nextafter([two_pi, -2 * two_pi, 3 * two_pi], 0)
    means = np.unique(
        np.r_[
            0.0,
            np.geomspace(1e-24, 0.1, 20),
            np.linspace(0.1, 2 * np.pi - 0.1, 20),
            np.nextafter(np.pi, 0),
            np.pi,
            np.nextafter(np.pi, 4),
            wraps,
        ]
    )
    e = precision.type(e)
    reference = np.array([_reference(m, e) for m in means]).T
    # Supply the host longitudes intact, including the point just below a turn.
    result = _pair(means, e)
    bound = float(ulps(32, precision.type(1)))
    np.testing.assert_allclose(np.array(result), reference[1:], rtol=0, atol=bound)
    seam_indices = np.searchsorted(means, wraps)
    np.testing.assert_allclose(
        np.asarray(result[0])[seam_indices],
        reference[1, seam_indices],
        rtol=bound,
        atol=0,
    )
    direct = Markley().solve(wraps, e)
    np.testing.assert_allclose(
        direct[0], reference[1, seam_indices], rtol=bound, atol=0
    )
    np.testing.assert_allclose(
        direct[1], reference[2, seam_indices], rtol=0, atol=bound
    )
    # The compiled grid sees represented inputs, including a float32 wrap
    # point above the seam where the original host input lay below it.
    represented = means.astype(precision)
    expected = (
        reference
        if np.array_equal(represented, means)
        else np.array([_reference(m, e) for m in represented]).T
    )
    for solve in (_pair, Markley().solve):
        compiled = jax.jit(solve)(jnp.array(represented), jnp.array(e))
        np.testing.assert_allclose(compiled, expected[1:], rtol=0, atol=bound)
        np.testing.assert_allclose(
            np.asarray(compiled[0])[seam_indices],
            expected[1, seam_indices],
            rtol=bound,
            atol=0,
        )


def test_tiny_signed_mean_anomalies(precision):
    """Keep both signs of the closest elliptic orbit accurate at periapsis.

    At each lane's largest e below one, allow 32 relative roundings for
    the small-angle polynomial, cubic starter, three corrections and trig
    rotation. Zero absolute slack exposes loss of the tiniest signed means.
    """
    means = np.geomspace(1e-24, 1e-8, 24, dtype=precision)
    e = np.nextafter(precision.type(1), precision.type(0))
    expected = np.array([_reference(m, e) for m in means]).T
    positive = jax.jit(_pair)(jnp.array(means), jnp.array(e))
    negative = jax.jit(_pair)(-jnp.array(means), jnp.array(e))
    relative = float(ulps(32, precision.type(1)))
    np.testing.assert_allclose(positive[0], expected[1], rtol=relative, atol=0)
    np.testing.assert_allclose(negative[0], -expected[1], rtol=relative, atol=0)
    np.testing.assert_allclose(negative[1], expected[2], rtol=relative, atol=0)


_broadcast_solve = jax.jit(_angle_and_pair)
_mapped_solve = jax.jit(
    jax.vmap(jax.vmap(_wrapped_angle, in_axes=(0, None)), in_axes=(None, 0))
)
_scalar_solve = jax.jit(_wrapped_angle)
_direct_solve = jax.jit(Markley().solve)


@example(
    means=np.array(
        [0.0, np.finfo(np.float32).tiny, 1.6e-43, -1.6e-43, 1.1e-39, -1.1e-39],
        dtype=np.float32,
    ),
    eccs=np.array([0.0, 0.4, 0.9], dtype=np.float32),
)
@example(
    means=np.array([0.1, 1.0, 3.0, 5.0, 0.1, 1.0]), eccs=np.array([0.0, 0.4, 0.999])
)
@example(
    means=np.array([-20.0, -4.0, 0.0, 0.1, 3.2, 20.0]), eccs=np.array([0.0, 0.4, 0.9])
)
@given(
    means=arrays(
        np.float64,
        (6,),
        elements=st.floats(-two_pi, two_pi, allow_nan=False, allow_infinity=False),
    ),
    eccs=arrays(
        np.float64,
        (3,),
        elements=st.floats(0.0, 0.99, allow_nan=False, allow_infinity=False),
    ),
)
def test_broadcast_wrapping_equation_and_trig(precision, means, eccs):
    """Fails if broadcast/vmap/scalar paths disagree or wrapping corrupts the root.

    Allow four roundings at a turn for angle recovery, wrapping and the
    equation residual, and eight at unit scale for the trig identity.
    A sorted interior pair checks monotonicity and continuity via analytic
    slope bounds; wrapped endpoints are lifted around their mean anomaly.
    """
    means = np.asarray(jnp.asarray(means)).astype(np.float64)
    eccs = np.asarray(jnp.asarray(eccs)).astype(np.float64)
    angle_bound = float(ulps(4, precision.type(two_pi)))
    trig_bound = float(ulps(8, precision.type(1)))
    eccentric, sine, cosine = map(
        np.asarray, _broadcast_solve(jnp.asarray(means), jnp.asarray(eccs[:, None]))
    )
    if eccs[0] == 0:
        reads_zero = np.asarray(jnp.asarray(means) == 0)
        nodes = reads_zero | (means == np.finfo(np.float32).tiny)
        expected = np.where(reads_zero, 0, means)
        direct = _direct_solve(jnp.asarray(means), jnp.asarray(eccs[0]))
        for s, c in ((sine[0], cosine[0]), direct):
            assert np.all(np.isfinite(s)) and np.all(np.isfinite(c))
            # At zero and the smallest normal, sin(M) rounds to M and cos(M)
            # to 1. Widen on the host so angle recovery preserves the normal.
            angle = np.arctan2(np.asarray(s, dtype=np.float64), np.asarray(c))
            np.testing.assert_array_equal(angle[nodes], expected[nodes])
    # Endpoint rounding (including subnormal flushing) may return 0 or 2*pi.
    assert np.all((eccentric >= 0) & (eccentric <= eccentric.dtype.type(two_pi)))
    residual = eccentric - eccs[:, None] * sine - means
    np.testing.assert_allclose(
        np.arctan2(np.sin(residual), np.cos(residual)), 0, rtol=0, atol=angle_bound
    )
    np.testing.assert_allclose(sine, np.sin(eccentric), rtol=0, atol=trig_bound)
    np.testing.assert_allclose(cosine, np.cos(eccentric), rtol=0, atol=trig_bound)
    np.testing.assert_allclose(sine**2 + cosine**2, 1, rtol=0, atol=trig_bound)
    np.testing.assert_allclose(
        _mapped_solve(jnp.asarray(means), jnp.asarray(eccs)),
        eccentric,
        rtol=0,
        atol=trig_bound,
    )
    np.testing.assert_allclose(
        _scalar_solve(jnp.asarray(means[0]), jnp.asarray(eccs[0])),
        eccentric[0, 0],
        rtol=0,
        atol=trig_bound,
    )
    pair = np.asarray(jnp.asarray([means[0], means[0] + 1e-7])).astype(np.float64)
    result = np.asarray(_scalar_solve(jnp.asarray(pair), jnp.asarray(eccs[0])))
    offset = result - pair
    lifted = pair + np.arctan2(np.sin(offset), np.cos(offset))
    delta = lifted[1] - lifted[0]
    step = pair[1] - pair[0]
    roundoff = 2 * angle_bound / (1 - eccs[0])
    assert step / (1 + eccs[0]) - roundoff <= delta <= step / (1 - eccs[0]) + roundoff


@pytest.mark.float64_only(reason="finite-difference derivative reference")
@pytest.mark.parametrize("e", [0.0, 0.3, 0.95])
def test_implicit_derivatives(e):
    """Both AD modes and second derivatives match finite differences (fp)."""
    means = jnp.array([0.2, 1.0, 2.5, 4.0])
    check_grads(
        _pair,
        (means, jnp.array(e)),
        order=2,
        modes=["fwd", "rev"],
        atol=2e-5,
        rtol=2e-5,
    )


def test_float32():
    """The float32 floor remains below 2e-6 on an interior grid (fp)."""
    means = np.linspace(0.01, 6.27, 100).astype(np.float32)
    eccentricity = np.float32(0.95)
    expected = np.array([_reference(m, eccentricity) for m in means]).T
    result = jax.jit(_pair)(jnp.array(means), jnp.array(eccentricity))
    assert all(x.dtype == jnp.float32 for x in result)
    np.testing.assert_allclose(result, expected[1:], rtol=0, atol=2e-6)


def test_gradients_with_integer_constants():
    """Integer constants preserve the circular analytic derivative.

    Four unit-scale roundings cover sin/cos, division and multiplication
    in the implicit derivative. The periapsis derivative is exactly zero.
    """
    circular = jax.jit(jax.grad(lambda m: _pair(m, 0)[0]))(1.0)
    periapsis = jax.jit(jax.grad(lambda e: _pair(0, e)[0]))(0.3)
    np.testing.assert_allclose(
        circular, np.cos(1.0), rtol=0, atol=float(ulps(4, circular.dtype.type(1)))
    )
    np.testing.assert_allclose(periapsis, 0.0, rtol=0, atol=0)


def test_tracing_precision_context(precision):
    """High-precision roots agree within 16 eps; catches int32 literal overflow.

    The tolerance covers argument reduction and conditioning at e=0.95 on
    these interior points. Explicit float32 inputs stay float32 under x64.
    """
    for dtype in dict.fromkeys((np.dtype(np.float32), precision)):
        means = np.array([-0.2, 0.0, 0.01, 1.0, 3.0, 6.1], dtype=dtype)
        eccentricity = dtype.type(0.95)
        expected = np.array(
            [_reference(float(m) % two_pi, eccentricity) for m in means]
        ).T
        result = jax.jit(_pair)(jnp.array(means), jnp.array(eccentricity))
        assert all(value.dtype == dtype for value in result)
        np.testing.assert_allclose(
            result, expected[1:], rtol=0, atol=16 * np.finfo(dtype).eps
        )
