"""Verification of the table-free solver against arbitrary-precision roots."""

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
from hwoutils.constants import two_pi
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from jax.test_util import check_grads

from orbix.kepler.markley import E_solve, E_solve_trig, solve_trig


def _reference(m, e):
    """Bisect the monotonic Kepler equation at 70 decimal digits."""
    with mp.workdps(70):
        mean = mp.mpf(float(m))
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
    "e", [0.0, 1e-8, 0.3, 0.78, 0.95, 0.9999, 1 - 1e-12, np.nextafter(1.0, 0.0)]
)
def test_matches_high_precision_roots(e):
    """E and its trig pair agree within 8e-15 rad (fp, sampled double floor).

    The logarithmic periapsis points expose cancellation that a small equation
    residual would hide when dM/dE approaches zero.
    """
    means = np.unique(
        np.r_[
            0.0,
            np.geomspace(1e-24, 0.1, 20),
            np.linspace(0.1, 2 * np.pi - 0.1, 20),
            np.nextafter(np.pi, 0),
            np.pi,
            np.nextafter(np.pi, 4),
            np.nextafter(2 * np.pi, 0),
        ]
    )
    reference = np.array([_reference(m, e) for m in means]).T
    result = jax.jit(E_solve_trig)(jnp.array(means), jnp.array(e))
    np.testing.assert_allclose(np.array(result), reference, rtol=0, atol=8e-15)


def test_tiny_signed_mean_anomalies():
    """Preserve relative precision at periapsis; no absolute tolerance (fp)."""
    means = np.geomspace(1e-24, 1e-8, 24)
    e = np.nextafter(1.0, 0.0)
    expected = np.array([_reference(m, e) for m in means]).T
    positive = jax.jit(solve_trig)(jnp.array(means), jnp.array(e))
    negative = jax.jit(solve_trig)(-jnp.array(means), jnp.array(e))
    np.testing.assert_allclose(positive[0], expected[1], rtol=5e-14, atol=0)
    np.testing.assert_allclose(negative[0], -expected[1], rtol=5e-14, atol=0)
    np.testing.assert_allclose(negative[1], expected[2], rtol=5e-14, atol=0)


_broadcast_solve = jax.jit(E_solve_trig)
_mapped_solve = jax.jit(
    jax.vmap(jax.vmap(E_solve, in_axes=(0, None)), in_axes=(None, 0))
)
_scalar_solve = jax.jit(E_solve)


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
def test_broadcast_wrapping_equation_and_trig(means, eccs):
    """Fails if broadcast/vmap/scalar paths disagree or wrapping corrupts the root.

    The 4e-15 residual and 2e-15 trig bounds retain the float64 regressions.
    A sorted interior pair checks monotonicity and continuity via analytic
    slope bounds; wrapped endpoints are lifted around their mean anomaly.
    """
    eccentric, sine, cosine = map(
        np.asarray, _broadcast_solve(jnp.asarray(means), jnp.asarray(eccs[:, None]))
    )
    # Endpoint rounding (including subnormal flushing) may return 0 or 2*pi.
    assert np.all((eccentric >= -2e-15) & (eccentric <= two_pi + 2e-15))
    residual = eccentric - eccs[:, None] * sine - means
    np.testing.assert_allclose(
        np.arctan2(np.sin(residual), np.cos(residual)), 0, rtol=0, atol=4e-15
    )
    np.testing.assert_allclose(sine, np.sin(eccentric), rtol=0, atol=2e-15)
    np.testing.assert_allclose(cosine, np.cos(eccentric), rtol=0, atol=2e-15)
    np.testing.assert_allclose(sine**2 + cosine**2, 1, rtol=0, atol=2e-15)
    np.testing.assert_allclose(
        _mapped_solve(jnp.asarray(means), jnp.asarray(eccs)),
        eccentric,
        rtol=0,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        _scalar_solve(jnp.asarray(means[0]), jnp.asarray(eccs[0])),
        eccentric[0, 0],
        rtol=0,
        atol=2e-15,
    )
    pair = np.array([means[0], means[0] + 1e-7])
    result = np.asarray(_scalar_solve(jnp.asarray(pair), jnp.asarray(eccs[0])))
    offset = result - pair
    lifted = pair + np.arctan2(np.sin(offset), np.cos(offset))
    delta = lifted[1] - lifted[0]
    step = pair[1] - pair[0]
    roundoff = 4e-15 / (1 - eccs[0])
    assert step / (1 + eccs[0]) - roundoff <= delta <= step / (1 - eccs[0]) + roundoff


@pytest.mark.parametrize("e", [0.0, 0.3, 0.95])
def test_implicit_derivatives(e):
    """Both AD modes and second derivatives match finite differences (fp)."""
    means = jnp.array([0.2, 1.0, 2.5, 4.0])
    check_grads(
        solve_trig,
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
    result = jax.jit(E_solve_trig)(jnp.array(means), jnp.array(eccentricity))
    assert all(x.dtype == jnp.float32 for x in result)
    np.testing.assert_allclose(result, expected, rtol=0, atol=2e-6)


def test_gradients_with_integer_constants():
    """Integer circular/periapsis constants must not break real AD (alg)."""
    circular = jax.jit(jax.grad(lambda m: E_solve(m, 0)))(1.0)
    periapsis = jax.jit(jax.grad(lambda e: E_solve(0, e)))(0.3)
    np.testing.assert_allclose(circular, 1.0, rtol=0, atol=2e-15)
    np.testing.assert_allclose(periapsis, 0.0, rtol=0, atol=0)
