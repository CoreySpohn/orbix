"""Kepler equation, continuity, near-circular, and reduced-trig regressions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import two_pi
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from orbix.kepler import kepler
from orbix.kepler._markley import _sin_cos


def _floats(low, high):
    return st.floats(low, high, allow_nan=False, allow_infinity=False)


@jax.jit
def _solutions(means, e):
    """Recover the angle for residual checks from the public solver pair."""
    sine, cosine = kepler(means, e, 0)
    return jnp.mod(jnp.arctan2(sine, cosine), two_pi), jnp.stack([sine, cosine])


@given(
    e=st.one_of(st.just(0.0), _floats(1e-100, 0.99)),
    mean=_floats(-two_pi, two_pi - 0.001),
    step=_floats(1e-8, 0.001),
)
def test_solver_equation_trig_and_continuity(e, mean, step):
    """Fails if any solver breaks Kepler's equation, trig, or monotone continuity.

    The 4e-14 rad float64 bound retains the polynomial-edge regression's
    accuracy. Tiny positive e is also pinned by an explicit regression.
    Lift wrapped E around M before checking slope bounds, including
    across the seam; 1/(1+e) <= dE/dM <= 1/(1-e).
    """
    means = np.array(
        [
            mean,
            mean + step,
            -two_pi,
            0.0,
            two_pi,
            np.nextafter(np.pi, 0),
            np.pi,
            np.nextafter(np.pi, 4),
        ]
    )
    # Retain the near-pi regression eccentricities and the 0.78 boundary.
    for ecc in (e, 0.0, 0.05, 0.2, 0.5, 0.9, np.nextafter(0.78, 0), 0.78, 0.99):
        eccentric, trig = map(
            np.asarray,
            _solutions(jnp.asarray(means), jnp.asarray(ecc)),
        )
        assert np.all((eccentric >= -4e-14) & (eccentric <= two_pi + 4e-14))
        sine, cosine = trig
        residual = eccentric - ecc * np.sin(eccentric) - means
        np.testing.assert_allclose(
            np.arctan2(np.sin(residual), np.cos(residual)), 0, atol=4e-14, rtol=0
        )
        np.testing.assert_allclose(sine, np.sin(eccentric), atol=4e-14, rtol=0)
        np.testing.assert_allclose(
            cosine,
            np.cos(eccentric),
            atol=4e-14,
            rtol=0,
        )
        np.testing.assert_allclose(sine**2 + cosine**2, 1, atol=4e-14, rtol=0)
        offset = eccentric[:2] - means[:2]
        lifted = means[:2] + np.arctan2(np.sin(offset), np.cos(offset))
        delta = lifted[1] - lifted[0]
        actual_step = means[1] - means[0]
        assert np.all(delta >= actual_step / (1 + ecc) - 8e-14)
        assert np.all(delta <= actual_step / (1 - ecc) + 8e-14)


_fast_trig = jax.jit(_sin_cos)


@given(angles=arrays(np.float64, (16,), elements=_floats(0.0, np.pi)))
def test_fast_trig_across_quadrants(angles):
    """Fails if reduced trig polynomials lose 8e-16 accuracy in any quadrant."""
    edges = np.array([0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    angles = np.r_[
        angles, edges, np.nextafter(edges[1:-1], 0), np.nextafter(edges[1:-1], np.pi)
    ]
    sine, cosine = _fast_trig(jnp.asarray(angles))
    np.testing.assert_allclose(sine, np.sin(angles), rtol=0, atol=8e-16)
    np.testing.assert_allclose(cosine, np.cos(angles), rtol=0, atol=8e-16)


@example(e=0.0)
@example(e=0.3)
@example(e=np.nextafter(0.78, 0))
@example(e=0.78)
@example(e=0.95)
@given(e=st.one_of(st.just(0.0), _floats(1e-100, 0.99)))
def test_eccentric_anomaly_nodes(e):
    """Fails if solving known roots loses 4e-14 rad at sampled anomaly nodes.

    Tiny positive e is covered by the explicit overflow regression.
    """
    nodes = np.linspace(0, np.pi, 13)
    eccentric = np.r_[
        nodes, np.nextafter(nodes[1:], 0), np.nextafter(nodes[:-1], np.pi)
    ]
    means = eccentric - e * np.sin(eccentric)
    actual, trig = _solutions(jnp.asarray(means), jnp.asarray(e))
    np.testing.assert_allclose(
        actual, np.broadcast_to(eccentric, actual.shape), rtol=0, atol=4e-14
    )
    expected = np.stack([np.sin(eccentric), np.cos(eccentric)])
    np.testing.assert_allclose(
        trig, np.broadcast_to(expected, trig.shape), rtol=0, atol=4e-14
    )


@pytest.mark.parametrize(
    "e",
    [
        0.0,
        pytest.param(
            1e-200,
            marks=pytest.mark.float64_only(
                reason="tiny positive eccentricity underflows float32"
            ),
        ),
        pytest.param(
            1e-100,
            marks=pytest.mark.float64_only(
                reason="tiny positive eccentricity underflows float32"
            ),
        ),
        np.nextafter(np.finfo(float).eps, 0),
        np.finfo(float).eps,
    ],
)
def test_tiny_positive_eccentricity_remains_finite(e):
    """Fails if a near-circular orbit becomes NaN through inverse-e overflow."""
    roots, trig = _solutions(jnp.array([1.3]), jnp.asarray(e))
    np.testing.assert_allclose(roots, 1.3, atol=4e-14, rtol=0)
    np.testing.assert_allclose(trig[0], np.sin(1.3), atol=4e-14, rtol=0)
    np.testing.assert_allclose(trig[1], np.cos(1.3), atol=4e-14, rtol=0)
    gradient = jax.grad(
        lambda ecc: sum(jnp.sum(x) for x in kepler(jnp.array([1.3]), ecc, 0))
    )(jnp.asarray(e))
    np.testing.assert_allclose(
        gradient, (np.cos(1.3) - np.sin(1.3)) * np.sin(1.3), atol=4e-14, rtol=0
    )
