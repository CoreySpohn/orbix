"""Properties for the Orvara solvers and their polynomial/trig boundaries."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import two_pi
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from orbix.kepler import core


def _floats(low, high):
    return st.floats(low, high, allow_nan=False, allow_infinity=False)


@jax.jit
def _solutions(means, e):
    """Exercise public wrappers and the appropriate low/high-e branch."""
    wrapped = jnp.mod(means, two_pi)
    branch_e = jnp.where(e == 0, 0.3, e)
    branch = jax.lax.cond(e < 0.78, core.le_E, core.he_E, wrapped, branch_e)
    branch_trig = jax.lax.cond(
        e < 0.78, core.le_E_trig, core.he_E_trig, wrapped, branch_e
    )
    branch = jnp.where(e == 0, core.identity_solver(wrapped, e), branch)
    identity = core.identity_solver_trig(wrapped, e)
    branch_trig = tuple(
        jnp.where(e == 0, x, y) for x, y in zip(identity, branch_trig, strict=False)
    )
    return (
        jnp.stack([core.E_solve(means, e), core.E_solve_trig(means, e)[0], branch]),
        jnp.stack(
            [
                jnp.stack(core.E_solve_trig(means, e)[1:]),
                jnp.stack(core.solve_trig(means, e)),
                jnp.stack(core.diff_solve_trig(means, e)),
                jnp.stack(branch_trig[1:]),
            ]
        ),
    )


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
    # Retain every eccentricity from the old near-pi grid and pin dispatch edges.
    for ecc in (e, 0.0, 0.05, 0.2, 0.5, 0.9, np.nextafter(0.78, 0), 0.78, 0.99):
        eccentric, trig = map(
            np.asarray,
            _solutions(jnp.asarray(means), jnp.asarray(ecc, dtype=jnp.float64)),
        )
        assert np.all((eccentric >= -4e-14) & (eccentric <= two_pi + 4e-14))
        sine, cosine = trig[:, 0], trig[:, 1]
        residual = eccentric - ecc * np.sin(eccentric) - means
        np.testing.assert_allclose(
            np.arctan2(np.sin(residual), np.cos(residual)), 0, atol=4e-14, rtol=0
        )
        np.testing.assert_allclose(
            sine, np.broadcast_to(np.sin(eccentric[0]), sine.shape), atol=4e-14, rtol=0
        )
        np.testing.assert_allclose(
            cosine,
            np.broadcast_to(np.cos(eccentric[0]), cosine.shape),
            atol=4e-14,
            rtol=0,
        )
        np.testing.assert_allclose(sine**2 + cosine**2, 1, atol=4e-14, rtol=0)
        offset = eccentric[:, :2] - means[:2]
        lifted = means[:2] + np.arctan2(np.sin(offset), np.cos(offset))
        delta = lifted[:, 1] - lifted[:, 0]
        actual_step = means[1] - means[0]
        assert np.all(delta >= actual_step / (1 + ecc) - 8e-14)
        assert np.all(delta <= actual_step / (1 - ecc) + 8e-14)


_fast_trig = jax.jit(core.fast_sinE_cosE)


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
def test_polynomial_interval_boundaries(e):
    """Fails if interval selection loses 4e-14 rad at either side of a table edge.

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
    [0.0, 1e-200, 1e-100, np.nextafter(np.finfo(float).eps, 0), np.finfo(float).eps],
)
def test_tiny_positive_eccentricity_remains_finite(e):
    """Fails if a near-circular orbit becomes NaN through inverse-e overflow."""
    result = core.E_solve_jit(jnp.array([1.3]), jnp.asarray(e))
    np.testing.assert_allclose(result, [1.3], atol=4e-14, rtol=0)
    roots, trig = _solutions(jnp.array([1.3]), jnp.asarray(e))
    np.testing.assert_allclose(roots, 1.3, atol=4e-14, rtol=0)
    np.testing.assert_allclose(trig[:, 0], np.sin(1.3), atol=4e-14, rtol=0)
    np.testing.assert_allclose(trig[:, 1], np.cos(1.3), atol=4e-14, rtol=0)
    gradient = jax.grad(
        lambda ecc: sum(
            jnp.sum(x) for x in core.E_solve_trig_jit(jnp.array([1.3]), ecc)
        )
    )(jnp.asarray(e))
    assert np.isfinite(gradient)
