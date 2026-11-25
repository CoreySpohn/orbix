"""Residual, independent iteration, and finite-difference longitude checks."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from orbix.kepler import AbstractKeplerSolver, Grid, Halley, Markley, kepler

_SOLVERS = (Markley(), Halley(), Grid())


def _sample(dtype):
    """Sample phases including both sides of periapsis and wrapped longitudes."""
    small = np.geomspace(1e-6, 0.1, 16)
    mean = np.r_[np.linspace(-np.pi, np.pi, 193), -small, small]
    e = np.linspace(0, 0.95, 25)
    p = np.array([-7.0, -0.3, 0.0, 0.7, 8.0])
    return tuple(
        jnp.asarray(value, dtype=dtype)
        for value in (
            mean[:, None, None] + p,
            e[None, :, None] * np.cos(p),
            e[None, :, None] * np.sin(p),
        )
    )


@pytest.mark.parametrize("solver", _SOLVERS)
def test_residual(solver, precision):
    """Implicit equation is truth: 16 eps roundoff, or 6e-6 grid truncation.

    A denser 281125-point survey over e <= 0.95 measured grid residuals
    4.68e-6 (float32) and 4.44e-6 (float64), rounded upward. The 6e-6 bound
    leaves margin for platform rounding. Check both trig components modulo 2*pi.
    """
    args = _sample(precision)
    sine, cosine = jax.jit(kepler)(*args, solver=solver)
    lam, ex, ey = map(np.asarray, args)
    # Evaluate the residual in float64 so the diagnostic has its own floor.
    sine, cosine = np.asarray(sine, dtype=float), np.asarray(cosine, dtype=float)
    residual = lam.astype(float) - (np.arctan2(sine, cosine) - ex * sine + ey * cosine)
    bound = 6e-6 if isinstance(solver, Grid) else 16 * np.finfo(precision).eps
    np.testing.assert_allclose(np.sin(residual), 0, rtol=0, atol=bound)
    np.testing.assert_allclose(np.cos(residual), 1, rtol=0, atol=bound)


def test_markley_against_halley_and_periapsis_rotation(precision):
    """Fails if independent Halley or periapsis rotations disagree within 24 eps/(1-e).

    The implicit slope bounds the amplification of input and rotation rounding;
    the grid spans longitude, eccentricity and periapsis independently.
    """
    args = _sample(precision)
    actual = jax.jit(kepler)(*args, solver=Markley())
    reference = jax.jit(kepler)(*args, solver=Halley())
    lam, ex, ey = args
    e = jnp.sqrt(ex * ex + ey * ey)
    bound = 24 * np.finfo(precision).eps / (1 - np.asarray(e))
    assert np.all(np.abs(np.asarray(actual) - np.asarray(reference)) <= bound)
    p = jnp.arctan2(ey, ex)
    rotated = (
        actual[0] * jnp.cos(p) - actual[1] * jnp.sin(p),
        actual[1] * jnp.cos(p) + actual[0] * jnp.sin(p),
    )
    expected = jax.jit(kepler)(lam - p, e, 0, solver=Halley())
    assert np.all(np.abs(np.asarray(rotated) - np.asarray(expected)) <= bound)


@pytest.mark.float64_only(reason="finite-difference derivative reference")
@pytest.mark.parametrize("solver", _SOLVERS)
def test_circular_value_gradient_and_hessian(solver):
    """Circular values are analytic; AD matches independent central differences.

    Float64 Halley values supply differences, including for the approximate
    grid's implicit rule. Step 2e-4 gives O(h^2) truncation and eps/h^2 noise;
    2e-7 absolute/relative covers both for gradients and Hessians.
    """
    point = jnp.array([0.7, 0.0, 0.0])

    def fun(x):
        return jnp.stack(kepler(*x, solver=solver))

    truth = jax.jit(lambda x: jnp.stack(kepler(*x, solver=Halley())))
    np.testing.assert_array_equal(fun(point), [jnp.sin(point[0]), jnp.cos(point[0])])
    jac = jax.jit(jax.jacfwd(fun))(point)
    hess = jax.jit(jax.jacfwd(jax.jacrev(fun)))(point)
    assert np.all(np.isfinite(jac)) and np.all(np.isfinite(hess))
    step = 2e-4
    offsets = np.eye(3) * step
    gradient = np.stack(
        [(truth(point + d) - truth(point - d)) / (2 * step) for d in offsets],
        axis=-1,
    )
    hessian = np.empty((2, 3, 3))
    for i, di in enumerate(offsets):
        for j, dj in enumerate(offsets):
            hessian[:, i, j] = (
                truth(point + di + dj)
                - truth(point + di - dj)
                - truth(point - di + dj)
                + truth(point - di - dj)
            ) / (4 * step**2)
    np.testing.assert_allclose(jac, gradient, rtol=2e-7, atol=2e-7)
    np.testing.assert_allclose(hess, hessian, rtol=2e-7, atol=2e-7)
    # Integer constants exercise float0 tangents in the public rule.
    gradient = jax.jit(jax.grad(lambda lam: kepler(lam, 0, 0, solver=solver)[0]))(0.7)
    np.testing.assert_allclose(gradient, np.cos(0.7), rtol=0, atol=2e-15)


@pytest.mark.float64_only(reason="finite-difference derivative reference")
@pytest.mark.parametrize("solver", _SOLVERS)
def test_second_derivatives(solver):
    """Central differences validate both AD modes through order two in float64.

    A 2e-5 tolerance covers finite-difference roundoff/truncation for converged
    solvers. Grid uses 3e-3 for O(cell width) slope error from bilinear values;
    its implicit derivatives deliberately approximate the equation's solution.
    """
    tolerance = 3e-3 if isinstance(solver, Grid) else 2e-5
    fun = jax.jit(lambda lam, ex, ey: kepler(lam, ex, ey, solver=solver))
    check_grads(
        fun,
        (jnp.array(1.1), jnp.array(0.2), jnp.array(-0.1)),
        order=2,
        modes=["fwd", "rev"],
        atol=tolerance,
        rtol=tolerance,
    )


@pytest.mark.parametrize("solver", _SOLVERS)
def test_dtype_shapes_and_equal_instance_trace(solver, precision):
    """Dtype, broadcasting and trace count are exact API contracts (zero slack).

    Scalar, vector and matrix broadcasts also agree with independent scalar
    calls within 4 eps, allowing floating operation fusion under JIT.
    """
    traces = []

    @partial(jax.jit, static_argnames=("strategy",))
    def evaluate(lam, ex, ey, strategy):
        traces.append(None)
        return kepler(lam, ex, ey, solver=strategy)

    equal = type(solver)()
    assert solver == equal and hash(solver) == hash(equal)
    assert jax.tree.leaves(solver) == []
    for dtype in dict.fromkeys((np.dtype(np.float32), precision)):
        for shape in ((), (3,), (2, 1)):
            lam = jnp.full(shape, 0.7, dtype=dtype)
            ex = jnp.asarray([0.0, 0.2, 0.5] if shape == (2, 1) else 0.2, dtype)
            ey = jnp.asarray(0.1, dtype)
            before = len(traces)
            result = evaluate(lam, ex, ey, solver)
            again = evaluate(lam, ex, ey, equal)
            assert len(traces) == before + 1
            expected_shape = np.broadcast_shapes(lam.shape, ex.shape, ey.shape)
            assert all(v.dtype == dtype and v.shape == expected_shape for v in result)
            np.testing.assert_array_equal(result, again)
            for index in np.ndindex(expected_shape):
                values = [a[index] for a in np.broadcast_arrays(lam, ex, ey)]
                scalar = kepler(*values, solver=solver)
                np.testing.assert_allclose(
                    np.asarray(result)[(slice(None), *index)],
                    scalar,
                    rtol=0,
                    atol=4 * np.finfo(dtype).eps,
                )
    # Equal closed-over strategies also share the underlying custom function.
    from orbix.kepler.longitude import _longitude_function

    assert _longitude_function(solver) is _longitude_function(equal)
    # Different settings must not reuse the old configuration's trace.
    other = Halley(steps=7) if isinstance(solver, Halley) else Grid(16, 64)
    before = len(traces)
    evaluate(jnp.asarray(0.7), jnp.asarray(0.2), jnp.asarray(0.1), other)
    assert len(traces) == before + 1


def test_abstract_strategy():
    """Python's abstract-method contract rejects incomplete strategies exactly."""
    with pytest.raises(TypeError, match="abstract"):
        AbstractKeplerSolver()
