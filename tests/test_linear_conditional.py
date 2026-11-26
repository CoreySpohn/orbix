"""The sky design matrix and the Gaussian conditional of the linear constants."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.numerics import roundoff
from hwoutils.precision import PrecisionError
from scipy.stats import multivariate_normal

from orbix.system import (
    FrameOrbit,
    fit_linear_map,
    linear_gaussian_conditional,
    sky_design,
)

RNG = np.random.default_rng(7)
EPS = np.finfo(float).eps


def _orbit():
    return FrameOrbit.from_classical(1.3, 0.3, 0.7, 0.4, 0.7, 0.2, mu_AU3_per_d2=1.0)


def _records(count=6):
    orbit = _orbit()
    time = jnp.linspace(0.0, 0.9 * orbit.period_d, count)
    design = sky_design(orbit, time)
    truth = jnp.array([0.4, -0.3, 0.8, 0.5])
    # Correlated, unequal noise: a precision matrix per record.
    cov = np.zeros((count, 2, 2))
    for t in range(count):
        s = RNG.uniform(0.01, 0.05, 2)
        rho = RNG.uniform(-0.6, 0.6)
        cov[t] = [[s[0] ** 2, rho * s[0] * s[1]], [rho * s[0] * s[1], s[1] ** 2]]
    noise = np.stack([RNG.multivariate_normal(np.zeros(2), c) for c in cov])
    target = np.asarray(design @ truth) + noise
    return (
        design,
        jnp.asarray(target),
        jnp.asarray(np.linalg.inv(cov)),
        cov,
        orbit,
        time,
    )


def _direct(design, target, cov, prior_precision, active=None):
    """The marginal and conditional by the textbook solve: no shared code."""
    design, target = np.asarray(design), np.asarray(target)
    keep = np.arange(design.shape[0]) if active is None else np.flatnonzero(active)
    h = design[keep].reshape(-1, design.shape[-1])
    d = target[keep].reshape(-1)
    n = d.size
    s = np.zeros((n, n))
    for i, t in enumerate(keep):
        s[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = cov[t]
    prior_cov = np.linalg.inv(prior_precision)
    marginal = multivariate_normal(np.zeros(n), s + h @ prior_cov @ h.T).logpdf(d)
    precision = np.linalg.inv(s)
    information = prior_precision + h.T @ precision @ h
    mean = np.linalg.solve(information, h.T @ precision @ d)
    return mean, information, marginal


def test_sky_design_reproduces_the_projected_positions():
    """Fails if the design matrix or the constants' ordering is wrong.

    Each projected coordinate sums two nonzero products. Each term can round
    in its multiply and again in the sum, on each of the two paths. The sum
    of absolute products in AU bounds the scale even when the terms cancel.
    """
    orbit = _orbit()
    time = jnp.linspace(-30.0, 200.0, 7)
    (B, G), (A, F) = np.asarray(orbit.linear_map[:2])
    design = sky_design(orbit, time)
    constants = jnp.array([A, B, F, G])
    predicted = design @ constants
    scale = jnp.max(jnp.abs(design) @ jnp.abs(constants))
    bound = roundoff(scale, n_ops=4)  # tol: floating-point, two roundings per path
    np.testing.assert_allclose(predicted, orbit.at(time)[0][:, :2], atol=bound, rtol=0)
    with pytest.raises(ValueError, match="unbatched"):
        sky_design(orbit, jnp.zeros((2, 2)))


@pytest.mark.float64_only(reason="Gaussian fit coefficients, precision and evidence")
@pytest.mark.parametrize("prior", ["scalar", "diagonal", "matrix"])
def test_conditional_matches_a_direct_numpy_solve(prior):
    """Fails if the mean, the precision factor or the evidence is wrong.

    Reference: the mean from ``solve(P, b)`` and the log marginal from the Gaussian
    density of the stacked data with covariance ``S + H prior_cov H^T``, written with
    numpy and scipy only. Tolerance 1e-9 relative for float64 solves of a system
    with condition number below 1e4 (checked in the test).
    """
    design, target, precision, cov, *_ = _records()
    lam = {"scalar": 4.0, "diagonal": np.array([1.0, 2.0, 3.0, 4.0])}.get(
        prior, np.diag([1.0, 2.0, 3.0, 4.0]) + 0.3
    )
    full = np.broadcast_to(lam, (4,)) if np.ndim(lam) < 2 else lam
    full = np.diag(full) if np.ndim(full) < 2 else full
    mean, chol, log_marginal = linear_gaussian_conditional(
        design, target, precision, lam
    )
    ref_mean, information, ref_marginal = _direct(design, target, cov, full)
    assert np.linalg.cond(information) < 1e4
    np.testing.assert_allclose(mean, ref_mean, rtol=1e-9, atol=0)
    np.testing.assert_allclose(chol @ chol.T, information, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(log_marginal, ref_marginal, rtol=1e-9, atol=0)


@pytest.mark.float64_only(reason="Gaussian fit coefficients, precision and evidence")
def test_a_zero_precision_record_is_inactive():
    """Fails if a masked record changes the mean, the factor or the evidence.

    The masked target is set to a huge number: it must have no effect, and the result
    must equal the direct solve on the other records (1e-9 relative).
    """
    design, target, precision, cov, *_ = _records()
    active = np.array([True, True, False, True, True, True])
    target = target.at[2].set(1e9)
    precision = precision.at[2].set(0.0)
    mean, chol, log_marginal = linear_gaussian_conditional(
        design, target, precision, 2.0
    )
    ref_mean, information, ref_marginal = _direct(
        design, target, cov, 2.0 * np.eye(4), active
    )
    np.testing.assert_allclose(mean, ref_mean, rtol=1e-9, atol=0)
    np.testing.assert_allclose(chol @ chol.T, information, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(log_marginal, ref_marginal, rtol=1e-9, atol=0)


@pytest.mark.float64_only(reason="Gaussian fit coefficients, precision and evidence")
def test_flat_prior_limit_is_the_least_squares_fit_of_fit_linear_map():
    """Fails if the conditional mean departs from the flat-prior fit.

    With a prior precision of 1e-12 the mean equals ``fit_linear_map``'s constants to
    1e-7 relative to their size (the prior shrinks by about 1e-12 over the data
    precision, which is at least 1e2 here), and the covariance ``P^-1`` equals its
    covariance to the same level.
    """
    design, target, precision, cov, orbit, time = _records()
    mean, chol, _ = linear_gaussian_conditional(design, target, precision, 1e-12)
    errors = (
        jnp.asarray(np.sqrt(cov[:, 0, 0])),
        jnp.asarray(np.sqrt(cov[:, 1, 1])),
        jnp.asarray(cov[:, 0, 1] / np.sqrt(cov[:, 0, 0] * cov[:, 1, 1])),
    )
    block, covariance, _, rank = fit_linear_map(
        orbit, time, target[:, 0], target[:, 1], errors
    )
    assert rank == 4
    ref = jnp.array([block[1, 0], block[0, 0], block[1, 1], block[0, 1]])
    np.testing.assert_allclose(mean, ref, rtol=1e-7, atol=1e-9)
    inverse = np.linalg.inv(np.asarray(chol @ chol.T))
    np.testing.assert_allclose(inverse, covariance, rtol=1e-6, atol=1e-12)


@pytest.mark.float64_only(reason="Gaussian fit coefficients, precision and evidence")
def test_conditional_is_differentiable_and_jits():
    """Fails if the evidence has a non-finite gradient in the design or the data."""
    design, target, precision, *_ = _records()
    f = jax.jit(
        jax.grad(
            lambda h, d: linear_gaussian_conditional(h, d, precision, 1.0)[2], (0, 1)
        )
    )
    grads = f(design, target)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads)


def test_fit_precision_refusals():
    """Both curvature fits refuse a float32 forward, even when float64 is available."""
    from hwoutils.leaves import canonical

    orbit = canonical(_orbit(), precision="float32")
    x = jnp.zeros(2, dtype=jnp.float32)
    with pytest.raises(PrecisionError, match="fit_linear_map requires float64"):
        fit_linear_map(orbit, x, x, x, jnp.eye(2))
    design = jnp.ones((1, 1, 1), dtype=jnp.float32)
    with pytest.raises(
        PrecisionError, match="linear_gaussian_conditional requires float64"
    ):
        linear_gaussian_conditional(design, jnp.zeros((1, 1)), jnp.ones((1, 1, 1)), 1.0)


@pytest.mark.float64_only(reason="Gaussian fit coefficients, precision and evidence")
def test_fits_widen_observations_before_arithmetic():
    """Exact widening of data and noise gives the same fits as an explicit cast."""
    from hwoutils.leaves import canonical

    design, target, precision, cov, orbit, time = _records()
    inputs = canonical((target, precision, 3.0), precision="float32")
    wide = canonical(inputs, precision="float64")
    for actual, expected in zip(
        linear_gaussian_conditional(design, *inputs),
        linear_gaussian_conditional(design, *wide),
        strict=True,
    ):
        assert actual.dtype == np.float64
        np.testing.assert_array_equal(actual, expected)

    for noise in (cov, (0.02, 0.04, 0.25)):
        inputs = canonical(
            (time, target[:, 0], target[:, 1], noise), precision="float32"
        )
        wide = canonical(inputs, precision="float64")
        for actual, expected in zip(
            fit_linear_map(orbit, *inputs), fit_linear_map(orbit, *wide), strict=True
        ):
            np.testing.assert_array_equal(actual, expected)
