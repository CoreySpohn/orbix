"""Tests for orbix grid-search utilities."""

import jax
import jax.numpy as jnp

from orbix.fitting.data import AstromData
from orbix.fitting.forward import predict_astrometry
from orbix.fitting.grid_search import ParamBounds
from orbix.fitting.likelihoods import loglike_astrom
from orbix.utils.quasi_random import roberts_sequence

TWO_PI = 2.0 * jnp.pi
MSUN = 1.98892e30


def test_roberts_shape_and_range():
    """Roberts sequence has correct shape and values in [0, 1)."""
    pts = roberts_sequence(1000, 3)
    assert pts.shape == (1000, 3)
    assert jnp.all((pts >= 0.0) & (pts < 1.0))


def test_roberts_low_discrepancy_1d_margin():
    """Max gap in sorted 1D projection is within 10x mean gap."""
    pts = roberts_sequence(2000, 2)
    xs = jnp.sort(pts[:, 0])
    gaps = jnp.diff(xs)
    assert float(jnp.max(gaps)) < 10.0 * float(jnp.mean(gaps))


def test_parambounds_scale_unit_cube():
    """ParamBounds.scale maps unit-cube corners and midpoint correctly."""
    b = ParamBounds(
        names=("logT", "cos_i"),
        low=jnp.array([0.0, -1.0]),
        high=jnp.array([4.0, 1.0]),
    )
    u = jnp.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    phys = b.scale(u)
    assert phys.shape == (3, 2)
    assert jnp.allclose(phys[0], jnp.array([0.0, -1.0]))
    assert jnp.allclose(phys[1], jnp.array([4.0, 1.0]))
    assert jnp.allclose(phys[2], jnp.array([2.0, 0.0]))


def test_eccvector_shape_maps_to_physical():
    """EccVectorShape converts unit-cube midpoint to expected physical values."""
    from orbix.fitting.grid_search import EccVectorShape

    shape = EccVectorShape()
    bounds = shape.default_bounds(log_T_range=(2.0, 3.0), e_max=0.5)
    u = jnp.full((1, len(bounds.names)), 0.5)
    phys = shape.to_physical(u, bounds, Ms=MSUN)
    for kkey in ("a", "e", "cos_i", "W", "cos_w", "sin_w", "tp"):
        assert kkey in phys
        assert phys[kkey].shape == (1,)
    # ex=ey=0 at midpoint of symmetric (-e_max, e_max) -> e=0, cos_w=1, sin_w=0
    assert float(phys["e"][0]) == 0.0
    assert jnp.allclose(phys["cos_w"][0], 1.0)
    assert jnp.allclose(phys["sin_w"][0], 0.0)
    # e <= e_max always
    u2 = jnp.array([[0.5, 1.0, 1.0, 0.5, 0.5, 0.5]])
    e2 = float(shape.to_physical(u2, bounds, Ms=MSUN)["e"][0])
    assert 0.0 <= e2 < 1.0


def test_ais_stage1_fills_unit_cube():
    """AdaptiveImportanceSampler.stage1 produces valid unit-cube points."""
    from orbix.fitting.grid_search import AdaptiveImportanceSampler

    s = AdaptiveImportanceSampler()
    u = s.stage1(jax.random.PRNGKey(0), ndim=6, n=4096)
    assert u.shape == (4096, 6)
    assert jnp.all((u >= 0.0) & (u < 1.0))


def _toy_astrom():
    """Build a small AstromData fixture from a known orbit."""
    t = jnp.array([0.0, 120.0, 240.0])
    a, e, cos_i, W = 1.0, 0.1, 0.7, 0.5
    cos_w, sin_w, tp = 1.0, 0.0, 30.0
    ra, dec = predict_astrometry(t, a, e, cos_i, W, cos_w, sin_w, tp, MSUN, 10.0)
    err = jnp.full(3, 1e-3)
    return AstromData(
        times=t,
        ra=ra,
        dec=dec,
        ra_err=err,
        dec_err=err,
        corr=jnp.zeros(3),
        planet_id=jnp.zeros(3, int),
        is_valid=jnp.ones(3, bool),
    )


def test_evaluator_matches_direct_loglike():
    """build_evaluator returns a function that matches loglike_astrom directly."""
    from orbix.fitting.grid_search import EccVectorShape, build_evaluator

    data = _toy_astrom()
    shape = EccVectorShape()
    ev = build_evaluator((data,), Ms=MSUN, dist_pc=10.0, shape=shape)
    phys = {
        "a": jnp.array(1.0),
        "e": jnp.array(0.1),
        "cos_i": jnp.array(0.7),
        "W": jnp.array(0.5),
        "cos_w": jnp.array(1.0),
        "sin_w": jnp.array(0.0),
        "tp": jnp.array(30.0),
    }
    ra, dec = predict_astrometry(
        data.times,
        phys["a"],
        phys["e"],
        phys["cos_i"],
        phys["W"],
        phys["cos_w"],
        phys["sin_w"],
        phys["tp"],
        MSUN,
        10.0,
    )
    assert jnp.allclose(ev(phys), loglike_astrom(ra, dec, data))


def test_batched_loglike_matches_unfused():
    """batched_loglike via scan+vmap matches direct vmap over all particles."""
    from orbix.fitting.grid_search import (
        AdaptiveImportanceSampler,
        EccVectorShape,
        batched_loglike,
        build_evaluator,
    )

    data = _toy_astrom()
    shape = EccVectorShape()
    ev = build_evaluator((data,), Ms=MSUN, dist_pc=10.0, shape=shape)
    bounds = shape.default_bounds(log_T_range=(2.0, 3.0), e_max=0.5)
    u = AdaptiveImportanceSampler().stage1(jax.random.PRNGKey(1), 6, 200)
    phys = shape.to_physical(u, bounds, Ms=MSUN)
    fused = batched_loglike(ev, phys, n_particles=200, chunk_size=50)
    ref = jax.vmap(ev)({k: v for k, v in phys.items()})
    assert fused.shape == (200,)
    assert jnp.allclose(fused, ref, atol=1e-5)


def test_stage2_returns_samples_and_logq():
    """stage2 returns unit-cube samples and finite log-densities."""
    from orbix.fitting.grid_search import AdaptiveImportanceSampler

    s = AdaptiveImportanceSampler(n_modes=3)
    survivors = jax.random.uniform(jax.random.PRNGKey(2), (50, 6))
    z, log_q = s.stage2(jax.random.PRNGKey(3), survivors, n=100)
    assert z.shape == (100, 6)
    assert log_q.shape == (100,)
    assert jnp.all(jnp.isfinite(log_q))
