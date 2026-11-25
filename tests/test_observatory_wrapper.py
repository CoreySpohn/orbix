"""Tests for orbix.Observatory composition wrapper."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.epoch import Epoch
from hwoutils.testing import ulps

from orbix.observatory import Observatory, ObservatoryL2Halo, is_observable

_EPOCH = Epoch(2460000.0, 575.75)
_compiled_interval = eqx.filter_jit(Observatory.observability)


def _interval(obs, ra, dec, epoch, start, stop, *, n_samples=64):
    """Keep numerical arguments dynamic so interval scenarios share a compile."""
    dtype = jnp.asarray(obs.keepout_min_deg).dtype
    ra, dec, start, stop = (
        jnp.asarray(value, dtype=dtype) for value in (ra, dec, start, stop)
    )
    return _compiled_interval(obs, ra, dec, epoch, start, stop, n_samples=n_samples)


@pytest.fixture(scope="module")
def keepout_observatory():
    """Reuse one halo structure, with an explicit 45-degree solar keepout."""
    return Observatory(
        ObservatoryL2Halo.from_default(),
        keepout_min_deg=jnp.asarray(45.0),
    )


def test_observatory_defaults():
    """Bare Observatory(orbit=...) uses documented scalar defaults."""
    orbit = ObservatoryL2Halo.from_default()
    obs = Observatory(orbit=orbit)

    assert obs.orbit is orbit
    assert obs.settling_time_d == 1.0
    assert obs.keepout_min_deg == 0.0
    assert obs.keepout_max_deg == 180.0


def test_observatory_custom_fields():
    """Scalar fields accept overrides via keyword arguments."""
    orbit = ObservatoryL2Halo.from_default()
    obs = Observatory(
        orbit=orbit,
        settling_time_d=2.0,
    )
    assert obs.settling_time_d == 2.0


def test_observatory_is_pytree():
    """Observatory flattens and unflattens cleanly as a JAX pytree."""
    orbit = ObservatoryL2Halo.from_default()
    obs = Observatory(orbit=orbit, settling_time_d=2.0)

    leaves, treedef = jax.tree_util.tree_flatten(obs)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.settling_time_d == 2.0


def test_observatory_delegates_orbit_method():
    """Observatory.orbit retains the full ObservatoryL2Halo API surface."""
    orbit = ObservatoryL2Halo.from_default()
    obs = Observatory(orbit=orbit)

    pos = obs.orbit.position_ecliptic(_EPOCH, 10.0)
    assert pos.shape == (3,)


def test_is_observable_at_matches_scalar_calls(keepout_observatory):
    """Fails if epoch conversion, broadcasting, or configured keepouts drift.

    Truth: scalar per-element calls of the epoch-relative API; boolean equality
    is exact (no tolerance).
    The split epoch and nonzero/negative day offsets detect lost epoch parts.
    """
    obs = keepout_observatory
    ra = jnp.array([[0.0], [100.0]])
    dec = jnp.array([0.0, 17.0, 63.0])
    time = jnp.array([-30.25, 95.125, 400.75])
    expected = jnp.stack(
        [
            jnp.stack(
                [
                    is_observable(
                        obs.orbit.position_ecliptic(_EPOCH, time[k]),
                        r,
                        dec[k],
                        _EPOCH,
                        time[k],
                        ko_sun_min_deg=obs.keepout_min_deg,
                        ko_sun_max_deg=obs.keepout_max_deg,
                    )
                    for k in range(3)
                ]
            )
            for r in ra[:, 0]
        ]
    )
    np.testing.assert_array_equal(obs.is_observable_at(ra, dec, _EPOCH, time), expected)
    np.testing.assert_array_equal(
        jax.jit(
            lambda r, d, epoch, t: obs.is_observable_at(r, d, epoch, t),
            static_argnames="epoch",
        )(ra, dec, _EPOCH, time),
        expected,
    )
    assert obs.is_observable_at(0.0, 0.0, _EPOCH, 0.0).shape == ()
    np.testing.assert_array_equal(
        obs.is_observable_at(ra, dec, _EPOCH, time.reshape(1, 1, 3)),
        expected[None],
    )


def test_annual_observability_against_dense_grid(keepout_observatory):
    """Fails if duration uses sample counts or refined boundaries are misplaced.

    Truth: 10^4 direct per-time decisions over a year for an ecliptic target.
    With 16 coarse samples and 8 bisections, delta=365.25/(15*256) d.
    Each midpoint errs by delta/2. Dense first/last samples err by at most
    h=365.25/9999 d < delta/2, so delta bounds their disagreement.
    Fraction tolerance sums delta/2+h/2 per transition, divided by the span.
    The coarse grid divides its span and scales the sample index, giving
    two rounding operations at the scale of the year.
    """
    obs = keepout_observatory
    year = 365.25
    result = _interval(obs, 180.0, 0.0, _EPOCH, 0.0, year, n_samples=16)
    times = jnp.linspace(0.0, year, 10000)
    direct = jax.jit(
        jax.vmap(
            lambda t: is_observable(
                obs.orbit.position_ecliptic(_EPOCH, t),
                180.0,
                0.0,
                _EPOCH,
                t,
                ko_sun_min_deg=45.0,
            )
        )
    )(times)
    dense = np.asarray(direct)
    times = np.asarray(times)
    transitions = np.count_nonzero(dense[1:] != dense[:-1])
    assert transitions == 2
    assert not dense[0] and not dense[-1]
    delta = year / (15 * 256)
    spacing = year / 9999
    dense_fraction = np.trapezoid(dense.astype(float), times) / year
    np.testing.assert_allclose(
        result.fraction,
        dense_fraction,
        rtol=0,
        atol=transitions * (delta + spacing) / (2 * year),
    )
    np.testing.assert_allclose(result.first_d, times[dense][0], rtol=0, atol=delta)
    np.testing.assert_allclose(result.last_d, times[dense][-1], rtol=0, atol=delta)
    np.testing.assert_allclose(
        result.time_d,
        np.linspace(0.0, year, 16),
        rtol=0,
        atol=float(ulps(2, result.time_d[-1])),
    )
    np.testing.assert_array_equal(
        result.is_observable,
        obs.is_observable_at(180.0, 0.0, _EPOCH, result.time_d),
    )


def test_full_and_empty_observability(keepout_observatory):
    """Fails if always/never visible intervals lose endpoint or sentinel semantics.

    Truth: the ecliptic pole stays outside a 45-degree solar keepout;
    RA=180 deg, Dec=0 is Sunward for the first ten days of this epoch.
    Constant decisions, fractions, endpoints, and sentinels are exact.
    """
    obs = keepout_observatory
    pole = _interval(obs, 270.0, 66.56, _EPOCH, 0.0, 365.25)
    assert pole.all and pole.any
    assert pole.fraction == 1.0
    assert pole.first_d == 0.0 and pole.last_d == 365.25
    blocked = _interval(obs, 180.0, 0.0, _EPOCH, 0.0, 10.0)
    assert not blocked.any and not blocked.all
    assert blocked.fraction == 0.0
    assert blocked.first_d == np.inf and blocked.last_d == -np.inf
    default = eqx.tree_at(lambda o: o.keepout_min_deg, obs, jnp.asarray(0.0))
    unrestricted = _interval(default, 0.0, 0.0, _EPOCH, 0.0, 365.25)
    assert unrestricted.all and unrestricted.fraction == 1.0
    instant = _interval(obs, 180.0, 0.0, _EPOCH, 0.0, 0.0)
    assert instant.fraction == 0.0 and instant.first_d == np.inf
    with pytest.raises(ValueError, match="n_samples"):
        obs.observability(0.0, 0.0, _EPOCH, 0.0, 1.0, n_samples=1)


def test_observability_jit_and_vmap(keepout_observatory):
    """Fails if interval summaries change under tracing or target vectorization.

    Truth: an eager scalar call and a scalar loop, including a target with
    no transitions. Three structures cover 16/64-point grids and 3 targets.
    Grid decisions/sentinels are exact; finite times allow 4e-11 d for x64
    fusion at MJD ~60000 (a few MJD ULPs). Fraction allows that duration
    roundoff divided by 365.25 d. Boundary brackets must stay unchanged.
    """
    obs = keepout_observatory
    ras = jnp.array([0.0, 75.0, 270.0])
    decs = jnp.array([0.0, 11.5, 66.56])
    results = [
        _interval(obs, r, d, _EPOCH, 0.0, 365.25)
        for r, d in zip(ras, decs, strict=True)
    ]
    eager_first = obs.observability(ras[0], decs[0], _EPOCH, 0.0, 365.25)
    mapped = jax.vmap(lambda r, d: _interval(obs, r, d, _EPOCH, 0.0, 365.25))(ras, decs)
    for k, eager in enumerate(results):
        alternatives = [jax.tree.map(lambda x, k=k: x[k], mapped)]
        if k == 0:
            alternatives.append(eager_first)
        for actual in alternatives:
            for field in ("is_observable", "any", "all"):
                np.testing.assert_array_equal(
                    getattr(actual, field), getattr(eager, field)
                )
            for field in ("time_d", "first_d", "last_d", "fraction"):
                np.testing.assert_allclose(
                    getattr(actual, field),
                    getattr(eager, field),
                    rtol=0,
                    atol=4e-11 / 365.25 if field == "fraction" else 4e-11,
                )
