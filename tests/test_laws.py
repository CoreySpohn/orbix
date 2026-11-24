"""Orbit laws: closed-form densities, the period law, draws and the one-size rule."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import priorities
import pytest
from hwoutils.constants import GM_SUN_AU3_PER_D2 as GM

from orbix.system import (
    ClassicalChart,
    FrameChart,
    FrameOrbit,
    IsotropicInclination,
    OrbitLaw,
)
from orbix.system.laws import _draw

KEY = jax.random.PRNGKey(3)
E_LAW = priorities.Uniform(0.05, 0.8)


def test_exactly_one_size_law():
    """Fails if a law with no size law, or two, can be built."""
    with pytest.raises(ValueError, match="exactly one"):
        OrbitLaw(E_LAW)
    with pytest.raises(ValueError, match="exactly one"):
        OrbitLaw(
            E_LAW,
            a_AU=priorities.LogUniform(1.0, 2.0),
            period_d=priorities.LogUniform(1.0, 2.0),
        )


def test_a_law_density_is_the_closed_form():
    """Fails if a factor of the density against the reference measure is off.

    ``log p_e(e) - log(2 pi e) + log p_a(a) + log a`` at a point, to 1e-13 relative.
    """
    a_law = priorities.LogUniform(0.8, 3.0)
    law = OrbitLaw(E_LAW, a_AU=a_law)
    orbit = FrameOrbit.from_classical(
        jnp.array([1.5]), 0.3, 0.7, 0.4, 0.7, 0.2, mu_AU3_per_d2=GM
    )
    expected = (
        -np.log(0.75)
        - np.log(2 * np.pi * 0.3)
        - np.log(1.5 * np.log(3.0 / 0.8))
        + np.log(1.5)
    )
    np.testing.assert_allclose(law.log_prob(orbit), expected, rtol=1e-13)


def test_a_period_law_is_the_same_law_as_the_size_law_it_implies():
    """Fails if the period Jacobian is not log(3/2) + log P.

    A log-uniform period at fixed mass is a log-uniform size, so the two laws have the
    same density against ``d(ln a)`` on the shared support.
    """
    mu = GM * 1.1
    period = (300.0, 3000.0)
    a_range = [(mu * (p / (2 * np.pi)) ** 2) ** (1 / 3) for p in period]
    by_period = OrbitLaw(E_LAW, period_d=priorities.LogUniform(*period))
    by_size = OrbitLaw(E_LAW, a_AU=priorities.LogUniform(*a_range))
    a = jnp.array(
        [0.9 * a_range[0] + 0.1 * a_range[1], 0.5 * (a_range[0] + a_range[1])]
    )
    orbit = FrameOrbit.from_classical(a, 0.3, 0.7, 0.4, 0.7, 0.2, mu_AU3_per_d2=mu)
    np.testing.assert_allclose(
        by_period.log_prob(orbit), by_size.log_prob(orbit), rtol=1e-12
    )


def test_draws_have_the_requested_batch_the_rate_and_the_law_support():
    """Fails if a draw ignores mu, the epoch or the support of its law.

    Parameters broadcast against ``mu`` to the batch; every draw is bound with the rate
    of its own mass, a unit quaternion and a period inside the law's range.
    """
    law = OrbitLaw(E_LAW, period_d=priorities.LogUniform(300.0, 3000.0))
    mu = GM * jnp.array([0.9, 1.1, 1.3])
    orbit = law.sample(KEY, mu_AU3_per_d2=mu, epoch_d=5.0)
    assert orbit.a_AU.shape == (3,) and orbit.quat.shape == (3, 4)
    np.testing.assert_allclose(orbit.n_rad_per_d, jnp.sqrt(mu / orbit.a_AU**3))
    np.testing.assert_allclose(jnp.linalg.norm(orbit.quat, axis=-1), 1.0)
    assert jnp.all(orbit.epoch_d == 5.0)
    period = 2 * np.pi / orbit.n_rad_per_d
    assert jnp.all((period >= 300.0) & (period <= 3000.0))
    e = jnp.hypot(orbit.ex, orbit.ey)
    assert jnp.all((e >= 0.05) & (e <= 0.8))
    wide = law.sample(KEY, mu_AU3_per_d2=mu[:1])
    assert wide.a_AU.shape == (1,)


def _a_law_spellings():
    """The same Uniform(1, 2) size law for two planets, spelled three ways."""
    one = lambda v: jnp.array([v])  # noqa: E731
    two = lambda v: jnp.array([v, v])  # noqa: E731
    return {
        "scalar": priorities.Uniform(1.0, 2.0),
        "shape1": priorities.Uniform(one(1.0), one(2.0)),
        "shape2": priorities.Uniform(two(1.0), two(2.0)),
    }


@pytest.mark.parametrize("spelling", ["scalar", "shape1", "shape2"])
def test_each_planet_gets_its_own_draw_whatever_the_parameter_shape(spelling):
    """Fails if a size law of shape (1,) shares one draw among the planets.

    Truth: independent continuous variates are equal with probability zero, and their
    sample correlation has standard error 1/sqrt(N); the band is 3 standard errors at
    N = 4096 (false rejection about 3 in 1000), with seed 0. Both ``sample`` and
    ``from_unit`` are held to it, and the density of a draw equals the scalar law's.
    No existing test touches broadcast draws, so this is a new function.
    """
    n, band = 4096, 3.0 / np.sqrt(4096)
    law = OrbitLaw(E_LAW, a_AU=_a_law_spellings()[spelling])
    scalar = OrbitLaw(E_LAW, a_AU=priorities.Uniform(1.0, 2.0))
    mu = jnp.full((n, 2), GM)
    key, ckey = jax.random.split(jax.random.PRNGKey(0))
    drawn = law.sample(key, mu_AU3_per_d2=mu)
    unit = law.from_unit(jax.random.uniform(ckey, (n, 2, 6)), mu_AU3_per_d2=mu)
    for orbit in (drawn, unit):
        a = orbit.a_AU
        assert not jnp.any(a[:64, 0] == a[:64, 1])
        assert abs(np.corrcoef(a[:, 0], a[:, 1])[0, 1]) < band
    score = law.log_prob(drawn)
    assert jnp.all(jnp.isfinite(score))
    np.testing.assert_allclose(score, scalar.log_prob(drawn), rtol=1e-12)


def _mixture():
    """Three components on [1, 2], [3, 4], [5, 6]; three, not the planet count."""
    return priorities.Mixture(
        jnp.zeros(3),
        priorities.Uniform(jnp.array([1.0, 3.0, 5.0]), jnp.array([2.0, 4.0, 6.0])),
    )


@pytest.mark.parametrize("size_law", [IsotropicInclination, _mixture])
def test_laws_without_a_planet_axis_still_draw_each_planet_independently(size_law):
    """Fails if a parameter-free law or a mixture's component axis shares one draw.

    A law with no parameters has no leaves to broadcast, and a mixture's component axis
    is not a planet axis. Truth and band as in the test above (3 standard errors at
    N = 4096, seed 0): no equal pair in 64 draws and the correlation of the two planets
    within the band; the mixture must also draw its first and last components. The
    scores of the draws are finite.
    """
    n, band = 4096, 3.0 / np.sqrt(4096)
    law = OrbitLaw(E_LAW, a_AU=size_law())
    a = law.sample(jax.random.PRNGKey(0), mu_AU3_per_d2=jnp.full((n, 2), GM)).a_AU
    assert a.shape == (n, 2)
    assert not jnp.any(a[:64, 0] == a[:64, 1])
    assert abs(np.corrcoef(a[:, 0], a[:, 1])[0, 1]) < band
    assert jnp.all(jnp.isfinite(law.log_prob(law.sample(KEY, mu_AU3_per_d2=GM))))
    if size_law is _mixture:
        assert jnp.any(a < 2.0) and jnp.any(a > 5.0)


@pytest.mark.parametrize("shape", [(64,), (64, 64)])
def test_a_batched_law_is_drawn_once_per_element_not_as_a_cross_product(shape):
    """Fails if drawing a (K,) or (B, K) law at (B, K) generates more than B K variates.

    Truth: one variate per element is 8 B K bytes of float64 output; a draw of the full
    sample shape against every parameter element is K (or B K) times that. The compiled
    temporary storage at (B, K) = (64, 64) is measured at 1.5 times the output (a
    spike); the bound is 4 times, far below the 64-fold cross product. ``_draw`` is
    called directly because it owns the draw; ``OrbitLaw.sample`` adds its own arrays.
    """
    batch = (64, 64)
    dist = priorities.Uniform(jnp.ones(shape), 2.0 * jnp.ones(shape))
    draw = jax.jit(lambda d, k: _draw(d, k, batch))
    stats = draw.lower(dist, KEY).compile().memory_analysis()
    assert stats.temp_size_in_bytes <= 4 * stats.output_size_in_bytes
    assert draw(dist, KEY).shape == batch


def test_a_vector_event_law_is_refused_for_the_shape_and_the_size():
    """Fails if a law with an event axis is accepted as a planet batch.

    A 2-D multivariate normal has event_dim 1; its event axis would be read as two
    planets and the covariance discarded.
    """
    vector = priorities.MultivariateNormal(jnp.zeros(2), jnp.eye(2))
    with pytest.raises(ValueError, match="eccentricity must be a scalar"):
        OrbitLaw(vector, a_AU=priorities.Uniform(1.0, 2.0))
    with pytest.raises(ValueError, match="a_AU must be a scalar"):
        OrbitLaw(E_LAW, a_AU=vector)
    with pytest.raises(ValueError, match="period_d must be a scalar"):
        OrbitLaw(E_LAW, period_d=vector)


@pytest.mark.parametrize("chart", [ClassicalChart(), FrameChart(), FrameChart(None)])
def test_sample_coordinates_invert_to_the_drawn_orbit(chart):
    """Fails if the coordinates drawn for an orbit do not rebuild that orbit.

    Positions agree to 1e-10 AU (float64 trigonometry through the chart), and a plain
    angle lies in [0, 2 pi).
    """
    law = OrbitLaw(E_LAW, a_AU=priorities.LogUniform(0.8, 3.0))
    mu = jnp.full((4,), GM)
    orbit = law.sample(KEY, mu_AU3_per_d2=mu)
    coords = chart.sample_coordinates(jax.random.PRNGKey(9), orbit)
    rebuilt = chart.to_orbit(coords, mu_AU3_per_d2=mu)
    times = jnp.array([0.0, 40.0])
    np.testing.assert_allclose(rebuilt.at(times)[0], orbit.at(times)[0], atol=1e-10)
    if isinstance(chart, ClassicalChart):
        assert jnp.all((coords[..., 3:] >= 0) & (coords[..., 3:] < 2 * np.pi))


@pytest.mark.parametrize(
    "law",
    [
        OrbitLaw(priorities.Uniform(0.0, 0.8), a_AU=priorities.LogUniform(0.8, 3.0)),
        OrbitLaw(
            priorities.Beta(0.867, 3.03), period_d=priorities.LogUniform(300.0, 3000.0)
        ),
    ],
    ids=["a", "period"],
)
def test_a_zero_of_the_law_wins_over_a_singular_factor(law):
    """Fails if a circular orbit reads +inf or an infinite size reads NaN.

    The exactly circular orbit has no density (``-inf``, not ``+inf`` from ``-log 0``),
    and an orbit of infinite size, where the size law is zero and the volume factor is
    infinite, has none either (``-inf``, not ``-inf + inf = NaN``).
    """
    mu = GM
    circular = FrameOrbit.from_classical(
        jnp.array([1.5]), 0.0, 0.7, 0.4, 0.7, 0.2, mu_AU3_per_d2=mu
    )
    assert float(law.log_prob(circular)[0]) == -np.inf
    eccentric = FrameOrbit.from_classical(
        jnp.array([1.5]), 0.3, 0.7, 0.4, 0.7, 0.2, mu_AU3_per_d2=mu
    )
    assert np.isfinite(float(law.log_prob(eccentric)[0]))
    huge = eqx.tree_at(
        lambda o: (o.a_AU, o.n_rad_per_d),
        eccentric,
        (jnp.array([jnp.inf]), jnp.array([0.0])),
    )
    assert float(law.log_prob(huge)[0]) == -np.inf
