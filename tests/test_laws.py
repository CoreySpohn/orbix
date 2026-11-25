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


@pytest.mark.float64_only(reason="orbital inference coordinates and log densities")
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


@pytest.mark.float64_only(reason="orbital inference coordinates and log densities")
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


class _Spy(priorities.AbstractDistribution):
    """A law that records every ``sample_shape`` it is asked for, then delegates."""

    inner: priorities.AbstractDistribution
    calls: list = eqx.field(static=True)
    event_dim = 0
    bounds = "interval"
    has_icdf = True

    def support(self):
        return self.inner.support()

    def log_prob(self, value):
        return self.inner.log_prob(value)

    def sample(self, key, sample_shape=()):
        self.calls.append(tuple(sample_shape))
        return self.inner.sample(key, sample_shape)

    def cdf(self, value):
        return self.inner.cdf(value)

    def icdf(self, q):
        return self.inner.icdf(q)


class _Tuple(priorities.AbstractDistribution):
    """A custom law whose bounds are static tuples, one per planet, and a spare leaf.

    It follows the shape rule ``sample_shape + batch`` (batch ``(2,)``) unless
    ``obey`` is false, in which case it ignores ``sample_shape``.
    """

    spare: jax.Array
    lows: tuple = eqx.field(static=True, default=(1.0, 10.0))
    highs: tuple = eqx.field(static=True, default=(2.0, 20.0))
    obey: bool = eqx.field(static=True, default=True)
    event_dim = 0
    bounds = "interval"
    has_icdf = True

    def support(self):
        return jnp.array(self.lows), jnp.array(self.highs)

    def log_prob(self, value):
        low, high = self.support()
        return -jnp.log(high - low) + 0.0 * value

    def sample(self, key, sample_shape=()):
        low, high = self.support()
        shape = (tuple(sample_shape) if self.obey else ()) + low.shape
        return jax.random.uniform(key, shape, minval=low, maxval=high)

    def cdf(self, value):
        low, high = self.support()
        return (value - low) / (high - low)

    def icdf(self, q):
        low, high = self.support()
        return low + q * (high - low)


@pytest.mark.parametrize("own", [(), (1,), (64,), (64, 1), (64, 64)])
def test_a_batched_law_is_drawn_once_per_element_not_as_a_cross_product(own):
    """Fails if a law of batch ``own`` is drawn at (64, 64) with extra variates.

    Truth: one variate per element is a count, independent of the compiler. The
    recording law must be asked exactly once, and the variates requested (the product
    of the sample shape and the law's batch) must be 64 * 64 = 4096. A draw of the
    whole batch followed by a diagonal asks for 64 * 64 * prod(own).
    """
    calls = []
    inner = priorities.Uniform(jnp.ones(own), 2.0 * jnp.ones(own))
    out = _draw(_Spy(inner, calls), KEY, (64, 64))
    assert out.shape == (64, 64)
    assert len(calls) == 1
    assert np.prod(calls[0]) * np.prod(own) == 4096


@pytest.mark.parametrize("own", [(3, 1), (1, 2)])
def test_each_axis_of_a_batched_law_lands_where_the_law_carries_it(own):
    """Fails if a drawn axis is moved or squeezed onto the wrong planet axis.

    A law of batch (3, 1) with three disjoint supports, or (1, 2) with two, drawn at
    (3, 2): every element must lie in the support of its own row, or its own column.
    """
    low = jnp.arange(np.prod(own), dtype=float).reshape(own) * 10.0
    out = _draw(priorities.Uniform(low, low + 1.0), KEY, (3, 2))
    expected = jnp.broadcast_to(low, (3, 2))
    assert jnp.all((out >= expected) & (out <= expected + 1.0))


@pytest.mark.parametrize("jit", [False, True])
def test_a_custom_law_with_static_bounds_draws_each_planet_independently(jit):
    """Fails if a law whose per-planet bounds are not array leaves shares one draw.

    Three systems of two planets: six distinct continuous sizes (equal with
    probability zero), each inside its own planet's support, eager and compiled.
    """
    law = OrbitLaw(E_LAW, a_AU=_Tuple(jnp.zeros(3)))
    sample = lambda k, mu: law.sample(k, mu_AU3_per_d2=mu).a_AU  # noqa: E731
    sample = eqx.filter_jit(sample) if jit else sample
    a = sample(KEY, jnp.full((3, 2), GM))
    assert a.shape == (3, 2) and len(set(np.asarray(a).ravel())) == 6
    assert jnp.all((a[:, 0] >= 1.0) & (a[:, 0] <= 2.0))
    assert jnp.all((a[:, 1] >= 10.0) & (a[:, 1] <= 20.0))


class _Spare(priorities.AbstractDistribution):
    """A scalar uniform law on [1, 2] that carries an unrelated (3,) array leaf."""

    spare: jax.Array
    event_dim = 0
    bounds = "interval"
    has_icdf = True

    def support(self):
        return jnp.asarray(1.0), jnp.asarray(2.0)

    def log_prob(self, value):
        return jnp.zeros_like(value)

    def sample(self, key, sample_shape=()):
        return jax.random.uniform(key, tuple(sample_shape), minval=1.0, maxval=2.0)

    def cdf(self, value):
        return value - 1.0

    def icdf(self, q):
        return 1.0 + q


def test_a_spare_array_leaf_of_a_law_is_not_a_batch_axis():
    """Fails if a non-parameter array leaf is broadcast to the target batch.

    A law with a (3,) leaf outside its support is drawn at (4, 2) under jit: shape
    (4, 2) and eight distinct variates (equal with probability zero).
    """
    draw = eqx.filter_jit(lambda d, k: _draw(d, k, (4, 2)))
    out = draw(_Spare(jnp.zeros(3)), KEY)
    assert out.shape == (4, 2) and len(set(np.asarray(out).ravel())) == 8


def test_a_law_that_ignores_the_sample_shape_is_refused():
    """Fails if a law off the shape rule is broadcast instead of refused.

    Its (2,) draw would be shared by the three systems; the draw must raise.
    """
    law = OrbitLaw(E_LAW, a_AU=_Tuple(jnp.zeros(3), obey=False))
    with pytest.raises(ValueError, match="sample returned shape"):
        law.sample(KEY, mu_AU3_per_d2=jnp.full((3, 2), GM))


def test_a_mixture_with_per_planet_weights_and_shared_components_is_refused():
    """Fails if the shape-rule failure is not reported by law name and rule.

    Weights (2, 2) over components of batch (2,) make ``Mixture.sample`` break the
    shape rule inside priorities. This case becomes an independence test when
    priorities draws mixtures over the joint batch of weights and components.
    """
    mix = priorities.Mixture(
        jnp.zeros((2, 2)),
        priorities.Uniform(jnp.array([1.0, 3.0]), jnp.array([2.0, 4.0])),
    )
    law = OrbitLaw(E_LAW, a_AU=mix)
    with pytest.raises(ValueError, match="must return draws of shape sample_shape"):
        law.sample(KEY, mu_AU3_per_d2=jnp.full((3, 2), GM))


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


@pytest.mark.float64_only(reason="orbital inference coordinates and log densities")
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


@pytest.mark.float64_only(reason="orbital inference coordinates and log densities")
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
