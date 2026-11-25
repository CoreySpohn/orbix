"""The unit-cube map of orbit blocks: same pushforward as ``sample``, deterministic."""

import jax
import jax.numpy as jnp
import numpy as np
import priorities
import pytest
from hwoutils.constants import GM_SUN_AU3_PER_D2 as GM
from scipy.stats import ks_1samp, ks_2samp, uniform
from test_blocks import (
    A_LAW,
    P_LAW,
    TI_A_LAW,
    TI_P_LAW,
    _block,
    _system_model,
    _ti_block,
    _TiltedLaw,
    chart_coordinates,
)

from orbix.system import (
    ClassicalChart,
    EccentricityVectorChart,
    FrameChart,
    OrbitBlock,
    OrbitLaw,
)

N = 40_000
ALPHA = 1e-3

CASES = {
    "classical-a": (ClassicalChart(), A_LAW, False),
    "classical-period": (ClassicalChart(), P_LAW, False),
    "eccentricity-vector": (EccentricityVectorChart(), A_LAW, False),
    "frame-gauged": (FrameChart(), A_LAW, False),
    "frame-ungauged": (FrameChart(gauge_sd=None), A_LAW, False),
    "thiele-innes-a": (None, TI_A_LAW, True),
    "thiele-innes-period": (None, TI_P_LAW, True),
}


def _make(name):
    chart, law, ti = CASES[name]
    return _ti_block(law) if ti else _block(chart, (1,), law)


def _two_samples(block, model):
    """``from_unit`` of uniform cube points and ``sample`` draws, as ``(N, n)``."""
    cube = jax.random.uniform(jax.random.PRNGKey(3), (N, block.unit_size))
    mapped = jax.jit(jax.vmap(lambda c: block.from_unit(c, model)))(cube)
    drawn = jax.jit(jax.vmap(lambda k: block.sample(k, model)))(
        jax.random.split(jax.random.PRNGKey(4), N)
    )
    names = block.coordinate_names
    stack = lambda out: np.asarray(jnp.stack([out[n][:, 0] for n in names], -1))  # noqa: E731
    return stack(mapped), stack(drawn)


@pytest.mark.float64_only(reason="orbital inference coordinates and log densities")
@pytest.mark.parametrize("name", list(CASES))
def test_from_unit_and_sample_have_one_distribution(name):
    """Fails if a coordinate of the cube map differs in law from ``sample``.

    Two-sample Kolmogorov-Smirnov on every coordinate, N = 40,000 each, p-value above
    1e-3 (fixed seeds, 7 to 8 coordinates per case). Coordinates whose sign is a gauge
    are compared through their square, and the quaternion components of the gauged
    frame chart are compared through absolute values: the gauge flips sign at random.
    """
    block, model = _make(name), _system_model()
    mapped, drawn = _two_samples(block, model)
    for k, coordinate in enumerate(block.coordinate_names):
        a, b = mapped[:, k], drawn[:, k]
        if coordinate in ("vw", "vx", "vy", "vz"):
            a, b = np.abs(a), np.abs(b)
        p = ks_2samp(a, b).pvalue
        assert p > ALPHA, (name, coordinate, p)


def _law_checks(block, model, mapped):
    """The mapped draws against the exact laws behind ``log_prob``, one-sample KS."""
    chart_x, _ = chart_coordinates(block, model, jnp.asarray(mapped))
    names = list(block.chart.coordinate_names)
    law = block.law
    cols = {n: np.asarray(chart_x[:, i]) for i, n in enumerate(names)}
    checks = []
    if "logit_e" in cols:
        e = 1 / (1 + np.exp(-cols["logit_e"]))
        checks.append(("e", e, lambda v: np.asarray(law.eccentricity.cdf(v))))
        z = np.tanh(cols["atanh_cos_i"])
        checks.append(("cos_i", z, uniform(-1, 2).cdf))
        for angle in ("W_rad", "w_rad", "M0_rad"):
            checks.append((angle, cols[angle], uniform(0, 2 * np.pi).cdf))
        if law.a_AU is not None:
            checks.append(
                ("a", np.exp(cols["ln_a_AU"]), lambda v: np.asarray(law.a_AU.cdf(v)))
            )
        else:
            mu = float(
                GM
                * (
                    model.dynamics.star.mass_Msun
                    + model.dynamics.planets.mass_Msun[:2].sum()
                )
            )
            period = 2 * np.pi * np.sqrt(np.exp(3 * cols["ln_a_AU"]) / mu)
            checks.append(("P", period, lambda v: np.asarray(law.period_d.cdf(v))))
    return checks


@pytest.mark.float64_only(reason="orbital inference coordinates and log densities")
@pytest.mark.parametrize("name", ["classical-a", "classical-period"])
def test_from_unit_matches_the_laws_that_log_prob_is_built_from(name):
    """Fails if the cube map's marginals are not the exact marginals of the law.

    The eccentricity, semi-major axis or period, cos(i) and the three plain angles of
    the classical chart have known distributions (the law's own, uniform on
    ``[-1, 1]`` and uniform on one cell); one-sample KS, p-value above 1e-3. The period
    case uses the rate of the model's masses at row 1.
    """
    block, model = _make(name), _system_model()
    mapped, _ = _two_samples(block, model)
    for label, values, cdf in _law_checks(block, model, mapped):
        assert ks_1samp(values, cdf).pvalue > ALPHA, (name, label)


@pytest.mark.float64_only(reason="orbital inference coordinates and log densities")
@pytest.mark.parametrize("name", list(CASES))
def test_from_unit_lands_where_log_prob_is_finite_and_is_deterministic(name):
    """Fails if a cube point maps outside the support, or the map is not a function."""
    block, model = _make(name), _system_model()
    cube = jax.random.uniform(jax.random.PRNGKey(5), (2000, block.unit_size))
    run = jax.jit(jax.vmap(lambda c: block.from_unit(c, model)))
    first, second = run(cube), run(cube)
    for n in block.coordinate_names:
        assert jnp.array_equal(first[n], second[n])
    logp = jax.jit(jax.vmap(lambda x: block.log_prob(x, model)))(first)
    assert bool(jnp.all(jnp.isfinite(logp)))


@pytest.mark.float64_only(reason="orbital inference coordinates and log densities")
@pytest.mark.parametrize("name", list(CASES))
def test_from_unit_preserves_locality(name):
    """Fails if nearby cube points are sent far apart on most of the cube.

    A cube step of 1e-7 per coordinate moves the coordinates by under 1e-3 for at
    least 99 percent of 4000 points; the rest sit at a seam (an angle wrap, the gauge
    sign at one half, the quaternion branch of the chart inverse).
    """
    block, model = _make(name), _system_model()
    cube = jax.random.uniform(jax.random.PRNGKey(6), (4000, block.unit_size))
    run = jax.jit(jax.vmap(lambda c: block.from_unit(c, model)))
    a, b = run(cube), run(cube + 1e-7)
    jump = jnp.max(
        jnp.stack([jnp.abs(a[n][:, 0] - b[n][:, 0]) for n in block.coordinate_names]),
        axis=0,
    )
    assert float(jnp.mean(jump < 1e-3)) >= 0.99


def test_unit_size_counts_rows_gauge_and_mass_and_refuses_without_an_inverse_cdf():
    """Fails if the cube length or the refusal of a law with no map is wrong."""
    model = _system_model()
    assert _block(ClassicalChart(), (0, 2)).unit_size == 12
    assert _block(FrameChart(), (0, 2)).unit_size == 16
    assert _ti_block().unit_size == 7
    tilted = OrbitBlock(
        ClassicalChart(), _TiltedLaw(A_LAW), rows=(0,), system="dynamics"
    )
    assert tilted.unit_size is None
    with pytest.raises(ValueError, match=r"shape \(6,\)"):
        _block(ClassicalChart(), (1,)).from_unit(jnp.zeros(5), model)
    mixture = priorities.Mixture(
        jnp.zeros(2), priorities.Uniform(jnp.array([0.0, 0.3]), jnp.array([0.2, 0.6]))
    )
    law = OrbitLaw(mixture, a_AU=priorities.LogUniform(0.8, 3.0))
    assert law.unit_size is None
    block = OrbitBlock(ClassicalChart(), law, rows=(0,), system="dynamics")
    assert block.unit_size is None
    with pytest.raises(NotImplementedError, match="no unit-cube map"):
        block.from_unit(jnp.zeros(1), model)
