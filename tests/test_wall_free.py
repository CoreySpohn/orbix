"""Law support and the wall-free coordinates of bounded laws."""

import jax
import jax.numpy as jnp
import numpy as np
import priorities
import pytest
from hwoutils.model_paths import assign
from scipy.stats import ks_1samp, logistic
from test_blocks import (
    A_LAW,
    P_LAW,
    _block,
    _system_model,
    _ti_block,
    _TiltedLaw,
)

from orbix.system import (
    CartesianChart,
    ClassicalChart,
    EccentricityVectorChart,
    FrameChart,
    OrbitBlock,
    OrbitLaw,
    ThieleInnesChart,
)

E0_LAW = priorities.Uniform(0.0, 0.8)
A0_LAW = OrbitLaw(E0_LAW, a_AU=priorities.LogUniform(0.8, 3.0))
P0_LAW = OrbitLaw(E0_LAW, period_d=priorities.LogUniform(300.0, 3000.0))
ALL = [
    ClassicalChart(),
    EccentricityVectorChart(),
    FrameChart(),
    CartesianChart(),
    ThieleInnesChart(),
]
SUPPORT = {"e": (jnp.asarray(0.0), jnp.asarray(0.8)), "ln_size": (-0.5, 1.5)}


def _id(chart):
    return type(chart).__name__


def test_law_support_reports_exactly_the_interval_bounded_elements():
    """Fails if a bounded element is missing, or a half-bounded one is reported."""
    support = A_LAW.support()
    assert set(support) == {"e", "a_AU"}
    np.testing.assert_allclose(support["e"], (0.05, 0.8), rtol=0, atol=0)
    np.testing.assert_allclose(support["a_AU"], (0.8, 3.0), rtol=0, atol=0)
    assert set(P_LAW.support()) == {"e", "period_d"}
    truncated = priorities.TruncatedDistribution(priorities.Beta(0.87, 3.03), high=0.7)
    law = OrbitLaw(truncated, period_d=priorities.LogUniform(300.0, 600.0))
    np.testing.assert_allclose(law.support()["e"], (0.0, 0.7), rtol=0, atol=1e-15)
    open_law = OrbitLaw(priorities.HalfNormal(0.2), a_AU=priorities.LogNormal(0.0, 1.0))
    assert open_law.support() == {}
    assert _TiltedLaw(A_LAW).support() == {}


@pytest.mark.parametrize("chart", ALL, ids=_id)
def test_constrain_log_jacobian_is_exact_and_inverts(chart):
    """Fails if the returned log derivative is not the true one, or the map not 1:1.

    The log absolute determinant of the automatic Jacobian of ``constrain`` equals the
    returned value to 1e-10 (float64 formulas through the logit), and
    ``unconstrain(constrain(y))`` returns ``y`` to 1e-9 for ``|y|`` up to 6; the
    chart coordinates stay inside the bounds even at ``|y| = 30``.
    """
    support = {} if chart.size_coordinate is None else dict(SUPPORT)
    if chart.eccentricity_coordinate is None:
        support.pop("e", None)
    else:
        support["e"] = SUPPORT["e"]
    rng = np.random.default_rng(1)
    for y in rng.normal(scale=2.0, size=(5, chart.n_coords)):
        y = jnp.asarray(y)
        x, log_jac = chart.constrain(y, support)
        jac = jax.jacfwd(lambda v: chart.constrain(v, support)[0])(y)
        np.testing.assert_allclose(jnp.linalg.slogdet(jac)[1], log_jac, atol=1e-10)
        np.testing.assert_allclose(chart.unconstrain(x, support), y, atol=1e-9)
    far = jnp.asarray(rng.choice([-30.0, 30.0], size=(chart.n_coords,)))
    x, log_jac = chart.constrain(far, support)
    assert bool(jnp.all(jnp.isfinite(x))) and bool(jnp.isfinite(log_jac))
    if "ln_size" in support and chart.size_coordinate:
        assert support["ln_size"][0] <= float(x[0]) <= support["ln_size"][1]
    if chart.eccentricity_coordinate == "logit":
        assert 0.0 <= float(jax.nn.sigmoid(x[1])) <= 0.8
    elif chart.eccentricity_coordinate == "disk":
        assert float(jnp.hypot(*x[1:3]) / jnp.sqrt(1 + x[1] ** 2 + x[2] ** 2)) <= 0.8


def test_an_empty_support_leaves_coordinates_untouched():
    """Fails if a walled block changes the chart coordinates in any way."""
    y = jnp.linspace(-1.0, 2.0, 6)
    x, log_jac = ClassicalChart().constrain(y, {})
    assert jnp.array_equal(x, y) and float(log_jac) == 0.0


def test_a_unit_upper_bound_is_the_identity_on_the_disk():
    """Fails if the disk map is not the identity when the law allows up to e = 1."""
    chart = EccentricityVectorChart()
    y = jnp.array([0.3, -2.0, 1.5, 0.4, 1.0, 2.0])
    x, log_jac = chart.constrain(y, {"e": (jnp.asarray(0.0), jnp.asarray(1.0))})
    np.testing.assert_allclose(x, y, atol=1e-15, rtol=0)
    assert abs(float(log_jac)) < 1e-15


def _wall_free(chart, law, **kw):
    return OrbitBlock(chart, law, rows=(1,), system="dynamics", wall_free=True, **kw)


def test_a_disk_chart_refuses_an_eccentricity_law_that_does_not_start_at_zero():
    """Fails if the lower wall of a disk eccentricity chart is silently kept."""
    with pytest.raises(ValueError, match="starts at zero"):
        _wall_free(EccentricityVectorChart(), A_LAW)
    _wall_free(ClassicalChart(), A_LAW)
    _wall_free(EccentricityVectorChart(), A0_LAW)


BLOCKS = {
    "classical-a": (ClassicalChart(), A_LAW),
    "classical-period": (ClassicalChart(), P_LAW),
    "eccentricity-vector-a": (EccentricityVectorChart(), A0_LAW),
    "eccentricity-vector-period": (EccentricityVectorChart(), P0_LAW),
    "frame-a": (FrameChart(), A0_LAW),
}


@pytest.mark.parametrize("name", list(BLOCKS))
def test_wall_free_block_is_the_walled_block_composed_with_the_map(name):
    """Fails if the wall-free density or forward map is not the walled one composed.

    At random wall-free coordinates, ``forward`` equals the walled block's orbits at
    the constrained coordinates (positions to 1e-12 AU) and ``log_prob`` equals the
    walled ``log_prob`` plus the map's log derivative (1e-10), and ``inverse`` returns
    the coordinates ``forward`` was given (1e-9, one angle cell).
    """
    chart, law = BLOCKS[name]
    model = _system_model()
    free, walled = _wall_free(chart, law), _block(chart, (1,), law)
    rng = np.random.default_rng(2)
    for _ in range(4):
        y = {
            n: jnp.asarray(rng.normal(scale=1.5, size=(1,)))
            for n in chart.coordinate_names
        }
        support = free._support(free._mu(model)[..., jnp.asarray(free.rows)])
        x, log_jac = chart.constrain(
            jnp.stack([y[n] for n in chart.coordinate_names], -1), support
        )
        xd = {n: x[..., k] for k, n in enumerate(chart.coordinate_names)}
        np.testing.assert_allclose(
            free.log_prob(y, model),
            walled.log_prob(xd, model) + jnp.sum(log_jac),
            atol=1e-10,
        )
        new = free.forward(y, model)["dynamics.orbits"]
        ref = walled.forward(xd, model)["dynamics.orbits"]
        times = jnp.array([0.0, 50.0])
        np.testing.assert_allclose(new.at(times)[0], ref.at(times)[0], atol=1e-12)


def _cell_difference(a, b, period):
    """Distance between angles modulo ``period``."""
    d = np.abs(np.asarray(a) - np.asarray(b)) % period
    return np.minimum(d, period - d)


@pytest.mark.parametrize("name", ["classical-a", "classical-period"])
def test_inverse_of_forward_recovers_wall_free_coordinates(name):
    """Fails if ``inverse`` does not undo ``forward`` for wall-free coordinates.

    Real coordinates to 1e-9, angles modulo ``2 pi`` to 1e-9 (float64 trigonometry).
    """
    chart, law = BLOCKS[name]
    model = _system_model()
    block = _wall_free(chart, law)
    y = {
        "ln_a_AU": jnp.array([0.7]),
        "logit_e": jnp.array([-1.2]),
        "atanh_cos_i": jnp.array([0.4]),
        "W_rad": jnp.array([1.0]),
        "w_rad": jnp.array([2.0]),
        "M0_rad": jnp.array([0.5]),
    }
    held = assign(model, "dynamics.orbits", block.forward(y, model)["dynamics.orbits"])
    back = block.inverse(held)
    for n in ("ln_a_AU", "logit_e", "atanh_cos_i"):
        np.testing.assert_allclose(back[n], y[n], atol=1e-9, err_msg=n)
    for n in ("W_rad", "w_rad", "M0_rad"):
        assert float(np.max(_cell_difference(back[n], y[n], 2 * np.pi))) < 1e-9, n


N = 300_000


def _sample(block, model, n=N, seed=11):
    out = jax.jit(jax.vmap(lambda k: block.sample(k, model)))(
        jax.random.split(jax.random.PRNGKey(seed), n)
    )
    return {k: np.asarray(v[:, 0]) for k, v in out.items()}


@pytest.mark.parametrize("name", ["classical-a", "classical-period"])
def test_size_and_eccentricity_coordinates_are_logistic_under_a_uniform_law(name):
    """Fails if a bounded element is not mapped onto the whole line by a logit.

    A log-uniform size and a uniform eccentricity are uniform on their interval, so
    their logit coordinates are standard logistic variables: one-sample KS on draws of
    ``sample`` and of ``from_unit``, p-value above 1e-3.
    """
    chart, law = BLOCKS[name]
    model = _system_model()
    block = _wall_free(chart, law)
    drawn = _sample(block, model)
    cube = jax.random.uniform(jax.random.PRNGKey(12), (50_000, block.unit_size))
    mapped = jax.vmap(lambda c: block.from_unit(c, model))(cube)
    mapped = {k: np.asarray(v[:, 0]) for k, v in mapped.items()}
    for source in (drawn, mapped):
        for n in ("ln_a_AU", "logit_e"):
            assert ks_1samp(source[n], logistic.cdf).pvalue > 1e-3, (name, n)


def _gauss_logpdf(x, sd):
    return -0.5 * (x / sd) ** 2 - np.log(sd * np.sqrt(2 * np.pi))


@pytest.mark.parametrize(
    "name", ["classical-a", "classical-period", "eccentricity-vector-a"]
)
def test_wall_free_density_integrates_to_one(name):
    """Fails if the wall-free density is not normalized over the real line.

    Draws ``y`` from ``sample`` and averages ``q / p`` with ``q`` a normalized
    product of Gaussians (scale 2) on the real coordinates and the uniform cell on
    the angles; this is ``integral of q = 1`` and needs ``q`` lighter-tailed than the
    density, which holds (logistic or power-law tails). Pass when within four standard
    errors of one and the standard error (the precision of the check, which the
    heavy-tailed ratio of the eccentricity-vector case limits to about 0.012) is
    below 0.02.
    """
    chart, law = BLOCKS[name]
    model = _system_model()
    block = _wall_free(chart, law)
    names = chart.coordinate_names
    drawn = _sample(block, model)
    p = np.asarray(
        jax.jit(
            jax.vmap(
                lambda *c: block.log_prob(
                    {n: v[None] for n, v in zip(names, c, strict=True)}, model
                )
            )
        )(*[drawn[n] for n in names])
    )
    log_q = np.zeros(N)
    for n in names:
        log_q += (
            -np.log(2 * np.pi)
            if n.endswith("_rad") or n == "M0_plus_w_rad"
            else _gauss_logpdf(drawn[n], 2.0)
        )
    weight = np.exp(log_q - p)
    error = weight.std() / np.sqrt(N)
    assert abs(weight.mean() - 1) < 4 * error and error < 0.02, (weight.mean(), error)


def test_gradients_are_finite_far_into_the_coordinates():
    """Fails if a gradient sampler could meet a wall or a non-finite derivative.

    The wall-free classical block has a finite density and gradient at 200 random
    points with coordinates up to 15 in magnitude; the walled block has none outside
    the law's bounds (density minus infinity).
    """
    model = _system_model()
    chart = ClassicalChart()
    block = _wall_free(chart, A_LAW)
    rng = np.random.default_rng(3)
    names = chart.coordinate_names

    def density(v):
        return block.log_prob({n: v[k][None] for k, n in enumerate(names)}, model)

    points = jnp.asarray(rng.uniform(-15, 15, size=(200, 6)))
    values = jax.vmap(density)(points)
    grads = jax.vmap(jax.grad(density))(points)
    assert bool(jnp.all(jnp.isfinite(values))) and bool(jnp.all(jnp.isfinite(grads)))
    walled = _block(chart, (1,), A_LAW)
    outside = {n: jnp.array([0.0]) for n in names} | {"ln_a_AU": jnp.array([5.0])}
    assert float(walled.log_prob(outside, model)) == -np.inf


@pytest.mark.medium
def test_nuts_on_the_prior_meets_no_wall():
    """Fails if NUTS on the wall-free prior diverges as it does against a wall.

    Short NUTS chains (400 warm-up with window adaptation, 400 draws, one fixed key) on
    the classical block of a log-uniform size and uniform eccentricity. Against the
    walls the review measured 95 percent divergent transitions; here the wall-free
    chain must stay under 5 percent, and the walled chain, as the control that the
    test can see the defect, must exceed 30 percent.
    """
    blackjax = pytest.importorskip("blackjax")
    model = _system_model()
    chart = ClassicalChart()
    names = chart.coordinate_names

    def divergent_fraction(block, start):
        def logdensity(v):
            return block.log_prob({n: v[k][None] for k, n in enumerate(names)}, model)

        warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
        key = jax.random.PRNGKey(0)
        (state, parameters), _ = warmup.run(key, start, num_steps=400)
        step = jax.jit(blackjax.nuts(logdensity, **parameters).step)

        def one(state, k):
            state, info = step(k, state)
            return state, info.is_divergent

        _, divergent = jax.lax.scan(one, state, jax.random.split(key, 400))
        return float(jnp.mean(divergent))

    start = jnp.array([0.1, 0.0, 0.2, 1.0, 2.0, 3.0])
    free = divergent_fraction(_wall_free(chart, A_LAW), start)
    walled_start = jnp.array(
        [0.5 * (np.log(0.8) + np.log(3.0)), -0.5, 0.2, 1.0, 2.0, 3.0]
    )
    walled = divergent_fraction(_block(chart, (1,), A_LAW), walled_start)
    assert free < 0.05, free
    assert walled > 0.30, walled


FAR_OUT = {
    **BLOCKS,
    "cartesian-a": (CartesianChart(), A_LAW),
}


@pytest.mark.parametrize("wall_free", [False, True])
@pytest.mark.parametrize("name", list(FAR_OUT))
def test_log_prob_far_out_is_minus_infinity_never_nan(name, wall_free):
    """Fails if a coordinate of 1e5 gives NaN or +inf instead of a finite value or -inf.

    Every coordinate set to +1e5 or to -1e5, and each coordinate alone at +-1e5 with
    the rest at zero, for the walled and the wall-free block. Overflowing charts and
    laws must report no density rather than ``inf - inf``.
    """
    chart, law = FAR_OUT[name]
    model = _system_model()
    block = _wall_free(chart, law) if wall_free else _block(chart, (1,), law)
    _far_out_is_not_nan(block, model, chart.coordinate_names, name, wall_free)


def test_thiele_innes_log_prob_far_out_is_minus_infinity_never_nan():
    """Fails if the Thiele-Innes block gives NaN or +inf at coordinates of 1e5."""
    block, model = _ti_block(rows=(1,)), _system_model()
    _far_out_is_not_nan(block, model, block.coordinate_names, "thiele-innes", False)


def _far_out_is_not_nan(block, model, names, name, wall_free):
    cases = [{n: 1e5 for n in names}, {n: -1e5 for n in names}]
    for k in names:
        cases += [{n: (s if n == k else 0.0) for n in names} for s in (1e5, -1e5)]
    for c in cases:
        y = {n: jnp.asarray([v]) for n, v in c.items()}
        value = float(block.log_prob(y, model))
        assert not np.isnan(value) and value != np.inf, (name, wall_free, c, value)
