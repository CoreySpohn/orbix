"""Law support and the unbounded coordinates a block gives a bounded law."""

import jax
import jax.numpy as jnp
import numpy as np
import priorities
import pytest
from hwoutils.constants import GM_SUN_AU3_PER_D2
from hwoutils.model_paths import assign
from scipy.stats import ks_1samp, logistic
from test_blocks import (
    A_LAW,
    P_LAW,
    _system_model,
    _ti_block,
    _TiltedLaw,
    chart_coordinates,
)

from orbix.system import (
    Body,
    CartesianChart,
    ClassicalChart,
    EccentricityVectorChart,
    FrameChart,
    OrbitBlock,
    OrbitLaw,
    ThieleInnesBlock,
    ThieleInnesChart,
    enclosed_mu,
)
from orbix.system.frame_orbit import FrameOrbit

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


def test_law_support_reports_exactly_the_interval_bounded_elements(precision):
    """Interval endpoints are the supplied values represented in the lane dtype."""
    support = A_LAW.support()
    assert set(support) == {"e", "a_AU"}
    np.testing.assert_array_equal(
        support["e"], np.asarray((0.05, 0.8), dtype=precision)
    )
    np.testing.assert_array_equal(
        support["a_AU"], np.asarray((0.8, 3.0), dtype=precision)
    )
    assert set(P_LAW.support()) == {"e", "period_d"}
    truncated = priorities.TruncatedDistribution(priorities.Beta(0.87, 3.03), high=0.7)
    law = OrbitLaw(truncated, period_d=priorities.LogUniform(300.0, 600.0))
    np.testing.assert_array_equal(
        law.support()["e"], np.asarray((0.0, 0.7), dtype=precision)
    )
    open_law = OrbitLaw(priorities.HalfNormal(0.2), a_AU=priorities.LogNormal(0.0, 1.0))
    assert open_law.support() == {}
    assert _TiltedLaw(A_LAW).support() == {}


@pytest.mark.float64_only(reason="orbital inference coordinates and log densities")
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
    """Fails if an empty support changes the chart coordinates in any way."""
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


def _mapped(chart, law):
    return OrbitBlock(chart, law, rows=(1,), system="dynamics")


@pytest.mark.float64_only(reason="orbital inference coordinates and log densities")
@pytest.mark.parametrize("chart", [EccentricityVectorChart(), FrameChart()], ids=_id)
def test_a_disk_chart_with_a_lower_eccentricity_bound_keeps_a_zero_density_hole(chart):
    """Fails if the block raises, or the hole is not -inf, or a draw lands in it.

    The law starts at e = 0.05, which the disk chart has no coordinate for: the block
    maps the upper bound and the disk of radius 0.05 around the origin stays with
    density minus infinity. Every ``sample`` and ``from_unit`` draw has finite density
    and eccentricity above 0.05.
    """
    model = _system_model()
    block = _mapped(chart, A_LAW)
    assert block.walls == ("e",)
    names = block.coordinate_names
    origin = {n: jnp.zeros(1) for n in names}
    assert float(block.log_prob(origin, model)) == -np.inf
    keys = jax.random.split(jax.random.PRNGKey(0), 200)
    draws = jax.vmap(lambda k: block.sample(k, model))(keys)
    cube = jax.random.uniform(jax.random.PRNGKey(1), (200, block.unit_size))
    mapped = jax.vmap(lambda c: block.from_unit(c, model))(cube)
    for out in (draws, mapped):
        values = jax.vmap(lambda x: block.log_prob(x, model))(out)
        assert bool(jnp.all(jnp.isfinite(values)))
        y = jnp.stack([out[n] for n in names], -1)[:, 0]
        x, _ = chart_coordinates(block, model, y)
        e = jnp.linalg.norm(x[:, 1:3], axis=-1) / jnp.sqrt(
            1 + jnp.sum(x[:, 1:3] ** 2, -1)
        )
        assert bool(jnp.all(e > 0.05)) and bool(jnp.all(e < 0.8))


BLOCKS = {
    "classical-a": (ClassicalChart(), A_LAW),
    "classical-period": (ClassicalChart(), P_LAW),
    "eccentricity-vector-a": (EccentricityVectorChart(), A0_LAW),
    "eccentricity-vector-period": (EccentricityVectorChart(), P0_LAW),
    "frame-a": (FrameChart(), A0_LAW),
}


@pytest.mark.float64_only(reason="orbital inference coordinates and log densities")
@pytest.mark.parametrize("name", list(BLOCKS))
def test_block_is_the_chart_composed_with_the_map(name):
    """Fails if the forward map or density is not the chart's at the mapped point.

    At random coordinates ``y``, ``forward`` equals the orbit of the chart at the
    constrained coordinates (positions to 1e-12 AU) and ``log_prob`` equals the law
    on that orbit plus the chart volume plus the map's log derivative (1e-10).
    """
    chart, law = BLOCKS[name]
    model = _system_model()
    block = _mapped(chart, law)
    rng = np.random.default_rng(2)
    mu = block._mu(model)[..., jnp.asarray(block.rows)]
    epoch = model.dynamics.orbits.epoch_d[jnp.asarray(block.rows)]
    for _ in range(4):
        y = {
            n: jnp.asarray(rng.normal(scale=1.5, size=(1,)))
            for n in block.coordinate_names
        }
        stacked = jnp.stack([y[n] for n in block.coordinate_names], -1)
        x, log_jac = chart_coordinates(block, model, stacked)
        orbit = chart.to_orbit(x, mu_AU3_per_d2=mu, epoch_d=epoch)
        expected = jnp.sum(law.log_prob(orbit) + chart.log_volume(x) + log_jac)
        np.testing.assert_allclose(block.log_prob(y, model), expected, atol=1e-10)
        new = block.forward(y, model)["dynamics.orbits"]
        times = jnp.array([0.0, 50.0])
        np.testing.assert_allclose(
            new.at(times)[0][1], orbit.at(times)[0][0], atol=1e-12
        )


def _cell_difference(a, b, period):
    """Distance between angles modulo ``period``."""
    d = np.abs(np.asarray(a) - np.asarray(b)) % period
    return np.minimum(d, period - d)


@pytest.mark.float64_only(reason="orbital inference coordinates and log densities")
@pytest.mark.parametrize("name", ["classical-a", "classical-period"])
def test_inverse_of_forward_recovers_the_coordinates(name):
    """Fails if ``inverse`` does not undo ``forward``.

    Real coordinates to 1e-9, angles modulo ``2 pi`` to 1e-9 (float64 trigonometry).
    """
    chart, law = BLOCKS[name]
    model = _system_model()
    block = _mapped(chart, law)
    y = {
        "mapped_ln_a_AU": jnp.array([0.7]),
        "logit_e": jnp.array([-1.2]),
        "atanh_cos_i": jnp.array([0.4]),
        "W_rad": jnp.array([1.0]),
        "w_rad": jnp.array([2.0]),
        "M0_rad": jnp.array([0.5]),
    }
    held = assign(model, "dynamics.orbits", block.forward(y, model)["dynamics.orbits"])
    back = block.inverse(held)
    for n in ("mapped_ln_a_AU", "logit_e", "atanh_cos_i"):
        np.testing.assert_allclose(back[n], y[n], atol=1e-9, err_msg=n)
    for n in ("W_rad", "w_rad", "M0_rad"):
        assert float(np.max(_cell_difference(back[n], y[n], 2 * np.pi))) < 1e-9, n


N = 300_000


def _sample(block, model, n=N, seed=11):
    out = jax.jit(jax.vmap(lambda k: block.sample(k, model)))(
        jax.random.split(jax.random.PRNGKey(seed), n)
    )
    return {k: np.asarray(v[:, 0]) for k, v in out.items()}


@pytest.mark.float64_only(reason="orbital inference coordinates and log densities")
@pytest.mark.parametrize("name", ["classical-a", "classical-period"])
def test_size_and_eccentricity_coordinates_are_logistic_under_a_uniform_law(name):
    """Fails if a bounded element is not mapped onto the whole line by a logit.

    A log-uniform size and a uniform eccentricity are uniform on their interval, so
    their logit coordinates are standard logistic variables: one-sample KS on draws of
    ``sample`` and of ``from_unit``, p-value above 1e-3.
    """
    chart, law = BLOCKS[name]
    model = _system_model()
    block = _mapped(chart, law)
    drawn = _sample(block, model)
    cube = jax.random.uniform(jax.random.PRNGKey(12), (50_000, block.unit_size))
    mapped = jax.vmap(lambda c: block.from_unit(c, model))(cube)
    mapped = {k: np.asarray(v[:, 0]) for k, v in mapped.items()}
    for source in (drawn, mapped):
        for n in ("mapped_ln_a_AU", "logit_e"):
            assert ks_1samp(source[n], logistic.cdf).pvalue > 1e-3, (name, n)


def _gauss_logpdf(x, sd):
    return -0.5 * (x / sd) ** 2 - np.log(sd * np.sqrt(2 * np.pi))


@pytest.mark.float64_only(reason="orbital inference coordinates and log densities")
@pytest.mark.parametrize(
    "name", ["classical-a", "classical-period", "eccentricity-vector-a"]
)
def test_density_integrates_to_one_over_the_real_coordinates(name):
    """Fails if the density is not normalized over the real line.

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
    block = _mapped(chart, law)
    names = block.coordinate_names
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


@pytest.mark.float64_only(reason="orbital inference coordinates and log densities")
def test_gradients_are_finite_far_into_the_coordinates():
    """Fails if a gradient sampler could meet a wall or a non-finite derivative.

    The classical block has a finite density and gradient at 200 random points with
    coordinates up to 15 in magnitude, where the law's own bounds would be walls.
    """
    model = _system_model()
    block = _mapped(ClassicalChart(), A_LAW)
    rng = np.random.default_rng(3)
    names = block.coordinate_names

    def density(v):
        return block.log_prob({n: v[k][None] for k, n in enumerate(names)}, model)

    points = jnp.asarray(rng.uniform(-15, 15, size=(200, 6)))
    values = jax.vmap(density)(points)
    grads = jax.vmap(jax.grad(density))(points)
    assert bool(jnp.all(jnp.isfinite(values))) and bool(jnp.all(jnp.isfinite(grads)))


@pytest.mark.float64_only(reason="orbital inference coordinates and log densities")
@pytest.mark.medium(reason="two NUTS chains of 800 steps, 25 s cold")
def test_nuts_on_the_prior_meets_no_wall():
    """Fails if NUTS on the prior diverges as it does against a wall.

    A short NUTS chain (400 warm-up with window adaptation, 400 draws, one fixed key)
    on the classical block of a log-uniform size and uniform eccentricity. Against the
    walls of the chart coordinates the earlier review measured 95 percent divergent
    transitions; here the chain must stay under 5 percent.
    """
    blackjax = pytest.importorskip("blackjax")
    model = _system_model()
    block = _mapped(ClassicalChart(), A_LAW)
    names = block.coordinate_names

    def logdensity(v):
        return block.log_prob({n: v[k][None] for k, n in enumerate(names)}, model)

    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
    key = jax.random.PRNGKey(0)
    start = jnp.array([0.1, 0.0, 0.2, 1.0, 2.0, 3.0])
    (state, parameters), _ = warmup.run(key, start, num_steps=400)
    step = jax.jit(blackjax.nuts(logdensity, **parameters).step)

    def one(state, k):
        state, info = step(k, state)
        return state, info.is_divergent

    _, divergent = jax.lax.scan(one, state, jax.random.split(key, 400))
    assert float(jnp.mean(divergent)) < 0.05


FAR_OUT = {
    **BLOCKS,
    "cartesian-a": (CartesianChart(), A_LAW),
}


@pytest.mark.float64_only(reason="orbital inference coordinates and log densities")
@pytest.mark.parametrize("name", list(FAR_OUT))
def test_log_prob_far_out_is_minus_infinity_never_nan(name):
    """Fails if a coordinate of 1e5 gives NaN or +inf instead of a finite value or -inf.

    Every coordinate set to +1e5 or to -1e5, and each coordinate alone at +-1e5 with
    the rest at zero. No guard wraps the block: the chart volumes, the law and the
    excluded sets (the Cartesian origin) must report no density themselves.
    """
    chart, law = FAR_OUT[name]
    model = _system_model()
    block = _mapped(chart, law)
    _far_out_is_not_nan(block, model, block.coordinate_names, name)


@pytest.mark.float64_only(reason="orbital inference coordinates and log densities")
def test_thiele_innes_log_prob_far_out_is_minus_infinity_never_nan():
    """Fails if the Thiele-Innes block gives NaN or +inf at coordinates of 1e5."""
    block, model = _ti_block(rows=(1,)), _system_model()
    _far_out_is_not_nan(block, model, block.coordinate_names, "thiele-innes")


def _far_out_is_not_nan(block, model, names, name):
    cases = [{n: 1e5 for n in names}, {n: -1e5 for n in names}]
    for k in names:
        cases += [{n: (s if n == k else 0.0) for n in names} for s in (1e5, -1e5)]
    for c in cases:
        y = {n: jnp.asarray([v]) for n, v in c.items()}
        value = float(block.log_prob(y, model))
        assert not np.isnan(value) and value != np.inf, (name, c, value)


BROAD_E = priorities.Uniform(0.0, 0.95)
BROAD_A = priorities.LogUniform(1.0, 100.0)
BROAD_LAW = OrbitLaw(BROAD_E, a_AU=BROAD_A)
INNER_E_LAW = OrbitLaw(priorities.Uniform(0.1, 0.9), a_AU=BROAD_A)
HALF_A_LAW = OrbitLaw(
    BROAD_E,
    a_AU=priorities.TruncatedDistribution(priorities.LogNormal(0.0, 1.0), low=1.0),
)
OPEN_LAW = OrbitLaw(priorities.Uniform(0.0, 1.0), a_AU=priorities.LogNormal(1.0, 1.0))
PERIOD_LAW = OrbitLaw(BROAD_E, period_d=priorities.LogUniform(300.0, 3000.0))
NEGATIVE_LOW_A = priorities.TruncatedDistribution(priorities.Normal(5.0, 3.0), low=-1.0)
NEGATIVE_LOW_LAW = OrbitLaw(BROAD_E, a_AU=NEGATIVE_LOW_A)


def _ti(law, mass):
    return ThieleInnesBlock(law, mass, rows=(1,), system="dynamics")


def _massive_planet_model():
    """The test system with a 0.01 solar-mass planet on the freed row."""
    model = _system_model()
    masses = jnp.array([1e-3, 1e-2, 1e-4])
    return assign(model, "dynamics.planets", Body(masses))


WALLS = {
    "classical-broad": (lambda: _mapped(ClassicalChart(), BROAD_LAW), ()),
    "classical-inner-e": (lambda: _mapped(ClassicalChart(), INNER_E_LAW), ()),
    "classical-half-a": (lambda: _mapped(ClassicalChart(), HALF_A_LAW), ("a_AU",)),
    "eccentricity-vector-broad": (
        lambda: _mapped(EccentricityVectorChart(), BROAD_LAW),
        (),
    ),
    "frame-broad": (lambda: _mapped(FrameChart(), BROAD_LAW), ()),
    "eccentricity-vector-inner-e": (
        lambda: _mapped(EccentricityVectorChart(), INNER_E_LAW),
        ("e",),
    ),
    "frame-inner-e": (lambda: _mapped(FrameChart(), INNER_E_LAW), ("e",)),
    "cartesian-broad": (lambda: _mapped(CartesianChart(), BROAD_LAW), ("a_AU", "e")),
    "cartesian-open": (lambda: _mapped(CartesianChart(), OPEN_LAW), ()),
    "classical-negative-low-a": (
        lambda: _mapped(ClassicalChart(), NEGATIVE_LOW_LAW),
        (),
    ),
    "thiele-innes-broad": (
        lambda: _ti(BROAD_LAW, priorities.Uniform(0.8, 1.2)),
        ("a_AU", "star_mass_Msun"),
    ),
    "thiele-innes-period-lognormal": (
        lambda: _ti(PERIOD_LAW, priorities.LogNormal(0.0, 0.2)),
        ("star_mass_Msun",),
    ),
}


@pytest.mark.parametrize("name", list(WALLS))
def test_walls_name_the_bounds_the_coordinates_do_not_map(name):
    """Fails if a block lists a mapped bound as a wall or misses an unmapped one.

    Truth: a wall is a size edge that is neither zero nor infinite with no mapped size
    coordinate, an eccentricity edge other than 0 and 1 that the chart does not map
    (a lower one is mapped only by the classical logit), or any finite edge of the
    Thiele-Innes star-mass law, zero included. Exact.
    """
    make, expected = WALLS[name]
    assert make().walls == expected


def _orbit_point(model, block, *, a, e, enclosed=None):
    """Coordinates of ``block`` for the model's orbits, row 1 at ``a`` and ``e``.

    The other elements of row 1 stay; its rate is set from ``enclosed`` (solar masses)
    when given, else from the model's masses.
    """
    system = model.dynamics
    elements = system.orbits.to_classical()
    mu = enclosed_mu(system.star.mass_Msun, system.planets.mass_Msun)
    if enclosed is not None:
        mu = mu.at[1].set(GM_SUN_AU3_PER_D2 * enclosed)
    orbit = FrameOrbit.from_classical(
        elements["a_AU"].at[1].set(a),
        elements["e"].at[1].set(e),
        elements["i_rad"],
        elements["W_rad"],
        elements["w_rad"],
        elements["M0_rad"],
        mu_AU3_per_d2=mu,
        epoch_d=system.orbits.epoch_d,
    )
    return block.inverse(assign(model, "dynamics.orbits", orbit))


def _wall_points(block, model, walls):
    """For each wall: coordinates just outside an edge and at the middle inside.

    Outside: size at half the lower edge, eccentricity midway between the upper edge
    and 1 or at half the lower edge, implied star mass at half the lower edge (an
    enclosed mass of half the inner planet mass when that edge is zero). Inside: the
    same orbit with that element in the middle of its support.
    """
    edges = {k: tuple(float(v) for v in b) for k, b in block.law.edges().items()}
    e_lo, e_hi = edges["e"]
    a_lo, a_hi = edges.get("a_AU", (1.0, 100.0))
    a_mid = np.sqrt(a_lo * a_hi) if np.isfinite(a_hi) else 2 * a_lo
    e_mid = 0.5 * (e_lo + e_hi)
    point = lambda **kw: _orbit_point(model, block, **kw)  # noqa: E731
    out = []
    if "a_AU" in walls:
        out.append(("a_AU", point(a=a_lo / 2, e=e_mid), point(a=a_mid, e=e_mid)))
    if "e" in walls:
        inside = point(a=a_mid, e=e_mid)
        mapped = "e" in block.chart_support(model)
        if e_hi < 1 and not mapped:
            out.append(("e", point(a=a_mid, e=0.5 * (e_hi + 1)), inside))
        if e_lo > 0:
            # inverse() refuses a point outside the support (NaN), so the hole point
            # scales the disk coordinates of the inside point toward the origin
            hole = {
                n: v * (e_lo / 2 / e_mid) if n in ("ux", "uy") else v
                for n, v in inside.items()
            }
            y = jnp.stack([hole[n] for n in block.coordinate_names], -1)
            x, _ = chart_coordinates(block, model, y)
            e = jnp.linalg.norm(x[..., 1:3], axis=-1) / jnp.sqrt(
                1 + jnp.sum(x[..., 1:3] ** 2, -1)
            )
            assert 0 < float(e[0]) < e_lo
            out.append(("e", hole, inside))
    if "star_mass_Msun" in walls:
        inner = float(jnp.sum(model.dynamics.planets.mass_Msun[:2]))
        low = float(block.star_mass_Msun.support()[0])
        star_out = low / 2 if low > 0 else -inner / 2

        def mass_point(enclosed):
            if "a_AU" in edges:
                return point(a=a_mid, e=e_mid, enclosed=enclosed)
            # a period law is mapped: keep the period at the middle of its support
            period = np.sqrt(np.prod(edges["period_d"]))
            mu = GM_SUN_AU3_PER_D2 * enclosed
            a = (mu * (period / (2 * np.pi)) ** 2) ** (1 / 3)
            return point(a=a, e=e_mid, enclosed=enclosed)

        out.append(
            (
                "star_mass_Msun",
                mass_point(star_out + inner),
                mass_point(1.0 + inner),
            )
        )
    return out


@pytest.mark.float64_only(reason="orbital inference coordinates and log densities")
@pytest.mark.parametrize("name", [k for k, (_, w) in WALLS.items() if w])
def test_log_prob_is_minus_infinity_just_outside_each_wall(name):
    """Fails if a declared wall is not where the density drops to zero.

    For every wall, ``log_prob`` is ``-inf`` at a constructed point whose element lies
    just outside the edge and finite at the same orbit with that element inside. The
    Thiele-Innes log-normal mass case uses a planet of 0.01 solar masses and an
    enclosed mass below it, so the implied star mass is negative: the zero edge of the
    mass law is a wall although the law has no finite edge but zero. Exact.
    """
    make, walls = WALLS[name]
    block = make()
    model = _massive_planet_model() if "lognormal" in name else _system_model()
    points = _wall_points(block, model, walls)
    assert {wall for wall, _, _ in points} == set(walls)
    for wall, outside, inside in points:
        assert float(block.log_prob(outside, model)) == -np.inf, wall
        assert np.isfinite(float(block.log_prob(inside, model))), wall


@pytest.mark.float64_only(reason="orbital inference coordinates and log densities")
@pytest.mark.parametrize("name", [k for k, (_, w) in WALLS.items() if not w])
def test_a_block_with_no_walls_has_density_at_random_coordinates(name):
    """Fails if a block with ``walls == ()`` has a zero-density region.

    ``log_prob`` is finite at 2000 points drawn Normal(0, 3) per coordinate (seed 0);
    measured no ``-inf`` there for the classical chart, against 0.2 to 48
    percent for the walled blocks. Exact.
    """
    make, _ = WALLS[name]
    block, model = make(), _system_model()
    rng = np.random.default_rng(0)
    points = {
        n: jnp.asarray(rng.normal(scale=3.0, size=(2000, 1)))
        for n in block.coordinate_names
    }
    values = jax.jit(jax.vmap(lambda x: block.log_prob(x, model)))(points)
    assert bool(jnp.all(jnp.isfinite(values)))


@pytest.mark.float64_only(reason="orbital inference coordinates and log densities")
@pytest.mark.parametrize(
    "support",
    [(-jnp.inf, 1.5), (-0.5, jnp.inf), (-jnp.inf, jnp.inf)],
    ids=["open-below", "open-above", "open-both"],
)
def test_a_size_interval_with_an_infinite_end_maps_by_a_softplus(support):
    """Fails if an infinite end of the size interval gives NaN or a wrong Jacobian.

    ``ln 0 = -inf`` is the lower end of the size coordinate for a size law whose
    support starts at zero (and ``ln n`` has an infinite upper end for a period law
    from zero). The log derivative equals the automatic one to 1e-10, the inverse
    returns ``y`` to 1e-9 for ``|y|`` up to 8, and the value stays inside the open end.
    """
    chart = ClassicalChart()
    low, high = (jnp.asarray(v) for v in support)
    bounds = {"ln_size": (low, high)}
    for y0 in np.linspace(-8.0, 8.0, 7):
        y = jnp.array([y0, 0.3, -0.2, 1.0, 2.0, 3.0])
        x, log_jac = chart.constrain(y, bounds)
        jac = jax.jacfwd(lambda v: chart.constrain(v, bounds)[0])(y)
        np.testing.assert_allclose(jnp.linalg.slogdet(jac)[1], log_jac, atol=1e-10)
        np.testing.assert_allclose(chart.unconstrain(x, bounds), y, atol=1e-9)
        assert float(low) < float(x[0]) < float(high)


ZERO_EDGE_SIZES = {
    "upper-truncated-log-normal-a": {
        "a_AU": priorities.TruncatedDistribution(
            priorities.LogNormal(1.0, 1.0), high=50.0
        )
    },
    "uniform-from-zero-a": {"a_AU": priorities.Uniform(0.0, 100.0)},
    "uniform-from-below-zero-a": {"a_AU": priorities.Uniform(-1.0, 100.0)},
    "truncated-normal-from-below-zero-a": {
        "a_AU": priorities.TruncatedDistribution(
            priorities.Normal(5.0, 3.0), low=-1.0, high=50.0
        )
    },
    "upper-truncated-log-normal-period": {
        "period_d": priorities.TruncatedDistribution(
            priorities.LogNormal(7.0, 1.0), high=5000.0
        )
    },
}


@pytest.mark.float64_only(reason="orbital inference coordinates and log densities")
@pytest.mark.parametrize("size", list(ZERO_EDGE_SIZES))
def test_a_size_law_from_zero_has_a_finite_density_everywhere(size):
    """Fails if a size law whose support starts at or below zero reads NaN or a wall.

    The logit on ``(ln 0, ln high)`` gave NaN at every point on the charts that map
    the size (measured: all of 2000 points), and a lower edge below zero gave the log
    of a negative number (all of 500 points). The lower end is the coordinate's own
    end (a size edge at or below zero is clipped at zero), so only the upper edge is
    mapped: ``log_prob`` is finite at 2000 points
    drawn Normal(0, 3) per coordinate (seed 0) on the classical and Frame charts and,
    for the period law, on the Thiele-Innes block with an unbounded mass law, and
    no block lists a wall. Exact.
    """
    law = OrbitLaw(BROAD_E, **ZERO_EDGE_SIZES[size])
    blocks = [_mapped(ClassicalChart(), law), _mapped(FrameChart(), law)]
    if "period" in size:
        blocks.append(_ti(law, priorities.Normal(1.0, 0.1)))
    model = _system_model()
    rng = np.random.default_rng(0)
    for block in blocks:
        assert block.walls == ()
        points = {
            n: jnp.asarray(rng.normal(scale=3.0, size=(2000, 1)))
            for n in block.coordinate_names
        }
        values = jax.jit(jax.vmap(lambda x, b=block: b.log_prob(x, model)))(points)
        assert bool(jnp.all(jnp.isfinite(values))), type(block.chart).__name__


@pytest.mark.parametrize(
    "chart", [ClassicalChart(), EccentricityVectorChart()], ids=_id
)
def test_inverse_support_mask_is_precision_independent(chart):
    """Four coarsest-precision epsilons at the interval scale resolve membership.

    An overshoot of a quarter epsilon is within the support tolerance; sixteen
    epsilons is outside even though it is below sqrt(float32 epsilon).
    """
    from hwoutils.numerics import COARSEST, eps

    high = 0.75
    radii = high * (1.0 + np.array([0.25, 16.0]) * eps(COARSEST))
    coords = jnp.zeros((2, chart.n_coords))
    if isinstance(chart, ClassicalChart):
        coords = coords.at[:, 1].set(np.log(radii / (1.0 - radii)))
    else:
        coords = coords.at[:, 1].set(radii / np.sqrt(1.0 - radii**2))
    for inverse in (chart.unconstrain, jax.jit(chart.unconstrain)):
        recovered = inverse(coords, {"e": (0.0, high)})
        assert np.all(np.isfinite(recovered[0]))
        assert np.all(np.isnan(recovered[1]))
        high_size = float(np.float32(1.2))
        offsets = np.array([4, 5]) * float(np.spacing(np.float32(high_size)))
        sizes = coords.at[:, 0].set(high_size + offsets)
        recovered = inverse(sizes, {"ln_size": (0.0, high_size)})
        assert np.all(np.isfinite(recovered[0]))
        assert np.all(np.isnan(recovered[1]))


@pytest.mark.float64_only(
    reason="interval endpoints outside the float32 exponent range"
)
@pytest.mark.parametrize("scale", [1e-100, 1e100])
def test_support_checks_preserve_the_float64_exponent_range(scale):
    """Only the comparison slack takes float32 resolution. Bounds retain their units."""
    chart = ClassicalChart()
    x = jnp.zeros((2, chart.n_coords)).at[:, 0].set(jnp.array([1.5, 3.0]) * scale)
    law = OrbitLaw(
        priorities.Uniform(0.0, 1.0), a_AU=priorities.Uniform(scale, 2 * scale)
    )
    block = _mapped(chart, law)
    assert block.same_map(block)
    result = chart.unconstrain(x, {"ln_size": (scale, 2 * scale)})
    assert np.all(np.isfinite(result[0]))
    assert np.all(np.isnan(result[1]))
