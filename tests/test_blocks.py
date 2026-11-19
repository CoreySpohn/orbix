"""Chart blocks: forward and inverse, normalized densities, draws and the contract."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import priorities
import pytest
from hwoutils.blocks import AbstractBlock
from hwoutils.model_paths import assign

from orbix.system import (
    AbstractOrbitLaw,
    Body,
    CartesianChart,
    ClassicalChart,
    EccentricityVectorChart,
    FrameChart,
    OrbitBlock,
    OrbitLaw,
    SkyFrame,
    System,
    ThieleInnesBlock,
    ThieleInnesChart,
    enclosed_mu,
)
from orbix.system.core import GM_SUN_AU3_PER_D2 as GM
from orbix.system.rotation import rotate

TIMES = jnp.array([-40.0, 0.0, 25.0, 90.0])
MASS_PATHS = ("dynamics.star.mass_Msun", "dynamics.planets.mass_Msun")


class _Model(eqx.Module):
    """The smallest model with the path layout of a skyscapes Scene."""

    dynamics: System


def _system_model():
    return _Model(
        System.from_classical(
            Body(jnp.asarray(1.1)),
            Body(jnp.asarray([1e-3, 3e-4, 1e-4])),
            SkyFrame(jnp.asarray(10.0)),
            a_AU=jnp.array([1.0, 3.0, 8.0]),
            e=jnp.array([0.1, 0.2, 0.3]),
            i_rad=jnp.array([0.7, 0.9, 1.2]),
            W_rad=jnp.array([0.5, 1.5, 2.5]),
            w_rad=jnp.array([1.0, 2.0, 3.0]),
            M0_rad=jnp.array([2.0, 0.3, 1.1]),
            epoch_d=jnp.array([3.0, 4.0, 5.0]),
        )
    )


def _scene_model():
    pytest.importorskip("skyscapes")
    from hwoutils.constants import Mearth2kg, Msun2kg
    from skyscapes.physical_model import LambertianPhysicalModel
    from skyscapes.scene import FlatStar, Planet, Scene

    def planet():
        return Planet(jnp.ones(1), LambertianPhysicalModel(jnp.array([0.3])))

    return Scene.from_classical(
        FlatStar(1e9),
        (planet(), planet(), planet()),
        star_mass_Msun=1.1,
        planet_mass_Msun=jnp.array([1e3, 3e2, 1e2]) * Mearth2kg / Msun2kg,
        dist_pc=10.0,
        a_AU=jnp.array([1.0, 3.0, 8.0]),
        e=jnp.array([0.1, 0.2, 0.3]),
        i_rad=jnp.array([0.7, 0.9, 1.2]),
        W_rad=jnp.array([0.5, 1.5, 2.5]),
        w_rad=jnp.array([1.0, 2.0, 3.0]),
        M0_rad=jnp.array([2.0, 0.3, 1.1]),
        epoch_d=jnp.array([3.0, 4.0, 5.0]),
    )


@pytest.fixture(params=["system", "scene"])
def model(request):
    """A bare orbix model, then a real skyscapes Scene, with the same orbits."""
    return _system_model() if request.param == "system" else _scene_model()


def _assign(model, updates):
    for path, value in updates.items():
        model = assign(model, path, value)
    return model


E_LAW = priorities.Uniform(0.05, 0.8)
A_LAW = OrbitLaw(E_LAW, a_AU=priorities.LogUniform(0.8, 3.0))
P_LAW = OrbitLaw(E_LAW, period_d=priorities.LogUniform(300.0, 3000.0))


def _block(chart, rows, law=A_LAW, system="dynamics"):
    return OrbitBlock(chart, law, rows=rows, system=system)


def _ti_block(law=A_LAW, system="dynamics", rows=(0,)):
    return ThieleInnesBlock(law, priorities.Uniform(0.9, 1.3), rows=rows, system=system)


def _positions(orbit, rows=None):
    r, _ = orbit.at(TIMES)
    return r if rows is None else r[jnp.asarray(rows)]


CHARTS = [
    ClassicalChart(),
    EccentricityVectorChart(),
    FrameChart(),
    FrameChart(gauge_sd=None),
    CartesianChart(scale_AU=3.0),
]


def _id(chart):
    return type(chart).__name__ + ("0" if getattr(chart, "gauge_sd", 1) is None else "")


@pytest.mark.parametrize("chart", CHARTS, ids=_id)
def test_contract_declares_owned_node_reads_and_writes(chart):
    """Fails if a block names a path other than its node and the masses it reads."""
    block = _block(chart, (0, 2))
    assert isinstance(block, AbstractBlock)
    assert block.owns == "dynamics.orbits"
    assert block.writes == ("dynamics.orbits",)
    assert block.reads == MASS_PATHS
    assert _block(chart, (1,), system="inner.dynamics").owns == "inner.dynamics.orbits"


def test_thiele_innes_block_writes_the_star_mass_and_reads_planet_masses():
    """Fails if the implied star mass is not a declared write."""
    block = _ti_block(rows=(1,))
    assert block.writes == ("dynamics.orbits", "dynamics.star.mass_Msun")
    assert block.reads == ("dynamics.planets.mass_Msun",)
    with pytest.raises(ValueError, match="exactly one row"):
        _ti_block(rows=(0, 1))
    with pytest.raises(ValueError, match="use ThieleInnesBlock"):
        _block(ThieleInnesChart(), (0,))
    with pytest.raises(ValueError, match="distinct"):
        _block(ClassicalChart(), (1, 1))


ALL_CHARTS = [*CHARTS, ThieleInnesChart()]

PERIODIC = {
    "ClassicalChart": {"W_rad", "w_rad", "M0_rad"},
    "EccentricityVectorChart": {"W_rad", "M0_plus_w_rad"},
}


@pytest.mark.parametrize("chart", ALL_CHARTS, ids=_id)
def test_periods_name_exactly_the_coordinates_the_orbit_repeats_in(chart):
    """Fails if a declared period is wrong, or a coordinate repeats without one.

    Shifting a coordinate by its declared period must leave the orbit (positions and
    rate) unchanged to 1e-9 AU; shifting any coordinate declared non-periodic by
    ``2 pi`` must move it by more than 1e-3 AU, so the declaration cannot be an
    accidental superset. Frame, Cartesian and Thiele-Innes charts have no angle.
    """
    names = chart.coordinate_names
    assert len(chart.periods) == len(names)
    expected = PERIODIC.get(type(chart).__name__, set())
    assert {n for n, p in zip(names, chart.periods, strict=True) if p} == expected
    x = jnp.array([0.3, 0.2, -0.4, 0.6, 0.5, 0.7, 0.9][: chart.n_coords])
    if isinstance(chart, ThieleInnesChart):
        x = jnp.array([-4.0, 0.2, -0.3, 0.4, 0.5, -0.3, 0.45])
    mu, epoch = 1.2e-3, 3.0

    def state(y):
        orbit = chart.to_orbit(y, mu_AU3_per_d2=mu, epoch_d=epoch)
        r, _ = orbit.at(jnp.array([0.0, 40.0]))
        return jnp.concatenate([jnp.ravel(r), jnp.atleast_1d(orbit.n_rad_per_d)])

    base = state(x)
    for k, (name, period) in enumerate(zip(names, chart.periods, strict=True)):
        shift = 2 * np.pi if period is None else period
        moved = state(x.at[k].add(shift))
        gap = float(jnp.max(jnp.abs(moved - base)))
        if period is None:
            assert gap > 1e-3, name
        else:
            assert period == 2 * np.pi and gap < 1e-9, (name, gap)


@pytest.mark.parametrize("chart", ALL_CHARTS, ids=_id)
def test_a_block_reports_the_periods_of_its_chart(chart):
    """Fails if the block drops the chart's names or periods."""
    block = (
        _ti_block() if isinstance(chart, ThieleInnesChart) else _block(chart, (0, 2))
    )
    assert block.coordinate_names == chart.coordinate_names
    assert block.periods == chart.periods


@pytest.mark.parametrize("chart", CHARTS, ids=_id)
@pytest.mark.parametrize("rows", [(1,), (0, 2)])
def test_forward_of_inverse_reproduces_the_orbits(model, chart, rows):
    """Fails if the freed rows move, a held row changes, or the rate leaves its mu.

    Positions agree to 1e-10 AU (float64 trigonometry through the chart); a held row
    keeps its geometry bit for bit, and its rate is the one its mass implies.
    """
    block = _block(chart, rows)
    x = block.inverse(model)
    assert tuple(x) == chart.coordinate_names
    assert all(v.shape == (len(rows),) for v in x.values())
    out = block.forward(x, model)
    assert tuple(out) == ("dynamics.orbits",)
    new, old = out["dynamics.orbits"], model.dynamics.orbits
    np.testing.assert_allclose(_positions(new), _positions(old), atol=1e-10)
    mu = enclosed_mu(model.dynamics.star.mass_Msun, model.dynamics.planets.mass_Msun)
    np.testing.assert_allclose(new.n_rad_per_d, old.n_rad_per_d, rtol=1e-12)
    np.testing.assert_allclose(new.n_rad_per_d, jnp.sqrt(mu / new.a_AU**3), rtol=1e-12)
    held = [k for k in range(3) if k not in rows]
    for name in ("a_AU", "ex", "ey", "epoch_d"):
        assert jnp.array_equal(
            getattr(new, name)[jnp.asarray(held)], getattr(old, name)[jnp.asarray(held)]
        )
    assert jnp.array_equal(new.quat[jnp.asarray(held)], old.quat[jnp.asarray(held)])


@pytest.mark.parametrize(
    "chart", [ClassicalChart(), EccentricityVectorChart()], ids=_id
)
def test_inverse_of_forward_recovers_the_coordinates(model, chart):
    """Fails if forward and inverse are not mutually inverse on one angle cell.

    The coordinates sit inside the cell the inverse returns (angles in (-pi, pi] or
    [0, 2 pi)); the tolerance is 1e-10 for float64 trigonometry.
    """
    block = _block(chart, (0, 2))
    x = {
        n: jnp.array([0.3 + 0.1 * k, -0.5 + 0.2 * k])
        for k, n in enumerate(chart.coordinate_names)
    }
    x["W_rad"] = jnp.array([1.0, 2.0])
    if "w_rad" in x:
        x["w_rad"], x["M0_rad"] = jnp.array([0.8, 3.0]), jnp.array([2.5, 1.0])
    else:
        x["M0_plus_w_rad"] = jnp.array([0.6, -2.0])
    back = block.inverse(_assign(model, block.forward(x, model)))
    for name in x:
        np.testing.assert_allclose(back[name], x[name], atol=1e-10, err_msg=name)


def test_forward_follows_the_masses_of_the_model(model):
    """Fails if a row keeps the old star mass after another entry changes it."""
    block = _block(ClassicalChart(), (1,))
    heavier = _assign(model, {MASS_PATHS[0]: jnp.asarray(1.6)})
    orbit = block.forward(block.inverse(model), heavier)["dynamics.orbits"]
    mu = enclosed_mu(jnp.asarray(1.6), model.dynamics.planets.mass_Msun)
    np.testing.assert_allclose(
        orbit.n_rad_per_d[1], jnp.sqrt(mu[1] / orbit.a_AU[1] ** 3), rtol=1e-12
    )
    # A held row keeps its geometry and follows the mass the block read.
    held = orbit.a_AU[0]
    assert held == model.dynamics.orbits.a_AU[0]
    np.testing.assert_allclose(
        orbit.n_rad_per_d[0], jnp.sqrt(mu[0] / held**3), rtol=1e-12
    )
    assert orbit.n_rad_per_d[0] != model.dynamics.orbits.n_rad_per_d[0]


def test_forward_is_a_per_draw_function_that_batches_with_vmap(model):
    """Fails if forward cannot be vmapped over draws, or is not traceable."""
    block = _block(ClassicalChart(), (0, 1))
    x = block.inverse(model)
    batch = {n: jnp.stack([v, v + 0.1, v - 0.1]) for n, v in x.items()}
    out = jax.jit(jax.vmap(lambda xi: block.forward(xi, model)["dynamics.orbits"]))(
        batch
    )
    assert out.a_AU.shape == (3, 3) and out.quat.shape == (3, 3, 4)
    one = block.forward({n: v[1] for n, v in batch.items()}, model)["dynamics.orbits"]
    np.testing.assert_allclose(out.a_AU[1], one.a_AU, rtol=1e-14)
    assert jax.jit(jax.vmap(lambda xi: block.log_prob(xi, model)))(batch).shape == (3,)
    draws = jax.jit(jax.vmap(lambda k: block.sample(k, model)))(
        jax.random.split(KEY, 3)
    )
    assert all(v.shape == (3, 2) for v in draws.values())


def test_thiele_innes_forward_of_inverse_reproduces_orbit_and_star_mass(model):
    """Fails if the implied star mass or the row is not reproduced.

    The model's rates satisfy n^2 a^3 = GM (Ms + sum m), so the round trip returns the
    stored star mass to 1e-12 relative and the positions to 1e-10 AU.
    """
    block = _ti_block(rows=(1,))
    out = block.forward(block.inverse(model), model)
    np.testing.assert_allclose(
        out["dynamics.star.mass_Msun"], model.dynamics.star.mass_Msun, rtol=1e-12
    )
    np.testing.assert_allclose(
        _positions(out["dynamics.orbits"]),
        _positions(model.dynamics.orbits),
        atol=1e-10,
    )
    # The chart's rate sets the orbit, and a new star mass is written.
    x = {
        n: v + (0.2 if n == "ln_n_rad_per_d" else 0.0)
        for n, v in block.inverse(model).items()
    }
    out = block.forward(x, model)
    mtot = out["dynamics.star.mass_Msun"] + jnp.sum(
        model.dynamics.planets.mass_Msun[:2]
    )
    moved = _assign(model, out)
    np.testing.assert_allclose(
        moved.dynamics.orbits.n_rad_per_d[1] ** 2 * moved.dynamics.orbits.a_AU[1] ** 3,
        GM * mtot,
        rtol=1e-12,
    )
    # Every row, held ones included, follows the star mass the block writes.
    mu = enclosed_mu(moved.dynamics.star.mass_Msun, moved.dynamics.planets.mass_Msun)
    np.testing.assert_allclose(
        moved.dynamics.orbits.n_rad_per_d,
        jnp.sqrt(mu / moved.dynamics.orbits.a_AU**3),
        rtol=1e-12,
    )


# --- normalization of the block density -----------------------------------------------

N = 400_000
KEY = jax.random.PRNGKey(0)
LOG_16 = np.log(16.0)


def _log_prob(block, model, coords):
    """Block log density of a batch of single-row coordinate draws."""
    names = block.chart.coordinate_names
    return jax.jit(
        jax.vmap(
            lambda c: block.log_prob(
                {n: c[k][None] for k, n in enumerate(names)}, model
            )
        )
    )(coords)


def _draws(block, model, n):
    """Coordinates of ``n`` single-row prior draws, shape ``(n, n_coords)``."""
    out = jax.jit(jax.vmap(lambda k: block.sample(k, model)))(jax.random.split(KEY, n))
    return jnp.stack([out[name][:, 0] for name in block.chart.coordinate_names], -1)


def _disk(key, radius, n):
    k1, k2 = jax.random.split(key)
    r = radius * jnp.sqrt(jax.random.uniform(k1, (n,)))
    t = jax.random.uniform(k2, (n,), maxval=2 * jnp.pi)
    return jnp.stack([r * jnp.cos(t), r * jnp.sin(t)], -1), -jnp.log(jnp.pi * radius**2)


def _uniform(key, n, low, high):
    return jax.random.uniform(key, (n, 1), minval=low, maxval=high)


def _ln_a_range(law, model):
    """The ln a support of the law on row 1 of the model."""
    if law.a_AU is not None:
        return np.log(0.8), np.log(3.0)
    mu = float(
        enclosed_mu(model.dynamics.star.mass_Msun, model.dynamics.planets.mass_Msun)[1]
    )
    return tuple(
        (np.log(mu) + 2 * np.log(period) - 2 * np.log(2 * np.pi)) / 3
        for period in (300.0, 3000.0)
    )


def _classical(key, n, ln_a):
    """Proposal covering the law: uniform ln a, logit e, atanh cos i, angles."""
    k = jax.random.split(key, 6)
    logit = [float(np.log(e / (1 - e))) for e in (0.05, 0.8)]
    coords = jnp.concatenate(
        [
            _uniform(k[0], n, *ln_a),
            _uniform(k[1], n, *logit),
            _uniform(k[2], n, -6.0, 6.0),
            *(_uniform(k[i], n, 0.0, 2 * np.pi) for i in (3, 4, 5)),
        ],
        -1,
    )
    log_q = -np.log((ln_a[1] - ln_a[0]) * (logit[1] - logit[0]) * 12 * (2 * np.pi) ** 3)
    return coords, log_q


def _eccentricity_vector(key, n, ln_a):
    k = jax.random.split(key, 5)
    u, log_area = _disk(k[1], 1.4, n)
    coords = jnp.concatenate(
        [
            _uniform(k[0], n, *ln_a),
            u,
            _uniform(k[2], n, -6.0, 6.0),
            *(_uniform(k[i], n, 0.0, 2 * np.pi) for i in (3, 4)),
        ],
        -1,
    )
    log_q = -np.log((ln_a[1] - ln_a[0]) * 12 * (2 * np.pi) ** 2) + log_area
    return coords, log_q


def _frame(gauge_sd):
    """Proposal for the frame chart: the radial law of the gauge, written out here."""

    def proposal(key, n, ln_a):
        k = jax.random.split(key, 4)
        u, log_area = _disk(k[1], 2.0, n)
        if gauge_sd is None:
            v = jax.random.normal(k[2], (n, 4))
            log_qv = -0.5 * jnp.sum(v**2, -1) - 2 * jnp.log(2 * jnp.pi)
        else:
            r = 1.0 + gauge_sd * jax.random.normal(k[2], (n,))
            d = jax.random.normal(k[3], (n, 4))
            v = r[:, None] * d / jnp.linalg.norm(d, axis=-1, keepdims=True)
            log_qv = (
                -0.5 * ((r - 1) / gauge_sd) ** 2
                - jnp.log(gauge_sd * jnp.sqrt(2 * jnp.pi))
                - jnp.log(2 * jnp.pi**2 * r**3)
            )
        coords = jnp.concatenate([_uniform(k[0], n, *ln_a), u, v], -1)
        return coords, -np.log(ln_a[1] - ln_a[0]) + log_area + log_qv

    return proposal


TI_A_LAW = OrbitLaw(E_LAW, a_AU=priorities.LogUniform(0.8, 1.6))
TI_P_LAW = OrbitLaw(E_LAW, period_d=priorities.LogUniform(300.0, 600.0))
STAR_MASS = (0.9, 1.3)


def _ti_support(law, model):
    """The ln n range and the largest a of a Thiele-Innes law with the star mass law."""
    inner = float(model.dynamics.planets.mass_Msun[0])
    mu = [GM * (m + inner) for m in STAR_MASS]
    if law.a_AU is not None:
        a = (0.8, 1.6)
        ln_n = (
            0.5 * (np.log(mu[0]) - 3 * np.log(a[1])),
            0.5 * (np.log(mu[1]) - 3 * np.log(a[0])),
        )
        return ln_n, a[1]
    period = (300.0, 600.0)
    ln_n = tuple(np.log(2 * np.pi / p) for p in period[::-1])
    return ln_n, (mu[1] * (period[1] / (2 * np.pi)) ** 2) ** (1 / 3)


def _thiele_innes(key, n, ln_n, cube):
    """Proposal for the Thiele-Innes chart: ln n, a disk and a cube of constants."""
    k = jax.random.split(key, 3)
    u, log_area = _disk(k[1], 1.4, n)
    coords = jnp.concatenate(
        [
            _uniform(k[0], n, *ln_n),
            u,
            jax.random.uniform(k[2], (n, 4), minval=-cube, maxval=cube),
        ],
        -1,
    )
    return coords, -np.log(ln_n[1] - ln_n[0]) + log_area - 4 * np.log(2 * cube)


def _summaries(*columns):
    """A summary function stacking the named columns of a coordinate row."""
    return lambda c: jnp.stack([f(c) for f in columns])


def _squared(lo, hi):
    return lambda c: jnp.sum(c[lo:hi] ** 2)


def _column(k):
    return lambda c: c[k]


CASES = {
    "classical-a": (
        ClassicalChart(),
        A_LAW,
        _classical,
        _summaries(_column(0), _column(1)),
    ),
    "classical-period": (
        ClassicalChart(),
        P_LAW,
        _classical,
        _summaries(_column(0), _column(1)),
    ),
    "eccentricity-vector": (
        EccentricityVectorChart(),
        A_LAW,
        _eccentricity_vector,
        _summaries(_column(0), _squared(1, 3)),
    ),
    "frame-ungauged": (
        FrameChart(gauge_sd=None),
        A_LAW,
        _frame(None),
        _summaries(_column(0), _squared(1, 3), _squared(3, 7)),
    ),
    "frame-gauged": (
        FrameChart(),
        A_LAW,
        _frame(0.1),
        _summaries(_column(0), _squared(1, 3), _squared(3, 7)),
    ),
}


@pytest.mark.parametrize("name", list(CASES))
def test_orbit_block_density_is_normalized_and_matches_its_draws(name):
    """Fails if the density does not integrate to one, or its draws follow another one.

    The proposal is uniform over a box (or the disk and gauge law, written out here)
    that contains the law's support, so ``mean(exp(log_prob) / q)`` estimates the
    integral; four standard errors of a fixed seed, with the error below 0.01 so that a
    missing factor 2 or a leftover 8 pi^2 is caught. The draws of ``sample`` must give
    the same means of a few coordinate summaries as the density does (four combined
    standard errors).
    """
    chart, law, proposal, summary = CASES[name]
    model = _system_model()
    block = _block(chart, (1,), law)
    coords, log_q = proposal(KEY, N, _ln_a_range(law, model))
    weight = jnp.exp(_log_prob(block, model, coords) - log_q)
    _check_unit(weight)
    _check_means(block, model, coords, weight, summary)


def _check_unit(weight, max_error=0.01):
    estimate = float(jnp.mean(weight))
    error = float(jnp.std(weight) / jnp.sqrt(weight.shape[0]))
    assert abs(estimate - 1.0) < 4 * error, (estimate, error)
    assert error < max_error


def _check_means(block, model, coords, weight, summary):
    """Means of the summaries under the density agree with the means of the draws."""
    values = jax.vmap(summary)(coords)
    by_density = np.asarray(jnp.mean(weight[:, None] * values, 0))
    error_density = np.asarray(jnp.std(weight[:, None] * values, 0) / np.sqrt(N))
    drawn = jax.vmap(summary)(_draws(block, model, N))
    by_draws = np.asarray(jnp.mean(drawn, 0))
    error_draws = np.asarray(jnp.std(drawn, 0) / np.sqrt(N))
    combined = np.sqrt(error_density**2 + error_draws**2)
    assert np.all(np.abs(by_density - by_draws) < 4 * combined), (
        by_density,
        by_draws,
        combined,
    )


class _TiltedLaw(AbstractOrbitLaw):
    """A law that is not symmetric under the line-of-sight reflection.

    The density against Haar is ``1 + 0.5 n_x`` with ``n_x`` the east component of the
    orbit normal, which flips under the reflection; it integrates to one.
    """

    base: OrbitLaw

    def log_prob(self, orbit):
        normal = rotate(orbit.quat, jnp.array([0.0, 0.0, 1.0]))
        return self.base.log_prob(orbit) + jnp.log1p(0.5 * normal[..., 0])

    def sample(self, key, *, mu_AU3_per_d2, epoch_d=0.0):
        raise NotImplementedError


THIELE_INNES = {
    "a": (TI_A_LAW, "symmetric"),
    "period": (TI_P_LAW, "symmetric"),
    "tilted": (_TiltedLaw(TI_A_LAW), "asymmetric"),
}


@pytest.mark.parametrize("name", list(THIELE_INNES))
def test_thiele_innes_block_density_is_normalized_and_matches_its_draws(name):
    """Fails if the sheet sum, the mass law or the star-mass Jacobian is wrong.

    Normalization as for the other charts, with a proposal over ``ln n``, a disk and a
    cube that contains every sky constant (each is at most ``a``). The tilted law is
    not symmetric under the reflection, so the chart's one sheet carries the sum over
    both sheets and still integrates to one. Draws are compared for the laws that can
    sample.
    """
    law, kind = THIELE_INNES[name]
    model = _system_model()
    block = _ti_block(law)
    ln_n, cube = _ti_support(TI_P_LAW if name == "period" else TI_A_LAW, model)
    coords, log_q = _thiele_innes(KEY, N, ln_n, cube)
    weight = jnp.exp(_log_prob(block, model, coords) - log_q)
    _check_unit(weight, max_error=0.04)
    if kind == "symmetric":
        summary = _summaries(_column(0), _squared(1, 3))
        _check_means(block, model, coords, weight, summary)


def test_the_tilted_law_differs_between_the_two_sheets():
    """Fails if the asymmetric test law is symmetric, which would make it vacuous."""
    x = jnp.array([[-4.0, 0.3, -0.2, 0.4, 0.5, -0.3, 0.45]])
    broad = OrbitLaw(
        priorities.Uniform(0.01, 0.99), a_AU=priorities.LogUniform(0.01, 100.0)
    )
    laws = [
        _TiltedLaw(broad).log_prob(
            ThieleInnesChart(branch=b).to_orbit(x, mu_AU3_per_d2=1.0)
        )
        for b in (1, -1)
    ]
    assert abs(float(laws[0][0] - laws[1][0])) > 0.1


def test_thiele_innes_sheet_sum_is_log_two_for_a_symmetric_law():
    """Fails if a symmetric law is not log 2 above the one-sheet density."""
    model = _system_model()
    block = _ti_block()
    x = block.inverse(model)
    chart = block.chart
    coords = jnp.stack([x[n] for n in chart.coordinate_names], -1)
    orbit = chart.to_orbit(
        coords, mu_AU3_per_d2=1.0, epoch_d=model.dynamics.orbits.epoch_d[:1]
    )
    enclosed = chart.implied_mu(coords)[..., 0] / GM
    star = enclosed - model.dynamics.planets.mass_Msun[0]
    one_sheet = (
        A_LAW.log_prob(orbit)[0]
        + chart.log_volume(coords)[0]
        + block.star_mass_Msun.log_prob(star)
        + jnp.log(enclosed)
    )
    np.testing.assert_allclose(
        block.log_prob(x, model), one_sheet + np.log(2.0), rtol=1e-12
    )


def test_a_two_row_block_is_the_sum_of_its_rows(model):
    """Fails if per-row law parameters are not broadcast over the freed rows.

    A law with a trailing rows axis on its parameters gives the same density as one
    single-row block per row with the matching scalar law.
    """
    low, high = jnp.array([0.8, 4.0]), jnp.array([1.4, 12.0])
    both = _block(
        ClassicalChart(),
        (0, 2),
        OrbitLaw(E_LAW, a_AU=priorities.LogUniform(low, high)),
    )
    x = both.inverse(model)
    total = both.log_prob(x, model)
    parts = 0.0
    for k, row in enumerate((0, 2)):
        single = _block(
            ClassicalChart(),
            (row,),
            OrbitLaw(E_LAW, a_AU=priorities.LogUniform(low[k], high[k])),
        )
        parts += single.log_prob(single.inverse(model), model)
    np.testing.assert_allclose(total, parts, rtol=1e-12)
    draw = both.sample(KEY, model)
    assert all(v.shape == (2,) for v in draw.values())
    assert np.isfinite(both.log_prob(draw, model))


def test_same_map_ignores_the_law_only():
    """Fails if a block with another chart, rows or system is taken for the same map."""
    base = _block(ClassicalChart(), (0,))
    other_law = _block(ClassicalChart(), (0,), P_LAW)
    assert base.same_map(other_law)
    assert not base.same_map(_block(FrameChart(), (0,)))
    assert not base.same_map(_block(ClassicalChart(), (1,)))
    assert not base.same_map(_block(ClassicalChart(), (0,), system="other"))
    assert not base.same_map(_ti_block())
    ti = _ti_block()
    assert ti.same_map(_ti_block(P_LAW))
    assert not ti.same_map(_ti_block(rows=(1,)))
    assert not ti.same_map(base)


def test_thiele_innes_star_mass_zero_wins_over_an_infinite_enclosed_mass():
    """Fails if an infinite size gives NaN (``-inf + log inf``) with the guard gone.

    At ``ln n = -1e5`` the implied enclosed mass is infinite, outside the star mass
    law's support; the block reports ``-inf``.
    """
    block, model = _ti_block(rows=(1,)), _system_model()
    x = {n: jnp.array([0.0]) for n in block.coordinate_names}
    x["ln_n_rad_per_d"] = jnp.array([-1e5])
    assert float(block.log_prob(x, model)) == -np.inf
