"""Chart blocks: forward and inverse, volume normalization and writer refusal."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.nn import sigmoid

from orbix.system import (
    AbstractBlock,
    Body,
    CartesianChart,
    ClassicalChart,
    EccentricityVectorChart,
    FrameChart,
    OrbitBlock,
    SkyFrame,
    System,
    ThieleInnesBlock,
    ThieleInnesChart,
    block_order,
    check_writers,
    enclosed_mu,
    paths_overlap,
    sky_block_geometry,
)
from orbix.system.core import GM_SUN_AU3_PER_D2 as GM
from orbix.system.orbit_blocks import _get

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
        model = eqx.tree_at(lambda m, p=path: _get(m, p), model, value)
    return model


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
    block = OrbitBlock(chart, rows=(0, 2))
    assert isinstance(block, AbstractBlock)
    assert block.owns == "dynamics.orbits"
    assert block.writes == ("dynamics.orbits",)
    assert block.reads == MASS_PATHS
    assert OrbitBlock(chart, rows=(1,), system="inner.dynamics").owns == (
        "inner.dynamics.orbits"
    )


def test_thiele_innes_block_writes_the_star_mass_and_reads_planet_masses():
    """Fails if the implied star mass is not a declared write."""
    block = ThieleInnesBlock(rows=(1,))
    assert block.writes == ("dynamics.orbits", "dynamics.star.mass_Msun")
    assert block.reads == ("dynamics.planets.mass_Msun",)
    with pytest.raises(ValueError, match="exactly one row"):
        ThieleInnesBlock(rows=(0, 1))
    with pytest.raises(ValueError, match="use ThieleInnesBlock"):
        OrbitBlock(ThieleInnesChart(), rows=(0,))
    with pytest.raises(ValueError, match="distinct"):
        OrbitBlock(ClassicalChart(), rows=(1, 1))


@pytest.mark.parametrize("chart", CHARTS, ids=_id)
@pytest.mark.parametrize("rows", [(1,), (0, 2)])
def test_forward_of_inverse_reproduces_the_orbits(model, chart, rows):
    """Fails if the freed rows move, a held row changes, or the rate leaves its mu.

    Positions agree to 1e-10 AU (float64 trigonometry through the chart); held rows
    are returned bit for bit.
    """
    block = OrbitBlock(chart, rows)
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
    for name in ("a_AU", "n_rad_per_d", "ex", "ey", "epoch_d"):
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
    block = OrbitBlock(chart, rows=(0, 2))
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
    """Fails if a freed row keeps the old star mass after another entry changes it."""
    block = OrbitBlock(ClassicalChart(), rows=(1,))
    heavier = _assign(model, {MASS_PATHS[0]: jnp.asarray(1.6)})
    orbit = block.forward(block.inverse(model), heavier)["dynamics.orbits"]
    mu = enclosed_mu(jnp.asarray(1.6), model.dynamics.planets.mass_Msun)
    np.testing.assert_allclose(
        orbit.n_rad_per_d[1], jnp.sqrt(mu[1] / orbit.a_AU[1] ** 3), rtol=1e-12
    )
    # A held row keeps its stored rate: the block never rebuilds what it does not own.
    assert orbit.n_rad_per_d[0] == model.dynamics.orbits.n_rad_per_d[0]


def test_forward_is_a_per_draw_function_that_batches_with_vmap(model):
    """Fails if forward cannot be vmapped over draws, or is not traceable."""
    block = OrbitBlock(ClassicalChart(), rows=(0, 1))
    x = block.inverse(model)
    batch = {n: jnp.stack([v, v + 0.1, v - 0.1]) for n, v in x.items()}
    out = jax.jit(jax.vmap(lambda xi: block.forward(xi, model)["dynamics.orbits"]))(
        batch
    )
    assert out.a_AU.shape == (3, 3) and out.quat.shape == (3, 3, 4)
    one = block.forward({n: v[1] for n, v in batch.items()}, model)["dynamics.orbits"]
    np.testing.assert_allclose(out.a_AU[1], one.a_AU, rtol=1e-14)
    assert jax.jit(jax.vmap(lambda xi: block.log_volume(xi, model)))(batch).shape == (
        3,
    )


def test_thiele_innes_forward_of_inverse_reproduces_orbit_and_star_mass(model):
    """Fails if the implied star mass or the row is not reproduced.

    The model's rates satisfy n^2 a^3 = GM (Ms + sum m), so the round trip returns the
    stored star mass to 1e-12 relative and the positions to 1e-10 AU.
    """
    block = ThieleInnesBlock(rows=(1,))
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
    x = {n: v + (0.2 if n == "ln_n" else 0.0) for n, v in block.inverse(model).items()}
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


# --- normalization against the chart volume -------------------------------------------

N = 600_000
KEY = jax.random.PRNGKey(0)


def _estimate(log_density, log_proposal, x):
    """Importance estimate of the integral of exp(log_density), with its std error."""
    w = jnp.exp(log_density - log_proposal)
    return float(jnp.mean(w)), float(jnp.std(w) / jnp.sqrt(w.shape[0]))


def _keys(n):
    return jax.random.split(KEY, n)


def _check_unit(estimate, error, max_error=0.01):
    # Four standard errors of the Monte Carlo mean (fixed seed, so deterministic).
    assert abs(estimate - 1.0) < 4 * error, (estimate, error)
    assert error < max_error  # sharp enough to catch a missing factor 2


def _disk(key, radius, n):
    k1, k2 = jax.random.split(key)
    r = radius * jnp.sqrt(jax.random.uniform(k1, (n,)))
    t = jax.random.uniform(k2, (n,), maxval=2 * jnp.pi)
    return jnp.stack([r * jnp.cos(t), r * jnp.sin(t)], -1), -jnp.log(jnp.pi * radius**2)


def _block_volume(block, model, coords):
    """Block log volume of a batch of single-row draws, as a dict per draw."""
    names = block.chart.coordinate_names
    return jax.jit(
        jax.vmap(
            lambda c: block.log_volume(
                {n: c[k][None] for k, n in enumerate(names)}, model
            )
        )
    )(coords)


def test_classical_block_density_is_normalized(model):
    """Fails if the chart volume is not the Jacobian to the reference measure.

    The prior is uniform in the reference measure (d ln a, dex dey, d cos i, dW, du)
    on a box of physical size nu; the box in chart coordinates is sampled uniformly.
    """
    block = OrbitBlock(ClassicalChart(), rows=(1,))
    k = jax.random.split(KEY, 6)
    la, l_e, z = (0.0, 2.0), (-2.0, 1.5), 3.0
    ang = jnp.stack(
        [jax.random.uniform(k[i], (N,), maxval=2 * jnp.pi) for i in (0, 1, 2)], -1
    )
    coords = jnp.concatenate(
        [
            jax.random.uniform(k[3], (N, 1), minval=la[0], maxval=la[1]),
            jax.random.uniform(k[4], (N, 1), minval=l_e[0], maxval=l_e[1]),
            jax.random.uniform(k[5], (N, 1), minval=-z, maxval=z),
            ang,
        ],
        -1,
    )
    e0, e1 = sigmoid(jnp.array(l_e[0])), sigmoid(jnp.array(l_e[1]))
    nu = (
        (la[1] - la[0]) * jnp.pi * (e1**2 - e0**2) * 2 * jnp.tanh(z) * (2 * jnp.pi) ** 2
    )
    log_q = -jnp.log((la[1] - la[0]) * (l_e[1] - l_e[0]) * 2 * z * (2 * jnp.pi) ** 3)
    _check_unit(
        *_estimate(
            _block_volume(block, model, coords) - jnp.log(nu), log_q, coords[:, 0]
        )
    )


def test_eccentricity_vector_block_density_is_normalized(model):
    """Fails if the disk-map or inclination Jacobian is wrong.

    Uniform in the reference measure over a disk of eccentricity vectors; the chart
    disk is the image of a plane disk of radius R.
    """
    block = OrbitBlock(EccentricityVectorChart(), rows=(1,))
    k = jax.random.split(KEY, 5)
    la, z, radius = (0.0, 2.0), 3.0, 4.0
    u, log_area = _disk(k[0], radius, N)
    ang = jnp.stack(
        [jax.random.uniform(k[i], (N,), maxval=2 * jnp.pi) for i in (1, 2)], -1
    )
    coords = jnp.concatenate(
        [
            jax.random.uniform(k[3], (N, 1), minval=la[0], maxval=la[1]),
            u,
            jax.random.uniform(k[4], (N, 1), minval=-z, maxval=z),
            ang,
        ],
        -1,
    )
    nu = (
        (la[1] - la[0])
        * jnp.pi
        * radius**2
        / (1 + radius**2)
        * 2
        * jnp.tanh(z)
        * (2 * jnp.pi) ** 2
    )
    log_q = -jnp.log(la[1] - la[0]) + log_area - jnp.log(2 * z * (2 * jnp.pi) ** 2)
    _check_unit(
        *_estimate(
            _block_volume(block, model, coords) - jnp.log(nu), log_q, coords[:, 0]
        )
    )


@pytest.mark.parametrize("gauge_sd", [None, 0.1])
def test_frame_block_density_is_normalized(model, gauge_sd):
    """Fails if the disk map or the quaternion gauge density is wrong.

    Uniform in the reference measure with normalized Haar orientation. With
    ``gauge_sd=None`` the chart density is normalized. With a gauge, the chart omits the
    constant ``2 pi^2 gauge_sd sqrt(2 pi)`` of the radial law, which the check restores.
    """
    block = OrbitBlock(FrameChart(gauge_sd), rows=(0,))
    k = jax.random.split(KEY, 5)
    la, radius = (0.0, 2.0), 4.0
    u, log_area = _disk(k[0], radius, N)
    if gauge_sd is None:
        v = jax.random.normal(k[1], (N, 4))
        log_qv = -0.5 * jnp.sum(v**2, -1) - 2 * jnp.log(2 * jnp.pi)
        constant = 1.0
    else:
        r = 1.0 + gauge_sd * jax.random.normal(k[1], (N,))
        d = jax.random.normal(k[2], (N, 4))
        v = r[:, None] * d / jnp.linalg.norm(d, axis=-1, keepdims=True)
        log_qv = (
            -0.5 * ((r - 1) / gauge_sd) ** 2
            - jnp.log(gauge_sd * jnp.sqrt(2 * jnp.pi))
            - jnp.log(2 * jnp.pi**2 * r**3)
        )
        constant = 2 * jnp.pi**2 * gauge_sd * jnp.sqrt(2 * jnp.pi)
    coords = jnp.concatenate(
        [jax.random.uniform(k[3], (N, 1), minval=la[0], maxval=la[1]), u, v], -1
    )
    nu = (la[1] - la[0]) * jnp.pi * radius**2 / (1 + radius**2)
    log_q = -jnp.log(la[1] - la[0]) + log_area + log_qv
    log_density = _block_volume(block, model, coords) - jnp.log(nu * constant)
    _check_unit(*_estimate(log_density, log_q, coords[:, 0]))


def test_thiele_innes_block_density_is_normalized(model):
    """Fails if the Thiele-Innes volume, with its star-mass Jacobian, is not normalized.

    Uniform in (d ln a, dex dey, Haar, dMs) on a physical box. The sky constants are
    sampled in a cube that contains the box, since each is at most ``a``.
    """
    block = ThieleInnesBlock(rows=(0,))
    chart = block.chart
    k = jax.random.split(KEY, 4)
    a_lim, ms_lim, radius = (0.8, 1.6), (0.9, 1.3), 3.0
    inner = float(model.dynamics.planets.mass_Msun[0])
    mu_lim = [GM * (m + inner) for m in ms_lim]
    # n = sqrt(mu / a^3) bounds ln n
    ln_n = (
        0.5 * (np.log(mu_lim[0]) - 3 * np.log(a_lim[1])),
        0.5 * (np.log(mu_lim[1]) - 3 * np.log(a_lim[0])),
    )
    u, log_area = _disk(k[0], radius, N)
    cube = a_lim[1]
    coords = jnp.concatenate(
        [
            jax.random.uniform(k[1], (N, 1), minval=ln_n[0], maxval=ln_n[1]),
            u,
            jax.random.uniform(k[2], (N, 4), minval=-cube, maxval=cube),
        ],
        -1,
    )
    a, _ = jax.vmap(lambda c: sky_block_geometry(*c[3:]))(coords)
    ms = jax.vmap(lambda c: chart.implied_mu(c) / GM)(coords) - inner
    inside = (a > a_lim[0]) & (a < a_lim[1]) & (ms > ms_lim[0]) & (ms < ms_lim[1])
    volume = _block_volume(block, model, coords)
    nu = (
        np.log(a_lim[1] / a_lim[0])
        * np.pi
        * radius**2
        / (1 + radius**2)
        * 8
        * np.pi**2
        * (ms_lim[1] - ms_lim[0])
    )
    log_q = -np.log(ln_n[1] - ln_n[0]) + log_area - 4 * np.log(2 * cube)
    estimate, error = _estimate(
        jnp.where(inside, volume - np.log(nu), -jnp.inf), log_q, coords[:, 0]
    )
    _check_unit(estimate, error, max_error=0.03)


# --- writers and order -------------------------------------------------------


def test_paths_overlap_matches_whole_nodes_only():
    """Fails if a path prefix is confused with a longer name."""
    assert paths_overlap("dynamics.orbits", "dynamics.orbits")
    assert paths_overlap("dynamics.orbits", "dynamics.orbits[0].ex")
    assert paths_overlap("dynamics.frame.dist_pc", "dynamics.frame")
    assert not paths_overlap("dynamics.orbits", "dynamics.orbits_extra")
    assert not paths_overlap("planets[0].Rp_Rearth", "planets[1].Rp_Rearth")


class _Mass(AbstractBlock):
    """A stand-in block that owns the star mass."""

    @property
    def owns(self):
        return "dynamics.star.mass_Msun"

    @property
    def reads(self):
        return ()

    @property
    def writes(self):
        return ("dynamics.star.mass_Msun",)

    def forward(self, x, model):
        return {}

    def log_volume(self, x, model):
        return 0.0

    def inverse(self, model):
        return {}


def test_two_writers_of_one_path_are_refused():
    """Fails if two entries may write one path, or a key is not the owned node."""
    classical = OrbitBlock(ClassicalChart(), rows=(0,))
    thiele = ThieleInnesBlock(rows=(1,))
    mass = _Mass()
    check_writers({"dynamics.orbits": classical})
    check_writers(
        {"dynamics.orbits": classical}, leaf_paths=["dynamics.star.mass_Msun"]
    )
    # a free leaf inside the node a block owns
    with pytest.raises(ValueError, match=r"dynamics\.orbits.*two writers"):
        check_writers({"dynamics.orbits": classical}, leaf_paths=["dynamics.orbits.ex"])
    # the star mass a Thiele-Innes block writes cannot also be a free leaf or a block
    with pytest.raises(ValueError, match=r"dynamics\.star\.mass_Msun.*two writers"):
        check_writers(
            {"dynamics.orbits": thiele}, leaf_paths=["dynamics.star.mass_Msun"]
        )
    with pytest.raises(ValueError, match=r"dynamics\.star\.mass_Msun.*two writers"):
        check_writers({"dynamics.orbits": thiele, mass.owns: mass})
    with pytest.raises(ValueError, match="must be keyed by the path it owns"):
        check_writers({"orbits": classical})


def test_blocks_run_after_the_blocks_that_write_what_they_read():
    """Fails if a block would read a mass before another block has written it."""
    orbit = OrbitBlock(ClassicalChart(), rows=(0,))
    mass = _Mass()
    expected = (mass.owns, "dynamics.orbits")
    assert block_order({"dynamics.orbits": orbit, mass.owns: mass}) == expected
    assert block_order({mass.owns: mass, "dynamics.orbits": orbit}) == expected
