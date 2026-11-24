"""The sky-position block: offsets at one time, the orbit shape held."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import priorities
import pytest
from hwoutils.blocks import AbstractBlock, check_ties
from hwoutils.model_paths import assign
from jax.test_util import check_grads

from orbix.system import (
    Body,
    SkyFrame,
    SkyPositionBlock,
    System,
    enclosed_mu,
)

TIME = 40.0
TIMES = jnp.array([-60.0, 0.0, TIME, 150.0])
DYN = "dynamics"
REAL = (priorities.Normal(0.0, 0.5), priorities.Normal(0.1, 0.4))
BOUNDED = (priorities.Uniform(-1.5, 1.5), priorities.Normal(0.0, 0.5))


class _Model(eqx.Module):
    """The smallest model with the path layout of a skyscapes Scene."""

    dynamics: System


# Inner planets of Jupiter mass and above, so that the center of mass the Jacobi rows
# are relative to sits about 1e-3 of an orbit from the star. Row 1 is circular and row
# 2 face-on, the two points where the classical elements are singular.
ORBITS = dict(
    a_AU=jnp.array([1.0, 3.0, 8.0]),
    e=jnp.array([0.1, 0.0, 0.3]),
    i_rad=jnp.array([0.7, 0.9, 0.0]),
    W_rad=jnp.array([0.5, 1.5, 2.5]),
    w_rad=jnp.array([1.0, 2.0, 3.0]),
    M0_rad=jnp.array([2.0, 0.3, 1.1]),
    epoch_d=jnp.array([3.0, 4.0, 5.0]),
)
MASSES = jnp.array([3e-3, 1e-3, 3e-4])


def _system_model(**changes):
    orbits = {**ORBITS, **changes}
    return _Model(
        System.from_classical(
            Body(jnp.asarray(1.1)), Body(MASSES), SkyFrame(jnp.asarray(10.0)), **orbits
        )
    )


def _scene_model():
    pytest.importorskip("skyscapes")
    from skyscapes.physical_model import LambertianPhysicalModel
    from skyscapes.scene import FlatStar, Planet, Scene

    def planet():
        return Planet(jnp.ones(1), LambertianPhysicalModel(jnp.array([0.3])))

    return Scene.from_classical(
        FlatStar(1e9),
        (planet(), planet(), planet()),
        star_mass_Msun=1.1,
        planet_mass_Msun=MASSES,
        dist_pc=10.0,
        **ORBITS,
    )


@pytest.fixture(params=["system", "scene"])
def model(request):
    """A bare orbix model, then a real skyscapes Scene, with the same orbits."""
    return _system_model() if request.param == "system" else _scene_model()


def _block(rows, laws=REAL, time_d=TIME, system=DYN):
    return SkyPositionBlock(*laws, time_d, rows=rows, system=system)


def _apply(model, block, x):
    return assign(model, block.owns, block.forward(x, model)[block.owns])


def _offsets(model, rows, time_d=TIME):
    """Star-relative (east, north) of the listed planets, read by ``System.solve``."""
    eph = model.dynamics.solve(jnp.array([time_d]))
    pairs = [eph.offset_arcsec(k + 1, 0) for k in rows]
    return jnp.stack([jnp.stack([e[0], n[0]]) for e, n in pairs])


def _states(orbits):
    r, v = orbits.at(TIMES)
    return r, v


@pytest.mark.parametrize("rows", [(1,), (2, 0)])
def test_forward_puts_each_planet_at_its_star_relative_offset(model, rows):
    """Fails if a freed planet misses its offset, or a held row's geometry changes.

    Truth: the star-relative offsets that ``System.solve`` reads at ``time_d`` through
    the barycentric map, which the block does not call; with real laws the coordinates
    are the offsets in arcseconds. Planets of 3e-3 and 1e-3 solar masses inside the
    freed rows put the Jacobi origin about 1e-3 of an orbit off the star, so skipping
    the center-of-mass correction, or applying it out of order (``rows=(2, 0)``), is
    off by more than 1e-4 arcsec. Tolerance: 1e-10 arcsec, float64 Kepler solve and
    trigonometry (measured 1e-16). A held row keeps its geometry bit for bit, and every
    row's rate is the one its enclosed mass implies (1e-12 relative, rounding).
    """
    block = _block(rows)
    x = {
        "east_arcsec": jnp.array([0.31, -0.07])[: len(rows)],
        "north_arcsec": jnp.array([-0.42, 0.05])[: len(rows)],
    }
    moved = _apply(model, block, x)
    np.testing.assert_allclose(
        _offsets(moved, rows),
        jnp.stack([x["east_arcsec"], x["north_arcsec"]], -1),
        rtol=0,
        atol=1e-10,
    )
    new, old = moved.dynamics.orbits, model.dynamics.orbits
    held = jnp.asarray([k for k in range(3) if k not in rows])
    for name in ("a_AU", "ex", "ey", "epoch_d", "quat"):
        assert jnp.array_equal(getattr(new, name)[held], getattr(old, name)[held])
    assert jnp.array_equal(new.epoch_d[jnp.asarray(rows)], jnp.full(len(rows), TIME))
    mu = enclosed_mu(model.dynamics.star.mass_Msun, model.dynamics.planets.mass_Msun)
    np.testing.assert_allclose(new.n_rad_per_d**2 * new.a_AU**3, mu, rtol=1e-12)


def test_forward_follows_a_mass_that_another_entry_frees():
    """Fails if a row keeps the rate of a star mass the model no longer has.

    Truth: Kepler's third law with the enclosed masses of the model (1e-12 relative,
    rounding), for the freed row and for the held rows, whose geometry is kept; and
    the freed planet still sits at its offset (1e-10 arcsec). The star mass moves from
    1.1 to 1.6 solar masses, so a stale rate is off by 20 percent.
    """
    model = assign(_system_model(), "dynamics.star.mass_Msun", jnp.asarray(1.6))
    block = _block((1,))
    x = {"east_arcsec": jnp.array([0.12]), "north_arcsec": jnp.array([0.25])}
    moved = _apply(model, block, x)
    new = moved.dynamics.orbits
    mu = enclosed_mu(jnp.asarray(1.6), MASSES)
    np.testing.assert_allclose(new.n_rad_per_d**2 * new.a_AU**3, mu, rtol=1e-12)
    assert jnp.array_equal(new.a_AU[0], model.dynamics.orbits.a_AU[0])
    np.testing.assert_allclose(_offsets(moved, (1,)), [[0.12, 0.25]], atol=1e-10)


def test_forward_and_inverse_are_mutually_inverse(model):
    """Fails if inverse is not the left and right inverse of forward.

    Truth: definitional. ``inverse(forward(x))`` returns ``x`` to 1e-10 (float64 trig
    through the Kepler solve and the logit; measured 1e-15). ``forward(inverse(model))``
    rebuilds each freed orbit: the same states at four times to 1e-10 AU and AU/day,
    although it is written at another epoch.
    """
    block = _block((2, 0), BOUNDED)
    x = {
        "mapped_east_arcsec": jnp.array([0.4, -1.3]),
        "north_arcsec": jnp.array([-0.6, 0.07]),
    }
    back = block.inverse(_apply(model, block, x))
    for name in x:
        np.testing.assert_allclose(back[name], x[name], rtol=0, atol=1e-10)
    rebuilt = _apply(model, block, block.inverse(model)).dynamics.orbits
    for new, old in zip(_states(rebuilt), _states(model.dynamics.orbits), strict=True):
        np.testing.assert_allclose(new, old, rtol=0, atol=1e-10)


SHAPES = {
    "generic": (0.3, 1.0),
    "circular": (0.0, 1.0),
    "face-on": (0.3, 0.0),
    "circular-face-on": (0.0, 0.0),
}


@pytest.mark.parametrize("shape", list(SHAPES))
def test_the_move_is_a_kepler_similarity_also_at_circular_and_face_on(shape):
    """Fails if the move changes the orbit's shape, or its gradient is wrong there.

    Truth: a rotation about the line of sight and a dilation by ``s``, the ratio of
    the projected separations, keep the eccentricity and the inclination (here the
    line-of-sight part of the unit angular momentum), multiply ``a`` by ``s``, and the
    written rate satisfies ``n^2 a^3 = mu`` of the enclosed mass. Row 0 is freed, so
    its Jacobi and star-relative positions coincide. Tolerance: 1e-12, float64
    rounding (exact algebra). The gradient of the moved positions with respect to the
    coordinates matches central differences (``check_grads``, first order, step 1e-4,
    tolerance 1e-4): circular and face-on orbits are where the classical elements and
    their volume are singular, and the map must stay smooth there.
    """
    e, inc = SHAPES[shape]
    model = _system_model(
        e=ORBITS["e"].at[0].set(e), i_rad=ORBITS["i_rad"].at[0].set(inc)
    )
    block = _block((0,))
    x = {"east_arcsec": jnp.array([-0.03]), "north_arcsec": jnp.array([0.17])}
    old = model.dynamics.orbits
    new = _apply(model, block, x).dynamics.orbits
    r_old, v_old = old.at_each(jnp.asarray(TIME))
    r_new, v_new = new.at_each(jnp.asarray(TIME))
    np.testing.assert_allclose(new.e[0], old.e[0], rtol=0, atol=1e-12)

    def cos_i(r, v):
        h = jnp.cross(r[0], v[0])
        return h[2] / jnp.linalg.norm(h)

    np.testing.assert_allclose(
        cos_i(r_new, v_new), cos_i(r_old, v_old), rtol=0, atol=1e-12
    )
    scale = jnp.hypot(*r_new[0, :2]) / jnp.hypot(*r_old[0, :2])
    np.testing.assert_allclose(new.a_AU[0] / old.a_AU[0], scale, rtol=1e-12)
    mu = enclosed_mu(model.dynamics.star.mass_Msun, model.dynamics.planets.mass_Msun)
    np.testing.assert_allclose(new.n_rad_per_d[0] ** 2 * new.a_AU[0] ** 3, mu[0])

    def positions(east, north):
        orbits = block.forward({"east_arcsec": east, "north_arcsec": north}, model)[
            block.owns
        ]
        return orbits.at(jnp.array([TIME + 30.0]))[0][0]

    check_grads(
        positions,
        (x["east_arcsec"], x["north_arcsec"]),
        order=1,
        modes=("fwd", "rev"),
        eps=1e-4,
        atol=1e-4,
        rtol=1e-4,
    )


def test_every_offset_on_the_field_gives_a_bound_orbit_there():
    """Fails if an offset leaves the bound set, as holding the velocity would.

    Truth: a bound orbit has finite ``a > 0``, ``e < 1`` and a unit quaternion; and
    it puts the planet at the offset (1e-10 arcsec, as above). The grid is 40 by 40
    over +-2 arcsec (0.2 to 28 AU of Jacobi separation at 10 pc for a 3 AU orbit),
    with no node at the Jacobi origin, plus one offset of 1e-6 arcsec. Exact.
    """
    model = _system_model()
    block = _block((1,))
    grid = jnp.linspace(-2.0, 2.0, 40)
    east, north = (v.ravel() for v in jnp.meshgrid(grid, grid))
    east, north = jnp.append(east, 1e-6), jnp.append(north, 0.0)

    @jax.jit
    @jax.vmap
    def one(e, n):
        x = {"east_arcsec": e[None], "north_arcsec": n[None]}
        moved = _apply(model, block, x)
        return moved.dynamics.orbits, _offsets(moved, (1,))[0]

    orbits, offsets = one(east, north)
    assert bool(jnp.all(jnp.isfinite(orbits.a_AU) & (orbits.a_AU > 0)))
    assert bool(jnp.all(orbits.e < 1))
    np.testing.assert_allclose(jnp.linalg.norm(orbits.quat, axis=-1), 1.0, rtol=1e-12)
    np.testing.assert_allclose(offsets, jnp.stack([east, north], -1), atol=1e-10)


LAWS = {
    "interval-real": BOUNDED,
    "lower-upper": (
        priorities.HalfNormal(0.8),
        priorities.TruncatedDistribution(priorities.Normal(0.0, 1.0), high=0.4),
    ),
}


@pytest.mark.parametrize("laws", list(LAWS))
def test_log_prob_is_the_density_of_the_offsets_and_is_normalized(laws):
    """Fails if the density is not the two laws carried onto the coordinates.

    Truth: the change of variables from the offsets to the coordinates. The offsets
    are read back through ``System.solve`` and differentiated by ``jax.jacfwd``, an
    independent path from the block's own map; ``log_prob`` equals the sum of the
    laws at those offsets plus ``log |det d offset / d x|`` to 1e-9 (float64 through
    the Kepler solve; measured 9e-16). Each row's offsets depend only on its own
    coordinates, so the off-diagonal blocks of the Jacobian are zero (1e-12). Second
    truth: normalization over the plane, by the trapezoid rule on a 401 by 401 grid
    over +-20 for one row, to 1e-6 (the integrand decays at least as ``exp(-|x|)``;
    measured 1e-13).
    """
    east_law, north_law = LAWS[laws]
    model = _system_model()
    block = _block((2, 0), LAWS[laws])
    names = block.coordinate_names
    x = {names[0]: jnp.array([0.3, -0.8]), names[1]: jnp.array([-0.4, 1.1])}

    def offsets(flat):
        y = {names[0]: flat[:2], names[1]: flat[2:]}
        return _offsets(_apply(model, block, y), (2, 0)).T.ravel()

    flat = jnp.concatenate([x[names[0]], x[names[1]]])
    value, jac = jax.jit(lambda f: (offsets(f), jax.jacfwd(offsets)(f)))(flat)
    np.testing.assert_allclose(jac - jnp.diag(jnp.diag(jac)), 0.0, atol=1e-12)
    expected = (
        jnp.sum(east_law.log_prob(value[:2]))
        + jnp.sum(north_law.log_prob(value[2:]))
        + jnp.sum(jnp.log(jnp.abs(jnp.diag(jac))))
    )
    np.testing.assert_allclose(block.log_prob(x, model), expected, rtol=0, atol=1e-9)

    one = _block((1,), LAWS[laws])
    grid = jnp.linspace(-20.0, 20.0, 401)
    density = jax.vmap(
        jax.vmap(
            lambda a, b: jnp.exp(
                one.log_prob({names[0]: a[None], names[1]: b[None]}, model)
            ),
            (None, 0),
        ),
        (0, None),
    )(grid, grid)
    total = jnp.trapezoid(jnp.trapezoid(density, grid, axis=1), grid)
    np.testing.assert_allclose(total, 1.0, rtol=0, atol=1e-6)


def test_a_free_distance_is_read_from_the_model():
    """Fails if the block keeps a distance other than the model's.

    Truth: an offset of ``x`` arcsec at ``d`` pc is ``x d`` AU (the parsec), so after
    a leaf law moves the distance from 10 to 14 pc the planet still sits at its
    coordinates (1e-10 arcsec), and the orbit of the innermost row, whose Jacobi origin
    is the star, is 1.4 times larger than the one written at 10 pc (1e-12 relative,
    exact algebra).
    """
    model = _system_model()
    far = assign(model, "dynamics.frame.dist_pc", jnp.asarray(14.0))
    block = _block((0,))
    x = {"east_arcsec": jnp.array([0.2]), "north_arcsec": jnp.array([-0.15])}
    moved = _apply(far, block, x)
    np.testing.assert_allclose(
        _offsets(moved, (0,)), [[0.2, -0.15]], rtol=0, atol=1e-10
    )
    near = _apply(model, block, x).dynamics.orbits
    np.testing.assert_allclose(
        moved.dynamics.orbits.a_AU[0] / near.a_AU[0], 1.4, rtol=1e-12
    )


def test_contract_paths_names_walls_and_ties():
    """Fails if the block declares a path, a name or a wall it does not have.

    Truth: definitional. The block reads both masses and the distance and writes the
    orbits, so ``check_ties`` accepts it beside free masses and a free distance; a
    bounded law takes the ``mapped_`` prefix and a real one does not; every bound is
    mapped, so there are no walls; a malformed block is refused at construction.
    """
    block = _block((0, 2), BOUNDED)
    assert isinstance(block, AbstractBlock)
    assert block.owns == "dynamics.orbits"
    assert block.writes == ("dynamics.orbits",)
    assert block.reads == (
        "dynamics.star.mass_Msun",
        "dynamics.planets.mass_Msun",
        "dynamics.frame.dist_pc",
    )
    assert block.coordinate_names == ("mapped_east_arcsec", "north_arcsec")
    assert block.periods == (None, None)
    assert block.walls == ()
    model = _system_model()
    leaves = ("dynamics.star.mass_Msun", "dynamics.planets.mass_Msun")
    leaves += ("dynamics.frame.dist_pc",)
    check_ties(
        model,
        {block.owns: block.writes, **{p: (p,) for p in leaves}},
        {block.owns: block.reads},
    )
    with pytest.raises(ValueError, match="distinct"):
        _block((1, 1))
    with pytest.raises(ValueError, match="one per freed row"):
        _block((0,), (priorities.Normal(jnp.zeros(3), 1.0), REAL[1]))
    with pytest.raises(ValueError, match="scalar"):
        SkyPositionBlock(*REAL, jnp.zeros(2), rows=(0,), system=DYN)


def test_sample_and_from_unit_draw_the_offsets_from_the_laws():
    """Fails if a draw does not follow the laws, or the unit map is not their icdf.

    Truth: ``from_unit`` gives offsets whose law CDF is the unit-cube point (1e-10,
    float64 through the Kepler solve); 2000 draws of ``sample`` under ``jit`` and
    ``vmap`` give law CDF values with mean 1/2 within 4 standard errors
    (``4 sqrt(1 / 12 / 2000) = 0.026``, Monte Carlo, seed 0).
    """
    model = _system_model()
    block = _block((2, 0), BOUNDED)
    laws = BOUNDED
    c = jnp.array([0.2, 0.7, 0.9, 0.35])
    x = block.from_unit(c, model)
    offsets = _offsets(_apply(model, block, x), (2, 0))
    cdf = jnp.stack([laws[k].cdf(offsets[:, k]) for k in range(2)], -1)
    np.testing.assert_allclose(cdf.ravel(), c, rtol=0, atol=1e-10)
    assert block.unit_size == 4

    @jax.jit
    @jax.vmap
    def draw(key):
        y = block.sample(key, model)
        return _offsets(_apply(model, block, y), (2, 0))

    drawn = draw(jax.random.split(jax.random.key(0), 2000))
    for k in range(2):
        u = laws[k].cdf(drawn[..., k])
        assert abs(float(jnp.mean(u)) - 0.5) < 0.026


def test_same_map_tracks_the_supports_and_the_time():
    """Fails if a block with another support or time is taken for the same map.

    Truth: definitional; the coordinates mean the same offsets only when both supports
    and ``time_d`` agree. A law with the same support (a moved mean) is the same map.
    """
    block = _block((1,), BOUNDED)
    moved_mean = _block((1,), (BOUNDED[0], priorities.Normal(0.3, 0.5)))
    assert block.same_layout(moved_mean) and block.same_map(moved_mean)
    wider = _block((1,), (priorities.Uniform(-2.0, 2.0), BOUNDED[1]))
    later = _block((1,), BOUNDED, time_d=TIME + 1.0)
    for other in (wider, later):
        assert block.same_layout(other) and not block.same_map(other)
    assert not block.same_layout(_block((2,), BOUNDED))
