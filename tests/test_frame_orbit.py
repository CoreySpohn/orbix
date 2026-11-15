"""Frame-state contracts against Cartesian mechanics and classical elements."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import G, Msun2kg
from jax.test_util import check_grads

from orbix.kepler import Halley, Markley
from orbix.system import FrameOrbit


@jax.jit
def _propagate_cartesian(state):
    orbit = FrameOrbit.from_cartesian(state[:3], state[3:], mu_AU3_per_d2=1.0)
    r, v = orbit.at(jnp.array([0.17, 0.53]))
    return jnp.concatenate([r.ravel(), v.ravel()])


@pytest.mark.parametrize(
    "e,inc", [(0.0, 0.7), (0.3, 0.0), (0.3, np.pi), (0.0, 0.0), (0.0, np.pi)]
)
def test_cartesian_singular_round_trip_and_derivatives(e, inc):
    """Fails if circular or face-on states lose phase or either derivative order.

    Truth: Cartesian two-body flow (Danby, Fundamentals of Celestial Mechanics).
    Set mu=1 to keep r and v order one. Central differences at h=1e-4
    have O(h**2) truncation: forward mode to order 2, reverse to order 1, at 1e-4.
    Epoch round trips allow 1e-12 relative plus 1e-14 for zero components,
    covering float64 solve/rotation roundoff. No classical angle is inverted.
    """
    original = FrameOrbit.from_classical(1.3, e, inc, 0.4, 0.7, 0.2, mu_AU3_per_d2=1.0)
    r, v = original.to_cartesian()
    state = jnp.concatenate([r, v])
    rebuilt = FrameOrbit.from_cartesian(r, v, mu_AU3_per_d2=1.0)
    np.testing.assert_allclose(rebuilt.to_cartesian(), (r, v), rtol=1e-12, atol=1e-14)
    jac = jax.jacfwd(_propagate_cartesian)(state)
    assert np.isfinite(jac).all()
    # Also check the chart itself; propagation could otherwise hide a bad leaf.
    chart_jac = jax.jacfwd(
        lambda x: FrameOrbit.from_cartesian(x[:3], x[3:], mu_AU3_per_d2=1.0)
    )(state)
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(chart_jac))
    for order, modes in ((2, ("fwd",)), (1, ("rev",))):
        check_grads(
            _propagate_cartesian,
            (state,),
            order=order,
            modes=modes,
            eps=1e-4,
            atol=1e-4,
            rtol=1e-4,
        )


@pytest.mark.parametrize("e,inc", [(1e-6, 0.7), (0.6, 1.1), (0.2, 2.6)])
def test_classical_round_trip_and_mechanical_invariants(e, inc):
    """Fails if the angle chart, analytic velocity or orbital scale is wrong.

    Truth: energy=-mu/(2a), h^2=mu*a*(1-e^2), and the initial classical
    elements (Murray and Dermott, Solar System Dynamics). Allow 3e-13
    relative/absolute for accumulated x64 rotation and Kepler roundoff;
    angles are compared modulo 2*pi, avoiding undefined circular views.
    """
    a, mu = 1.7, G * Msun2kg
    expected = [a, e, inc, 0.4, 0.7, 0.2]
    orbit = FrameOrbit.from_classical(*expected, mu_AU3_per_d2=mu, epoch_d=17.0)
    elements = orbit.to_classical()
    actual = np.array(
        [elements[k] for k in ("a_AU", "e", "i_rad", "W_rad", "w_rad", "M0_rad")]
    )
    np.testing.assert_allclose(actual[:3], expected[:3], rtol=3e-13, atol=3e-13)
    delta = actual[3:] - expected[3:]
    np.testing.assert_allclose(np.arctan2(np.sin(delta), np.cos(delta)), 0, atol=3e-13)
    times = 17.0 + np.linspace(0, 2, 9) * float(orbit.period_d)
    r, v = map(np.asarray, orbit.at(times))
    energy = np.sum(v * v, -1) / 2 - mu / np.linalg.norm(r, axis=-1)
    h = np.cross(r, v)
    np.testing.assert_allclose(energy, -mu / (2 * a), rtol=3e-13)
    np.testing.assert_allclose(
        h, np.broadcast_to(h[0], h.shape), rtol=3e-13, atol=3e-16
    )
    np.testing.assert_allclose(np.sum(h * h, -1), mu * a * (1 - e * e), rtol=3e-13)
    np.testing.assert_allclose(orbit.mu_AU3_per_d2, mu, rtol=3e-15)
    # The same closed-form ellipse has pericenter/apocenter radii a*(1-/+e).
    tp = 17.0 - 0.2 / float(orbit.n_rad_per_d)
    extrema = orbit.at(jnp.array([tp, tp + float(orbit.period_d) / 2]))[0]
    np.testing.assert_allclose(
        np.linalg.norm(extrema, axis=-1), a * np.array([1 - e, 1 + e]), rtol=3e-13
    )


@pytest.mark.parametrize("solver", [Markley(), Halley()])
def test_circular_closed_form(solver):
    """Fails if the frame's epoch phase or time/velocity units drift.

    Truth: r=a*(sin(l), cos(l), 0), l=n*(t-epoch), in the pinned sky
    convention. Absolute tolerances 2e-14 AU and 2e-16 AU/day allow x64
    trig/rotation roundoff; a circular solve is exact for either strategy.
    """
    a, n, epoch = 1.4, 0.03, 31.0
    orbit = FrameOrbit.from_classical(
        a, 0.0, 0.0, 0.0, 0.0, 0.0, mu_AU3_per_d2=n * n * a**3, epoch_d=epoch
    )
    times = jnp.array([-100.0, 31.0, 200.0])
    angle = n * (np.asarray(times) - epoch)
    r = a * np.stack([np.sin(angle), np.cos(angle), np.zeros(3)], -1)
    v = a * n * np.stack([np.cos(angle), -np.sin(angle), np.zeros(3)], -1)
    actual_r, actual_v = orbit.at(times, solver=solver)
    np.testing.assert_allclose(actual_r, r, atol=2e-14, rtol=0)
    np.testing.assert_allclose(actual_v, v, atol=2e-16, rtol=0)


@pytest.mark.parametrize("batch", [(), (3,), (2, 3)])
def test_batch_contract_and_vmap(batch):
    """Fails if time and orbit axes pair silently or external vmap changes values.

    Truth: scalar evaluation and elementwise broadcasting; allow 2e-14 AU
    for x64 fused arithmetic. Nonconstant phases distinguish draw and planet
    axes; two-dimensional times pin the complete outer-product contract.
    """
    mean = jnp.arange(max(1, np.prod(batch))).reshape(batch) * 0.1
    orbit = FrameOrbit.from_classical(1.4, 0.2, 0.7, 0.4, 0.3, mean, mu_AU3_per_d2=1.0)
    time = jnp.array([[0.0, 0.2], [0.5, 1.0]])
    r, v = orbit.at(time)
    assert r.shape == v.shape == (*batch, 2, 2, 3)
    flat = jax.tree.map(
        lambda x: x.reshape((-1, 4)) if x.ndim == len(batch) + 1 else x.reshape(-1),
        orbit,
    )
    expected = jax.vmap(lambda o: o.at(time))(flat)
    for actual, value in zip((r, v), expected, strict=True):
        np.testing.assert_allclose(
            actual, value.reshape(actual.shape), atol=2e-14, rtol=0
        )
    paired_time = jnp.arange(3) * 0.1 if batch else jnp.array([0.1, 0.2, 0.3])
    paired = orbit.at_each(paired_time)
    assert paired[0].shape == (*jnp.broadcast_shapes(batch, paired_time.shape), 3)
    for k, t in enumerate(np.asarray(paired_time)):
        single_time = orbit.at(t)
        for actual, value in zip(paired, single_time, strict=True):
            np.testing.assert_allclose(
                actual[..., k, :] if batch else actual[k],
                value[..., k, :] if batch else value,
                atol=2e-14,
                rtol=0,
            )
    if len(batch) == 2:
        mapped = jax.vmap(lambda draw: draw.at(time))(orbit)
        np.testing.assert_allclose(mapped, (r, v), atol=2e-14, rtol=0)


def test_float32_long_baseline():
    """Fails if day offsets lose more than 1e-5 of a over twenty years.

    Truth: identical float64 state and Markley evaluation. For a=1 AU and
    e=0.2, phase multiplication over 126 radians dominates float32 error.
    The measured maximum vector residual is 7.93222044e-6*a; the 1e-5*a
    bound covers that rounding floor without using Julian dates.
    """

    def evaluate():
        orbit = FrameOrbit.from_classical(
            1.0, 0.2, 0.7, 0.4, 0.3, 0.2, mu_AU3_per_d2=G * Msun2kg
        )
        return orbit.at(jnp.linspace(0.0, 20 * 365.25, 33))[0]

    expected = np.asarray(evaluate())
    with jax.enable_x64(False):
        actual = evaluate()
        assert actual.dtype == jnp.float32
        residual = np.max(np.linalg.norm(np.asarray(actual) - expected, axis=-1))
    print(f"float32 max residual/a: {residual:.17g}")
    assert residual < 1e-5


def test_shape_validation():
    """Fails if inconsistent state batches are accepted; exact structural checks."""
    with pytest.raises(ValueError, match="batch shape"):
        FrameOrbit(jnp.ones(2), 1.0, 0.0, 0.0, jnp.ones(4), 0.0)
    with pytest.raises(ValueError, match="quat"):
        FrameOrbit(1.0, 1.0, 0.0, 0.0, jnp.ones(3), 0.0)
    with pytest.raises(ValueError, match="Cartesian"):
        FrameOrbit.from_cartesian(jnp.ones(2), jnp.ones(3), mu_AU3_per_d2=1.0)
    # A different quaternion sign describes exactly the same rotation.
    orbit = FrameOrbit.from_classical(1.0, 0.2, 0.7, 0.4, 0.3, 0.2, mu_AU3_per_d2=1.0)
    flipped = eqx.tree_at(lambda o: o.quat, orbit, -orbit.quat)
    np.testing.assert_array_equal(orbit.at(0.3), flipped.at(0.3))


@pytest.mark.parametrize("solver", [Markley(), Halley()])
@pytest.mark.parametrize("time", [0.37, np.array([-0.3, 17.0, 300.0])])
def test_plane_factorization(time, solver):
    """Fails if trig construction or plane factors lose state, scale, or time axes.

    Truth: the propagated state for the generic and singular element sets
    above, batched together. Trig construction must match every state leaf
    and propagated position/rate to 1e-13 in float64.
    Both solvers must supply the same position and analytic-rate contract.
    """
    elements = np.array(
        [
            (1e-6, 0.7),
            (0.6, 1.1),
            (0.2, 2.6),
            (0.0, 0.7),
            (0.3, 0.0),
            (0.3, np.pi),
            (0.0, 0.0),
            (0.0, np.pi),
        ]
    ).reshape(2, 4, 2)
    mean = 0.2 + np.arange(8).reshape(2, 4) * 0.1
    orbit = FrameOrbit.from_classical(
        1.3,
        elements[..., 0],
        elements[..., 1],
        0.4,
        0.7,
        mean,
        mu_AU3_per_d2=1.0,
        epoch_d=17.0,
    )
    trig = FrameOrbit.from_classical_trig(
        a_AU=1.3,
        e=elements[..., 0],
        cos_i=np.cos(elements[..., 1]),
        sin_i=np.sin(elements[..., 1]),
        cos_W=np.cos(0.4),
        sin_W=np.sin(0.4),
        cos_w=np.cos(0.7),
        sin_w=np.sin(0.7),
        cos_M0=np.cos(mean),
        sin_M0=np.sin(mean),
        mu_AU3_per_d2=1.0,
        epoch_d=17.0,
    )
    for actual, expected in zip(
        jax.tree.leaves(trig), jax.tree.leaves(orbit), strict=True
    ):
        assert actual.dtype == jnp.float64
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        trig.at(time, solver=solver),
        orbit.at(time, solver=solver),
        rtol=1e-13,
        atol=1e-13,
    )
    xy, xy_rate = orbit.plane_coordinates(time, solver=solver)
    basis = orbit.plane_basis()
    assert basis.shape == (2, 4, 3, 2)
    assert xy.shape == xy_rate.shape == (2, 4, *np.shape(time), 2)
    basis = np.asarray(basis).reshape(2, 4, *(1,) * np.ndim(time), 3, 2)
    expected = orbit.at(time, solver=solver)
    for coordinates, state in zip((xy, xy_rate), expected, strict=True):
        actual = (basis @ np.asarray(coordinates)[..., None])[..., 0]
        np.testing.assert_allclose(actual, state, rtol=1e-13, atol=0)


@pytest.mark.parametrize("e,inc", [(0.05, 0.7), (0.6, 1.1), (0.95, 2.6)])
def test_perifocal_coordinates_and_thiele_innes(e, inc):
    """Fails if perifocal rates or periapsis/east/north rotations change signs.

    Truth: NumPy Newton iteration for E-e*sin(E)=M and the classical
    Thiele-Innes formulas, including a retrograde orbit. Float64 permits
    1e-12 relative plus 1e-14 absolute near coordinate zeros. Rates match
    central differences at h=1e-5 days to 1e-7 relative.
    """
    a, node, peri, mean, epoch = 1.7, 0.4, 0.7, 0.2, 17.0
    orbit = FrameOrbit.from_classical(
        a, e, inc, node, peri, mean, mu_AU3_per_d2=1.0, epoch_d=epoch
    )
    times = epoch + np.array([-0.2, 0.0, 0.13, 0.5, 0.9]) * float(orbit.period_d)
    anomaly = mean + float(orbit.n_rad_per_d) * (times - epoch)
    anomaly = (anomaly + np.pi) % (2 * np.pi) - np.pi
    eccentric = np.where(anomaly >= 0, np.pi, -np.pi)
    for _ in range(20):
        eccentric -= (eccentric - e * np.sin(eccentric) - anomaly) / (
            1 - e * np.cos(eccentric)
        )
    expected = np.stack(
        [np.cos(eccentric) - e, np.sqrt(1 - e * e) * np.sin(eccentric)], -1
    )
    actual, rate = orbit.perifocal_coordinates(times)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    step = 1e-5
    difference = (
        orbit.perifocal_coordinates(times + step)[0]
        - orbit.perifocal_coordinates(times - step)[0]
    ) / (2 * step)
    np.testing.assert_allclose(rate, difference, rtol=1e-7, atol=0)
    cO, sO, cw, sw, ci = (
        np.cos(node),
        np.sin(node),
        np.cos(peri),
        np.sin(peri),
        np.cos(inc),
    )
    A = a * (cO * cw - sO * sw * ci)
    B = a * (sO * cw + cO * sw * ci)
    F = a * (-cO * sw - sO * cw * ci)
    G = a * (-sO * sw + cO * cw * ci)
    sky = np.asarray(actual) @ np.array([[B, A], [G, F]])
    np.testing.assert_allclose(sky, orbit.at(times)[0][..., :2], rtol=1e-12, atol=1e-14)
    sky_rate = np.asarray(rate) @ np.array([[B, A], [G, F]])
    np.testing.assert_allclose(
        sky_rate, orbit.at(times)[1][..., :2], rtol=1e-12, atol=1e-14
    )


@pytest.mark.parametrize("component", [0, 1])
def test_plane_coordinates_circular_gradient(component):
    """Fails if circular plane coordinates have a singular eccentricity gradient.

    Truth: a centered difference in ex at zero with h=1e-5; its O(h**2)
    truncation permits 1e-8 relative and 1e-10 absolute in float64.
    The undefined circular periapsis view must also have a finite fallback.
    """
    orbit = FrameOrbit.from_classical(1.3, 0.0, 0.7, 0.4, 0.7, 0.2, mu_AU3_per_d2=1.0)

    def evaluate(ex):
        changed = eqx.tree_at(lambda o: o.ex, orbit, jnp.asarray(ex))
        return changed.plane_coordinates(0.37)[0][component]

    derivative = jax.grad(evaluate)(0.0)
    step = 1e-5
    difference = (evaluate(step) - evaluate(-step)) / (2 * step)
    assert np.isfinite(derivative)
    np.testing.assert_allclose(derivative, difference, rtol=1e-8, atol=1e-10)
    np.testing.assert_array_equal(
        orbit.perifocal_coordinates(0.37), orbit.plane_coordinates(0.37)
    )
    for kind in (0, 1):
        derivative = jax.grad(
            lambda ex, kind=kind: eqx.tree_at(
                lambda o: o.ex, orbit, ex
            ).perifocal_coordinates(0.37)[kind][component]
        )(0.0)
        assert np.isfinite(derivative)


@pytest.mark.parametrize("parameter", ["cos_i", "cos_w", "sin_w"])
def test_classical_trig_gradients(parameter):
    """Fails if angle inversion makes face-on or periapsis gradients singular.

    Truth: central differences with independently varied trig inputs at e=0.3.
    Float64 and h=1e-5 permit 1e-6 relative error from O(h**2) truncation,
    plus 1e-11 absolute for roundoff in the zero face-on derivative.
    """
    inputs = dict(
        a_AU=1.3,
        e=0.3,
        cos_i=1.0,
        sin_i=0.0,
        cos_W=np.cos(0.4),
        sin_W=np.sin(0.4),
        cos_w=np.cos(0.7),
        sin_w=np.sin(0.7),
        cos_M0=np.cos(0.2),
        sin_M0=np.sin(0.2),
        mu_AU3_per_d2=1.0,
    )

    def evaluate(value):
        return FrameOrbit.from_classical_trig(**(inputs | {parameter: value})).at(0.37)[
            0
        ][0]

    value = jnp.asarray(inputs[parameter], dtype=jnp.float64)
    derivative = jax.grad(evaluate)(value)
    step = 1e-5
    difference = (evaluate(value + step) - evaluate(value - step)) / (2 * step)
    assert np.isfinite(derivative)
    np.testing.assert_allclose(derivative, difference, rtol=1e-6, atol=1e-11)
