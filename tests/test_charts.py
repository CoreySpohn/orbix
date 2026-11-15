"""Chart interchange and reference-measure contracts in float64."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import chi2

from orbix.system import (
    CartesianChart,
    ClassicalChart,
    EccentricityVectorChart,
    FrameChart,
    FrameOrbit,
)
from orbix.system.rotation import rotate


@eqx.filter_jit
def _round_trip(chart, x):
    orbit = chart.to_orbit(x, mu_AU3_per_d2=1.0, epoch_d=17.0)
    recovered = chart.from_orbit(orbit)
    rebuilt = chart.to_orbit(recovered, mu_AU3_per_d2=1.0, epoch_d=orbit.epoch_d)
    times = jnp.array([16.7, 17.0, 17.8])
    return recovered, orbit.at(times), rebuilt.at(times)


@pytest.mark.parametrize(
    "chart",
    [
        FrameChart(),
        FrameChart(gauge_sd=None),
        EccentricityVectorChart(),
        ClassicalChart(),
        CartesianChart(),
        CartesianChart(scale_AU=3.0),
    ],
)
def test_round_trip(chart):
    """Fails if chart coordinates, phase, sign convention or batch axes change.

    Truth: the supplied coordinates and the state's Kepler flow at three
    times, in AU and days with mu=1. A 1e-12 absolute/relative tolerance
    covers float64 inversion, rotations and solve roundoff at generic
    points. Angles are compared modulo 2*pi; frame vectors are normalized
    with their largest component positive. Two leading axes pin batching.
    """
    x = np.random.default_rng(0).normal(size=(2, 3, chart.n_coords))
    recovered, original, rebuilt = _round_trip(chart, jnp.asarray(x))
    expected = x.copy()
    if isinstance(chart, FrameChart):
        q = expected[..., 3:]
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        pivot = np.take_along_axis(q, np.argmax(np.abs(q), axis=-1)[..., None], -1)
        q *= np.where(pivot < 0, -1.0, 1.0)
        delta = np.asarray(recovered) - expected
    elif isinstance(chart, CartesianChart):
        delta = np.asarray(recovered) - expected
        r = chart.scale_AU * x[..., :3]
        w = x[..., 3:]
        v = (
            w
            * np.sqrt(2 / (np.linalg.norm(r, axis=-1) * (1 + np.sum(w * w, axis=-1))))[
                ..., None
            ]
        )
        np.testing.assert_allclose(original[0][..., 1, :], r, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(original[1][..., 1, :], v, atol=1e-12, rtol=1e-12)
    else:
        delta = np.asarray(recovered) - expected
        first_angle = 4 if isinstance(chart, EccentricityVectorChart) else 3
        angles = delta[..., first_angle:]
        delta[..., first_angle:] = np.arctan2(np.sin(angles), np.cos(angles))
    np.testing.assert_allclose(delta, 0.0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(rebuilt, original, atol=1e-12, rtol=1e-12)
    assert not jax.tree.leaves(chart)


@pytest.mark.parametrize("e,inc", [(0.0, 0.7), (0.3, 0.0), (0.3, np.pi)])
@pytest.mark.parametrize("gauge_sd", [0.1, None])
def test_frame_singular_round_trip(e, inc, gauge_sd):
    """Fails if circular or either face-on states lose phase in frame inversion.

    Truth: FrameOrbit.from_classical at e=0, i=0 or i=pi, followed by its
    Kepler flow at three times. The 1e-12 AU and AU/day absolute/relative
    tolerance covers x64 rotations and solves without inverting singular
    classical angles. The vector chart also preserves circular phase.
    """
    orbit = FrameOrbit.from_classical(
        1.3, e, inc, 0.4, 0.7, 0.2, mu_AU3_per_d2=1.0, epoch_d=17.0
    )
    charts = [FrameChart(gauge_sd=gauge_sd)]
    if e == 0.0:
        charts.append(EccentricityVectorChart())
    for chart in charts:
        x = chart.from_orbit(orbit)
        recovered, original, rebuilt = _round_trip(chart, x)
        np.testing.assert_allclose(recovered, x, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(rebuilt, original, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(
            rebuilt, orbit.at(jnp.array([16.7, 17.0, 17.8])), atol=1e-12, rtol=1e-12
        )


@pytest.mark.parametrize("scale_AU", [1.0, 3.0])
def test_cartesian_orbit_round_trip(scale_AU):
    """Fails if Cartesian inversion loses generic, circular or face-on phase.

    Truth: independently constructed classical states at their epoch and
    two nearby times. Both AU positions and AU/day velocities agree to
    1e-12 absolute/relative, allowing float64 rotation and solve roundoff.
    """
    orbit = FrameOrbit.from_classical(
        1.3,
        jnp.array([0.3, 0.0, 0.3, 0.3]),
        jnp.array([0.7, 0.7, 0.0, np.pi]),
        0.4,
        0.7,
        0.2,
        mu_AU3_per_d2=1.0,
        epoch_d=17.0,
    )
    chart = CartesianChart(scale_AU=scale_AU)
    x = chart.from_orbit(orbit)
    recovered, original, rebuilt = _round_trip(chart, x)
    np.testing.assert_allclose(recovered, x, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(rebuilt, original, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        rebuilt, orbit.at(jnp.array([16.7, 17.0, 17.8])), atol=1e-12, rtol=1e-12
    )


@eqx.filter_jit
def _volume_comparison(source, target, x, mu=1.0):
    def phi(point):
        return target.from_orbit(source.to_orbit(point, mu_AU3_per_d2=mu))

    y = jax.vmap(phi)(x)
    jac = jax.vmap(jax.jacfwd(phi))(x)
    return source.log_volume(x) - target.log_volume(y), jac


@pytest.mark.parametrize(
    "source,target",
    [
        (EccentricityVectorChart(), ClassicalChart()),
        (ClassicalChart(), EccentricityVectorChart()),
        (FrameChart(), ClassicalChart()),
        (FrameChart(gauge_sd=None), ClassicalChart()),
    ],
)
def test_volume_against_automatic_jacobian(source, target):
    """Fails if volume factors disagree with AD or omit disk area contraction.

    Truth: change of variables for one shared reference measure; automatic
    Jacobians independently differentiate orbit interchange at five seed-0
    generic points. Residual spread <2e-11 allows x64 rotation/inverse and
    determinant conditioning; the additive normalization is not asserted.

    For the 6x7 frame Jacobian J, normalization kills the radial direction
    and scales three quaternion tangent directions by 1/r. The coarea
    Jacobian is sqrt(det(J J.T)). Cartesian dv=r^3 dr dS3, so removing
    the radial gauge requires subtracting log g(r), the radial density
    per unit r: log g(r) = log lift(v) + 3 log r, up to a constant.
    The remainder minus Classical.log_volume must
    equal log coarea Jacobian plus a constant. Dropping the disk term is
    a negative control required to fail this very same spread assertion.
    """
    x = jnp.asarray(np.random.default_rng(0).normal(size=(5, source.n_coords)))
    difference, jac = map(np.asarray, _volume_comparison(source, target, x))
    if isinstance(source, FrameChart):
        v = np.asarray(x[..., 3:])
        radius = np.linalg.norm(v, axis=-1)
        radial = np.concatenate([np.zeros((5, 3)), v / radius[:, None]], axis=-1)
        np.testing.assert_allclose(
            np.einsum("bij,bj->bi", jac, radial), 0, atol=2e-12, rtol=0
        )
        if source.gauge_sd is None:
            radial_log_density = -(radius**2) / 2 + 3 * np.log(radius) - np.log(2)
        else:
            radial_log_density = -0.5 * ((radius - 1) / source.gauge_sd) ** 2
        difference = difference - radial_log_density
        sign, logdet = np.linalg.slogdet(jac @ np.swapaxes(jac, -1, -2))
        np.testing.assert_array_equal(sign, 1)
        logdet /= 2
    else:
        sign, logdet = np.linalg.slogdet(jac)
        assert np.all(np.abs(sign) == 1)
    residual = difference - logdet
    spread = np.ptp(residual)
    print(f"{type(source).__name__} -> {type(target).__name__} spread: {spread:.17g}")
    assert spread < 2e-11
    if isinstance(source, FrameChart):
        disk_logdet = -2 * np.log1p(np.sum(np.asarray(x[..., 1:3]) ** 2, axis=-1))
        wrong_spread = np.ptp(residual - disk_logdet)
        print(f"Missing disk term spread: {wrong_spread:.17g}")
        with pytest.raises(AssertionError):
            assert wrong_spread < 2e-11


@pytest.mark.parametrize("scale_AU", [1.0, 3.0])
@pytest.mark.parametrize("mu", [0.25, 4.0])
def test_cartesian_volume_against_automatic_jacobian(scale_AU, mu):
    """Fails for missing radius/ball factors or hidden mu normalization.

    Truth: the 6x6 AD change-of-variables determinant at five seed-0
    generic points. The same 2e-11 spread bound as the other chart pair
    covers x64 inverse/determinant conditioning. The residual's constant
    is also -log(16) at both mu and scale values, proving cancellation
    rather than hiding mu in an untested additive constant. Dropping the
    ball Jacobian once must fail the same spread assertion.
    """
    chart = CartesianChart(scale_AU=scale_AU)
    x = jnp.asarray(np.random.default_rng(0).normal(size=(5, chart.n_coords)))
    difference, jac = map(
        np.asarray, _volume_comparison(chart, ClassicalChart(), x, jnp.asarray(mu))
    )
    sign, logdet = np.linalg.slogdet(jac)
    assert np.all(np.abs(sign) == 1)
    residual = difference - logdet
    spread = np.ptp(residual)
    print(f"Cartesian scale={scale_AU}, mu={mu} spread: {spread:.17g}")
    assert spread < 2e-11
    np.testing.assert_allclose(residual, -np.log(16.0), atol=2e-11, rtol=0)
    if scale_AU == 1.0 and mu == 0.25:
        ball_logdet = -2.5 * np.log1p(np.sum(np.asarray(x[..., 3:]) ** 2, axis=-1))
        wrong_spread = np.ptp(residual - ball_logdet)
        print(f"Missing ball term spread: {wrong_spread:.17g}")
        with pytest.raises(AssertionError):
            assert wrong_spread < 2e-11


@eqx.filter_jit
def _orientation_sample(v):
    x = jnp.concatenate([jnp.zeros((*v.shape[:-1], 3)), v], axis=-1)
    orbit = FrameChart(gauge_sd=None).to_orbit(x, mu_AU3_per_d2=1.0)
    normal = rotate(orbit.quat, jnp.array([0.0, 0.0, 1.0]))
    return -normal[..., 2], jnp.arctan2(-normal[..., 1], normal[..., 0])


def test_frame_isotropy():
    """Fails if normalized Gaussian quaternions favor an inclination or node.

    Truth: rotational invariance of N(0,I4) induces Haar measure on SO(3),
    giving uniform cos(i) and node. Use 200000 seed-0 draws and 20 bins:
    Pearson chi-square with 19 degrees of freedom, false rejection 5e-7
    per marginal (union bound 1e-6 overall). Expected bin count is 10000,
    making the chi-square approximation accurate. No propagation is needed.

    At five points the default gauge density times r^3 must factor into
    a constant sphere density and the stated radial normal. This catches
    a missing coarea correction or the old Gaussian lift as the default.
    """
    radius = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
    direction = np.random.default_rng(1).normal(size=(5, 4))
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    x = jnp.asarray(np.concatenate([np.zeros((5, 3)), radius[:, None] * direction], -1))
    radial_density = np.asarray(FrameChart().log_volume(x)) + 3 * np.log(radius)
    np.testing.assert_allclose(
        radial_density - radial_density[2], [-2, -0.5, 0, -0.5, -2], atol=2e-14, rtol=0
    )
    v = jnp.asarray(np.random.default_rng(0).normal(size=(200_000, 4)))
    cosine, node = map(np.asarray, _orientation_sample(v))
    threshold = chi2.isf(5e-7, 19)
    for values, bounds in [(cosine, (-1.0, 1.0)), (node, (-np.pi, np.pi))]:
        counts = np.histogram(values, bins=20, range=bounds)[0]
        statistic = np.sum((counts - 10_000) ** 2 / 10_000)
        assert counts.sum() == 200_000
        assert statistic < threshold


@pytest.mark.parametrize(
    "chart",
    [
        FrameChart(),
        FrameChart(gauge_sd=None),
        EccentricityVectorChart(),
        ClassicalChart(),
        CartesianChart(),
        CartesianChart(scale_AU=3.0),
    ],
)
def test_unconstrained(chart):
    """Fails if scale-5 inputs produce unbound or nonfinite frame states.

    Truth: disk/ball maps, the exponential and quaternion normalization
    enforce e<1, a>0 and |q|=1 away from degenerate states. Use 512 seed-0
    draws; strict bounds test the constraints, and 2e-15 absolute covers
    x64 norm roundoff.
    """
    x = jnp.asarray(
        np.random.default_rng(0).normal(scale=5, size=(512, chart.n_coords))
    )
    orbit = eqx.filter_jit(chart.to_orbit)(x, mu_AU3_per_d2=1.0)
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(orbit))
    assert np.isfinite(chart.log_volume(x)).all()
    assert np.all(np.asarray(orbit.e) < 1)
    assert np.all(np.asarray(orbit.a_AU) > 0)
    np.testing.assert_allclose(
        np.linalg.norm(orbit.quat, axis=-1), 1, atol=2e-15, rtol=0
    )


@pytest.mark.parametrize("gauge_sd", [0.1, None])
def test_frame_origin_autodiff(gauge_sd):
    """Fails if guarded zero branches leak NaNs into position autodiff.

    Truth: the disk map has identity Jacobian at u=0, and the frame chart
    explicitly assigns identity orientation and zero quaternion tangent at
    v=0. Normalization has no continuous extension there: this checks an
    autodiff convention, not a mathematical derivative at the origin.
    Both AD modes must be finite and agree to 2e-14 absolute (x64 roundoff).
    At epoch and identity orientation, dr/dln(a)=(1,0,0), proving the
    assertion is not satisfied by zeroing every tangent.
    """
    chart = FrameChart(gauge_sd=gauge_sd)
    x = jnp.zeros(7)

    def position(point):
        return chart.to_orbit(point, mu_AU3_per_d2=1.0).at(0.0)[0]

    forward = eqx.filter_jit(jax.jacfwd(position))(x)
    reverse = eqx.filter_jit(jax.jacrev(position))(x)
    assert np.isfinite(forward).all() and np.isfinite(reverse).all()
    np.testing.assert_allclose(forward, reverse, atol=2e-14, rtol=0)
    np.testing.assert_allclose(forward[:, 0], [1, 0, 0], atol=2e-14, rtol=0)
    np.testing.assert_array_equal(
        chart.to_orbit(x, mu_AU3_per_d2=1.0).quat, [1, 0, 0, 0]
    )
    if gauge_sd is None:
        np.testing.assert_allclose(chart.log_volume(x), -2 * np.log(2 * np.pi))
    else:
        assert np.isneginf(chart.log_volume(x))


def test_uniform_in_e():
    """Fails if the 1/e prior retains an extra eccentricity-area factor.

    Truth: (1/e)*e de=de; pulling this into logit coordinates leaves only
    de/dlogit(e)=e*(1-e). Compare e=0.2 and 0.7 with identical orientation
    at 2e-14 absolute tolerance for float64 logarithm/subtraction roundoff.
    """
    e = jnp.array([0.2, 0.7])
    x = jnp.zeros((2, 6)).at[:, 1].set(jnp.log(e) - jnp.log1p(-e))
    chart = ClassicalChart()
    orbit = chart.to_orbit(x, mu_AU3_per_d2=1.0)
    density = -jnp.log(orbit.e) + chart.log_volume(x)
    jacobian = jnp.log(e * (1 - e))
    np.testing.assert_allclose(
        density[1] - density[0], jacobian[1] - jacobian[0], atol=2e-14, rtol=0
    )
