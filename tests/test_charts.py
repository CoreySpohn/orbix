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


def test_sky_block_completion():
    """Fails if completion swaps sky axes or loses face-on/edge-on states.

    Truth: independently constructed orthonormal frames, including exact
    face-on matrices. 32 eps times a covers the small matrix arithmetic;
    the branch reflects only the away row and leaves the sky block intact.
    """
    from orbix.system import frame_to_sky_block, sky_block_to_frame

    # Rows are east, north, away; columns are the epoch plane axes.
    bases = jnp.array(
        [
            [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            [[0.6, -0.8], [0.8, 0.6], [0.0, 0.0]],
            [[0.6, 0.8], [0.8, -0.6], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]],
            [[0.6, 0.0], [0.0, 1.0], [0.8, 0.0]],
        ]
    )
    constants = 1.7 * bases[:, [1, 0, 1, 0], [0, 0, 1, 1]]
    for branch in (-1, 1):
        a, q = eqx.filter_jit(sky_block_to_frame)(
            *jnp.moveaxis(constants, -1, 0), branch=branch
        )
        actual = frame_to_sky_block(a, q)
        np.testing.assert_allclose(
            actual, constants, atol=32 * np.finfo(float).eps * 1.7, rtol=0
        )
        f, g = (
            rotate(q, jnp.array([1.0, 0.0, 0.0])),
            rotate(q, jnp.array([0.0, 1.0, 0.0])),
        )
        expected = np.asarray(bases).copy()
        expected[:, 2] *= branch
        np.testing.assert_allclose(
            jnp.stack([f, g], -1), expected, atol=32 * np.finfo(float).eps, rtol=0
        )
        np.testing.assert_allclose(a, 1.7, atol=8 * np.finfo(float).eps, rtol=0)


def test_sky_block_geometry_of_projected_circular_frames():
    """Fails if the geometry loses precision or a finite scale derivative.

    A circular frame projects to singular values a and a*abs(cos(i)).
    Allow 32 float64 eps at the scale of each quantity for projection and
    norm roundoff. A uniform rescaling gives the directional derivative.
    """
    from orbix.system import frame_to_sky_block, sky_block_geometry

    a_true = 1.7
    cos_i = jnp.array([1.0, 1.0 - 2.0**-35, 0.85, 0.0, -0.75, -1.0])
    orbit = FrameOrbit.from_classical_trig(
        a_AU=a_true,
        e=0.0,
        cos_i=cos_i,
        sin_i=jnp.sqrt((1.0 - cos_i) * (1.0 + cos_i)),
        cos_W=0.6,
        sin_W=0.8,
        cos_w=0.8,
        sin_w=0.6,
        cos_M0=1.0,
        sin_M0=0.0,
        mu_AU3_per_d2=1.0,
    )
    block = jnp.moveaxis(frame_to_sky_block(a_true, orbit.quat), -1, 0)
    (a, gap), (da, dgap) = jax.jvp(sky_block_geometry, tuple(block), tuple(block))
    gap_true = a_true**2 * (1.0 - cos_i**2)
    eps = np.finfo(float).eps
    np.testing.assert_allclose(a, a_true, atol=32 * eps * a_true, rtol=0)
    np.testing.assert_allclose(gap, gap_true, atol=32 * eps * a_true**2, rtol=0)
    np.testing.assert_allclose(da, a_true, atol=32 * eps * a_true, rtol=0)
    np.testing.assert_allclose(dgap, 2 * gap_true, atol=64 * eps * a_true**2, rtol=0)


@pytest.mark.parametrize("branch", [-1, 1])
def test_thiele_innes_round_trip(branch):
    """Fails if a branch, circular phase, free mu, or batch broadcasting is lost.

    Generic seed-0 blocks have no angle windings; 2e-13 relative/absolute
    allows x64 rank-one completion and quaternion roundoff. The circular
    member pins the eccentricity disk origin without a periapsis angle.
    """
    from orbix.system import ThieleInnesChart

    chart = ThieleInnesChart(branch=branch)
    x = jnp.asarray(np.random.default_rng(0).normal(size=(2, 3, 7)))
    x = x.at[0, 0, 1:3].set(0.0)
    recovered, original, rebuilt = _round_trip(chart, x)
    np.testing.assert_allclose(recovered, x, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(rebuilt, original, rtol=2e-13, atol=2e-13)
    orbit = chart.to_orbit(x, mu_AU3_per_d2=100.0, epoch_d=jnp.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(orbit.mu_AU3_per_d2, chart.implied_mu(x), rtol=8e-15)
    np.testing.assert_allclose(chart.from_orbit(orbit), x, rtol=2e-13, atol=2e-13)
    # Independent classical source in the chosen branch, including e=0.
    source = FrameOrbit.from_classical(
        1.3,
        jnp.array([0.0, 0.4, 0.9, 0.3, 0.0]),
        jnp.array([0.7, 1.57, 2.6, 0.0, np.pi]),
        0.4,
        0.7,
        0.2,
        mu_AU3_per_d2=0.3,
        epoch_d=17.0,
    )
    q = source.quat
    f, g = rotate(q, jnp.array([1.0, 0.0, 0.0])), rotate(q, jnp.array([0.0, 1.0, 0.0]))
    z = jnp.stack([f[:, 2], g[:, 2]], -1)
    pivot = jnp.take_along_axis(z, jnp.argmax(jnp.abs(z), -1)[:, None], -1)[:, 0]
    from orbix.system.rotation import from_basis

    sign = branch * jnp.where(pivot < 0, -1.0, 1.0)
    f, g = f.at[:, 2].multiply(sign), g.at[:, 2].multiply(sign)
    source = eqx.tree_at(lambda o: o.quat, source, from_basis(f, g, jnp.cross(f, g)))
    result = chart.to_orbit(chart.from_orbit(source), mu_AU3_per_d2=100.0, epoch_d=17.0)
    np.testing.assert_allclose(
        result.at(jnp.array([17.0, 18.0])),
        source.at(jnp.array([17.0, 18.0])),
        atol=2e-13,
        rtol=2e-13,
    )


@pytest.mark.parametrize("branch", [-1, 1])
def test_thiele_innes_volume(branch):
    """Fails if sky determinant or the d ln(mu) factor of two is omitted.

    Truth: jacfwd into the already tested ClassicalChart plus ln(mu).
    Five seed-0 generic points allow 2e-11 absolute for inverse/determinant
    conditioning, as in the existing chart-volume tests; constants included.
    """
    from orbix.system import ThieleInnesChart

    chart = ThieleInnesChart(branch=branch)
    target = ClassicalChart()

    def phi(x):
        orbit = chart.to_orbit(x, mu_AU3_per_d2=1.0)
        return jnp.concatenate(
            [target.from_orbit(orbit), jnp.log(orbit.mu_AU3_per_d2)[None]]
        )

    @jax.jit
    def compare(x):
        y = jax.vmap(phi)(x)
        jac = jax.vmap(jax.jacfwd(phi))(x)
        return chart.log_volume(x) - target.log_volume(y[:, :6]), jnp.linalg.slogdet(
            jac
        )[1]

    x = jnp.asarray(np.random.default_rng(0).normal(size=(5, 7)))
    difference, logdet = compare(x)
    np.testing.assert_allclose(difference, logdet, atol=2e-11, rtol=0)


def test_fit_linear_map_rank_and_masks():
    """Fails if AU/axis ordering, correlated weighting, or rank fallback is wrong.

    Noiseless truth is propagated astrometry. The whitened design condition
    number is printed and bounded by 5; 128 eps times a covers QR/SVD and
    the Kepler/rotation products. Singular covariance means a pseudoinverse
    on the estimable subspace, verified against NumPy's independent pinv.
    """
    from orbix.system import fit_linear_map

    orbit = FrameOrbit.from_classical(1.3, 0.3, 0.7, 0.4, 0.7, 0.2, mu_AU3_per_d2=1.0)
    time = jnp.arange(8) * orbit.period_d / 8
    xy = orbit.at(time)[0][:, :2]
    errors = (jnp.full(8, 0.01), jnp.full(8, 0.02), jnp.full(8, 0.3))
    fit = eqx.filter_jit(fit_linear_map)
    block, covariance, chi, rank = fit(orbit, time, xy[:, 0], xy[:, 1], errors)
    condition = np.sqrt(np.linalg.cond(covariance))
    print(f"Linear fit whitened-design condition: {condition:.9g}")
    assert condition < 5 and rank == 4
    floor = 128 * np.finfo(float).eps * 1.3
    np.testing.assert_allclose(block, orbit.linear_map[:2], atol=floor, rtol=0)
    assert chi < (floor / 0.01) ** 2 * 16
    cov = jnp.array([[0.0001, 0.00006], [0.00006, 0.0004]])
    masked = fit(
        orbit, time, xy[:, 0].at[0].set(jnp.nan), xy[:, 1], cov, mask=jnp.arange(8) != 0
    )
    np.testing.assert_allclose(masked[0], block, atol=floor, rtol=0)
    one = fit(orbit, time[:1], xy[:1, 0], xy[:1, 1], cov)
    coords = np.asarray(orbit.in_plane(time[:1])[0])[0]
    np.testing.assert_allclose(
        one[0], np.outer(xy[0], coords) / (coords @ coords), atol=floor, rtol=0
    )
    assert one[3] == 2
    assert all(np.isfinite(value).all() for value in one)
    X, Y = coords
    design = np.array([[0.0, X, 0.0, Y], [X, 0.0, Y, 0.0]])
    whitened = np.linalg.solve(np.linalg.cholesky(cov), design)
    inverse = np.linalg.pinv(whitened)
    np.testing.assert_allclose(
        one[1], inverse @ inverse.T, rtol=128 * np.finfo(float).eps, atol=0
    )
    np.testing.assert_allclose(one[0] @ coords, xy[0], atol=floor, rtol=0)
    none = fit(orbit, time, xy[:, 0], xy[:, 1], errors, mask=jnp.zeros(8, dtype=bool))
    for value in none:
        np.testing.assert_array_equal(value, np.zeros_like(value))
    empty = fit(orbit, time[:0], xy[:0, 0], xy[:0, 1], cov)
    for value in empty:
        np.testing.assert_array_equal(value, np.zeros_like(value))


def test_fit_linear_map_noise_distribution():
    """Fails if error correlation or covariance normalization biases pulls/chi2.

    512 seed-0 Gaussian draws, eight epochs, four parameters: residual sums
    have chi2(6144), whitened pull squares chi2(2048), and scaled mean-pull
    squares chi2(4). Each exact two-sided test has false rejection 1e-6;
    union bound 3e-6. N=512 resolves mean bias 4/sqrt(N)=0.177 sigma and
    variance bias 4*sqrt(2/(4*N))=0.125. No fitted noise scale is used.
    """
    from orbix.system import fit_linear_map

    orbit = FrameOrbit.from_classical(1.3, 0.3, 0.7, 0.4, 0.7, 0.2, mu_AU3_per_d2=1.0)
    time = jnp.arange(8) * orbit.period_d / 8
    xy = orbit.at(time)[0][:, :2]
    cov = jnp.array([[0.0001, 0.00006], [0.00006, 0.0004]])
    noise = (
        np.random.default_rng(0).normal(size=(512, 8, 2)) @ np.linalg.cholesky(cov).T
    )

    @jax.jit
    def fits(data):
        return jax.vmap(
            lambda rows: fit_linear_map(orbit, time, rows[:, 0], rows[:, 1], cov)
        )(data)

    blocks, covariance, chis, ranks = fits(xy + noise)
    delta = np.asarray(blocks - orbit.linear_map[:2])[:, [1, 0, 1, 0], [0, 0, 1, 1]]
    pulls = np.linalg.solve(np.linalg.cholesky(covariance[0]), delta.T).T
    statistics = [np.sum(chis), np.sum(pulls**2), 512 * np.sum(pulls.mean(axis=0) ** 2)]
    for statistic, dof in zip(statistics, [6144, 2048, 4], strict=True):
        assert chi2.ppf(5e-7, dof) < statistic < chi2.isf(5e-7, dof)
    np.testing.assert_array_equal(ranks, 4)


def test_disk_map_round_trip():
    """Fails if the public disk map and its inverse stop being exact inverses.

    The origin and a near-boundary point are included; 1e-13 allows two roundings.
    """
    from orbix.system import disk_to_plane, plane_to_disk

    u = jnp.asarray(np.random.default_rng(0).normal(size=(5, 2)) * 3.0)
    u = u.at[0].set(0.0)
    evec = plane_to_disk(u)
    assert bool(jnp.all(jnp.linalg.norm(evec, axis=-1) < 1.0))
    np.testing.assert_allclose(disk_to_plane(evec), u, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        jnp.linalg.norm(plane_to_disk(jnp.array([1e3, 0.0]))), 1.0, atol=1e-6
    )


def test_periapsis_chart_shape():
    """Fails if the periapsis time is not the point where the orbit is at periapsis.

    An orbit built from the shape passes through periapsis at tp: the chart
    orbit's mean anomaly at the epoch is n*(epoch - tp), and e is recovered.
    """
    from orbix.system import ThieleInnesChart, disk_to_plane, periapsis_chart_shape

    T, e, tp, epoch = 400.0, 0.3, 120.0, 50.0
    ln_n, ex, ey = periapsis_chart_shape(T, e, tp, epoch)
    n = 2 * np.pi / T
    np.testing.assert_allclose(jnp.exp(ln_n), n, rtol=1e-14)
    np.testing.assert_allclose(jnp.hypot(ex, ey), e, rtol=1e-14)
    # In-plane the orbit reaches periapsis (x = a(1-e) along the periapsis axis,
    # distance a(1-e)) at tp.
    chart = ThieleInnesChart()
    u = disk_to_plane(jnp.stack([ex, ey]))
    x = jnp.concatenate([ln_n[None], u, jnp.array([1.0, 0.2, -0.3, 0.9])])
    orbit = chart.to_orbit(x, mu_AU3_per_d2=1.0, epoch_d=epoch)
    r_peri, _ = orbit.at(jnp.array([tp]))
    r_other, _ = orbit.at(jnp.array([tp + T / 2]))
    ratio = jnp.linalg.norm(r_other) / jnp.linalg.norm(r_peri)
    np.testing.assert_allclose(ratio, (1 + e) / (1 - e), rtol=1e-10)
    # Broadcasting over a batch of periods.
    batch = periapsis_chart_shape(jnp.array([100.0, 200.0]), 0.1, 10.0)
    assert batch[0].shape == (2,)


@pytest.mark.parametrize("branch", [-1, 1])
def test_line_of_sight_branch(branch):
    """Fails if the branch read back from an orbit differs from the chart's sheet.

    Orbits of a ThieleInnesChart sheet report that sheet, and reflecting an orbit
    through the sky plane flips the sign while the sky block stays the same.
    """
    from orbix.system import ThieleInnesChart, frame_to_sky_block, line_of_sight_branch

    x = jnp.asarray(np.random.default_rng(0).normal(size=(6, 7)))
    chart = ThieleInnesChart(branch=branch)
    orbit = chart.to_orbit(x, mu_AU3_per_d2=1.0)
    np.testing.assert_array_equal(line_of_sight_branch(orbit), branch)
    other = ThieleInnesChart(branch=-branch).to_orbit(x, mu_AU3_per_d2=1.0)
    np.testing.assert_array_equal(line_of_sight_branch(other), -branch)
    np.testing.assert_allclose(
        frame_to_sky_block(orbit.a_AU, orbit.quat),
        frame_to_sky_block(other.a_AU, other.quat),
        atol=1e-13,
    )
