"""Jacobi hierarchy and ephemeris observables against independent mechanics."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import (
    GM_SUN_AU3_PER_D2,
    AU2m,
    G,
    Mjup2kg,
    Msun2kg,
    arcsec2mas,
    d2s,
    pc2AU,
    rad2arcsec,
    yr2s,
)
from hwoutils.epoch import Epoch
from hwoutils.numerics import roundoff
from hwoutils.testing import ulps

from orbix.kepler import Markley
from orbix.system import (
    Body,
    Ephemeris,
    FrameOrbit,
    SkyFrame,
    System,
    barycentric_to_jacobi,
    enclosed_mu,
    jacobi_to_barycentric,
    observables,
)

ELEMENTS = dict(
    a_AU=[1.0, 5.0],
    e=[0.05, 0.08],
    i_rad=[0.5, 0.55],
    W_rad=[0.3, 0.33],
    w_rad=[0.7, 1.1],
    M0_rad=[0.4, 1.7],
)


def _system(second_mass=0.005):
    return System.from_classical(
        Body(1.0), Body([0.001, second_mass]), SkyFrame(10.0), **ELEMENTS
    )


@pytest.mark.parametrize("anchor", ["face_on", "ascending", "reflex", "phase"])
def test_sky_sign_anchors(anchor):
    """Fails only on sky-motion, ascending-node, reflex or illumination reversal.

    Truth: the pinned east/north/away geometry. Assert strict signs, not
    numerical proximity, so no floating tolerance can hide an axis reversal.
    For a planet behind the star, phase must be on the zero side of pi/2.
    """
    inclination = 0.0 if anchor == "face_on" else np.pi / 2
    mean = np.pi if anchor == "reflex" else (np.pi / 2 if anchor == "phase" else 0.0)
    sys = System.from_classical(
        Body(1.0),
        Body([0.001]),
        SkyFrame(10.0),
        a_AU=1.0,
        e=0.0,
        i_rad=inclination,
        W_rad=0.0,
        w_rad=0.0,
        M0_rad=mean,
    )
    eph = sys.solve(jnp.array([0.0, 1.0]))
    if anchor == "face_on":
        assert eph.position_angle_deg(1)[1] > eph.position_angle_deg(1)[0]
        assert eph.position_AU[1, 0, 1] > 0
        assert eph.velocity_AU_per_d[1, 0, 0] > 0
    elif anchor == "ascending":
        assert eph.velocity_AU_per_d[1, 0, 2] > 0
    elif anchor == "reflex":
        assert eph.radial_velocity_m_per_s(1, 0)[0] < 0
        assert eph.radial_velocity_m_per_s(0)[0] > 0
    else:
        assert eph.position_AU[1, 0, 2] > eph.position_AU[0, 0, 2]
        assert eph.phase_angle_rad(1)[0] < np.pi / 2
        assert observables.phase_angle_rad(jnp.array([0.0, 0.0, 1.0])) == 0.0
        assert observables.phase_angle_rad(jnp.array([0.0, 0.0, -1.0])) > np.pi / 2


@pytest.mark.parametrize("second_mass", [0.005, 0.0])
def test_hierarchy_mass_conservation_and_third_law(second_mass):
    """Fails if cumulative masses or the Jacobi barycentric map are incorrect.

    Truth: sum(m*r)=sum(m*v)=0 and n^2*a^3=G*M_enclosed (Murray and
    Dermott). Allow two steps of six operations each (mass fraction,
    weighted state and ordered prefix/total sums) at the summed mass-moment
    scale. Eight powers, products and divisions bound the third law.
    A massless outer companion must leave the inner states bit-identical.
    """
    sys = _system(second_mass)
    time = jnp.array([0.0, 200.0, 700.0])
    eph = sys.solve(time)
    masses = np.array([1.0, 0.001, second_mass])
    for state in (eph.position_AU, eph.velocity_AU_per_d):
        moments = np.asarray(state) * masses[:, None, None]
        scale = np.max(np.sum(np.abs(moments), axis=0))
        np.testing.assert_allclose(
            moments.sum(axis=0),
            0,
            atol=float(roundoff(scale, n_ops=12, dtype=state.dtype)),
            rtol=0,
        )
    tol = float(roundoff(1.0, n_ops=8, dtype=sys.orbits.n_rad_per_d.dtype))
    expected_mu = G * Msun2kg * np.array([1.001, 1.001 + second_mass])
    np.testing.assert_allclose(
        enclosed_mu(1.0, jnp.array([0.001, second_mass])), expected_mu, rtol=tol, atol=0
    )
    np.testing.assert_allclose(
        (2 * np.pi / np.asarray(sys.orbits.period_d)) ** 2 * np.array([1.0, 5.0]) ** 3,
        expected_mu,
        rtol=tol,
        atol=0,
    )
    if second_mass == 0.0:
        single = System.from_classical(
            Body(1.0),
            Body([0.001]),
            SkyFrame(10.0),
            **{k: v[0] for k, v in ELEMENTS.items()},
        ).solve(time)
        np.testing.assert_array_equal(eph.position_AU[:2], single.position_AU)
        np.testing.assert_array_equal(
            eph.velocity_AU_per_d[:2], single.velocity_AU_per_d
        )


# IAS15 reference, AU, body zero; full generation recipe is in the test below.
NBODY_STAR_AU = np.array(
    [
        [0.0034416947817952517, 0.024620254942484886, -0.0028517973766051751],
        [0.011659987373103555, 0.023259618176159735, 0.0020836945032919733],
        [0.01570152356768233, 0.020640461819412585, 0.005065554121319431],
        [0.020746672175181387, 0.013866989210599029, 0.0092245260467275266],
        [0.022910179629047015, 0.0092704471872226274, 0.011475086600783908],
        [0.022864810778155217, 0.0002541571088228319, 0.013203099553590412],
        [0.02266513148941832, -0.0054919465791158318, 0.01421484388660991],
        [0.017810501529668663, -0.012974265797931586, 0.01295179845602033],
    ]
)


@pytest.mark.float64_only(reason="float64 reference values for orbital validation")
def test_two_planets_against_nbody():
    """Fails if the hierarchy regresses to independent star-planet two-body motion.

    Truth: REBOUND 5.0.0 IAS15, Jacobi elements, AU/day/Msun, G from
    hwoutils. Max 3D stellar residual: chain 4.55874285e-6 AU versus
    independent 2.02401225e-4 AU. Use 1.4e-5 AU (3.07 times chain max).
    This bounds the neglected perturbations over four years, not roundoff.
    Initial elements have mutual inclination about 3 degrees. The control
    uses the SAME initial barycentric r,v converted to astrocentric states.

    One-time generating snippet (rebound is not a test dependency)::

        import rebound
        sim = rebound.Simulation()
        sim.units = ('AU', 'day', 'Msun')
        sim.G = G * Msun2kg
        sim.integrator = 'ias15'
        sim.add(m=1.)
        for j, mass in enumerate([.001, .005]):
            sim.add(m=mass, a=ELEMENTS['a_AU'][j], e=ELEMENTS['e'][j],
                    inc=ELEMENTS['i_rad'][j], Omega=ELEMENTS['W_rad'][j],
                    omega=ELEMENTS['w_rad'][j], M=ELEMENTS['M0_rad'][j])
        sim.move_to_com()
        for t in np.linspace(0., 4*365.25, 8):
            sim.integrate(t)
            p = sim.particles[0]
            print([format(x, '.17g') for x in (p.y, p.x, p.z)])

    REBOUND defaults to Jacobi insertion when primary is omitted. Its
    classical axes are reordered to (east, north, away) after integration.
    """
    sys = _system()
    times = np.linspace(0.0, 4 * 365.25, 8)
    eph = sys.solve(times)
    r, v = eph.position_AU[:, 0], eph.velocity_AU_per_d[:, 0]
    independent = FrameOrbit.from_cartesian(
        r[1:] - r[0],
        v[1:] - v[0],
        mu_AU3_per_d2=G * Msun2kg * (1 + np.array([0.001, 0.005])),
    )
    weights = np.array([0.001 / 1.001, 0.005 / 1.005])
    control = -np.sum(weights[:, None, None] * independent.at(times)[0], axis=0)
    chain_residual = np.linalg.norm(eph.position_AU[0] - NBODY_STAR_AU, axis=-1)
    control_residual = np.linalg.norm(control - NBODY_STAR_AU, axis=-1)
    print(
        "N-body max/RMS AU:",
        chain_residual.max(),
        np.sqrt(np.mean(chain_residual**2)),
        control_residual.max(),
        np.sqrt(np.mean(control_residual**2)),
    )
    assert chain_residual.max() < 1.4e-5
    assert control_residual.max() > 1.4e-5
    np.testing.assert_allclose(
        eph.position_AU[0, 0], NBODY_STAR_AU[0], atol=4e-17, rtol=0
    )


def test_host_time_observables_and_solver_trace():
    """Fails if JD precision, paired rows, solve reuse or observable references drift.

    Truth: host float64 subtraction and Euclidean vector projections; allow
    eight operations for vector norms and angle/length conversions. At each
    time, allow 32 operations for Kepler solution, frame rotation and the
    Jacobi sums, at the largest position scale. Equal solver configurations
    must add exactly one trace on the first call and none on the second.
    Row pairing is checked against separately evaluated systems at each time.
    """
    frame = SkyFrame(10.0, Epoch(2460000.0, 0.125, "tdb"))
    dates = np.array([2460000.125, 2460001.25])
    np.testing.assert_array_equal(frame.time_d(jd=dates), [0.0, 1.125])
    assert frame.time_d(jd=dates).dtype == np.float64
    assert len(jax.tree.leaves(frame)) == 1
    sys = _system()
    times = jnp.array([0.0, 200.0, 700.0])
    traces = []

    @eqx.filter_jit
    def solve(system, time, solver):
        traces.append(None)
        return system.solve(time, solver=solver)

    baseline = len(traces)
    eph = solve(sys, times, Markley())
    same = solve(sys, times, Markley())
    assert len(traces) - baseline == 1
    np.testing.assert_array_equal(eph.position_AU, same.position_AU)
    # Stacked systems are caller-owned; vmap pairs each draw with one time.
    draws = jax.tree.map(lambda x: jnp.stack([x] * 3), sys)
    paired = jax.vmap(lambda system, time: system.solve(time))(draws, times)
    np.testing.assert_allclose(
        paired.position_AU,
        jnp.moveaxis(eph.position_AU, 1, 0),
        atol=float(ulps(32, jnp.max(jnp.abs(eph.position_AU)))),
        rtol=0,
    )
    r = eph.position_AU[1] - eph.position_AU[0]
    v = eph.velocity_AU_per_d[1] - eph.velocity_AU_per_d[0]
    tol = float(roundoff(1.0, n_ops=8, dtype=r.dtype))
    east, north = eph.offset_arcsec(1)
    np.testing.assert_allclose(
        eph.separation_arcsec(1), np.hypot(east, north), rtol=tol, atol=0
    )
    np.testing.assert_allclose(
        eph.distance_AU(1), np.sqrt(np.sum(r * r, -1)), rtol=tol, atol=0
    )
    np.testing.assert_allclose(
        eph.radial_velocity_m_per_s(1, 0), v[..., 2] * AU2m / d2s, rtol=tol, atol=0
    )
    np.testing.assert_allclose(
        eph.phase_angle_rad(1),
        np.arctan2(np.hypot(r[..., 0], r[..., 1]), r[..., 2]),
        rtol=tol,
        atol=0,
    )
    np.testing.assert_allclose(
        eph.position_angle_deg(1),
        np.rad2deg(np.arctan2(r[..., 0], r[..., 1])) % 360,
        rtol=tol,
        atol=0,
    )
    with pytest.raises(ValueError, match="reference"):
        eph.distance_AU(1, "invalid")


@pytest.mark.parametrize("batch_shape", [(), (2, 3)])
def test_batch_and_massless_tail(batch_shape):
    """Fails if leading axes couple systems or zeros alter the ordered sum.

    Exact algebra: trailing zero weights must preserve all preceding bits.
    Batch/vectorized and scalar solves use the same arithmetic (2 ULP floor).
    """
    base = _system()
    batch = jax.tree.map(lambda x: jnp.broadcast_to(x, batch_shape + x.shape), base)
    times = jnp.array([0.0, 31.0])
    result = batch.solve(times)
    truth = base.solve(times)
    np.testing.assert_array_equal(
        result.position_AU, np.broadcast_to(truth.position_AU, result.position_AU.shape)
    )
    np.testing.assert_array_equal(
        result.separation_arcsec(1),
        np.broadcast_to(truth.separation_arcsec(1), (*batch_shape, 2)),
    )
    padded = System.from_classical(
        Body(jnp.ones(batch_shape)),
        Body(
            jnp.broadcast_to(
                jnp.array([0.001, 0.005, 0.0, 0.0, 0.0]), (*batch_shape, 5)
            )
        ),
        SkyFrame(jnp.full(batch_shape, 10.0)),
        **{
            name: jnp.broadcast_to(
                jnp.array([*values, *([1.0 if name == "a_AU" else 0.0] * 3)]),
                (*batch_shape, 5),
            )
            for name, values in ELEMENTS.items()
        },
    )
    extra = padded.solve(times)
    np.testing.assert_array_equal(extra.position_AU[..., :3, :, :], result.position_AU)
    np.testing.assert_array_equal(
        extra.velocity_AU_per_d[..., :3, :, :], result.velocity_AU_per_d
    )
    if batch_shape:
        expected = np.stack([truth.separation_arcsec(i) for i in (1, 2)])
        np.testing.assert_array_equal(
            result.separation_arcsec(jnp.array([1, 2])),
            np.broadcast_to(expected, (*batch_shape, *expected.shape)),
        )
        paired = batch.solve(jnp.broadcast_to(times, batch_shape + times.shape))
        np.testing.assert_array_equal(paired.position_AU, result.position_AU)


@pytest.mark.parametrize("planet_count", [1, 2, 3, 4])
def test_jacobi_inverse_removes_only_barycenter(planet_count):
    """Truth: mass-weighted Cartesian barycenter.

    Each companion adds a weight division, two products and three prefix
    additions in the two Jacobi maps. Four more operations remove the
    origin and recover the inertial states.

    Random masses span four decades, and extra time axes exercise the
    same inverse for positions or velocities in an arbitrary common origin.
    Error is relative to the full centered vector norm in each system,
    avoiding componentwise cancellation at the small stellar reflex.
    """
    rng = np.random.default_rng(74 + planet_count)
    star = rng.uniform(0.7, 1.3, 2)
    planets = 10 ** rng.uniform(-6, -2, (2, planet_count))
    values = rng.normal(size=(2, planet_count + 1, 3, 2, 3))
    masses = np.concatenate([star[:, None], planets], axis=1)
    center = np.sum(values * masses[..., None, None, None], axis=1, keepdims=True)
    center /= masses.sum(axis=1)[:, None, None, None, None]
    inverse = jax.jit(barycentric_to_jacobi)(values, star, planets)
    actual = jacobi_to_barycentric(inverse, star, planets)
    expected = values - center
    error = np.linalg.norm((actual - expected).reshape(2, -1), axis=-1)
    scale = np.linalg.norm(expected.reshape(2, -1), axis=-1)
    assert np.all(
        error / scale
        < float(roundoff(1.0, n_ops=6 * planet_count + 4, dtype=actual.dtype))
    )


def test_absolute_cartesian_snapshot_preserves_small_relative_orbit():
    """A float64 host snapshot fixes the radius and velocity before transfer.

    A 1/16 AU orbit and its circular velocity are both smaller than the
    float32 spacing at the common origin. Eight operations for the mass,
    norms, and energy give the semimajor-axis bound; the Jacobi chart and
    rotation add up to 32 operations at the position and velocity scales.
    """
    star, planets = Body(1.0), Body([0.0])
    radius = 0.0625
    speed = np.sqrt(GM_SUN_AU3_PER_D2 / radius)
    origin = 2.0**21
    position = np.array([[origin, 0, 0], [origin + radius, 0, 0]])
    velocity = np.array([[0, origin, 0], [0, origin + speed, 0]])
    r, v = position[1] - position[0], velocity[1] - velocity[0]
    # Host subtraction must also work for a float32 owner in an x64 process.
    jacobi = barycentric_to_jacobi(position, jnp.float32(1), jnp.zeros(1, jnp.float32))
    assert jacobi.dtype == jnp.float32
    np.testing.assert_array_equal(jacobi[0], r)
    expected_a = 1 / (2 / np.linalg.norm(r) - v @ v / GM_SUN_AU3_PER_D2)
    system = System.from_cartesian(
        star, planets, SkyFrame(10.0), position_AU=position, velocity_AU_per_d=velocity
    )
    np.testing.assert_allclose(
        system.orbits.a_AU,
        expected_a,
        atol=float(roundoff(radius, n_ops=8, dtype=system.orbits.a_AU.dtype)),
        rtol=0,
    )
    for values, expected in zip(system.orbits.to_cartesian(), (r, v), strict=True):
        np.testing.assert_allclose(
            values[0],
            expected,
            atol=float(
                roundoff(np.linalg.norm(expected), n_ops=32, dtype=values.dtype)
            ),
            rtol=0,
        )


@pytest.mark.parametrize("batch_shape", [(), (2,)])
def test_system_from_cartesian_epoch_round_trip(batch_shape):
    """Truth: input Cartesian snapshot in AU and AU/day.

    Allow 64 operations at each state scale for the Jacobi sums, norms,
    eccentric longitude, quaternion construction and final rotation.

    Measured: 1.8e-15 AU in position and 4.3e-18 AU/day in velocity, so
    the epoch round trip is exact to rounding. A translated origin and
    velocity boost cancel in star-relative states. JIT construction and
    batched solves exercise traced per-system leaves.
    """
    epoch = 23.0
    base = _system().solve(jnp.asarray(epoch))
    r = base.position_AU + jnp.array([2.0, -3.0, 0.7])
    v = base.velocity_AU_per_d + jnp.array([0.01, -0.02, 0.03])
    r = jnp.broadcast_to(r, batch_shape + r.shape)
    v = jnp.broadcast_to(v, batch_shape + v.shape)
    star = Body(jnp.ones(batch_shape))
    planets = Body(jnp.broadcast_to(jnp.array([0.001, 0.005]), (*batch_shape, 2)))
    frame = SkyFrame(jnp.full(batch_shape, 10.0))
    system = eqx.filter_jit(System.from_cartesian)(
        star, planets, frame, position_AU=r, velocity_AU_per_d=v, epoch_d=epoch
    )
    result = system.solve(jnp.array([epoch]))
    for actual, expected in (
        (result.position_AU[..., 0, :], r),
        (result.velocity_AU_per_d[..., 0, :], v),
    ):
        np.testing.assert_allclose(
            actual[..., 1:, :] - actual[..., :1, :],
            expected[..., 1:, :] - expected[..., :1, :],
            atol=float(ulps(64, jnp.max(jnp.abs(expected)))),
            rtol=0,
        )


def test_system_from_cartesian_per_system_epochs():
    """Truth: snapshots at 23 and 35 days.

    Allow 64 operations at the state scale for the Jacobi sums,
    Cartesian chart, Kepler solve and quaternion rotation.
    """
    epochs = jnp.array([23.0, 35.0])
    source = _system().solve(epochs)
    r = jnp.moveaxis(source.position_AU, 1, 0)
    v = jnp.moveaxis(source.velocity_AU_per_d, 1, 0)
    system = eqx.filter_jit(System.from_cartesian)(
        Body(jnp.ones(2)),
        Body(jnp.broadcast_to(jnp.array([0.001, 0.005]), (2, 2))),
        SkyFrame(jnp.full(2, 10.0)),
        position_AU=r,
        velocity_AU_per_d=v,
        epoch_d=epochs,
    )
    result = system.solve(epochs[:, None])
    for actual, expected in (
        (result.position_AU[..., 0, :], r),
        (result.velocity_AU_per_d[..., 0, :], v),
    ):
        np.testing.assert_allclose(
            actual[:, 1:] - actual[:, :1],
            expected[:, 1:] - expected[:, :1],
            atol=float(ulps(64, jnp.max(jnp.abs(expected)))),
            rtol=0,
        )


@pytest.mark.parametrize("shape", [(3,), (2, 3), (1,)])
def test_system_from_cartesian_rejects_planet_epochs(shape):
    """Truth: one epoch per system, B=(2,); exact rejection needs no tolerance."""
    with pytest.raises(ValueError, match="epoch_d"):
        System.from_cartesian(
            Body(jnp.ones(2)),
            Body(jnp.full((2, 3), 0.001)),
            SkyFrame(jnp.full(2, 10.0)),
            position_AU=jnp.zeros((2, 4, 3)),
            velocity_AU_per_d=jnp.zeros((2, 4, 3)),
            epoch_d=jnp.ones(shape),
        )


@pytest.mark.parametrize("field", ["position_AU", "velocity_AU_per_d"])
@pytest.mark.parametrize("shape", [(2, 3), (3, 2), (1, 3, 3)])
def test_system_from_cartesian_rejects_wrong_shapes(field, shape):
    """Truth: B + (K+1, 3) shape contract; exact rejection needs no tolerance."""
    states = dict(position_AU=jnp.zeros((3, 3)), velocity_AU_per_d=jnp.zeros((3, 3)))
    states[field] = jnp.zeros(shape)
    with pytest.raises(ValueError, match=r"B \+ \(K\+1, 3\)"):
        System.from_cartesian(Body(1.0), Body([0.001, 0.001]), SkyFrame(10.0), **states)


@pytest.mark.float64_only(reason="float64 reference values for orbital validation")
def test_cartesian_chain_against_nbody(record_property):
    """Truth: independent Newtonian RK4 of the three bodies.

    Two initially circular Jupiter-mass planets at 1 and 5.2 AU share a
    60-degree inclination with phases 90 degrees apart, at 10 pc. The
    chain of Kepler orbits omits the planet-planet pull, so it departs
    from N-body by a residual that grows with time. Basis, from a spike:
    in this geometry the residual measured 0.014/0.151 mas (inner) and
    0.002/0.171 mas (outer) at 1/10 yr; over the probed geometries
    (inclinations 0 to 85 deg, phase separations 0 to 180 deg) it ranged
    0.04 to 0.20 mas at 10 yr and stayed within 0.015 mas at 1 yr. The
    limits are 0.05 mas at 1 yr and 0.3 mas at 5 and 10 yr; all three
    kill a chain that treats the planets as star-relative
    (0.65/2.4/1.4 mas). Halving the 0.75-day RK4 step must move every
    star-relative sky offset by less than 0.001 mas.
    """
    masses = np.array([1.0, Mjup2kg / Msun2kg, Mjup2kg / Msun2kg])
    inclination = np.deg2rad(60.0)
    phase = np.array([0.0, np.pi / 2])
    radial = np.array([0.0, np.cos(inclination), np.sin(inclination)])
    tangent = np.array([1.0, 0.0, 0.0])
    position = np.zeros((3, 3))
    position[1:] = np.array([1.0, 5.2])[:, None] * (
        np.cos(phase)[:, None] * radial + np.sin(phase)[:, None] * tangent
    )
    velocity = np.zeros_like(position)
    speed = np.sqrt(GM_SUN_AU3_PER_D2 * (1 + masses[1:]) / [1.0, 5.2])
    velocity[1:] = speed[:, None] * (
        -np.sin(phase)[:, None] * radial + np.cos(phase)[:, None] * tangent
    )
    position -= np.sum(masses[:, None] * position, axis=0) / masses.sum()
    velocity -= np.sum(masses[:, None] * velocity, axis=0) / masses.sum()
    times = np.array([1.0, 5.0, 10.0]) * yr2s / d2s
    factor = rad2arcsec * arcsec2mas / (10.0 * pc2AU)

    def acceleration(r):
        delta = r[None, :, :] - r[:, None, :]
        squared = np.sum(delta * delta, axis=-1)
        np.fill_diagonal(squared, np.inf)
        return GM_SUN_AU3_PER_D2 * np.sum(
            delta * (masses[None, :] / squared**1.5)[..., None], axis=1
        )

    def integrate(step):
        r, v = position.copy(), velocity.copy()
        snapshots = []
        previous = 0.0
        for time in times:
            # Both fixed steps hit every requested year exactly.
            count = round((time - previous) / step)
            dt = step
            assert count * dt == time - previous
            for _ in range(count):
                a1 = acceleration(r)
                v2 = v + dt * a1 / 2
                a2 = acceleration(r + dt * v / 2)
                v3 = v + dt * a2 / 2
                a3 = acceleration(r + dt * v2 / 2)
                v4 = v + dt * a3
                a4 = acceleration(r + dt * v3)
                r += dt * (v + 2 * v2 + 2 * v3 + v4) / 6
                v += dt * (a1 + 2 * a2 + 2 * a3 + a4) / 6
            snapshots.append((r[1:] - r[0])[:, :2] * factor)
            previous = time
        return np.stack(snapshots, axis=1)

    coarse, truth = integrate(0.75), integrate(0.375)
    convergence = np.linalg.norm(coarse - truth, axis=-1)
    assert convergence.max() < 1e-3
    system = System.from_cartesian(
        Body(masses[0]),
        Body(masses[1:]),
        SkyFrame(10.0),
        position_AU=position,
        velocity_AU_per_d=velocity,
    )
    eph = system.solve(jnp.asarray(times))
    actual = np.stack(eph.offset_arcsec(jnp.array([1, 2])), axis=-1) * arcsec2mas
    residual = np.linalg.norm(actual - truth, axis=-1)
    record_property("nbody_sky_offsets_mas", truth.tolist())
    record_property("jacobi_residual_mas", residual.tolist())
    record_property("step_halving_max_mas", float(convergence.max()))
    print("N-body east/north offsets [planet, year, coordinate] (mas):", truth)
    print("Jacobi residuals [planet, year] (mas):", residual)
    print("Step-halving maximum (mas):", convergence.max())
    assert np.all(residual[:, 0] < 0.05)
    assert np.all(residual[:, 1:] < 0.3)


def test_ephemeris_rejects_distance_history_and_uses_scalar_distance():
    """Truth: one AU at one pc subtends one arcsec; 2e-15 allows conversion.

    Three bodies at three dates: a distance per date, [1, 2, 3] pc, has
    the length of the body axis and must still be refused (it would move
    body indexing onto the time axis), as must a (2,) distance. Shape
    rejection is exact. With a scalar 1 pc distance, [2, 4, 6] AU must
    yield [2, 4, 6] arcsec at the three dates.
    """
    position = jnp.zeros((3, 3, 3)).at[1, :, 0].set(jnp.array([2.0, 4.0, 6.0]))
    velocity = jnp.zeros_like(position)
    for history in ([1.0, 2.0, 3.0], [1.0, 2.0]):
        with pytest.raises(ValueError, match=r"distance history.*batch shape"):
            Ephemeris(position, velocity, jnp.array(history))
    eph = Ephemeris(position, velocity, 1.0)
    east, north = eph.offset_arcsec(1, 0)
    np.testing.assert_allclose(east, [2.0, 4.0, 6.0], rtol=2e-15, atol=0)
    np.testing.assert_array_equal(north, 0.0)


def test_ephemeris_accepts_batch_distances():
    """Truth: separate systems at 1 and 2 pc; 2e-15 allows angular conversion.

    A batch is declared by batch_ndim; without it the (2,) distance is a
    distance history and is refused.
    """
    position = jnp.zeros((2, 3, 3, 3)).at[:, 1, :, 0].set(2.0)
    with pytest.raises(ValueError, match="distance history"):
        Ephemeris(position, jnp.zeros_like(position), jnp.array([1.0, 2.0]))
    eph = Ephemeris(
        position.tolist(), jnp.zeros_like(position), [1.0, 2.0], batch_ndim=1
    )
    assert _system().solve(jnp.arange(3.0)).batch_ndim == 0
    east, north = eph.offset_arcsec(1, 0)
    np.testing.assert_allclose(east, [[2.0] * 3, [1.0] * 3], rtol=2e-15, atol=0)
    np.testing.assert_array_equal(north, 0.0)


def test_ephemeris_batch_rank_follows_vmap():
    """Truth: vmap adds one batch axis; ranks and stored leaves agree exactly.

    Reconstruction and positional subclass construction must preserve the
    arrays bit for bit, so no numerical tolerance is needed.
    """

    class _SceneEphemeris(Ephemeris):
        scene_planet_indices: jax.Array

    systems = jax.tree.map(lambda x: jnp.stack([x, x]), _system())
    solve = jax.vmap(lambda system: system.solve(jnp.array([0.0, 23.0])))
    for evaluate in (solve, jax.jit(solve)):
        eph = evaluate(systems)
        assert eph.batch_ndim == eph.dist_pc.ndim == 1
        rebuilt = Ephemeris(
            eph.position_AU,
            eph.velocity_AU_per_d,
            eph.dist_pc,
            batch_ndim=eph.batch_ndim,
        )
        scene = _SceneEphemeris(
            eph.position_AU,
            eph.velocity_AU_per_d,
            eph.dist_pc,
            jnp.array([1, 2]),
            batch_ndim=eph.batch_ndim,
        )
        assert rebuilt.batch_ndim == scene.batch_ndim == 1
        np.testing.assert_array_equal(rebuilt.position_AU, eph.position_AU)
        np.testing.assert_array_equal(scene.scene_planet_indices, [1, 2])


@pytest.mark.parametrize("rank", [-2, -1, 2, 1.0, True])
def test_ephemeris_rejects_invalid_batch_rank(rank):
    """Truth: Python integer rank in [0, ndim - 2]; exact rejection, no tolerance."""
    position = jnp.zeros((3, 3, 3)).at[1, :, 0].set(jnp.array([2.0, 4.0, 6.0]))
    with pytest.raises(ValueError, match="must be an integer between"):
        Ephemeris(position, jnp.zeros_like(position), [1.0, 2.0, 3.0], batch_ndim=rank)
