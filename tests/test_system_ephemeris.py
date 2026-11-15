"""Jacobi hierarchy and ephemeris observables against independent mechanics."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import AU2m, G, Msun2kg, d2s

from orbix.kepler import Markley
from orbix.kepler.markley import solve_trig
from orbix.orbit import KeplerianOrbit
from orbix.system import Body, FrameOrbit, SkyFrame, System, enclosed_mu, observables

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


@pytest.mark.parametrize("e,inc", [(1e-6, 0.7), (0.6, 1.1), (0.2, 2.6)])
def test_existing_orbit_agreement(e, inc):
    """Fails if sky convention, epoch, mass, reflex or velocity units change.

    Truth: the existing KeplerianOrbit with explicit Markley (not its grid
    default), equal star/planet masses and a shared day origin. Its t0_d
    and new epoch_d both equal 23 days, with M0=.2 at that instant.
    Allow 3e-13 arcsec, 3e-14 AU/day and 3e-10 m/s: several hundred x64
    ULPs for independently arranged phase, basis and mass arithmetic.
    """
    elements = dict(a_AU=1.7, e=e, i_rad=inc, W_rad=0.4, w_rad=0.7, M0_rad=0.2)
    sys = System.from_classical(
        Body(1.2), Body([0.01]), SkyFrame(12.0), **elements, epoch_d=23.0
    )
    old = KeplerianOrbit(**elements, t0_d=23.0)
    times = jnp.array([-100.0, 23.0, 200.0, 700.0])
    context = dict(Ms_kg=1.2 * Msun2kg, Mp_kg=0.01 * Msun2kg)
    eph = sys.solve(times)
    offsets = np.asarray(eph.offset_arcsec(1))
    old_offsets = np.asarray(
        old.position_arcsec(solve_trig, times, **context, dist_pc=12.0)
    )[:, 0]
    reflex = np.asarray(eph.offset_arcsec(0, "barycenter"))
    old_reflex = np.asarray(
        old.reflex_position_arcsec(solve_trig, times, **context, dist_pc=12.0)
    )[:, 0]
    rv = eph.radial_velocity_m_per_s(0)
    old_rv = old.radial_velocity_m_s(solve_trig, times, **context)[0]
    velocity = eph.velocity_AU_per_d[1] - eph.velocity_AU_per_d[0]
    old_velocity = old.velocity_AU_d(solve_trig, times, **context)[0].T
    np.testing.assert_allclose(offsets, old_offsets, atol=3e-13, rtol=0)
    np.testing.assert_allclose(reflex, old_reflex, atol=3e-13, rtol=0)
    np.testing.assert_allclose(rv, old_rv, atol=3e-10, rtol=0)
    np.testing.assert_allclose(velocity, old_velocity, atol=3e-14, rtol=0)
    print(
        "parity max (offset, reflex, RV, velocity):",
        *(
            float(np.max(np.abs(a - b)))
            for a, b in [
                (offsets, old_offsets),
                (reflex, old_reflex),
                (rv, old_rv),
                (velocity, old_velocity),
            ]
        ),
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
    Dermott). Allow 4e-17 Msun AU, 4e-19 Msun AU/day and 3e-15 relative
    in the third law for x64 cancellation. A massless outer companion
    leaves inner states unchanged to 2e-15 AU and AU/day roundoff.
    """
    sys = _system(second_mass)
    time = jnp.array([0.0, 200.0, 700.0])
    eph = sys.solve(time)
    masses = np.array([1.0, 0.001, second_mass])
    np.testing.assert_allclose(
        np.einsum("k,ktc->tc", masses, eph.position_AU), 0.0, atol=4e-17
    )
    np.testing.assert_allclose(
        np.einsum("k,ktc->tc", masses, eph.velocity_AU_per_d), 0.0, atol=4e-19
    )
    expected_mu = G * Msun2kg * np.array([1.001, 1.001 + second_mass])
    np.testing.assert_allclose(
        enclosed_mu(1.0, jnp.array([0.001, second_mass])), expected_mu, rtol=3e-15
    )
    np.testing.assert_allclose(
        (2 * np.pi / np.asarray(sys.orbits.period_d)) ** 2 * np.array([1.0, 5.0]) ** 3,
        expected_mu,
        rtol=3e-15,
    )
    if second_mass == 0.0:
        single = System.from_classical(
            Body(1.0),
            Body([0.001]),
            SkyFrame(10.0),
            **{k: v[0] for k, v in ELEMENTS.items()},
        ).solve(time)
        np.testing.assert_allclose(
            eph.position_AU[:2], single.position_AU, atol=2e-15, rtol=0
        )
        np.testing.assert_allclose(
            eph.velocity_AU_per_d[:2], single.velocity_AU_per_d, atol=2e-15, rtol=0
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
    2e-15 relative for x64 unit conversion. Equal solver configurations
    must add exactly one trace on the first call and none on the second.
    Row pairing is checked against separately evaluated systems at each time.
    """
    frame = SkyFrame(10.0, 2460000.0, 0.125, "tdb")
    dates = np.array([2460000.125, 2460001.25])
    with jax.enable_x64(False):
        np.testing.assert_array_equal(frame.time_d(dates), [0.0, 1.125])
        assert frame.time_d(dates).dtype == np.float64
    assert type(frame.epoch_jd1) is type(frame.epoch_jd2) is float
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
    np.testing.assert_array_equal(
        sys.solve_at(times).position_AU, sys.solve(times).position_AU
    )
    # Stacked systems are caller-owned; solve_at pairs each draw with one row.
    draws = jax.tree.map(lambda x: jnp.stack([x] * 3), sys)
    paired = jax.vmap(lambda system, time: system.solve_at(time))(draws, times)
    np.testing.assert_allclose(
        paired.position_AU, jnp.moveaxis(eph.position_AU, 1, 0), atol=2e-14, rtol=0
    )
    r = eph.position_AU[1] - eph.position_AU[0]
    v = eph.velocity_AU_per_d[1] - eph.velocity_AU_per_d[0]
    east, north = eph.offset_arcsec(1)
    np.testing.assert_allclose(
        eph.separation_arcsec(1), np.hypot(east, north), rtol=2e-15
    )
    np.testing.assert_allclose(
        eph.distance_AU(1), np.sqrt(np.sum(r * r, -1)), rtol=2e-15
    )
    np.testing.assert_allclose(
        eph.radial_velocity_m_per_s(1, 0), v[..., 2] * AU2m / d2s, rtol=2e-15
    )
    np.testing.assert_allclose(
        eph.phase_angle_rad(1),
        np.arctan2(np.hypot(r[..., 0], r[..., 1]), r[..., 2]),
        rtol=2e-15,
    )
    np.testing.assert_allclose(
        eph.position_angle_deg(1),
        np.rad2deg(np.arctan2(r[..., 0], r[..., 1])) % 360,
        rtol=2e-15,
    )
    with pytest.raises(ValueError, match="reference"):
        eph.distance_AU(1, "invalid")
