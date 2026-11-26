"""Trace public orbit and sky geometry with float32 leaves in an x64 process."""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import priorities
import pytest
from hwoutils.constants import Msun2kg
from hwoutils.epoch import Epoch
from hwoutils.leaves import canonical
from hwoutils.precision import float_dtype
from hwoutils.testing import assert_precision_closed, precision_pair, ulps
from hwoutils.testing._numerical import _first_wide, _subjaxprs, _wide_dtype
from jax.extend.source_info_util import summarize

from orbix import equations as eq
from orbix.kepler import Grid, Halley, Markley, kepler
from orbix.observatory import (
    Observatory,
    ObservatoryL2Halo,
    body_angle_deg,
    ecliptic_pointing,
    is_observable,
    sun_ecliptic_longitude_deg,
)
from orbix.observatory.keepout import target_ecliptic_direction
from orbix.observatory.solar_system import (
    _rot1,
    _rot3,
    earth_position_ecliptic,
    earth_position_equatorial,
    eclip2equat,
    equat2eclip,
    equatorial_to_ecliptic_deg,
    planet_position_ecliptic,
    planet_position_equatorial,
)
from orbix.system import (
    Body,
    CartesianChart,
    ClassicalChart,
    EccentricityVectorChart,
    Ephemeris,
    FrameChart,
    FrameOrbit,
    IsotropicInclination,
    OrbitLaw,
    SkyFrame,
    System,
    ThieleInnesChart,
    barycentric_to_jacobi,
    disk_to_plane,
    enclosed_mu,
    jacobi_to_barycentric,
    periapsis_chart_shape,
    plane_to_disk,
)
from orbix.system import observables as obs
from orbix.system.linear import (
    frame_to_sky_block,
    line_of_sight_sheet,
    reflect_line_of_sight,
    sky_block_geometry,
    sky_block_to_frame,
    sky_design,
)
from orbix.system.rotation import from_axis_angle, from_basis, rotate

_EPOCH = Epoch(2451545.0, 8123.0)


@pytest.fixture(scope="module")
def geometry(tmp_path_factory):
    """Use a nonsingular, mildly eccentric two-planet system for paired checks."""
    star = Body(jnp.array(1.0))
    planets = Body(jnp.array([3e-6, 1e-3]))
    frame = SkyFrame(jnp.array(10.0), _EPOCH, jnp.array(110.0), jnp.array(32.0))
    classical = tuple(map(jnp.array, ([1.2, 3.1], [0.2, 0.4], 0.6, 0.7, 0.9, 0.4)))
    a, eccentricity, inc, node, peri, mean = classical
    mu = enclosed_mu(star.mass_Msun, planets.mass_Msun)
    orbit = FrameOrbit.from_classical(*classical, mu_AU3_per_d2=mu)
    system = System(star, planets, orbit, frame)
    time = jnp.array([11.0, 47.0, 83.0])
    ephemeris = system.solve(time)
    snapshot = system.solve(jnp.array(0.0))
    r, v = orbit.to_cartesian()
    halo_file = tmp_path_factory.mktemp("halo") / "halo.npz"
    np.savez(
        halo_file,
        t_yr=[0.0, 0.2, 0.5],
        r_earth_AU=[[0.01, 0.001, 0.005], [0.01, 0.002, 0.0], [0.01, 0.001, 0.005]],
        period_yr=0.5,
        L2_dist_AU=1.01,
        mu=3e-6,
    )
    return {
        "star": star,
        "planets": planets,
        "frame": frame,
        "classical": classical,
        "trig": (a, eccentricity, inc, node, peri, mean, mu),
        "orbit": orbit,
        "system": system,
        "batch": jax.tree.map(lambda x: jnp.stack([x, x]), system),
        "time": time,
        "ephemeris": ephemeris,
        "snapshot": snapshot,
        "r": r[0],
        "v": v[0],
        "mu": mu[0],
        "halo": ObservatoryL2Halo.from_default(),
        "custom_halo": ObservatoryL2Halo.from_npz(halo_file),
    }


@pytest.fixture(scope="module")
def cases(geometry):
    """Provide a callable and typed inputs for each public numerical route."""
    g = geometry
    a, e, inc, node, peri, mean, mu = g["trig"]
    orbit, time, system = g["orbit"], g["time"], g["system"]
    frame, star, planets = g["frame"], g["star"], g["planets"]
    halo = g["halo"]
    platform = Observatory(halo, keepout_min_deg=45.0)
    r, v, mu0 = g["r"], g["v"], g["mu"]
    day = time[0]
    ra, dec = frame.ra_deg, frame.dec_deg
    distance = frame.dist_pc
    position = jnp.array([1.1, 0.4, 0.01])
    sun = jnp.zeros_like(position)
    quat = orbit.quat[0]
    axis = jnp.array([0.0, 0.0, 1.0])
    basis = jnp.eye(3)
    constants = eq.thiele_innes_constants(node, inc, peri)
    period, mass, companion = map(jnp.array, (400.0, Msun2kg, 3e-6 * Msun2kg))
    direction = IsotropicInclination()
    law = OrbitLaw(direction, a_AU=direction)
    relative = orbit.at(time)[0]
    masses = (star.mass_Msun, planets.mass_Msun)
    snap = g["snapshot"]
    single = jax.tree.map(lambda x: x[0], orbit)
    charts = (
        FrameChart(),
        CartesianChart(),
        ClassicalChart(),
        EccentricityVectorChart(),
        ThieleInnesChart(),
    )
    coordinates = {type(chart).__name__: chart.from_orbit(single) for chart in charts}
    date = _EPOCH.jd1 + _EPOCH.jd2 + np.array([11.0, 47.0, 83.0])

    def chart_to_orbit(chart, x, mu):
        return chart.to_orbit(x, mu_AU3_per_d2=mu)

    def clock_solve(model):
        return model.solve(model.frame.time_d(jd=date))

    def longitude_tangent(lam, ex, ey):
        return jax.jvp(kepler, (lam, ex, ey), (jnp.ones_like(lam), ex, ey))

    return {
        "Markley.solve": (Markley().solve, (mean, e), 32),
        "Halley.solve": (Halley().solve, (mean, e), 32),
        "Grid.solve": (Grid(16, 64).solve, (mean, e), 32),
        "kepler": (kepler, (time / 32, orbit.ex[0], orbit.ey[0]), 32),
        "kepler.weak_zero": (
            lambda lam, ex: kepler(lam, ex, 0.0),
            (time / 32, e[0]),
            32,
        ),
        "kepler.jvp": (longitude_tangent, (time / 32, orbit.ex[0], orbit.ey[0]), 64),
        "kepler.Halley": (
            partial(kepler, solver=Halley()),
            (time / 32, orbit.ex[0], orbit.ey[0]),
            32,
        ),
        "kepler.Grid": (
            partial(kepler, solver=Grid(16, 64)),
            (time / 32, orbit.ex[0], orbit.ey[0]),
            32,
        ),
        "period_a": (eq.period_a, (a, mu), 16),
        "period_n": (eq.period_n, (orbit.n_rad_per_d,), 16),
        "mean_motion": (eq.mean_motion, (a, mu), 16),
        "period_to_sma": (eq.period_to_sma, (period, mass), 16),
        "semi_amplitude": (eq.semi_amplitude, (period, mass, companion, e, inc), 64),
        "semi_amplitude_reduced": (
            eq.semi_amplitude_reduced,
            (period, mass, companion * jnp.sin(inc), jnp.sqrt(1 - e**2)),
            64,
        ),
        "mean_anomaly_t0": (
            eq.mean_anomaly_t0,
            (day, single.n_rad_per_d, mean, day / 2),
            16,
        ),
        "mean_anomaly_tp": (eq.mean_anomaly_tp, (day, single.n_rad_per_d, day / 2), 16),
        "thiele_innes_constants": (eq.thiele_innes_constants, (node, inc, peri), 16),
        "thiele_innes_to_elements": (eq.thiele_innes_to_elements, constants, 64),
        "thiele_innes_constants_reduced": (
            eq.thiele_innes_constants_reduced,
            (
                jnp.sin(node),
                jnp.cos(node),
                jnp.sin(peri),
                jnp.cos(peri),
                jnp.sin(peri) * jnp.cos(inc),
                jnp.cos(peri) * jnp.cos(inc),
            ),
            16,
        ),
        "state_vector_to_keplerian": (eq.state_vector_to_keplerian, (r, v, mu0), 128),
        "lambert_solve": (
            eq.lambert_solve,
            (r, jnp.array([0.1, -1.2, 0.4]), day * 52, mu0, jnp.array(1)),
            128,
        ),
        "lambert_tof_min": (
            eq.lambert_tof_min,
            (r, jnp.array([0.1, -1.2, 0.4]), mu0, jnp.array(1)),
            128,
        ),
        "lambert_solve.zero_short": (
            eq.lambert_solve,
            (basis[0], basis[1], jnp.array(1.4), jnp.array(1.0)),
            128,
        ),
        "lambert_solve.zero_long": (
            partial(eq.lambert_solve, long_way=True),
            (basis[0], basis[1], jnp.array(1.4), jnp.array(1.0)),
            128,
        ),
        "lambert_tof_min.zero_short": (
            eq.lambert_tof_min,
            (basis[0], basis[1], jnp.array(1.0)),
            16,
        ),
        "lambert_tof_min.zero_long": (
            partial(eq.lambert_tof_min, long_way=True),
            (basis[0], basis[1], jnp.array(1.0)),
            16,
        ),
        "offset_arcsec": (obs.offset_arcsec, (r, distance), 16),
        "offset_arcsec.host_distance": (lambda r: obs.offset_arcsec(r, 10.0), (r,), 16),
        "separation_arcsec": (obs.separation_arcsec, (r, distance), 16),
        "position_angle_deg": (obs.position_angle_deg, (r,), 16),
        "phase_angle_rad": (obs.phase_angle_rad, (r,), 16),
        "radial_velocity_m_per_s": (obs.radial_velocity_m_per_s, (v,), 16),
        "distance_AU": (obs.distance_AU, (r,), 16),
        "from_axis_angle": (from_axis_angle, (axis, inc), 16),
        "from_axis_angle.host_axis": (
            lambda t: from_axis_angle([0.0, 0.0, 1.0], t),
            (inc,),
            16,
        ),
        "from_axis_angle.host_angle": (lambda a: from_axis_angle(a, 0.6), (axis,), 16),
        "rotate": (rotate, (quat, r), 16),
        "rotate.host_vector": (lambda q: rotate(q, [0.0, 0.0, 1.0]), (quat,), 16),
        "from_basis": (from_basis, tuple(basis), 32),
        "IsotropicInclination.cdf": (direction.cdf, (inc,), 16),
        "IsotropicInclination.icdf": (direction.icdf, (e,), 16),
        "IsotropicInclination.log_prob": (direction.log_prob, (inc,), 16),
        "OrbitLaw.from_unit": (
            lambda law, c, mu: law.from_unit(c, mu_AU3_per_d2=mu),
            (law, jnp.array([0.1, 0.5, 0.3, 0.4, 0.2, 0.7]), mu0),
            64,
        ),
        "FrameOrbit.from_classical": (
            lambda a, e, i, W, w, M, mu: FrameOrbit.from_classical(
                a, e, i, W, w, M, mu_AU3_per_d2=mu
            ),
            (*g["classical"], mu),
            32,
        ),
        "FrameOrbit.from_classical_trig": (
            lambda a, e, i, W, w, M, mu: FrameOrbit.from_classical_trig(
                a_AU=a,
                e=e,
                cos_i=jnp.cos(i),
                sin_i=jnp.sin(i),
                cos_W=jnp.cos(W),
                sin_W=jnp.sin(W),
                cos_w=jnp.cos(w),
                sin_w=jnp.sin(w),
                cos_M0=jnp.cos(M),
                sin_M0=jnp.sin(M),
                mu_AU3_per_d2=mu,
            ),
            g["trig"],
            32,
        ),
        "FrameOrbit.from_periapsis_time": (
            lambda a, e, i, W, w, t, mu: FrameOrbit.from_periapsis_time(
                a, e, jnp.cos(i), W, jnp.cos(w), jnp.sin(w), t, mu_AU3_per_d2=mu
            ),
            (a, e, inc, node, peri, day, mu),
            32,
        ),
        "FrameOrbit.from_unit_plane": (
            FrameOrbit.from_unit_plane,
            (e, day, orbit.n_rad_per_d),
            16,
        ),
        "FrameOrbit.from_unit_plane_phase": (
            FrameOrbit.from_unit_plane_phase,
            (e, jnp.log(orbit.period_d), e, day),
            16,
        ),
        "FrameOrbit.from_cartesian": (
            lambda r, v, mu: FrameOrbit.from_cartesian(r, v, mu_AU3_per_d2=mu),
            (r, v, mu0),
            128,
        ),
        "FrameOrbit.to_cartesian": (FrameOrbit.to_cartesian, (orbit,), 64),
        "FrameOrbit.to_classical": (FrameOrbit.to_classical, (orbit,), 64),
        "FrameOrbit.to_periapsis_elements": (
            FrameOrbit.to_periapsis_elements,
            (orbit,),
            64,
        ),
        "FrameOrbit.at": (FrameOrbit.at, (orbit, time), 64),
        "FrameOrbit.at_each": (FrameOrbit.at_each, (orbit, time[:2]), 64),
        "FrameOrbit.offsets_arcsec": (
            FrameOrbit.offsets_arcsec,
            (orbit, time, distance),
            64,
        ),
        "FrameOrbit.plane_basis": (FrameOrbit.plane_basis, (orbit,), 32),
        "FrameOrbit.linear_map": (lambda o: o.linear_map, (orbit,), 32),
        "FrameOrbit.plane_coordinates": (
            FrameOrbit.plane_coordinates,
            (orbit, time),
            64,
        ),
        "FrameOrbit.perifocal_coordinates": (
            FrameOrbit.perifocal_coordinates,
            (orbit, time),
            64,
        ),
        "FrameOrbit.in_plane": (FrameOrbit.in_plane, (orbit, time), 64),
        "FrameOrbit.in_plane_each": (FrameOrbit.in_plane_each, (orbit, time[:2]), 64),
        "FrameOrbit.e": (lambda o: o.e, (orbit,), 16),
        "FrameOrbit.period_d": (lambda o: o.period_d, (orbit,), 16),
        "FrameOrbit.mu_AU3_per_d2": (lambda o: o.mu_AU3_per_d2, (orbit,), 16),
        "enclosed_mu": (enclosed_mu, masses, 16),
        "enclosed_mu.host_star": (lambda p: enclosed_mu(1.0, p), (masses[1],), 16),
        "jacobi_to_barycentric": (jacobi_to_barycentric, (relative, *masses), 32),
        "barycentric_to_jacobi": (
            barycentric_to_jacobi,
            (g["ephemeris"].position_AU, *masses),
            32,
        ),
        "System.from_classical": (
            lambda s, p, f, a, e, i, W, w, M: System.from_classical(
                s, p, f, a_AU=a, e=e, i_rad=i, W_rad=W, w_rad=w, M0_rad=M
            ),
            (star, planets, frame, *g["classical"]),
            32,
        ),
        "System.from_cartesian": (
            lambda s, p, f, r, v: System.from_cartesian(
                s, p, f, position_AU=r, velocity_AU_per_d=v
            ),
            (star, planets, frame, snap.position_AU, snap.velocity_AU_per_d),
            128,
        ),
        "System.solve": (System.solve, (system, time), 64),
        "System.solve.clock": (clock_solve, (system,), 64),
        "System.solve.shared_times": (System.solve, (g["batch"], time), 64),
        "System.solve.paired_times": (
            System.solve,
            (g["batch"], jnp.stack([time, time + day])),
            64,
        ),
        **{
            f"Ephemeris.{name}": (getattr(Ephemeris, name), (g["ephemeris"], 1), 16)
            for name in (
                "offset_arcsec",
                "separation_arcsec",
                "position_angle_deg",
                "phase_angle_rad",
                "radial_velocity_m_per_s",
                "distance_AU",
            )
        },
        "frame_to_sky_block": (frame_to_sky_block, (a, orbit.quat), 32),
        "sky_block_geometry": (sky_block_geometry, constants, 32),
        "sky_block_to_frame": (sky_block_to_frame, constants, 64),
        "line_of_sight_sheet": (line_of_sight_sheet, (orbit,), 16),
        "reflect_line_of_sight": (reflect_line_of_sight, (orbit, 0), 16),
        "sky_design": (sky_design, (single, time), 64),
        "plane_to_disk": (plane_to_disk, (jnp.array([0.3, -0.4]),), 16),
        "disk_to_plane": (disk_to_plane, (jnp.array([0.3, -0.4]),), 16),
        "periapsis_chart_shape": (periapsis_chart_shape, (period, e, day), 16),
        "ThieleInnesChart.implied_mu": (
            ThieleInnesChart().implied_mu,
            (coordinates["ThieleInnesChart"],),
            64,
        ),
        **{
            type(chart).__name__ + ".to_orbit": (
                chart_to_orbit,
                (chart, coordinates[type(chart).__name__], mu0),
                128,
            )
            for chart in charts
        },
        **{
            type(chart).__name__ + ".from_orbit": (chart.from_orbit, (single,), 128)
            for chart in charts
        },
        "equat2eclip": (equat2eclip, (position,), 16),
        "eclip2equat": (eclip2equat, (position,), 16),
        "equatorial_to_ecliptic_deg": (equatorial_to_ecliptic_deg, (ra, dec), 32),
        "planet_position_ecliptic": (
            partial(planet_position_ecliptic, "Mars", _EPOCH),
            (day,),
            128,
        ),
        "planet_position_equatorial": (
            partial(planet_position_equatorial, "Jupiter", _EPOCH),
            (day,),
            128,
        ),
        "earth_position_ecliptic": (
            partial(earth_position_ecliptic, _EPOCH),
            (day,),
            128,
        ),
        "earth_position_equatorial": (
            partial(earth_position_equatorial, _EPOCH),
            (day,),
            128,
        ),
        "sun_ecliptic_longitude_deg": (
            partial(sun_ecliptic_longitude_deg, _EPOCH),
            (time,),
            128,
        ),
        "ecliptic_pointing": (
            lambda ra, dec, t: ecliptic_pointing(ra, dec, _EPOCH, t),
            (ra, dec, time),
            128,
        ),
        "ecliptic_pointing.host_sky": (
            lambda t: ecliptic_pointing(110.0, 32.0, _EPOCH, t),
            (time,),
            128,
        ),
        "target_ecliptic_direction": (target_ecliptic_direction, (ra, dec), 32),
        "body_angle_deg": (body_angle_deg, (position, sun, ra, dec), 32),
        "body_angle_deg.host_sky": (
            lambda p, s: body_angle_deg(p, s, 110.0, 32.0),
            (position, sun),
            32,
        ),
        "is_observable": (
            lambda p, ra, dec, t: is_observable(p, ra, dec, _EPOCH, t),
            (position, ra, dec, day),
            16,
        ),
        "ObservatoryL2Halo.position_ecliptic": (
            lambda h, t: h.position_ecliptic(_EPOCH, t),
            (halo, day),
            128,
        ),
        "ObservatoryL2Halo.position_ecliptic.custom": (
            lambda h, t: h.position_ecliptic(_EPOCH, t),
            (g["custom_halo"], day),
            128,
        ),
        "ObservatoryL2Halo.solar_elongation_deg": (
            lambda h, ra, dec, t: h.solar_elongation_deg(ra, dec, _EPOCH, t),
            (halo, ra, dec, day),
            128,
        ),
        "Observatory.is_observable_at": (
            lambda h, ra, dec, t: h.is_observable_at(ra, dec, _EPOCH, t),
            (platform, ra, dec, time),
            16,
        ),
        "Observatory.is_observable_at.host_sky": (
            lambda h, t: h.is_observable_at(110.0, 32.0, _EPOCH, t),
            (platform, time),
            16,
        ),
        "Observatory.observability": (
            lambda h, ra, dec, t: h.observability(
                ra, dec, _EPOCH, t[0], t[-1], n_samples=8
            ),
            (platform, ra, dec, time),
            128,
        ),
        "Observatory.observability.host_interval": (
            lambda h, ra, dec: h.observability(
                ra, dec, _EPOCH, 11.0, 83.0, n_samples=8
            ),
            (platform, ra, dec),
            128,
        ),
    }


@pytest.mark.float64_only(reason="Detect float64 intermediates in an x64 process")
@pytest.mark.parametrize(
    "entry",
    """
    Markley.solve Halley.solve Grid.solve kepler kepler.Halley kepler.Grid
    kepler.weak_zero kepler.jvp
    period_a period_n mean_motion period_to_sma semi_amplitude semi_amplitude_reduced
    mean_anomaly_t0 mean_anomaly_tp thiele_innes_constants thiele_innes_to_elements
    thiele_innes_constants_reduced state_vector_to_keplerian
    lambert_solve lambert_tof_min lambert_solve.zero_short lambert_solve.zero_long
    lambert_tof_min.zero_short lambert_tof_min.zero_long
    offset_arcsec offset_arcsec.host_distance separation_arcsec
    position_angle_deg phase_angle_rad
    radial_velocity_m_per_s distance_AU from_axis_angle rotate from_basis
    from_axis_angle.host_axis from_axis_angle.host_angle rotate.host_vector
    IsotropicInclination.cdf IsotropicInclination.icdf IsotropicInclination.log_prob
    OrbitLaw.from_unit
    FrameOrbit.from_classical FrameOrbit.from_classical_trig
    FrameOrbit.from_periapsis_time FrameOrbit.from_unit_plane
    FrameOrbit.from_unit_plane_phase FrameOrbit.from_cartesian
    FrameOrbit.to_cartesian FrameOrbit.to_classical FrameOrbit.to_periapsis_elements
    FrameOrbit.at FrameOrbit.at_each FrameOrbit.offsets_arcsec FrameOrbit.plane_basis
    FrameOrbit.linear_map FrameOrbit.plane_coordinates FrameOrbit.perifocal_coordinates
    FrameOrbit.in_plane FrameOrbit.in_plane_each FrameOrbit.e FrameOrbit.period_d
    FrameOrbit.mu_AU3_per_d2 enclosed_mu enclosed_mu.host_star
    jacobi_to_barycentric barycentric_to_jacobi
    System.from_classical System.from_cartesian System.solve System.solve.clock
    System.solve.shared_times
    System.solve.paired_times Ephemeris.offset_arcsec Ephemeris.separation_arcsec
    Ephemeris.position_angle_deg Ephemeris.phase_angle_rad
    Ephemeris.radial_velocity_m_per_s
    Ephemeris.distance_AU frame_to_sky_block sky_block_geometry sky_block_to_frame
    line_of_sight_sheet reflect_line_of_sight sky_design
    plane_to_disk disk_to_plane periapsis_chart_shape ThieleInnesChart.implied_mu
    FrameChart.to_orbit FrameChart.from_orbit CartesianChart.to_orbit
    CartesianChart.from_orbit ClassicalChart.to_orbit ClassicalChart.from_orbit
    EccentricityVectorChart.to_orbit EccentricityVectorChart.from_orbit
    ThieleInnesChart.to_orbit ThieleInnesChart.from_orbit
    equat2eclip eclip2equat
    equatorial_to_ecliptic_deg planet_position_ecliptic planet_position_equatorial
    earth_position_ecliptic earth_position_equatorial sun_ecliptic_longitude_deg
    ecliptic_pointing target_ecliptic_direction body_angle_deg is_observable
    ecliptic_pointing.host_sky body_angle_deg.host_sky
    ObservatoryL2Halo.position_ecliptic ObservatoryL2Halo.solar_elongation_deg
    ObservatoryL2Halo.position_ecliptic.custom
    Observatory.is_observable_at Observatory.observability
    Observatory.is_observable_at.host_sky Observatory.observability.host_interval
    """.split(),
)
def test_forward_precision(entry, cases):
    """Check closure and float32 roundoff on moderate, nondegenerate inputs.

    Allow 16 epsilons at the leaf scale for elementary dot products, norms,
    trig and unit conversions. Double that for chained products or Kepler
    residuals, whose inverse slope is at most 1/(1-e) on the test ellipses.
    Allow 64 for propagating through two such stages or inverting angles.
    The amplitude formulas also use 64 because rounding a fractional power
    of a stellar mass in kg amplifies exponent error by ln(M) < 70.
    Allow 128 for Cartesian energy subtraction, Lambert velocity recovery,
    and the chained polynomial, angle, and rotation operations in solar and
    halo ephemerides. The grid has the same resolution in both runs, so
    interpolation error is common to both. Boolean results agree exactly.
    """
    fn, args, n = cases[entry]
    assert_precision_closed(fn, *canonical(args, precision="float32"))
    precision_pair(fn, args, bound=float(ulps(n, np.float32(1))))


@pytest.mark.parametrize("route", ["outer", "paired", "shared", "batched"])
@pytest.mark.parametrize("mode", ["eager", "captured", "traced"])
@pytest.mark.parametrize("narrow", [False, True], ids=["mixed", "narrow"])
def test_host_clock_conditioning(route, mode, narrow):
    """A circular orbit's transverse displacement is a*sin(n*dt).

    Sixteen epsilons at the displacement scale cover the Kepler, trig and
    projection operations. Zero absolute slack exposes lost small offsets.
    """
    dtype = np.dtype("float32") if narrow else float_dtype()
    epoch = 10000.0
    dates = epoch + np.array([0.0001, 0.0003])
    offset = dates - epoch
    # A narrow absolute tracer would already have lost the offsets on entry.
    if mode == "traced" and dtype == np.float32:
        epoch, dates = 0.0, offset
    n = dtype.type(0.017)
    orbit = FrameOrbit(
        jnp.float32(1),
        n,
        dtype.type(0),
        dtype.type(0),
        jnp.array([1, 0, 0, 0], dtype=dtype),
        dtype.type(epoch),
    )
    expected = np.sin(np.float64(n) * offset)
    if route in ("shared", "batched"):
        model = System(
            Body(jnp.ones(2, dtype=dtype)),
            Body(jnp.zeros((2, 1), dtype=dtype)),
            jax.tree.map(lambda x: jnp.broadcast_to(x, (2, 1, *x.shape)), orbit),
            SkyFrame(jnp.ones(2, dtype=dtype)),
        )

        def query(system, times):
            return system.solve(times).position_AU[:, 1, :, 1]

        if route == "batched":
            dates = np.broadcast_to(dates, (2, 2))
        expected = np.broadcast_to(expected, (2, 2))
    else:
        model = orbit
        at = FrameOrbit.at if route == "outer" else FrameOrbit.at_each

        def query(orbit, times):
            return at(orbit, times)[0][..., 1]

    if mode == "captured":
        result = jax.jit(lambda model: query(model, dates))(model)
        if narrow and float_dtype() == np.float64:
            assert_precision_closed(lambda model: query(model, dates), model)
    elif mode == "traced":
        result = jax.jit(query)(model, jnp.asarray(dates, dtype=dtype))
    else:
        result = query(model, dates)
    assert result.dtype == dtype
    np.testing.assert_allclose(
        result, expected, rtol=float(ulps(16, dtype.type(1))), atol=0
    )


@pytest.mark.parametrize("mode", ["eager", "jit"])
@pytest.mark.parametrize("form", ["scalar", "sequence", "typed"])
def test_cartesian_snapshot_epoch(mode, form):
    """The stored snapshot date equals the input rounded once at its final dtype."""
    dtype = float_dtype()
    expected = np.array(
        [10000.0001, 10000.0003] if form != "scalar" else [10000.0001] * 2
    )
    epochs = (
        float(expected[0])
        if form == "scalar"
        else [np.float64(x) for x in expected]
        if form == "typed"
        else expected.tolist()
    )
    state_dtype = np.dtype("float32") if form == "typed" else dtype
    r = jnp.broadcast_to(
        jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=state_dtype), (2, 2, 3)
    )
    v = jnp.broadcast_to(
        jnp.array([[0.0, 0.0, 0.0], [0.0, 0.017, 0.001]], dtype=r.dtype), r.shape
    )
    stars = Body(jnp.ones(2, dtype=jnp.float32))
    planets = Body(jnp.full((2, 1), 1e-4, dtype=jnp.float32))
    frame = SkyFrame(jnp.ones(2, dtype=jnp.float32))

    def build(star, planets, frame, r, v):
        return System.from_cartesian(
            star, planets, frame, position_AU=r, velocity_AU_per_d=v, epoch_d=epochs
        ).orbits.epoch_d

    result = (jax.jit(build) if mode == "jit" else build)(stars, planets, frame, r, v)
    assert result.dtype == dtype
    np.testing.assert_array_equal(result[:, 0], expected.astype(dtype))


@pytest.mark.parametrize("long_way", [False, True], ids=["short", "long"])
@pytest.mark.parametrize("mode", ["eager", "jit"])
@pytest.mark.parametrize("mixed", [False, True], ids=["narrow", "mixed"])
def test_lambert_parabolic_boundary(long_way, mode, mixed):
    """Compare N=0 against the parabolic limit for perpendicular unit radii.

    The limit is sqrt(2)/3 * (s**1.5 +/- (s-c)**1.5) at unit mu.
    Sixteen epsilons cover chord, semiperimeter and fractional powers.
    Flights 50 ppm to either side independently test the validity decision.
    """
    dtype = float_dtype() if mixed else np.dtype("float32")
    r1 = jnp.array([1, 0, 0], dtype=jnp.float32)
    r2 = jnp.array([0, 1, 0], dtype=jnp.float32)
    mu = jnp.asarray(1, dtype=dtype)
    chord = np.sqrt(2)
    s = 1 + chord / 2
    sign = 1 if long_way else -1
    limit = chord / 3 * (s**1.5 + sign * (s - chord) ** 1.5)

    def solve(r1, r2, mu):
        minimum = eq.lambert_tof_min(r1, r2, mu, long_way=long_way)
        below = eq.lambert_solve(
            r1, r2, dtype.type(limit * 0.99995), mu, long_way=long_way
        )
        above = eq.lambert_solve(
            r1, r2, dtype.type(limit * 1.00005), mu, long_way=long_way
        )
        return minimum, below[-1], above[-1]

    minimum, below, above = (jax.jit(solve) if mode == "jit" else solve)(r1, r2, mu)
    assert minimum.dtype == dtype
    assert (bool(below), bool(above)) == (False, True)
    np.testing.assert_allclose(
        minimum, limit, rtol=float(ulps(16, dtype.type(1))), atol=0
    )


@pytest.mark.parametrize("mode", ["eager", "jit"])
def test_signed_host_axis_angle(mode):
    """The quaternion retains the signed sine of the original host half-angle.

    Four float32 epsilons at the transverse scale cover rounding of the
    trigonometric result and the axis product.
    """
    angle = 2 * np.pi - 1e-9

    def quat(axis):
        return from_axis_angle(axis, angle)

    axis = jnp.array([0, 0, 1], dtype=jnp.float32)
    result = (jax.jit(quat) if mode == "jit" else quat)(axis)
    assert result.dtype == np.float32
    np.testing.assert_allclose(
        result[-1], np.sin(angle / 2), rtol=float(ulps(4, np.float32(1))), atol=0
    )


@pytest.mark.parametrize("mode", ["eager", "jit"])
def test_classical_wrap_follows_angle_precision(mode):
    """A negative node wraps by a full turn at the quaternion's precision.

    Sixteen epsilons cover basis rotation, angle recovery and one remainder.
    """
    dtype = float_dtype()
    orbit = FrameOrbit.from_classical(
        dtype.type(1), 0.2, 0.7, -0.4, -1.1, 0.5, mu_AU3_per_d2=0.0003
    )
    orbit = FrameOrbit(
        orbit.a_AU,
        orbit.n_rad_per_d,
        jnp.float32(orbit.ex),
        jnp.float32(orbit.ey),
        orbit.quat,
        orbit.epoch_d,
    )
    convert = FrameOrbit.to_classical
    angles = (jax.jit(convert) if mode == "jit" else convert)(orbit)
    node = angles["W_rad"]
    assert node.dtype == dtype
    assert angles["w_rad"].dtype == dtype
    assert angles["M0_rad"].dtype == np.float32
    np.testing.assert_allclose(
        node, 2 * np.pi - 0.4, rtol=float(ulps(16, dtype.type(1))), atol=0
    )


@pytest.mark.parametrize("mode", ["eager", "jit"])
def test_orbit_law_samples_period_with_sequence_mu(mode):
    """Kepler's third law recovers each list entry to 16 rounding operations."""
    dtype = float_dtype()
    law = OrbitLaw(
        IsotropicInclination(),
        period_d=priorities.Uniform(jnp.asarray(100, dtype), jnp.asarray(200, dtype)),
    )
    expected = [0.0003, 0.0004]

    def sample(law, key):
        return law.sample(key, mu_AU3_per_d2=expected)

    orbit = (jax.jit(sample) if mode == "jit" else sample)(law, jax.random.key(0))
    assert orbit.a_AU.shape == (2,)
    np.testing.assert_allclose(
        orbit.mu_AU3_per_d2, expected, rtol=float(ulps(16, dtype.type(1))), atol=0
    )


def test_orbit_law_sample_draws_at_owner_precision():
    """Match inverse-CDF draws and orbital invariants using the same split keys.

    For this seed both eccentricities are below one. Sixteen epsilons cover
    the inverse-CDF, eccentricity-vector norm and mean-motion products.
    """
    key = jax.random.key(3)
    mu = jnp.float32(0.0003)
    direction = IsotropicInclination()
    law = OrbitLaw(direction, a_AU=direction)

    def sample(law, key, mu):
        return law.sample(key, mu_AU3_per_d2=mu)

    if float_dtype() == np.float64:
        assert_precision_closed(sample, law, key, mu)
    orbit = jax.jit(sample)(law, key, mu)
    ke, ka, _, _ = jax.random.split(key, 4)
    expected = 2 * np.arcsin(np.sqrt(jax.random.uniform(ke, dtype=mu.dtype)))
    expected_a = 2 * np.arcsin(np.sqrt(jax.random.uniform(ka, dtype=mu.dtype)))
    bound = float(ulps(16, np.float32(1)))
    assert all(x.dtype == np.float32 for x in jax.tree.leaves(orbit))
    np.testing.assert_allclose(orbit.e, expected, rtol=bound, atol=0)
    np.testing.assert_allclose(orbit.a_AU, expected_a, rtol=bound, atol=0)
    np.testing.assert_allclose(orbit.mu_AU3_per_d2, mu, rtol=bound, atol=0)
    assert direction.sample(key).dtype == float_dtype()


@pytest.mark.parametrize("long_way", [False, True], ids=["short", "long"])
@pytest.mark.parametrize("mode", ["eager", "jit"])
def test_lambert_near_parabolic_velocity(long_way, mode):
    """Recover the terminal velocity of a known high-eccentricity ellipse.

    The unit-radius endpoints are symmetric about periapsis. Compute the
    flight time from their eccentric anomalies and Kepler's area law in
    extended host precision. Allow 64 epsilons at the velocity scale for the
    Lambert root, the Lagrange coefficients and the endpoint geometry.
    """
    dtype = float_dtype()
    direction = -1 if long_way else 1
    with mp.workdps(40):
        e = mp.mpf("0.9999")
        f = mp.pi * (3 if long_way else 1) / 4
        p = 1 + e * mp.cos(f)
        a = p / (1 - e**2)
        anomaly = 2 * mp.atan(mp.sqrt((1 - e) / (1 + e)) * mp.tan(f / 2))
        duration = float(2 * (anomaly - e * mp.sin(anomaly)) * a ** mp.mpf("1.5"))
        radial = float(e * mp.sin(f) / mp.sqrt(p))
        transverse = direction * float(mp.sqrt(p))
    r1 = jnp.array([1, 0, 0], dtype=jnp.float32)
    r2 = jnp.array([0, 1, 0], dtype=jnp.float32)
    # Only TOF sets the wider working precision in the float64 lane.
    solve = partial(eq.lambert_solve, long_way=long_way)
    if mode == "jit":
        solve = jax.jit(solve)
    v1, v2, valid = solve(r1, r2, jnp.asarray(duration, dtype=dtype), jnp.float32(1))
    assert v1.dtype == v2.dtype == dtype
    assert bool(valid)
    np.testing.assert_allclose(
        (v1, v2),
        ((-radial, transverse, 0), (-transverse, radial, 0)),
        rtol=float(ulps(64, dtype.type(1))),
        atol=0,
    )


@pytest.mark.parametrize("rotate_basis", [_rot1, _rot3])
def test_obliquity_rotation_helpers_are_precision_closed(rotate_basis):
    """Check float32 quarter-turn geometry and the private helper trace.

    The matrix entries have at most one trigonometric operation and one
    rounding of pi/2. Four epsilons cover zeros as well as the unit entries.
    """
    angle = jnp.float32(np.pi / 2)
    if float_dtype() == np.float64:
        assert_precision_closed(rotate_basis, angle)
    expected = (
        [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
        if rotate_basis is _rot1
        else [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    )
    np.testing.assert_allclose(
        rotate_basis(angle), expected, rtol=0, atol=float(ulps(4, np.float32(1)))
    )


class _PrioritiesPromotion(AssertionError):
    """A float64 intermediate at the known priorities guard."""


_PRIORITIES_WIDE_SITE = (
    "/priorities/_distribution.py:126:17 (bounded.<locals>.<genexpr>)"
)


def _widening_caller(jaxpr):
    """Find the first attributable frame above an anonymous nested conversion."""
    for eqn in jaxpr.eqns:
        for child in _subjaxprs(eqn.params):
            if _first_wide(child) is not None:
                return _widening_caller(child) or summarize(eqn.source_info)
        if _wide_dtype((*eqn.invars, *eqn.outvars)) is not None:
            return summarize(eqn.source_info)
    return ""


def _assert_orbit_law_closed(fn, *args):
    try:
        assert_precision_closed(fn, *args)
    except AssertionError as error:
        trace, _, _ = eqx.filter_make_jaxpr(fn)(*args)
        hit = _first_wide(trace.jaxpr)
        if (
            hit is not None
            and hit[1] == np.float64
            and _widening_caller(trace.jaxpr).endswith(_PRIORITIES_WIDE_SITE)
        ):
            raise _PrioritiesPromotion(
                f"{_PRIORITIES_WIDE_SITE} widens to float64"
            ) from error
        raise


@pytest.mark.xfail(
    float_dtype() == np.float64,
    reason="priorities/_distribution.py:126 bounded widens to float64",
    raises=_PrioritiesPromotion,
    strict=True,
)
@pytest.mark.parametrize("method", ["sample", "from_unit"])
def test_orbit_law_with_typed_uniforms(method):
    """Check real typed scalar laws against the affine uniform quantile.

    Allow 16 epsilons for scalar quantiles and the eccentricity-vector norm.
    Sampled values use the matching width's uniforms because JAX generates
    different random streams for different widths. The unit map additionally
    compares every leaf across both widths.
    """
    dtype = float_dtype()
    e_min, e_max, a_min, a_max, mu = (
        jnp.asarray(x, dtype=dtype) for x in (0.1, 0.5, 1, 2, 0.0003)
    )
    law = OrbitLaw(
        priorities.Uniform(e_min, e_max), a_AU=priorities.Uniform(a_min, a_max)
    )
    bound = float(ulps(16, dtype.type(1)))
    if method == "sample":
        source = jax.random.key(0)

        def draw(law, key, mu):
            return law.sample(key, mu_AU3_per_d2=mu)

        keys = jax.random.split(source, 4)[:2]
        units = jax.vmap(
            lambda key: jax.random.uniform(
                key, dtype=dtype, minval=np.finfo(dtype).tiny
            )
        )(keys)
    else:
        source = jnp.full(6, 0.3, dtype=dtype)

        def draw(law, cube, mu):
            return law.from_unit(cube, mu_AU3_per_d2=mu)

        units = source[:2]
    orbit = draw(law, source, mu)
    expected = (
        np.array([e_min, a_min]) + np.array([e_max - e_min, a_max - a_min]) * units
    )
    assert all(leaf.dtype == dtype for leaf in jax.tree.leaves(orbit))
    np.testing.assert_allclose((orbit.e, orbit.a_AU), expected, rtol=bound, atol=0)
    if dtype == np.float64:

        def narrow(x):
            return x.astype(jnp.float32) if jnp.issubdtype(x.dtype, jnp.inexact) else x

        args = (law, source, mu)
        if method == "from_unit":
            precision_pair(draw, args, bound=float(ulps(16, np.float32(1))))
        _assert_orbit_law_closed(draw, *jax.tree.map(narrow, args))
