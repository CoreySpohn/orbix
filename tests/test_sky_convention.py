"""Sky coordinates against Green (1985) and Kervella et al. (2016)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import G, Msun2kg, two_pi
from hypothesis import example, given
from hypothesis import strategies as st

from orbix.equations.orbit import AB_matrices, mean_motion, state_vector_to_keplerian
from orbix.equations.propagation import single_r, single_r_v, system_r, system_r_v
from orbix.kepler.markley import solve_trig
from orbix.orbit import KeplerianOrbit

jax.config.update("jax_enable_x64", True)

# Kervella et al. (2016), A&A 594, A107, arXiv:1610.06079, Table A.1.
# All 25 rows: epoch (year), PA (deg), sigma_PA (deg), rho and sigma_rho (arcsec).
ALPHA_CEN_ASTROMETRY = np.array(
    [
        [2016.1893, 305.19, 0.30, 4.013, 0.02],
        [2015.3326, 293.30, 0.60, 4.020, 0.04],
        [2014.9574, 288.28, 0.70, 4.081, 0.05],
        [2014.5426, 282.91, 0.15, 4.208, 0.01],
        [2014.5126, 282.84, 0.30, 4.184, 0.02],
        [2014.2410, 279.20, 0.30, 4.330, 0.05],
        [2012.7100, 262.70, 0.40, 5.050, 0.05],
        [2009.7140, 244.20, 0.40, 7.020, 0.05],
        [2009.1517, 241.07, 0.60, 7.400, 0.07],
        [2008.7190, 239.80, 0.40, 7.780, 0.05],
        [2007.3670, 235.70, 0.60, 8.780, 0.10],
        [2004.2739, 228.26, 0.20, 11.122, 0.02],
        [2002.6880, 224.30, 1.00, 12.500, 0.40],
        [2000.7400, 223.80, 1.00, 13.300, 0.40],
        [2000.1950, 222.63, 0.80, 13.933, 0.20],
        [1991.2500, 215.29, 0.15, 19.073, 0.05],
        [1989.5370, 214.68, 0.16, 19.740, 0.05],
        [1988.2880, 213.75, 0.15, 20.180, 0.05],
        [1987.2900, 213.21, 0.15, 20.540, 0.05],
        [1986.5090, 212.77, 0.15, 20.784, 0.05],
        [1985.6030, 212.08, 0.15, 21.087, 0.05],
        [1984.4250, 211.54, 0.14, 21.385, 0.05],
        [1982.4900, 210.60, 0.40, 21.910, 0.13],
        [1980.4000, 210.50, 1.00, 21.710, 0.50],
        [1980.2870, 210.04, 0.06, 21.614, 0.05],
    ]
)

# Table 1, K16 column (relative orbit of B about A).
K16_A_ARCSEC = 17.592
K16_PARALLAX_ARCSEC = 0.74717
K16_PERIOD_YR = 79.929
K16_TP_YR = 1955.604
K16_E = 0.5208
K16_I_RAD, K16_W_RAD, K16_OMEGA_RAD = np.deg2rad([79.320, 232.006, 205.064])


@pytest.fixture
def alpha_cen_orbit():
    """Use K16 Table 1's exact period rather than the rounded published mass."""
    dist_pc = 1.0 / K16_PARALLAX_ARCSEC
    a_AU = K16_A_ARCSEC * dist_pc
    period_d = K16_PERIOD_YR * 365.25
    mass_kg = 4.0 * np.pi**2 * a_AU**3 / (G * period_d**2)
    # A common day origin is sufficient; Julian years preserve the tabulated P.
    orbit = KeplerianOrbit.from_period(
        T_d=period_d,
        e=K16_E,
        cos_i=np.cos(K16_I_RAD),
        W_rad=K16_OMEGA_RAD,
        cos_w=np.cos(K16_W_RAD),
        sin_w=np.sin(K16_W_RAD),
        tp_d=(K16_TP_YR - 2000.0) * 365.25,
        Ms_kg=mass_kg,
    )
    return orbit, dict(Ms_kg=mass_kg, dist_pc=dist_pc)


def _numpy_k16_position(epoch_yr):
    """Independent Newton/Thiele-Innes solution; Green (1985), ch. 17."""
    mean_anomaly = 2.0 * np.pi * (epoch_yr - K16_TP_YR) / K16_PERIOD_YR
    eccentric_anomaly = mean_anomaly.copy()
    for _ in range(20):
        step = (
            eccentric_anomaly - K16_E * np.sin(eccentric_anomaly) - mean_anomaly
        ) / (1.0 - K16_E * np.cos(eccentric_anomaly))
        eccentric_anomaly -= step
        if np.max(np.abs(step)) < 1e-14:
            break
    X = np.cos(eccentric_anomaly) - K16_E
    Y = np.sqrt(1.0 - K16_E**2) * np.sin(eccentric_anomaly)
    cw, sw = np.cos(K16_W_RAD), np.sin(K16_W_RAD)
    cO, sO = np.cos(K16_OMEGA_RAD), np.sin(K16_OMEGA_RAD)
    ci = np.cos(K16_I_RAD)
    A = K16_A_ARCSEC * (cO * cw - sO * sw * ci)
    B = K16_A_ARCSEC * (sO * cw + cO * sw * ci)
    F = K16_A_ARCSEC * (-cO * sw - sO * cw * ci)
    G_ti = K16_A_ARCSEC * (-sO * sw + cO * cw * ci)
    north = A * X + F * Y
    east = B * X + G_ti * Y
    return east, north


@pytest.mark.parametrize("node_deg, expected", [(0.0, [0.0, 0.1]), (90.0, [0.1, 0.0])])
def test_ascending_node_cardinal_directions(node_deg, expected):
    """Omega runs north through east: Green, Spherical Astronomy (1985), ch. 17."""
    orbit = KeplerianOrbit(
        a_AU=1.0,
        e=0.0,
        W_rad=np.deg2rad(node_deg),
        i_rad=np.pi / 3,
        w_rad=0.0,
        M0_rad=0.0,
        t0_d=0.0,
    )
    # For e=w=M0=0, t=t0 is the ascending node.
    ra, dec = orbit.position_arcsec(
        time_jd=jnp.array([0.0]), Ms_kg=Msun2kg, dist_pc=10.0
    )
    assert ra.shape == dec.shape == (1, 1)
    np.testing.assert_allclose([ra[0, 0], dec[0, 0]], expected, atol=1e-12, rtol=0)


def test_ascending_node_moves_away():
    """Ascending means positive dz/dt (away); Green (1985), ch. 17."""
    orbit = KeplerianOrbit(
        a_AU=1.0,
        e=0.0,
        W_rad=0.0,
        i_rad=np.pi / 3,
        w_rad=0.0,
        M0_rad=0.0,
        t0_d=0.0,
    )
    r, _, _ = orbit.propagate(solve_trig, jnp.array([-0.01, 0.0, 0.01]), Ms_kg=Msun2kg)
    z = np.asarray(r)[0, 2]
    assert z[0] < 0.0 < z[2]
    assert z[1] == pytest.approx(0.0, abs=1e-14)
    dz_dt = (z[2] - z[0]) / 0.02
    assert dz_dt > 0.0
    np.testing.assert_allclose(
        dz_dt, np.sqrt(G * Msun2kg) * np.sin(np.pi / 3), rtol=1e-8
    )


def test_alpha_cen_published_astrometry(alpha_cen_orbit):
    """Match Kervella et al. (2016), A&A 594, A107, Tables 1 and A.1.

    Require PA RMS < 1 deg and chi2 per epoch < 4, summing the two
    squared residuals normalized by the published PA/rho uncertainties.
    This aggregate allows individual outliers and rounded orbital elements;
    it is not a claim to reproduce the paper's full-fit reduced chi2.
    """
    orbit, stellar = alpha_cen_orbit
    epoch, pa_deg, sigma_pa, rho_arcsec, sigma_rho = ALPHA_CEN_ASTROMETRY.T
    ra, dec = orbit.position_arcsec(
        time_jd=jnp.asarray((epoch - 2000.0) * 365.25), **stellar
    )
    delta_pa = (np.rad2deg(np.arctan2(ra[0], dec[0])) - pa_deg + 180.0) % 360.0 - 180.0
    delta_rho = np.hypot(ra[0], dec[0]) - rho_arcsec
    pa_rms = np.sqrt(np.mean(delta_pa**2))
    chi2_per_epoch = np.mean((delta_pa / sigma_pa) ** 2 + (delta_rho / sigma_rho) ** 2)
    assert pa_rms < 1.0, f"PA RMS = {pa_rms:.6f} deg"
    assert chi2_per_epoch < 4.0, f"chi2 per epoch = {chi2_per_epoch:.6f}"


def test_alpha_cen_independent_thiele_innes(alpha_cen_orbit):
    """Pin east/north to 1e-9 arcsec: Green (1985), ch. 17; K16 Table 1."""
    orbit, stellar = alpha_cen_orbit
    # Include a whole revolution as well as every observed epoch.
    epoch = np.concatenate(
        [
            ALPHA_CEN_ASTROMETRY[:, 0],
            np.linspace(K16_TP_YR, K16_TP_YR + K16_PERIOD_YR, 101),
        ]
    )
    expected_ra, expected_dec = _numpy_k16_position(epoch)
    ra, dec = orbit.position_arcsec(
        solve_trig, jnp.asarray((epoch - 2000.0) * 365.25), **stellar
    )
    assert ra.shape == dec.shape == (1, epoch.size)
    np.testing.assert_allclose(ra[0], expected_ra, atol=1e-9, rtol=0)
    np.testing.assert_allclose(dec[0], expected_dec, atol=1e-9, rtol=0)
    r, _, _ = orbit.propagate(
        solve_trig, jnp.asarray((epoch - 2000.0) * 365.25), Ms_kg=stellar["Ms_kg"]
    )
    # Public raw rows now follow the right-handed east, north, away frame,
    # rather than the classical north-first Thiele-Innes letter ordering.
    np.testing.assert_allclose(
        r[0, 0] / stellar["dist_pc"], expected_ra, atol=1e-9, rtol=0
    )
    np.testing.assert_allclose(
        r[0, 1] / stellar["dist_pc"], expected_dec, atol=1e-9, rtol=0
    )


def _propagated_state(orbit, time_d, mass_kg):
    """Differentiate the public position with respect to time in days."""
    return jax.jvp(
        lambda t: orbit.propagate(solve_trig, t, Ms_kg=mass_kg)[0][0, :, 0],
        (jnp.asarray(time_d),),
        (jnp.asarray(1.0),),
    )


def test_increasing_pa_has_angular_momentum_toward_observer():
    """Perryman (2018), Fig. 2.2: east, north, away is right-handed."""
    orbit = KeplerianOrbit(
        a_AU=1.0,
        e=0.2,
        i_rad=1e-3,
        W_rad=0.4,
        w_rad=0.3,
        M0_rad=0.2,
        t0_d=0.0,
    )
    dra, ddec = orbit.position_arcsec(
        solve_trig, jnp.array([0.0, 1.0]), Ms_kg=Msun2kg, dist_pc=10.0
    )
    pa = np.unwrap(np.arctan2(dra[0], ddec[0]))
    assert pa[1] > pa[0]
    r, v = _propagated_state(orbit, 0.0, Msun2kg)
    assert np.cross(r, v)[2] < 0.0, "Increasing PA must give h_z < 0 (toward observer)"


def _bounded(low, high):
    return st.floats(low, high, allow_nan=False, allow_infinity=False)


def _angles_away_from_axes(margin):
    """Sample all four half-turns without rejection near multiples of pi."""
    return st.one_of(
        *(_bounded(k * np.pi + margin, (k + 1) * np.pi - margin) for k in range(-2, 2))
    )


@jax.jit
def _round_trip_state(elements, mass):
    a, e, inc, node, peri, mean = elements
    orbit = KeplerianOrbit(
        a_AU=a, e=e, i_rad=inc, W_rad=node, w_rad=peri, M0_rad=mean, t0_d=0.0
    )
    r, v = _propagated_state(orbit, 0.0, mass)
    actual = jnp.array(state_vector_to_keplerian(r, v, G * mass))
    jac = jax.jacfwd(state_vector_to_keplerian, argnums=(0, 1))(r, v, G * mass)
    dra, ddec = orbit.position_arcsec(
        solve_trig, jnp.array([0.0]), Ms_kg=mass, dist_pc=1.0
    )
    sine, cosine = solve_trig(jnp.atleast_1d(mean), e)
    A, B = AB_matrices(a, e, inc, node, peri)
    n = mean_motion(a, G * mass)
    sr, sv = single_r_v(A, B, e, sine, cosine, n)
    args = (
        A[:, None, None],
        B[:, None, None],
        e.reshape(1, 1),
        sine.reshape(1, 1),
        cosine.reshape(1, 1),
    )
    br, bv = system_r_v(*args, n.reshape(1, 1))
    return (
        actual,
        jac,
        r,
        v,
        jnp.array([dra[0, 0], ddec[0, 0]]),
        sr[:, 0],
        sv[:, 0],
        br[:, 0, 0],
        bv[:, 0, 0],
        system_r(*args)[:, 0, 0],
        single_r(A, B, e, sine, cosine)[:, 0],
    )


# Node wrapping is nonsingular at -pi, unlike the clipped anomaly axes.
@example(a=2.3, e=0.31, inc=0.7, node=-np.pi, peri=0.7, mean=0.4, mass=1.0)
@example(
    a=2.3,
    e=0.31,
    inc=0.7,
    node=np.nextafter(-np.pi, -np.inf),
    peri=0.7,
    mean=0.4,
    mass=1.0,
)
@example(
    a=2.3,
    e=0.31,
    inc=0.7,
    node=np.nextafter(-np.pi, np.inf),
    peri=0.7,
    mean=0.4,
    mass=1.0,
)
@given(
    a=_bounded(0.1, 30.0),
    e=_bounded(0.01, 0.99),
    inc=_bounded(0.05, np.pi - 0.05),
    node=_bounded(-two_pi, two_pi),
    peri=_angles_away_from_axes(0.002),
    mean=_angles_away_from_axes(0.05),
    mass=_bounded(0.1, 3.0),
)
def test_public_state_round_trip(a, e, inc, node, peri, mean, mass):
    """Fails if elements/state inversion or any propagation path corrupts the sky frame.

    Bounded strategies exclude circular e < 0.01 and inclinations within
    0.05 rad of 0 or pi, where elements are singular. Periapsis and mean
    angles avoid arccos-clipped axes by construction. For e <= 0.99, a
    0.05 rad mean-anomaly margin keeps abs(sin(nu)) > 0.001 even at apoapsis.
    Explicit regressions cover the axes. Compare angles modulo 2*pi.
    """
    sine, cosine = map(float, solve_trig(jnp.asarray(mean), jnp.asarray(e)))
    sin_nu = np.sqrt(1 - e**2) * sine / (1 - e * cosine)
    assert abs(np.sin(peri)) > 0.001 and abs(sin_nu) > 0.001
    expected = np.array([a, e, inc, node, peri, mean])
    actual, jac, r, v, sky, sr, sv, br, bv, only_r, single_only_r = _round_trip_state(
        jnp.asarray(expected), jnp.asarray(mass * Msun2kg)
    )
    np.testing.assert_allclose(actual[:3], expected[:3], atol=1e-11, rtol=1e-11)
    diff = np.asarray(actual[3:]) - expected[3:]
    np.testing.assert_allclose(
        np.arctan2(np.sin(diff), np.cos(diff)), 0, atol=1e-10, rtol=0
    )
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(jac))
    np.testing.assert_allclose(r[:2], sky, atol=1e-11, rtol=0)
    for position in (sr, br, only_r, single_only_r):
        np.testing.assert_allclose(position, r, atol=1e-11, rtol=1e-12)
    for velocity in (sv, bv):
        np.testing.assert_allclose(velocity, v, atol=1e-14, rtol=1e-12)
    # Independent frame anchor: ascending node PA is north through east,
    # crossing toward positive away. A shared axis swap cannot pass this.
    A, B = AB_matrices(a, e, inc, node, 0.0)
    np.testing.assert_allclose(
        np.asarray(A) / a, [np.sin(node), np.cos(node), 0], atol=1e-14, rtol=0
    )
    assert B[2] > 0
    np.testing.assert_allclose(
        np.cross(np.asarray(A), np.asarray(B))[2],
        -(a**2) * np.sqrt(1 - e**2) * np.cos(inc),
        atol=1e-11,
        rtol=1e-12,
    )


@jax.jit
def _anomaly_round_trip(mean, e):
    """Recover anomalies through the inverse's embedded true-to-mean conversion."""
    sine, cosine = solve_trig(mean, e)
    nu = jnp.arctan2(jnp.sqrt(1 - e**2) * sine, cosine - e)
    # Independent state from polar radius and radial/transverse velocities.
    p = 1 - e**2
    radius = p / (1 + e * jnp.cos(nu))
    radial = e * jnp.sin(nu) / jnp.sqrt(p)
    transverse = (1 + e * jnp.cos(nu)) / jnp.sqrt(p)
    u = jnp.array([jnp.cos(nu), 0.0, jnp.sin(nu)])
    tangent = jnp.array([-jnp.sin(nu), 0.0, jnp.cos(nu)])
    recovered = state_vector_to_keplerian(
        radius * u, radial * u + transverse * tangent, 1.0
    )[-1]
    recovered_sine, recovered_cosine = solve_trig(recovered, e)
    recovered_nu = jnp.arctan2(
        jnp.sqrt(1 - e**2) * recovered_sine, recovered_cosine - e
    )
    return sine, cosine, nu, recovered, recovered_sine, recovered_cosine, recovered_nu


@example(e=0.0, mean=1.0)
@given(e=_bounded(0.0, 0.99), mean=_angles_away_from_axes(0.05))
def test_anomaly_round_trip(e, mean):
    """Fails if mean/eccentric/true anomaly conversions lose phase or a quadrant.

    Orbix embeds true-to-mean in state_vector_to_keplerian rather than a
    standalone conversion API. Construct means away from true-anomaly axes
    covered by explicit regressions below; circular orbits use its
    argument-of-latitude branch.
    """
    result = _anomaly_round_trip(jnp.asarray(mean), jnp.asarray(e))
    sine, cosine, nu, recovered, rs, rc, rnu = map(float, result)
    assert abs(np.sin(nu)) > 0.001
    # Recovering the periapsis direction divides state roundoff by e.
    # Below the inverse's 1e-9 circular threshold, phase conventions also
    # differ by O(e); this bound covers both effects without dropping e=0.
    tolerance = 5e-11 + 16 * np.finfo(float).eps / max(e, 1e-9)
    difference = recovered - mean
    assert abs(np.arctan2(np.sin(difference), np.cos(difference))) < tolerance
    np.testing.assert_allclose([rs, rc], [sine, cosine], atol=tolerance, rtol=0)
    difference = rnu - nu
    assert abs(np.arctan2(np.sin(difference), np.cos(difference))) < tolerance


@pytest.mark.parametrize("peri, mean, index", [(0.0, 0.4, 4), (0.7, 0.0, 5)])
def test_state_round_trip_on_anomaly_axes(peri, mean, index):
    """Fails if arccos safety clipping biases valid periapsis or mean anomaly."""
    expected = jnp.array([2.3, 0.31, 0.7, 0.4, peri, mean])
    actual = _round_trip_state(expected, jnp.asarray(Msun2kg))[0]
    np.testing.assert_allclose(actual[index], expected[index], atol=1e-11, rtol=0)


@pytest.mark.parametrize(
    "r, v",
    [([0.0, 0.5, 0.0], [0.0, 0.0, np.sqrt(3)]), ([0.3, 0.5, 0.2], [0.2, -0.4, 0.9])],
)
def test_state_inversion_reverse_gradients_are_finite(r, v):
    """Axis-safe anomaly extraction must not poison reverse-mode derivatives."""
    gradient = jax.jit(
        jax.grad(lambda r, v: sum(state_vector_to_keplerian(r, v, 1.0)), argnums=(0, 1))
    )(jnp.asarray(r), jnp.asarray(v))
    assert np.isfinite(np.asarray(gradient)).all()


def test_periapsis_state_has_zero_anomalies():
    """An exact node/periapsis state must retain both zero angles."""
    actual = state_vector_to_keplerian(
        jnp.array([0.0, 0.5, 0.0]), jnp.array([0.0, 0.0, np.sqrt(3)]), 1.0
    )
    np.testing.assert_allclose(
        actual, [1.0, 0.5, np.pi / 2, 0, 0, 0], atol=1e-14, rtol=0
    )


def _numpy_k16_state(eccentric_anomaly):
    """Independent classical Thiele-Innes state in public (east, north, away)."""
    a = K16_A_ARCSEC / K16_PARALLAX_ARCSEC
    n = 2.0 * np.pi / (K16_PERIOD_YR * 365.25)
    cw, sw = np.cos(K16_W_RAD), np.sin(K16_W_RAD)
    cO, sO = np.cos(K16_OMEGA_RAD), np.sin(K16_OMEGA_RAD)
    ci, si = np.cos(K16_I_RAD), np.sin(K16_I_RAD)
    # Green (1985), ch. 17: A,F north; B,G east; C,H away.
    A = a * (cO * cw - sO * sw * ci)
    B = a * (sO * cw + cO * sw * ci)
    F = a * (-cO * sw - sO * cw * ci)
    G_ti = a * (-sO * sw + cO * cw * ci)
    C, H = a * sw * si, a * cw * si
    sinE, cosE = np.sin(eccentric_anomaly), np.cos(eccentric_anomaly)
    X, Y = cosE - K16_E, np.sqrt(1.0 - K16_E**2) * sinE
    dE_dt = n / (1.0 - K16_E * cosE)
    dX_dt = -sinE * dE_dt
    dY_dt = np.sqrt(1.0 - K16_E**2) * cosE * dE_dt
    x_north, y_east, z = A * X + F * Y, B * X + G_ti * Y, C * X + H * Y
    v_north = A * dX_dt + F * dY_dt
    v_east = B * dX_dt + G_ti * dY_dt
    v_away = C * dX_dt + H * dY_dt
    return np.array([y_east, x_north, z]), np.array([v_east, v_north, v_away])


def test_independent_right_handed_k16_state():
    """Kervella et al. (2016), Table 1: recover the published orientation."""
    E = 1.1
    r, v = _numpy_k16_state(E)
    a = K16_A_ARCSEC / K16_PARALLAX_ARCSEC
    mu = (2.0 * np.pi / (K16_PERIOD_YR * 365.25)) ** 2 * a**3
    actual = jax.jit(state_vector_to_keplerian)(jnp.asarray(r), jnp.asarray(v), mu)
    expected = [a, K16_E, K16_I_RAD, K16_OMEGA_RAD, K16_W_RAD, E - K16_E * np.sin(E)]
    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=0)


@pytest.mark.parametrize("path", ["single_r_v", "system_r_v", "system_r"])
def test_propagation_helpers_use_public_frame(path):
    """All matrix propagation paths preserve east/north for positions and velocities."""
    E = np.array([0.2, 1.1, 3.4])
    expected_r, expected_v = _numpy_k16_state(E)
    a = K16_A_ARCSEC / K16_PARALLAX_ARCSEC
    n = 2.0 * np.pi / (K16_PERIOD_YR * 365.25)
    A, B = AB_matrices(a, K16_E, K16_I_RAD, K16_OMEGA_RAD, K16_W_RAD)
    sinE, cosE = jnp.sin(E), jnp.cos(E)
    if path == "single_r_v":
        r, v = single_r_v(A, B, K16_E, sinE, cosE, n)
    else:
        args = (
            A[:, None, None],
            B[:, None, None],
            jnp.array([[K16_E]]),
            sinE[None, :],
            cosE[None, :],
        )
        if path == "system_r":
            r = system_r(*args)[:, 0]
            v = None
        else:
            r, v = system_r_v(*args, jnp.array([[n]]))
            r, v = r[:, 0], v[:, 0]
    if v is not None:
        np.testing.assert_allclose(v, expected_v, atol=1e-14, rtol=0)
    np.testing.assert_allclose(r, expected_r, atol=1e-11, rtol=0)
