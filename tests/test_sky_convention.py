"""Sky coordinates against Green (1985) and Kervella et al. (2016)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import G, Msun2kg

from orbix.equations.orbit import AB_matrices, state_vector_to_keplerian
from orbix.equations.propagation import single_r_v, system_r, system_r_v
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
    ra, dec = orbit.position_arcsec(t_jd=jnp.array([0.0]), Ms_kg=Msun2kg, dist_pc=10.0)
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
        t_jd=jnp.asarray((epoch - 2000.0) * 365.25), **stellar
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


@pytest.mark.parametrize("inc_deg", [30.0, 150.0])
@pytest.mark.parametrize("node_deg", [10.0, 200.0])
def test_public_state_round_trip(inc_deg, node_deg):
    """Recover visual-binary elements without hiding a shared frame error."""
    expected = np.array(
        [2.3, 0.31, np.deg2rad(inc_deg), np.deg2rad(node_deg), 0.7, 0.4]
    )
    orbit = KeplerianOrbit(
        a_AU=expected[0],
        e=expected[1],
        i_rad=expected[2],
        W_rad=expected[3],
        w_rad=expected[4],
        M0_rad=expected[5],
        t0_d=0.0,
    )
    r, v = _propagated_state(orbit, 0.0, Msun2kg)
    actual = jax.jit(state_vector_to_keplerian)(r, v, G * Msun2kg)
    # Markley's float64 solve is near machine precision; 1e-11 allows
    # roundoff in the inverse trig/energy operations, far below grid error.
    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=0)
    jac = jax.jit(jax.jacfwd(state_vector_to_keplerian, argnums=(0, 1)))(
        r, v, G * Msun2kg
    )
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(jac))
    dra, ddec = orbit.position_arcsec(
        solve_trig, jnp.array([0.0]), Ms_kg=Msun2kg, dist_pc=1.0
    )
    # A round trip alone also passes when both ends use the old wrong frame.
    np.testing.assert_allclose(r[:2], [dra[0, 0], ddec[0, 0]], atol=1e-11, rtol=0)


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
