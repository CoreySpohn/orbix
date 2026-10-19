"""Sky coordinates against Green (1985) and Kervella et al. (2016)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import G, Msun2kg

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
