"""Lambert boundary-value solver: round-trips, multi-rev aliases, gradients."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.testing import ulps
from jax.test_util import check_grads

from orbix.equations import lambert_solve, lambert_tof_min

MU = 1.0


def _rot_z(t):
    """Active rotation matrix about the z axis."""
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_x(t):
    """Active rotation matrix about the x axis."""
    c, s = np.cos(t), np.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rv_from_elements(a, e, inc, W, w, M, mu):
    """Independent float64 (r, v, nu) from Newton iteration and perifocal rotation."""
    E = M
    for _ in range(60):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    sinE, cosE = np.sin(E), np.cos(E)
    rmag = a * (1.0 - e * cosE)
    sq1me2 = np.sqrt(1.0 - e**2)
    r_pf = np.array([a * (cosE - e), a * sq1me2 * sinE, 0.0])
    v_pf = (np.sqrt(mu * a) / rmag) * np.array([-sinE, sq1me2 * cosE, 0.0])
    R = _rot_z(W) @ _rot_x(inc) @ _rot_z(w)
    nu = 2.0 * np.arctan2(
        np.sqrt(1.0 + e) * np.sin(E / 2.0), np.sqrt(1.0 - e) * np.cos(E / 2.0)
    )
    return R @ r_pf, R @ v_pf, nu


def _two_epoch_fixture(dM, n_rev=0, a=1.0, e=0.35, inc=0.6, W=1.1, w=0.7, M1=0.4):
    """Truth (r1, v1, r2, v2, tof, long_way) for a transfer of dM (+ n_rev turns)."""
    M2 = M1 + dM + 2.0 * np.pi * n_rev
    r1, v1, nu1 = _rv_from_elements(a, e, inc, W, w, M1, MU)
    r2, v2, nu2 = _rv_from_elements(a, e, inc, W, w, M2, MU)
    n = np.sqrt(MU / a**3)
    tof = (M2 - M1) / n
    sweep = np.mod(nu2 - nu1, 2.0 * np.pi)
    return (
        jnp.asarray(r1),
        v1,
        jnp.asarray(r2),
        v2,
        jnp.asarray(tof),
        bool(sweep > np.pi),
    )


def _implied_tof(r1, v1, r2, v2, mu, n_rev):
    """Convention-free time of flight of the orbit (r1, v1) -> (r2, v2)."""
    a = 1.0 / (2.0 / jnp.linalg.norm(r1) - jnp.dot(v1, v1) / mu)
    n = jnp.sqrt(mu / a**3)

    def ecc_anomaly(r, v):
        rn = jnp.linalg.norm(r)
        evec = ((jnp.dot(v, v) - mu / rn) * r - jnp.dot(r, v) * v) / mu
        e = jnp.linalg.norm(evec)
        cosE = (1.0 - rn / a) / e
        sinE = jnp.dot(r, v) / (e * jnp.sqrt(mu * a))
        return jnp.arctan2(sinE, cosE), e

    E1, e = ecc_anomaly(r1, v1)
    E2, _ = ecc_anomaly(r2, v2)
    dE = jnp.mod(E2 - E1, 2.0 * jnp.pi) + 2.0 * jnp.pi * n_rev
    dM = dE - e * (jnp.sin(E1 + dE) - jnp.sin(E1))
    time = dM / n
    # Relative condition of (2/r - v^2/mu)^(-3/2) in the recovered period.
    condition = 1.5 * a * (2 / jnp.linalg.norm(r1) + jnp.dot(v1, v1) / mu)
    return time, ulps(32, time) * condition


def test_n0_short_way_recovers_velocities():
    """A single-rev short-way transfer recovers both terminal velocities.

    Allow 32 roundings at the speed norm for two 3D rotations, terminal
    geometry (norms and dot), f/g coefficients and component recovery.
    """
    r1, v1, r2, v2, tof, long_way = _two_epoch_fixture(dM=1.6)
    assert not long_way
    v1_l, v2_l, valid = lambert_solve(r1, r2, tof, MU, 0, long_way)
    assert bool(valid)
    np.testing.assert_allclose(
        v1_l, v1, rtol=0, atol=float(ulps(32, v1_l.dtype.type(np.linalg.norm(v1))))
    )
    np.testing.assert_allclose(
        v2_l, v2, rtol=0, atol=float(ulps(32, v2_l.dtype.type(np.linalg.norm(v2))))
    )


def test_n0_long_way_recovers_velocities():
    """A single-rev long-way (sweep > pi) transfer recovers both velocities.

    Allow 32 roundings at the speed norm for two 3D rotations, terminal
    geometry (norms and dot), f/g coefficients and component recovery.
    """
    r1, v1, r2, v2, tof, long_way = _two_epoch_fixture(dM=4.4)
    assert long_way
    v1_l, v2_l, valid = lambert_solve(r1, r2, tof, MU, 0, long_way)
    assert bool(valid)
    np.testing.assert_allclose(
        v1_l, v1, rtol=0, atol=float(ulps(32, v1_l.dtype.type(np.linalg.norm(v1))))
    )
    np.testing.assert_allclose(
        v2_l, v2, rtol=0, atol=float(ulps(32, v2_l.dtype.type(np.linalg.norm(v2))))
    )


@pytest.mark.parametrize("radius", [1.0, 1e10])
def test_multirev_circular_velocities_independent_of_units(radius):
    """A circle with mu=r has unit speed and a period of 2*pi*r.

    The high branch completes one turn plus a quarter turn. Allow 32
    roundings at unit speed for the root, f/g coefficients and component
    recovery, including cancellation of the radial velocity.
    """
    r1 = jnp.asarray([radius, 0.0, 0.0])
    r2 = jnp.asarray([0.0, radius, 0.0])
    tof = 2.5 * np.pi * radius
    for solve in (lambert_solve, jax.jit(lambert_solve)):
        v1, v2, valid = solve(r1, r2, tof, radius, N=1, high_branch=True)
        assert valid
        atol = float(ulps(32, r1.dtype.type(1.0)))
        np.testing.assert_allclose(v1, [0.0, 1.0, 0.0], atol=atol, rtol=0)
        np.testing.assert_allclose(v2, [-1.0, 0.0, 0.0], atol=atol, rtol=0)


def test_multirev_one_branch_matches_truth():
    """For a 3.x-revolution transfer, exactly one N=3 branch is the true orbit.

    As for N=0, allow 32 roundings at the speed norm for the terminal
    geometry and rotated Kepler state. The other branch is physically distinct.
    """
    r1, v1, r2, _v2, tof, long_way = _two_epoch_fixture(dM=1.6, n_rev=3)
    errs = []
    for high in (False, True):
        v1_l, _, valid = lambert_solve(r1, r2, tof, MU, 3, long_way, high)
        assert bool(valid)
        errs.append(np.max(np.abs(np.asarray(v1_l) - v1)))
    assert min(errs) <= ulps(32, v1_l.dtype.type(np.linalg.norm(v1)))
    assert max(errs) > 1e-3


def test_valid_aliases_all_satisfy_the_boundary_value_problem():
    """Every valid (N, branch) alias connects r1 to r2 in the requested time.

    Allow 32 roundings for the two reconstructed anomalies, trig and phase
    difference, scaled by the period condition from the energy subtraction.
    Eight roundings at the angular momentum norm cover two cross products
    and their upstream f/g velocity expressions.
    """
    r1, _, r2, _, tof, long_way = _two_epoch_fixture(dM=1.6, n_rev=3)
    n_valid = 0
    for N in range(9):
        for high in (False, True):
            v1_l, v2_l, valid = lambert_solve(r1, r2, tof, MU, N, long_way, high)
            if not bool(valid):
                continue
            n_valid += 1
            tof_back, bound = _implied_tof(r1, v1_l, r2, v2_l, MU, N)
            assert jnp.allclose(tof_back, tof, rtol=0, atol=bound)
            h1 = jnp.cross(r1, v1_l)
            h2 = jnp.cross(r2, v2_l)
            assert jnp.allclose(h1, h2, rtol=0, atol=ulps(8, jnp.linalg.norm(h1)))
    assert n_valid > 2
    invalid_count = 0
    for high in (False, True):
        _, _, valid = lambert_solve(r1, r2, tof, MU, 30, long_way, high)
        invalid_count += int(not bool(valid))
    assert invalid_count == 2


def test_long_way_flips_orbit_normal():
    """Long-way solutions orbit in the opposite sense to short-way ones."""
    r1, _, r2, _, tof, _ = _two_epoch_fixture(dM=1.6)
    normal_ref = jnp.cross(r1, r2)
    v1_s, _, valid_s = lambert_solve(r1, r2, tof, MU, 0, False)
    v1_l, _, valid_l = lambert_solve(r1, r2, tof, MU, 0, True)
    assert bool(valid_s) and bool(valid_l)
    assert jnp.dot(jnp.cross(r1, v1_s), normal_ref) > 0.0
    assert jnp.dot(jnp.cross(r1, v1_l), normal_ref) < 0.0


def test_below_parabolic_tof_is_invalid():
    """A time of flight below the parabolic limit has no elliptic solution."""
    r1, _, r2, _, tof, long_way = _two_epoch_fixture(dM=1.6)
    _, _, valid = lambert_solve(r1, r2, 1e-3 * tof, MU, 0, long_way)
    assert not bool(valid)


def test_high_branch_requires_multirev():
    """The high-x branch does not exist for N=0 and is flagged invalid."""
    r1, _, r2, _, tof, long_way = _two_epoch_fixture(dM=1.6)
    _, _, valid = lambert_solve(r1, r2, tof, MU, 0, long_way, True)
    assert not bool(valid)


def test_tof_min_increases_with_rev_count():
    """The minimum achievable TOF grows with the revolution count N."""
    r1, _, r2, _, _, long_way = _two_epoch_fixture(dM=1.6)
    tmins = jnp.array([lambert_tof_min(r1, r2, MU, N, long_way) for N in range(5)])
    assert bool(jnp.all(jnp.diff(tmins) > 0.0))


def test_valid_flag_matches_tof_min_boundary():
    """Solutions exist exactly when tof exceeds the N-rev minimum TOF."""
    r1, _, r2, _, _, long_way = _two_epoch_fixture(dM=1.6)
    tmin = lambert_tof_min(r1, r2, MU, 2, long_way)
    for high in (False, True):
        _, _, valid_above = lambert_solve(r1, r2, 1.05 * tmin, MU, 2, long_way, high)
        _, _, valid_below = lambert_solve(r1, r2, 0.95 * tmin, MU, 2, long_way, high)
        assert bool(valid_above)
        assert not bool(valid_below)


@pytest.mark.float64_only(reason="finite-difference derivative reference")
def test_gradients_match_finite_differences():
    """Implicit-function gradients through the solve match finite differences."""
    r1, _, r2, _, tof, long_way = _two_epoch_fixture(dM=1.6)

    def loss(r1_, r2_, tof_, mu_):
        v1_l, v2_l, _ = lambert_solve(r1_, r2_, tof_, mu_, 0, long_way)
        return jnp.sum(v1_l * 1.3 + v2_l * 0.7)

    check_grads(loss, (r1, r2, tof, MU), order=1, modes=["rev"], atol=1e-4, rtol=1e-4)


@pytest.mark.float64_only(reason="finite-difference derivative reference")
def test_gradients_match_finite_differences_multirev():
    """Gradients are also implicit-function-correct on a multi-rev branch."""
    r1, _, r2, _, tof, long_way = _two_epoch_fixture(dM=1.6, n_rev=2)

    def loss(r1_, tof_):
        v1_l, _, _ = lambert_solve(r1_, r2, tof_, MU, 2, long_way, True)
        return jnp.sum(v1_l)

    check_grads(loss, (r1, tof), order=1, modes=["rev"], atol=1e-4, rtol=1e-4)


def test_vmap_over_depth_batch():
    """The scalar core vmaps over a batch of line-of-sight depth hypotheses.

    Each recovered TOF has the same 32-rounding phase reconstruction and
    energy-to-period condition as in the individual alias test.
    """
    r1, _, r2, _, tof, long_way = _two_epoch_fixture(dM=1.6)
    dz = jnp.linspace(-0.3, 0.3, 8)
    r1_batch = r1[None, :] + dz[:, None] * jnp.array([0.0, 0.0, 1.0])

    v1_b, v2_b, valid_b = jax.vmap(
        lambda r1_: lambert_solve(r1_, r2, tof, MU, 0, long_way)
    )(r1_batch)
    assert v1_b.shape == (8, 3) and v2_b.shape == (8, 3) and valid_b.shape == (8,)
    assert bool(jnp.all(jnp.isfinite(v1_b[valid_b])))
    tof_back, bounds = jax.vmap(
        lambda r1_, v1_, v2_: _implied_tof(r1_, v1_, r2, v2_, MU, 0)
    )(r1_batch, v1_b, v2_b)
    assert jnp.all(jnp.abs(tof_back[valid_b] - tof) <= bounds[valid_b])
