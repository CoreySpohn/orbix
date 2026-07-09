"""Lambert boundary-value solver: round-trips, multi-rev aliases, gradients."""

import jax
import jax.numpy as jnp
from jax.test_util import check_grads

from orbix.equations import lambert_solve, lambert_tof_min

jax.config.update("jax_enable_x64", True)

MU = 1.0


def _rot_z(t):
    """Active rotation matrix about the z axis."""
    c, s = jnp.cos(t), jnp.sin(t)
    return jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_x(t):
    """Active rotation matrix about the x axis."""
    c, s = jnp.cos(t), jnp.sin(t)
    return jnp.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rv_from_elements(a, e, inc, W, w, M, mu):
    """Independent (r, v, nu) reference: Newton Kepler solve + perifocal rotation."""
    E = M
    for _ in range(60):
        E = E - (E - e * jnp.sin(E) - M) / (1.0 - e * jnp.cos(E))
    sinE, cosE = jnp.sin(E), jnp.cos(E)
    rmag = a * (1.0 - e * cosE)
    sq1me2 = jnp.sqrt(1.0 - e**2)
    r_pf = jnp.array([a * (cosE - e), a * sq1me2 * sinE, 0.0])
    v_pf = (jnp.sqrt(mu * a) / rmag) * jnp.array([-sinE, sq1me2 * cosE, 0.0])
    R = _rot_z(W) @ _rot_x(inc) @ _rot_z(w)
    nu = 2.0 * jnp.arctan2(
        jnp.sqrt(1.0 + e) * jnp.sin(E / 2.0), jnp.sqrt(1.0 - e) * jnp.cos(E / 2.0)
    )
    return R @ r_pf, R @ v_pf, nu


def _two_epoch_fixture(dM, n_rev=0, a=1.0, e=0.35, inc=0.6, W=1.1, w=0.7, M1=0.4):
    """Truth (r1, v1, r2, v2, tof, long_way) for a transfer of dM (+ n_rev turns)."""
    M2 = M1 + dM + 2.0 * jnp.pi * n_rev
    r1, v1, nu1 = _rv_from_elements(a, e, inc, W, w, M1, MU)
    r2, v2, nu2 = _rv_from_elements(a, e, inc, W, w, M2, MU)
    n = jnp.sqrt(MU / a**3)
    tof = (M2 - M1) / n
    sweep = jnp.mod(nu2 - nu1, 2.0 * jnp.pi)
    return r1, v1, r2, v2, tof, bool(sweep > jnp.pi)


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
    return dM / n


def test_n0_short_way_recovers_velocities():
    """A single-rev short-way transfer recovers both terminal velocities."""
    r1, v1, r2, v2, tof, long_way = _two_epoch_fixture(dM=1.6)
    assert not long_way
    v1_l, v2_l, valid = lambert_solve(r1, r2, tof, MU, 0, long_way)
    assert bool(valid)
    assert jnp.allclose(v1_l, v1, rtol=1e-8, atol=1e-10)
    assert jnp.allclose(v2_l, v2, rtol=1e-8, atol=1e-10)


def test_n0_long_way_recovers_velocities():
    """A single-rev long-way (sweep > pi) transfer recovers both velocities."""
    r1, v1, r2, v2, tof, long_way = _two_epoch_fixture(dM=4.4)
    assert long_way
    v1_l, v2_l, valid = lambert_solve(r1, r2, tof, MU, 0, long_way)
    assert bool(valid)
    assert jnp.allclose(v1_l, v1, rtol=1e-8, atol=1e-10)
    assert jnp.allclose(v2_l, v2, rtol=1e-8, atol=1e-10)


def test_multirev_one_branch_matches_truth():
    """For a 3.x-revolution transfer, exactly one N=3 branch is the true orbit."""
    r1, v1, r2, v2, tof, long_way = _two_epoch_fixture(dM=1.6, n_rev=3)
    errs = []
    for high in (False, True):
        v1_l, _, valid = lambert_solve(r1, r2, tof, MU, 3, long_way, high)
        assert bool(valid)
        errs.append(float(jnp.max(jnp.abs(v1_l - v1))))
    assert min(errs) < 1e-8
    assert max(errs) > 1e-3


def test_valid_aliases_all_satisfy_the_boundary_value_problem():
    """Every valid (N, branch) alias connects r1 to r2 in exactly tof."""
    r1, _, r2, _, tof, long_way = _two_epoch_fixture(dM=1.6, n_rev=3)
    n_valid = 0
    for N in range(9):
        for high in (False, True):
            v1_l, v2_l, valid = lambert_solve(r1, r2, tof, MU, N, long_way, high)
            if not bool(valid):
                continue
            n_valid += 1
            tof_back = _implied_tof(r1, v1_l, r2, v2_l, MU, N)
            assert jnp.allclose(tof_back, tof, rtol=1e-8)
            h1 = jnp.cross(r1, v1_l)
            h2 = jnp.cross(r2, v2_l)
            assert jnp.allclose(h1, h2, rtol=1e-8, atol=1e-12)
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


def test_gradients_match_finite_differences():
    """Implicit-function gradients through the solve match finite differences."""
    r1, _, r2, _, tof, long_way = _two_epoch_fixture(dM=1.6)

    def loss(r1_, r2_, tof_, mu_):
        v1_l, v2_l, _ = lambert_solve(r1_, r2_, tof_, mu_, 0, long_way)
        return jnp.sum(v1_l * 1.3 + v2_l * 0.7)

    check_grads(loss, (r1, r2, tof, MU), order=1, modes=["rev"], atol=1e-4, rtol=1e-4)


def test_gradients_match_finite_differences_multirev():
    """Gradients are also implicit-function-correct on a multi-rev branch."""
    r1, _, r2, _, tof, long_way = _two_epoch_fixture(dM=1.6, n_rev=2)

    def loss(r1_, tof_):
        v1_l, _, _ = lambert_solve(r1_, r2, tof_, MU, 2, long_way, True)
        return jnp.sum(v1_l)

    check_grads(loss, (r1, tof), order=1, modes=["rev"], atol=1e-4, rtol=1e-4)


def test_vmap_over_depth_batch():
    """The scalar core vmaps over a batch of line-of-sight depth hypotheses."""
    r1, _, r2, _, tof, long_way = _two_epoch_fixture(dM=1.6)
    dz = jnp.linspace(-0.3, 0.3, 8)
    r1_batch = r1[None, :] + dz[:, None] * jnp.array([0.0, 0.0, 1.0])

    v1_b, v2_b, valid_b = jax.vmap(
        lambda r1_: lambert_solve(r1_, r2, tof, MU, 0, long_way)
    )(r1_batch)
    assert v1_b.shape == (8, 3) and v2_b.shape == (8, 3) and valid_b.shape == (8,)
    assert bool(jnp.all(jnp.isfinite(v1_b[valid_b])))
    tof_back = jax.vmap(lambda r1_, v1_, v2_: _implied_tof(r1_, v1_, r2, v2_, MU, 0))(
        r1_batch, v1_b, v2_b
    )
    assert jnp.allclose(tof_back[valid_b], tof, rtol=1e-8)
