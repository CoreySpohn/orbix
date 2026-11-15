"""Resolution-aware properties and contracts for grid shortcut solvers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import two_pi
from hypothesis import example, given
from hypothesis import strategies as st

from orbix.kepler.markley import E_solve, E_solve_trig
from orbix.kepler.shortcuts.grid import get_grid_solver


@pytest.fixture(scope="module")
def grid_solvers():
    """Construct each output/kind once; examples reuse fixed-shape executables."""
    result = {}
    for kind in ("linear", "bilinear"):
        solvers = [
            get_grid_solver(kind=kind, E=E, trig=trig)
            for E, trig in ((True, True), (True, False), (False, True))
        ]

        def evaluate(means, e, solvers=solvers):
            return tuple(jax.vmap(s, in_axes=(0, None))(means, e) for s in solvers)

        result[kind] = jax.jit(evaluate)
    return result


@pytest.mark.parametrize("kind", ["linear", "bilinear"])
@example(e=0.3, mean=-0.001)
@example(e=0.9985, mean=1.0)
@given(
    e=st.floats(0.0, 0.99, allow_nan=False, allow_infinity=False),
    mean=st.floats(-two_pi, two_pi, allow_nan=False, allow_infinity=False),
)
def test_grid_equation_trig_and_wrapping(grid_solvers, kind, e, mean):
    """Fails if any grid output uses the wrong cell, wrapping, or trig table.

    Linear snaps e within half a row; bilinear spans one row and one M cell.
    Residual bounds reflect that resolution, including float32 table storage.
    Both kinds must be continuous and monotone in M.
    Lookup approximation is not an exact Kepler solve.
    """
    node = np.floor(np.mod(mean, two_pi) * 2048 / two_pi) * two_pi / 2048
    means = np.array(
        [
            mean,
            mean + two_pi,
            mean - two_pi,
            mean + 1e-7,
            node - 1e-7,
            node + 1e-7,
            -1e-7,
            1e-7,
            two_pi - 1e-7,
            two_pi + 1e-7,
        ]
    )
    triple, E_only, trig_only = grid_solvers[kind](jnp.asarray(means), jnp.asarray(e))
    eccentric, sine, cosine = map(np.asarray, triple)
    np.testing.assert_allclose(E_only, eccentric, atol=2e-6, rtol=0)
    np.testing.assert_allclose(trig_only, [sine, cosine], atol=2e-6, rtol=0)
    if e == 0.3 and mean == -0.001:
        # Absolute wrapped roots catch a full-turn offset that residual and
        # periodicity checks cannot; retain the original 1e-4 E-error guard.
        truth = float(E_solve(jnp.asarray(np.mod(mean, two_pi)), jnp.asarray(e)))
        for roots in (eccentric[:3], np.asarray(E_only)[:3]):
            assert np.all((0 <= roots) & (roots < two_pi))
            assert np.max(np.abs(roots - truth)) < 1e-4
    residual = eccentric - e * np.sin(eccentric) - means
    if kind == "linear":
        # Taylor remainder in M plus nearest-row error in e.
        bound = (
            0.5 / 512
            + e / 2 * (two_pi / 2048 / (1 - min(e + 0.5 / 512, 511 / 512))) ** 2
        )
    else:
        angular_span = (two_pi / 2048 + 1 / 512) / (1 - min(e + 1 / 512, 0.999999))
        # Taylor/Jensen remainder for sin(E), plus the e/sin(E) covariance.
        bound = min(
            two_pi / 2048 + 1 / 512, e * angular_span**2 / 8 + angular_span / 512 / 4
        )
    assert (
        np.max(np.abs(np.arctan2(np.sin(residual), np.cos(residual)))) <= bound + 3e-6
    )
    # Float32 M-grid spacing error is amplified by dE/dM near periapsis.
    phase_roundoff = 2 * np.finfo(np.float32).eps * two_pi / (1 - e)
    difference = eccentric[:3] - eccentric[0]
    np.testing.assert_allclose(
        np.arctan2(np.sin(difference), np.cos(difference)),
        0,
        atol=phase_roundoff,
        rtol=0,
    )
    for values in (sine, cosine):
        np.testing.assert_allclose(values[:3], values[0], atol=phase_roundoff, rtol=0)
    # A trig pair need not be unit length after interpolation, but its defect
    # is bounded by the angular cell size (chord or tangent remainder).
    trig_bound = (two_pi / 2048 / (1 - min(e + 1 / 512, 0.999999))) ** 2 + 3e-6
    assert np.max(np.abs(sine**2 + cosine**2 - 1)) <= trig_bound
    # Pin phase too: unit norm alone admits a swapped or constant trig table.
    angular_cell = (two_pi / 2048 + (1 / 512 if kind == "bilinear" else 0)) / (
        1 - min(e + 1 / 512, 0.999999)
    )
    assert (
        np.max(np.hypot(sine - np.sin(eccentric), cosine - np.cos(eccentric)))
        <= angular_cell**2 / 2 + phase_roundoff
    )
    if e == 0.9985 and mean == 1.0:
        # At M=1 the local solve is well-conditioned even in the top e row.
        # Linear snaps e by < 0.5/512; bilinear interpolates across that row.
        # Compare to float64 truth: global periapsis bounds are vacuous here.
        truth = float(E_solve(jnp.float64(mean), jnp.float64(e)))
        tolerance = 1e-3 if kind == "linear" else 1e-4
        np.testing.assert_allclose(
            [sine[0], cosine[0]],
            [np.sin(truth), np.cos(truth)],
            atol=tolerance,
            rtol=0,
        )
    if kind == "bilinear":
        for left, right in ((0, 3), (4, 5), (6, 7), (8, 9)):
            pair = means[[left, right]]
            difference = eccentric[[left, right]] - pair
            lifted = pair + np.arctan2(np.sin(difference), np.cos(difference))
            delta = lifted[1] - lifted[0]
            limit = (pair[1] - pair[0]) / (1 - min(e + 1 / 512, 0.999999))
            assert -phase_roundoff <= delta <= limit + phase_roundoff
        # Keep the top-row regression's tighter accuracy at a well-conditioned M.
        if mean == 1.0:
            truth = float(E_solve(jnp.asarray(mean), jnp.asarray(e)))
            error = abs(eccentric[0] - truth)
            assert error < 1e-4
            reference = grid_solvers[kind](jnp.asarray(means), jnp.asarray(0.995))[1][0]
            reference_error = abs(
                float(reference) - float(E_solve(jnp.asarray(mean), jnp.asarray(0.995)))
            )
            assert error < 100 * max(reference_error, 1e-7)


def test_get_grid_solver_rejects_no_output():
    """Fails if requesting no output stops raising a useful ValueError."""
    with pytest.raises(ValueError, match=r"E|trig"):
        get_grid_solver(E=False, trig=False)


@pytest.mark.parametrize("cell", [0, 100, 2047, 2048])
def test_linear_grid_monotone_at_cell_boundary(grid_solvers, cell):
    """Fails if the linear grid jumps backwards as mean anomaly increases."""
    node = cell * two_pi / 2048
    means = jnp.array([node - 1e-10, node + 1e-10] + [node] * 8)
    triple, result, trig_only = grid_solvers["linear"](means, jnp.asarray(0.5))
    result = np.asarray(result)
    result = means + np.arctan2(np.sin(result - means), np.cos(result - means))
    assert 0 <= result[1] - result[0] <= 2e-10 / (1 - 0.5) + 2e-12
    for trig in (np.asarray(triple[1:]), np.asarray(trig_only)):
        assert np.all(np.abs(trig[:, 1] - trig[:, 0]) <= 2e-10 / (1 - 0.5) + 2e-12)


@pytest.mark.parametrize("kind", ["linear", "bilinear"])
@pytest.mark.parametrize("outputs", [(True, True), (True, False), (False, True)])
def test_grid_precision_and_cache(kind, outputs):
    """Dtypes match input promotion exactly and caches separate default dtypes.

    Compare all builders with the combined output to 2 eps (floating summation
    floor). At a table node, 8 eps agreement with Markley checks table precision,
    not just the output cast. Frozen float32 values pin the pre-fix behavior.
    """
    low = high = None
    for x64 in (False, True, False):
        with jax.enable_x64(x64):
            solver = get_grid_solver(
                kind=kind, E=outputs[0], trig=outputs[1], n_e=16, n_M=64
            )
            if x64:
                high = solver
                assert high is not low
            elif low is None:
                low = solver
            else:
                assert solver is low
            for dtype in [jnp.float32, jnp.float64] if x64 else [jnp.float32]:
                mean, e = jnp.asarray(2.0, dtype), jnp.asarray(0.3, dtype)
                values = jax.jit(solver)(mean, e)
                assert all(v.dtype == dtype for v in jax.tree.leaves(values))
                full = get_grid_solver(kind=kind, n_e=16, n_M=64)(mean, e)
                expected = full if all(outputs) else full[0] if outputs[0] else full[1:]
                np.testing.assert_allclose(
                    values, expected, rtol=0, atol=2 * np.finfo(dtype).eps
                )
            if x64:
                mixed = solver(jnp.float32(2.0), jnp.float64(0.3))
                assert all(v.dtype == jnp.float64 for v in jax.tree.leaves(mixed))
                node = (jnp.float64(two_pi / 4), jnp.float64(0.25))
                truth = E_solve_trig(*node)
                expected = (
                    truth if all(outputs) else truth[0] if outputs[0] else truth[1:]
                )
                np.testing.assert_allclose(
                    solver(*node), expected, rtol=0, atol=8 * np.finfo(np.float64).eps
                )


@pytest.mark.parametrize("kind", ["linear", "bilinear"])
def test_grid_float32_values_unchanged(kind):
    """Frozen pre-fix float32 results agree exactly (zero-tolerance regression)."""
    expected = {
        "linear": [
            [0.10000000149011612, 2.2441091537475586, 5.218289852142334],
            [0.09982188791036606, 0.7811498641967773, -0.874165952205658],
            [0.9949170351028442, -0.6230855584144592, 0.484312504529953],
        ],
        "bilinear": [
            [0.10000000894069672, 2.235628128051758, 5.180873394012451],
            [0.09982188791036606, 0.7862817645072937, -0.8907663822174072],
            [0.9949170351028442, -0.6163423657417297, 0.4508167803287506],
        ],
    }
    with jax.enable_x64(False):
        solver = get_grid_solver(kind=kind, n_e=16, n_M=64)
        result = jax.jit(jax.vmap(solver))(
            jnp.array([0.1, 2.0, -0.3]), jnp.array([0.0, 0.3, 0.9])
        )
        np.testing.assert_array_equal(
            result, np.asarray(expected[kind], dtype=np.float32)
        )
