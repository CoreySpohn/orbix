"""Resolution-aware properties and contracts for grid shortcut solvers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import two_pi
from hypothesis import example, given
from hypothesis import strategies as st

from orbix.kepler import Grid, Markley, kepler


@jax.jit
def _grid(means, e):
    return kepler(means, e, 0, solver=Grid())


def _reference(mean, e):
    sine, cosine = kepler(mean, e, 0, solver=Markley())
    return jnp.mod(jnp.arctan2(sine, cosine), two_pi)


def _reference_grid(means, e):
    sine, cosine = _grid(means, e)
    return jnp.mod(jnp.arctan2(sine, cosine), two_pi)


@example(e=0.3, mean=-0.001)
@example(e=0.9985, mean=1.0)
@given(
    e=st.floats(0.0, 0.99, allow_nan=False, allow_infinity=False),
    mean=st.floats(-two_pi, two_pi, allow_nan=False, allow_infinity=False),
)
def test_grid_equation_trig_and_wrapping(e, mean):
    """Fails if any grid output uses the wrong cell, wrapping, or trig table.

    Bilinear interpolation spans one row and one M cell. Residual bounds
    reflect that resolution, including float32 table storage. Values must
    remain continuous and monotone in M.
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
    sine, cosine = map(np.asarray, _grid(jnp.asarray(means), jnp.asarray(e)))
    eccentric = np.mod(np.arctan2(sine, cosine), two_pi)
    if e == 0.3 and mean == -0.001:
        # Absolute wrapped roots catch a full-turn offset that residual and
        # periodicity checks cannot; retain the original 1e-4 E-error guard.
        truth = float(_reference(jnp.asarray(np.mod(mean, two_pi)), jnp.asarray(e)))
        for roots in (eccentric[:3],):
            assert np.all((0 <= roots) & (roots < two_pi))
            assert np.max(np.abs(roots - truth)) < 1e-4
    residual = eccentric - e * np.sin(eccentric) - means
    angular_span = (two_pi / 2048 + 1 / 512) / (1 - min(e + 1 / 512, 0.999999))
    # Interpolated trig lies on a chord; angle error is bounded by the cell span.
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
    angular_cell = (two_pi / 2048 + 1 / 512) / (1 - min(e + 1 / 512, 0.999999))
    assert (
        np.max(np.hypot(sine - np.sin(eccentric), cosine - np.cos(eccentric)))
        <= angular_cell**2 / 2 + phase_roundoff
    )
    if e == 0.9985 and mean == 1.0:
        # At M=1 the local solve is well-conditioned even in the top e row.
        # Bilinear interpolation must use the appended row near e=1.
        # Host reference: solve E - 0.9985*sin(E) = 1 at 60 decimal digits.
        truth = 1.933528471382059
        tolerance = 1e-4
        np.testing.assert_allclose(
            [sine[0], cosine[0]],
            [np.sin(truth), np.cos(truth)],
            atol=tolerance,
            rtol=0,
        )
    for left, right in ((0, 3), (4, 5), (6, 7), (8, 9)):
        pair = means[[left, right]]
        difference = eccentric[[left, right]] - pair
        lifted = pair + np.arctan2(np.sin(difference), np.cos(difference))
        delta = lifted[1] - lifted[0]
        limit = (pair[1] - pair[0]) / (1 - min(e + 1 / 512, 0.999999))
        assert -phase_roundoff <= delta <= limit + phase_roundoff
    # Keep the top-row regression's tighter accuracy at a well-conditioned M.
    if mean == 1.0:
        truth = float(_reference(jnp.asarray(mean), jnp.asarray(e)))
        error = abs(eccentric[0] - truth)
        assert error < 1e-4
        reference = _reference_grid(jnp.asarray(means), jnp.asarray(0.995))[0]
        reference_error = abs(
            float(reference) - float(_reference(jnp.asarray(mean), jnp.asarray(0.995)))
        )
        assert error < 100 * max(reference_error, 1e-7)


@pytest.mark.float64_only(
    reason="1e-10 rad probes must stay distinct across each grid node"
)
@pytest.mark.parametrize("cell", [0, 100, 2047, 2048])
def test_grid_monotone_at_cell_boundary(cell):
    """Fails if the bilinear trig table jumps backwards at a periodic M node."""
    node = cell * two_pi / 2048
    means = jnp.array([node - 1e-10, node + 1e-10])
    sine, cosine = _grid(means, jnp.asarray(0.5))
    angle = np.arctan2(sine, cosine)
    lifted = means + np.arctan2(np.sin(angle - means), np.cos(angle - means))
    assert 0 <= lifted[1] - lifted[0] <= 2e-10 / (1 - 0.5) + 2e-12
    assert np.all(
        np.abs(np.asarray([sine, cosine])[:, 1] - np.asarray([sine, cosine])[:, 0])
        <= 2e-10 / (1 - 0.5) + 2e-12
    )


@pytest.mark.float64_only(
    reason="cache isolation between float32 and float64 table builds"
)
def test_grid_precision_context_and_nodes():
    """Fails if cached float32 tables leak into float64 construction.

    Table-node agreement within 8 eps checks storage precision as well as
    output casting. Switching back to float32 also exercises cache reuse.
    """
    for x64 in (False, True, False):
        with jax.enable_x64(x64):
            solver = Grid(16, 64)
            for dtype in [jnp.float32, jnp.float64] if x64 else [jnp.float32]:
                node = (jnp.asarray(two_pi / 4, dtype), jnp.asarray(0.25, dtype))
                values = jax.jit(kepler)(*node, 0, solver=solver)
                assert all(v.dtype == dtype for v in values)
                expected = kepler(*node, 0, solver=Markley())
                np.testing.assert_allclose(
                    values, expected, rtol=0, atol=8 * np.finfo(dtype).eps
                )
            if x64:
                mixed = jax.jit(kepler)(
                    jnp.float32(2.0), jnp.float64(0.3), 0, solver=solver
                )
                assert all(v.dtype == jnp.float64 for v in mixed)
