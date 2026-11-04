"""Properties for fixed-e shortcuts, with resolution-aware error bounds."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import two_pi
from hypothesis import example, given
from hypothesis import strategies as st

from orbix.kepler.shortcuts.fixed_e import E_hermite_interp, E_linear_interp, E_lookup


@jax.jit
def _shortcuts(means, e):
    """Build tables with dynamic e inside one fixed-shape compiled program."""
    return (
        E_lookup(e, 512)(means),
        E_linear_interp(e, 2048)(means),
        E_hermite_interp(e, 2048)(means),
    )


@example(e=0.2, mean=(10.4 * 0.999999 + 0.6) * two_pi / 512)
@example(e=0.3, mean=two_pi - two_pi / 2048 / 2)
@given(
    e=st.one_of(
        st.just(0.0), st.floats(1e-100, 0.99, allow_nan=False, allow_infinity=False)
    ),
    mean=st.floats(-two_pi, two_pi, allow_nan=False, allow_infinity=False),
)
def test_shortcut_equation_and_wrapping(e, mean):
    """Fails if lookup rounds down, interpolation exceeds cell error, or seams blend.

    Lookup quantizes M to half a cell; linear interpolation has residual
    <= e/2 * (dM/(1-e))**2 by Taylor's theorem. Hermite must stay within
    one M cell. Tiny positive e is covered by the core overflow regression.
    Lookup is intentionally discrete; interpolation must remain continuous.
    """
    means = np.array([mean, mean + two_pi, mean - two_pi, two_pi - two_pi / 4096])
    lookup, linear, hermite = map(
        np.asarray, _shortcuts(jnp.asarray(means), jnp.asarray(e))
    )
    for result, bound in (
        (lookup, two_pi / 512 / 2),
        (linear, e / 2 * (two_pi / 2048 / (1 - e)) ** 2),
        (hermite, two_pi / 2048),
    ):
        residual = result - e * np.sin(result) - means
        np.testing.assert_array_less(
            np.abs(np.arctan2(np.sin(residual), np.cos(residual))), bound + 2e-12
        )
        phase_difference = result[:3] - result[0]
        np.testing.assert_allclose(
            np.arctan2(np.sin(phase_difference), np.cos(phase_difference)),
            0,
            atol=2 * bound / (1 - e) + 2e-11,
            rtol=0,
        )
    # Preserve the tighter last-cell regression, now across eccentricities.
    if e <= 0.3:
        residual = hermite[-1] - e * np.sin(hermite[-1]) - means[-1]
        assert abs(residual) / (1 - e) < 1e-6

    # Hermite is continuous, unlike the intentionally discrete lookup.
    node = np.floor(np.mod(mean, two_pi) * 2048 / two_pi) * two_pi / 2048
    centers = np.array([mean, node, 0.0, two_pi])
    means = np.stack([centers - 1e-8, centers + 1e-8]).ravel()
    result = np.asarray(_shortcuts(jnp.asarray(means), jnp.asarray(e))[2])
    offset = result - means
    lifted = means + np.arctan2(np.sin(offset), np.cos(offset))
    differences = lifted[4:] - lifted[:4]
    assert np.all(differences >= -2e-12)
    assert np.all(differences <= 2e-8 / (1 - e) + 2e-12)


@pytest.mark.parametrize("kind", [1, 2], ids=["linear", "hermite"])
@pytest.mark.parametrize("cell", [0, 100, 2047, 2048])
def test_linear_monotone_at_cell_boundary(kind, cell):
    """Fails if the purported linear interpolant jumps backwards at an M node."""
    node = cell * two_pi / 2048
    means = jnp.array([node - 1e-10, node + 1e-10])
    result = np.asarray(_shortcuts(means, jnp.asarray(0.5))[kind])
    result = means + np.arctan2(np.sin(result - means), np.cos(result - means))
    assert 0 <= result[1] - result[0] <= 2e-10 / (1 - 0.5) + 2e-12
