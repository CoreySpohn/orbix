"""Campbell inversion with classical north/east Thiele-Innes letters."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import two_pi

from orbix.equations import thiele_innes_constants, thiele_innes_to_elements


@pytest.mark.parametrize(
    "dtype",
    [
        jnp.float32,
        pytest.param(
            jnp.float64,
            marks=pytest.mark.float64_only(reason="explicit float64 orbital inputs"),
        ),
    ],
)
def test_thiele_innes_inverse_roundtrip_and_degeneracies(dtype):
    """Floating-point basis: 64 eps covers sums, products and angle recovery.

    Independent expected constants are the two rotated orbital-plane axes,
    projected onto north/east. Compare identifiable W+w and W-w angles;
    at i=0 or pi only the respective sum or difference is observable. e=0
    retains the arbitrary input phase axis. The generic inverse/forward
    Jacobian must be the identity to the same floating-point tolerance.
    """
    rng = np.random.default_rng(19)
    a = jnp.asarray(rng.uniform(0.1, 10.0, 64), dtype)
    c = jnp.asarray(rng.uniform(-0.9, 0.9, 64), dtype)
    c = c.at[:4].set(jnp.array([1.0, -1.0, 0.0, 0.0], dtype))
    W = jnp.asarray(rng.uniform(0.0, two_pi, 64), dtype)
    w = jnp.asarray(rng.uniform(0.0, two_pi, 64), dtype)
    e = jnp.asarray(rng.uniform(0.0, 0.95, 64), dtype).at[::2].set(0.0)
    cw, sw, cW, sW = np.cos(w), np.sin(w), np.cos(W), np.sin(W)
    expected = np.array(
        [
            a * (cW * cw - sW * sw * c),
            a * (sW * cw + cW * sw * c),
            a * (-cW * sw - sW * cw * c),
            a * (-sW * sw + cW * cw * c),
        ]
    )
    constants = a * jnp.stack(thiele_innes_constants(W, jnp.arccos(c), w))
    tol = 64 * np.finfo(dtype).eps
    np.testing.assert_allclose(constants, expected, atol=tol * float(a.max()), rtol=0)
    # Exercise the eccentric-anomaly sin(E) basis used by angular fitters.
    scaled = constants.at[2:].multiply(jnp.sqrt(1 - e**2))
    unscaled = scaled.at[2:].divide(jnp.sqrt(1 - e**2))
    aa, cc, node, omega = jax.jit(thiele_innes_to_elements)(*unscaled)
    np.testing.assert_allclose(aa, a, rtol=tol)
    np.testing.assert_allclose(cc, c, atol=tol, rtol=0)
    rebuilt = aa * jnp.stack(thiele_innes_constants(node, jnp.arccos(cc), omega))
    np.testing.assert_allclose(rebuilt, expected, atol=tol * float(a.max()), rtol=0)
    np.testing.assert_array_equal(node[:2], 0.0)
    for sign, mask in [(1, c != -1), (-1, c != 1)]:
        np.testing.assert_allclose(
            np.exp(1j * np.asarray(node + sign * omega))[mask],
            np.exp(1j * np.asarray(W + sign * w))[mask],
            atol=tol,
            rtol=0,
        )

    def roundtrip(x):
        block = x[0] * jnp.stack(thiele_innes_constants(x[2], jnp.arccos(x[1]), x[3]))
        return jnp.stack(thiele_innes_to_elements(*block))

    jac = jax.jit(jax.jacfwd(roundtrip))(jnp.array([1.3, -0.4, 0.2, 0.6], dtype))
    np.testing.assert_allclose(jac, np.eye(4), atol=tol, rtol=0)
    np.testing.assert_array_equal(thiele_innes_to_elements(0.0, 0.0, 0.0, 0.0), 0.0)
