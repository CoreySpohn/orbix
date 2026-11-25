"""Campbell inversion with classical north/east Thiele-Innes letters."""

import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import two_pi
from hwoutils.testing import precision_pair, ulps

from orbix.equations import thiele_innes_constants, thiele_innes_to_elements


def test_thiele_innes_inverse_roundtrip_and_degeneracies(precision):
    """Floating-point basis: 64 eps covers sums, products and angle recovery.

    Independent float64 host constants are the two rotated orbital-plane axes,
    projected onto north/east. Compare identifiable W+w and W-w angles;
    at i=0 or pi only the respective sum or difference is observable. e=0
    retains the arbitrary input phase axis. The generic inverse/forward
    Jacobian must be the identity to the same floating-point tolerance.
    With x64 available, a float32 round trip also matches its float64
    evaluation to this operation budget at the unit angle/length scale.
    """
    dtype = precision
    rng = np.random.default_rng(19)
    a, c = rng.uniform(0.1, 10.0, 64), rng.uniform(-0.9, 0.9, 64)
    c[:4] = [1.0, -1.0, 0.0, 0.0]
    W, w = rng.uniform(0.0, two_pi, (2, 64))
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
    a, c, W, w = (jnp.asarray(x, dtype) for x in (a, c, W, w))
    constants = a * jnp.stack(thiele_innes_constants(W, jnp.arccos(c), w))
    tol = float(ulps(64, jnp.ones((), dtype=dtype)))
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

    point = jnp.array([1.3, -0.4, 0.2, 0.6], dtype)
    if dtype == np.float64:
        precision_pair(
            roundtrip, (point,), bound=float(ulps(64, np.float32(1))), scale=1.0
        )
    jac = jax.jit(jax.jacfwd(roundtrip))(point)
    np.testing.assert_allclose(jac, np.eye(4), atol=tol, rtol=0)
    np.testing.assert_array_equal(thiele_innes_to_elements(0.0, 0.0, 0.0, 0.0), 0.0)
