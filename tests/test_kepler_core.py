"""Regression tests for the Kepler solver trig reconstruction near M = pi.

The initial-guess table edge plus a sequential sinE-then-cosE update produced a
badly wrong (sinE, cosE) pair at bit-exact M = pi while being machine-accurate
one ulp to either side. These tests pin that corner.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from orbix.kepler.core import E_solve, diff_solve_trig, solve_trig


def _newton_E(M, e, iters=80):
    """High-accuracy Kepler reference: Newton iteration on M = E - e*sin(E)."""
    M = float(np.mod(M, 2.0 * np.pi))
    E = M if e < 0.8 else np.pi
    for _ in range(iters):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    return E


# M = pi exactly, and the two representable neighbours one ulp away.
_M_POINTS = [
    np.nextafter(np.pi, 0.0),  # pi - 1 ulp
    np.pi,
    np.nextafter(np.pi, 4.0),  # pi + 1 ulp
]
_E_VALUES = [0.0, 0.05, 0.2, 0.5, 0.9]


@pytest.mark.parametrize("e", _E_VALUES)
@pytest.mark.parametrize("M", _M_POINTS)
def test_solve_trig_matches_newton_near_mpi(M, e):
    """(sinE, cosE) match a converged Newton reference at M = pi +/- 1 ulp."""
    sinE, cosE = solve_trig(jnp.array([M]), e)
    E_ref = _newton_E(M, e)
    np.testing.assert_allclose(float(sinE[0]), np.sin(E_ref), atol=1e-9)
    np.testing.assert_allclose(float(cosE[0]), np.cos(E_ref), atol=1e-9)


@pytest.mark.parametrize("e", _E_VALUES)
@pytest.mark.parametrize("M", _M_POINTS)
def test_solve_trig_pair_is_unit_norm(M, e):
    """The returned (sinE, cosE) lie on the unit circle."""
    sinE, cosE = solve_trig(jnp.array([M]), e)
    np.testing.assert_allclose(float(sinE[0] ** 2 + cosE[0] ** 2), 1.0, atol=1e-9)


@pytest.mark.parametrize("e", _E_VALUES)
@pytest.mark.parametrize("M", _M_POINTS)
def test_solve_trig_consistent_with_returned_E(M, e):
    """The trig pair is the actual sine/cosine of the E that E_solve returns."""
    E = E_solve(jnp.array([M]), e)
    sinE, cosE = solve_trig(jnp.array([M]), e)
    np.testing.assert_allclose(float(sinE[0]), float(jnp.sin(E[0])), atol=1e-9)
    np.testing.assert_allclose(float(cosE[0]), float(jnp.cos(E[0])), atol=1e-9)


@pytest.mark.parametrize("e", _E_VALUES)
@pytest.mark.parametrize("M", _M_POINTS)
def test_diff_solve_trig_matches_solve_trig_near_mpi(M, e):
    """diff_solve_trig agrees with solve_trig at the M = pi corner."""
    s_ref, c_ref = solve_trig(jnp.array([M]), e)
    sinE, cosE = diff_solve_trig(jnp.array([M]), e)
    np.testing.assert_allclose(float(sinE[0]), float(s_ref[0]), atol=1e-12)
    np.testing.assert_allclose(float(cosE[0]), float(c_ref[0]), atol=1e-12)
