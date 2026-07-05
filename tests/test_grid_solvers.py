"""Accuracy and contract tests for the grid shortcut solvers."""

import jax
import jax.numpy as jnp
import pytest

from orbix.kepler.core import E_solve
from orbix.kepler.shortcuts.grid import get_grid_solver

jax.config.update("jax_enable_x64", True)


def test_bilinear_high_e_no_accuracy_cliff():
    """The top e cell must interpolate, not silently use the wrong row."""
    solver = get_grid_solver(level="scalar", E=True, trig=False, jit=True)
    M = jnp.float64(1.0)
    # e in the top grid cell (defaults n_e=512 -> cell width ~2e-3)
    e_top = 0.9985
    e_ref = 0.995  # a mid-grid e for the error scale we should match
    err_top = abs(
        float(solver(M, jnp.float64(e_top)) - E_solve(jnp.array([1.0]), e_top)[0])
    )
    err_ref = abs(
        float(solver(M, jnp.float64(e_ref)) - E_solve(jnp.array([1.0]), e_ref)[0])
    )
    # Pre-fix: err_top ~ 1.35e-3 vs err_ref ~ 5e-7 (a ~3500x cliff).
    # Post-fix both are interpolation-limited; allow one order of freedom
    # for the larger dE/de near e=1.
    assert err_top < 100 * max(err_ref, 1e-7)
    assert err_top < 1e-4


def test_get_grid_solver_rejects_no_output():
    """Requesting neither E nor trig output raises ValueError, not UnboundLocalError."""
    with pytest.raises(ValueError, match="E|trig"):
        get_grid_solver(E=False, trig=False)


def test_negative_M_wraps():
    """The bilinear grid path must wrap negative M like E_solve does."""
    solver = get_grid_solver(level="scalar", E=True, trig=False, jit=True)
    e = jnp.float64(0.3)
    M_neg = jnp.float64(-0.001)
    truth = E_solve(jnp.array([jnp.mod(-0.001, 2 * jnp.pi)]), 0.3)[0]
    got = solver(M_neg, e)
    assert abs(float(got - truth)) < 1e-4


def test_negative_M_wraps_linear_kind():
    """The linear-kind grid path must also wrap negative M like E_solve does."""
    solver = get_grid_solver(
        level="scalar", kind="linear", E=True, trig=False, jit=True
    )
    e = jnp.float64(0.3)
    M_neg = jnp.float64(-0.001)
    truth = E_solve(jnp.array([jnp.mod(-0.001, 2 * jnp.pi)]), 0.3)[0]
    got = solver(M_neg, e)
    assert abs(float(got - truth)) < 1e-4
