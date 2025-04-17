"""Factory functions for creating Kepler solvers."""

from functools import partial

import jax

from orbix.kepler.core import E_solve, E_solve_trig
from orbix.kepler.shortcuts.guess import E_guess


def discrete_e_kepler_factory(e_vals, func):
    """Creates E solvers for a discrete set of `e` values using the provided function.

    The main way to use this is to provide the func as one of:
    - fixed_e_solver
    - fixed_e_trig_solver
    - fixed_e_guesser
    since those are optimized for a single eccentricity value.

    Args:
        e_vals (jnp.ndarray): The eccentricity values to create solvers for.
        func (function): The function to create solvers for.

    Returns:
        function: A function that takes an eccentricity and mean anomaly and
                  returns the eccentric anomaly.
    """
    # Create a dictionary of functions for each eccentricity value
    funcs = {e: func(e) for e in e_vals}

    # Closure that contains the dictionary of functions and can chooses
    # the correct function based on the eccentricity value
    def E_calc(e, M):
        """Call the correct single-eccentricity function."""
        return funcs[e](M)

    return E_calc


def gen_fixed_e_solver(base_func, e: float):
    """Creates a JIT-compiled solver with a fixed eccentricity value.

    Args:
        base_func: The base function to compile (E_solve, E_solve_trig, or E_guess)
        e: The eccentricity value

    Returns:
        A JIT-compiled function that takes mean anomaly as input
    """
    partial_func = partial(base_func, e=e)
    return jax.jit(partial_func)


def fixed_e_solver(e: float):
    """Create a jit-compiled form of solve_E with a specific eccentricity value."""
    return gen_fixed_e_solver(E_solve, e)


def fixed_e_trig_solver(e: float):
    """Create a jit-compiled form of solve_E_trig with a specific eccentricity value."""
    return gen_fixed_e_solver(E_solve_trig, e)


def fixed_e_guesser(e: float):
    """Create a jit-compiled form of E_guess with a specific eccentricity value."""
    return gen_fixed_e_solver(E_guess, e)
