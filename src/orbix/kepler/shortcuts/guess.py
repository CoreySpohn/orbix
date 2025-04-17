"""Methods for generating initial guesses for the eccentric anomaly.

`E_guess` returns the initial guesses for the eccentric anomaly calculated from
the constructed polynomials defined by RPP and orvara, based on the mean
anomaly and eccentricity. There is also a jit-compiled version available
as `E_guess_jit`.
"""

import jax
import jax.numpy as jnp

from orbix.constants import two_pi
from orbix.kepler.core import cut_M, getbounds, init_E_coeffs


def E_guess(M: jnp.ndarray, e: float):
    """Gives the initial guess based on the constructed polynomials.

    Args:
        M (jnp.ndarray): Mean anomaly. Shape: (n,).
        e (float): Eccentricity.

    Returns:
        E (jnp.ndarray): Eccentric anomaly. Shape: (n,).
    """
    bounds, coeffs = getbounds(e)
    Esigns, _M = cut_M(M)
    init_E = init_E_coeffs(_M, bounds, coeffs)
    return jnp.fmod(Esigns * init_E + two_pi, two_pi)


E_guess_jit = jax.jit(E_guess)
