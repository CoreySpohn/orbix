"""JAX-compiled functions to compute the eccentric anomaly based on the orvara method.

The primary entrance point is the `solve_E` function, which chooses the correct
method based on the eccentricity value. When repeated calls will be made with
the same eccentricity value, it is recommended to use the `get_E_solver`
function which will return a JIT-compiled function with the eccentricity value
fixed.

If the sine and cosine of the eccentric anomaly are also required, such as in
RV calculations, the `solve_E_trig` function can be used. This function returns
the eccentric anomaly along with the sine and cosine of the eccentric anomaly.
Similarly, a JIT-compiled function can be obtained using the `get_E_trig_solver`
and a specific eccentricity value to speed up repeated calculations.

If high precision is not required, the `guess_E` function returns the initial
guess for the eccentric anomaly calculated with a lookup table of polynomials.
A compiled form for a specific `e` value can be obtained using the
`get_E_guesser` function.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

# Define coefficients used in the shortsin function
if3 = 1.0 / 6.0
if5 = 1.0 / (6.0 * 20.0)
if7 = 1.0 / (6.0 * 20.0 * 42.0)
if9 = 1.0 / (6.0 * 20.0 * 42.0 * 72.0)
if11 = 1.0 / (6.0 * 20.0 * 42.0 * 72.0 * 110.0)
if13 = 1.0 / (6.0 * 20.0 * 42.0 * 72.0 * 110.0 * 156.0)
if15 = 1.0 / (6.0 * 20.0 * 42.0 * 72.0 * 110.0 * 156.0 * 210.0)
pi = jnp.pi
pi_d_12 = pi / 12.0
pi_d_6 = pi / 6.0
pi_d_4 = pi / 4.0
pi_d_3 = pi / 3.0
fivepi_d_12 = 5.0 * pi / 12.0
pi_d_2 = pi / 2.0
sevenpi_d_12 = 7.0 * pi / 12.0
twopi_d_3 = 2.0 * pi / 3.0
threepi_d_4 = 3.0 * pi / 4.0
fivepi_d_6 = 5.0 * pi / 6.0
elevenpi_d_12 = 11.0 * pi / 12.0
two_pi = 2.0 * pi


def create_E_func_multi_e(e_vals, func):
    """Creates E solvers for all provided `e` values using the provided function."""
    # Assure that the provided e_range is linearly spaced
    funcs = {e: func(e) for e in e_vals}

    def E_calc(e, M):
        return funcs[e](M)

    return E_calc


def get_E_solver(e: float):
    """Get the eccentric anomaly solver for a specific eccentricity value."""
    solve_E_partial = partial(solve_E, e=e)
    solve_E_jitted = jax.jit(solve_E_partial)
    return solve_E_jitted


@jax.jit
def solve_E(M, e):
    # Not handling e == 0 case for now, would look something like this:
    # e_ind = jnp.select([e == 0, e < 0.78], [0, 1], default=2)
    # return lax.switch(e_ind, [identity_solver, le_E, he_E], M, e)
    # where identity_solver returns M as E
    M = jnp.mod(M, two_pi)
    return lax.cond(e < 0.78, le_E, he_E, M, e)


def get_E_trig_solver(e: float):
    """Get the eccentric anomaly solver for a specific eccentricity value."""
    solve_E_trig_partial = partial(solve_E_trig, e=e)
    solve_E_trig_jitted = jax.jit(solve_E_trig_partial)
    return solve_E_trig_jitted


@jax.jit
def solve_E_trig(M, e):
    return lax.cond(e < 0.78, le_E_trig, he_E_trig, M, e)


def get_E_guesser(e: float):
    """Creates a JIT-compiled function for a specific e value, treating e as static.

    Args:
        e (float): The eccentricity value.

    Returns:
        function: A JIT-compiled function that takes M as input.
    """
    # JIT-compile E_guess with e treated as static
    guess_E_partial = partial(guess_E, e=e)
    guess_E_jitted = jax.jit(guess_E_partial)

    # Return a lambda function with e fixed
    return guess_E_jitted


@jax.jit
def guess_E(M: jnp.ndarray, e: float):
    """Gives the initial guess based on the constructed polynomials."""
    bounds, coeffs = getbounds(e)
    Esigns, _M = cut_M(M)
    init_E = init_E_coeffs(_M, bounds, coeffs)
    return jnp.fmod(Esigns * init_E + two_pi, two_pi)


def get_E_lookup(e, n=1800):
    M_vals = jnp.linspace(0, two_pi, n, endpoint=False)
    E_vals = solve_E(M_vals, e)
    dM = M_vals[1] - M_vals[0]
    inv_dM = 1.0 / dM
    n_int = jnp.int32(n)

    @jax.jit
    def E_lookup(M):
        return E_vals[(M * inv_dM).astype(jnp.int32) % n_int]

    return E_lookup


def get_E_lookup_d(e, n=1800):
    # Linearly spaced mean anomaly values
    M_vals = jnp.linspace(0, two_pi, n, endpoint=False)
    dM = M_vals[1] - M_vals[0]
    # Use the solve_E function to get the real eccentric anomaly values
    # and the derivative of E (with respect to the array indices)
    E_vals = solve_E(M_vals, e)
    dE_dM = 1 / (1 - e * jnp.cos(E_vals))
    dE_dind = dE_dM * dM

    inv_dM = 1.0 / dM
    n_int = jnp.int32(n)

    @jax.jit
    def interp_E(M):
        # Scale M to index space
        ind_M = M * inv_dM
        # Get integer part of the index
        ind_M_int = ind_M.astype(jnp.int32)
        # Get the fractional part of the index
        dind = ind_M - ind_M_int
        # Wrap to [0, 2pi) indices with modulo n
        final_inds = ind_M_int % n_int
        # Linear interpolation
        return E_vals[final_inds] + dE_dind[final_inds] * dind

    return interp_E


def get_E_lookup_hermite(e, n=1800):
    # Generate linearly spaced mean anomaly values
    M_vals = jnp.linspace(0, two_pi, n, endpoint=False)
    # Compute the step size in mean anomaly per index
    dM = M_vals[1] - M_vals[0]
    # Solve for eccentric anomaly for each mean anomaly
    E_vals = solve_E(M_vals, e)
    # Compute the derivative of E with respect to the index
    dE_dind = jnp.gradient(E_vals)

    @jax.jit
    def E_lookup(M):
        # Wrap M to the range [0, two_pi]
        _M = jnp.mod(M, two_pi)
        # Determine the fractional index and the integer part
        # dind, ind_float = jnp.modf(_M / dind_dM)
        dind, ind_float = jnp.modf(jnp.true_divide(_M, dM))
        ind_int = ind_float.astype(int)
        # Handle wrap-around for the next index
        ind_next = (ind_int + 1) % n

        # Retrieve function values and derivatives at the surrounding indices
        E_i = E_vals[ind_int]
        E_ip1 = E_vals[ind_next]
        dE_i = dE_dind[ind_int]
        dE_ip1 = dE_dind[ind_next]

        # Hermite basis functions
        h00 = 2 * dind**3 - 3 * dind**2 + 1
        h10 = dind**3 - 2 * dind**2 + dind
        h01 = -2 * dind**3 + 3 * dind**2
        h11 = dind**3 - dind**2

        # Perform Hermite interpolation
        E_interp = h00 * E_i + h10 * dE_i + h01 * E_ip1 + h11 * dE_ip1

        return E_interp

    return E_lookup


def shortsin(x):
    """Approximates the sine function using a short polynomial.

    This is only valid between [0, π].
    """
    x2 = x * x
    return x * (
        1
        - x2
        * (
            if3
            - x2
            * (if5 - x2 * (if7 - x2 * (if9 - x2 * (if11 - x2 * (if13 - x2 * if15)))))
        )
    )


def cut_M(M: jnp.ndarray):
    """Cut M to be between 0 and pi.

    Also returns the sign of the eccentric anomaly.

    Args:
        M (jnp.ndarray):
            Mean anomalies (rad). Shape: (n,).

    Returns:
        Esigns (jnp.ndarray):
            Sign of the eccentric anomaly. Shape: (n,).
        _M (jnp.ndarray):
            Modified mean anomalies. Shape: (n,).
    """
    mask = M > pi
    Esigns = jnp.where(mask, -1, 1)
    _M = jnp.where(mask, two_pi - M, M)
    return Esigns, _M


def getbounds(e: float):
    """Create bounds and coefficients for the eccentric anomaly polynomial.

    Args:
        e (float): Eccentricity

    Returns:
        tuple:
            bounds (jnp.ndarray):
                Array of bounds for the eccentric anomaly intervals. Shape: (13,)
            coeffs (jnp.ndarray)
                Lookup table containing coefficients for the Taylor series
                expansion. Shape: (13, 6)
    """
    # Compute scaled constants
    g2s_e = 0.2588190451025207623489 * e
    g3s_e = 0.5 * e
    g4s_e = 0.7071067811865475244008 * e
    g5s_e = 0.8660254037844386467637 * e
    g6s_e = 0.9659258262890682867497 * e

    g2c_e = g6s_e
    g3c_e = g5s_e
    g4c_e = g4s_e
    g5c_e = g3s_e
    g6c_e = g2s_e

    # Initialize bounds array
    bounds = jnp.array(
        [
            0.0,
            pi_d_12 - g2s_e,
            pi_d_6 - g3s_e,
            pi_d_4 - g4s_e,
            pi_d_3 - g5s_e,
            fivepi_d_12 - g6s_e,
            pi_d_2 - e,
            sevenpi_d_12 - g6s_e,
            twopi_d_3 - g5s_e,
            threepi_d_4 - g4s_e,
            fivepi_d_6 - g3s_e,
            elevenpi_d_12 - g2s_e,
            pi,
        ]
    )

    # Initialize coeffs array with shape (13, 6)
    ai1 = jnp.array(
        [
            1.0 / (1.0 - e),
            1.0 / (1.0 - g2c_e),
            1.0 / (1.0 - g3c_e),
            1.0 / (1.0 - g4c_e),
            1.0 / (1.0 - g5c_e),
            1.0 / (1.0 - g6c_e),
            1.0,
            1.0 / (1.0 + g6c_e),
            1.0 / (1.0 + g5c_e),
            1.0 / (1.0 + g4c_e),
            1.0 / (1.0 + g3c_e),
            1.0 / (1.0 + g2c_e),
            1.0 / (1.0 + e),
        ]
    )
    ai2 = (
        jnp.array(
            [
                0,
                -0.5 * g2s_e,
                -0.5 * g3s_e,
                -0.5 * g4s_e,
                -0.5 * g5s_e,
                -0.5 * g6s_e,
                -0.5 * e,
                -0.5 * g6s_e,
                -0.5 * g5s_e,
                -0.5 * g4s_e,
                -0.5 * g3s_e,
                -0.5 * g2s_e,
                0,
            ]
        )
        * ai1**3
    )

    # Index of the lower bound of the interval
    i = jnp.arange(12)
    # Set the 0th coefficient of the polynomials
    ai0 = i * pi_d_12

    # Set the 3rd, 4th, and 5th coefficients of the polynomials with array
    # operations since they are solved based on the 1st and 2nd coefficients
    ii = i + 1
    idx = 1.0 / (bounds[ii] - bounds[i])
    B0 = idx * (-ai2[i] - idx * (ai1[i] - idx * pi_d_12))
    B1 = idx * (-2.0 * ai2[i] - idx * (ai1[i] - ai1[ii]))
    B2 = idx * (ai2[ii] - ai2[i])
    ai3 = B2 - 4.0 * B1 + 10.0 * B0
    ai4 = (-2.0 * B2 + 7.0 * B1 - 15.0 * B0) * idx
    ai5 = (B2 - 3.0 * B1 + 6.0 * B0) * idx**2
    coeffs = jnp.stack([ai0, ai1[:-1], ai2[:-1], ai3, ai4, ai5], axis=1)

    return bounds, coeffs


def init_E_poly(M, e):
    """Initial guess for the eccentric anomaly.

    Calculates the initial guess for the eccentric anomaly based on the mean
    anomaly and eccentricity. Translated from the C implementation into JAX.

    Parameters:
        M (jnp.ndarray):
            Mean anomaly in radians.
        e (float):
            Eccentricity of the orbit.

    Returns:
        jnp.ndarray:
            Initial estimate of the eccentric anomaly in radians.
    """
    ome = 1.0 - e
    sqrt_ome = lax.sqrt(ome)
    chi = M / (sqrt_ome * ome)
    Lam = lax.sqrt(8.0 + 9.0 * chi**2)
    S = lax.cbrt(Lam + 3.0 * chi)
    S_squared = S * S
    sigma = 6.0 * chi / (2.0 + S_squared + 4.0 / S_squared)
    s2 = sigma * sigma
    denom = s2 + 2.0
    E = sigma * (
        1.0
        + s2
        * ome
        * (
            (s2 + 20.0) / (60.0 * denom)
            + s2
            * ome
            * (s2**3 + 25.0 * s2**2 + 340.0 * s2 + 840.0)
            / (1400.0 * denom**3)
        )
    )
    return E * sqrt_ome


def init_E_coeffs(M: jnp.ndarray, bounds: jnp.ndarray, coeffs: jnp.ndarray):
    """Create the initial guess for the eccentric anomaly using the polynomials."""
    # j_inds = jnp.searchsorted(bounds, M, side="right") - 1
    j_inds = jnp.digitize(M, bounds) - 1
    dx = M - bounds[j_inds]
    return coeffs[j_inds, 0] + dx * (
        coeffs[j_inds, 1]
        + dx
        * (
            coeffs[j_inds, 2]
            + +dx
            * (coeffs[j_inds, 3] + dx * (coeffs[j_inds, 4] + dx * coeffs[j_inds, 5]))
        )
    )


def dE_num_denom(M, E, e_inv, sinE, cosE):
    """Compute the numerator and denominator for dE."""
    num = (M - E) * e_inv + sinE
    denom = e_inv - cosE
    return num, denom


def dE_2nd(M, E, e_inv, sinE, cosE):
    """Compute the second order approximation of dE."""
    num, denom = dE_num_denom(M, E, e_inv, sinE, cosE)
    return num * denom / (denom * denom + 0.5 * sinE * num)


def dE_3rd(M, E, e_inv, sinE, cosE):
    """Compute the third order approximation of dE."""
    num, denom = dE_num_denom(M, E, e_inv, sinE, cosE)
    dE = (
        num
        * (denom * denom + 0.5 * num * sinE)
        / (denom * denom * denom + num * (denom * sinE + if3 * num * cosE))
    )
    return dE


def compute_dE_single(M, init_E_val, e_inv_val, sinE_val, cosE_val):
    """Computes dE for a single element based on the condition M > 0.4.

    Args:
        M (float): Single element from _M.
        init_E_val (float): Corresponding element from init_E.
        e_inv_val (float): Inverse of eccentricity.
        sinE_val (float): Sine of E.
        cosE_val (float): Cosine of E.

    Returns:
        float: Computed dE for the element.
    """
    return jax.lax.cond(
        M > 0.4, dE_2nd, dE_3rd, M, init_E_val, e_inv_val, sinE_val, cosE_val
    )


# Vectorize the compute_dE_single function
compute_dE_vectorized = jax.vmap(
    compute_dE_single, in_axes=(0, 0, None, 0, 0), out_axes=0
)


def le_E(M: jnp.ndarray, e: float):
    """Inverts Kepler's time equation for elliptical orbits using Orvara's method.

    Args:
        M (jnp.ndarray): Mean anomalies (rad). Shape: (n,).
        e (float): Eccentricity. Must satisfy 0 <= e < 1.

    Returns:
        - E (jnp.ndarray): Eccentric anomalies (rad). Shape: (n,).
    """
    # Get bounds and coeffs
    bounds, coeffs = getbounds(e)
    # Cut M to be between 0 and pi
    Esigns, _M = cut_M(M)

    # Get initial guess
    init_E = init_E_coeffs(_M, bounds, coeffs)
    sinE, cosE = fast_sinE_cosE(init_E)
    dE = dE_2nd(_M, init_E, 1.0 / e, sinE, cosE)
    E = jnp.fmod(Esigns * (init_E + dE) + two_pi, two_pi)
    return E


def le_E_trig(M: jnp.ndarray, e: float):
    """Inverts Kepler's time equation for elliptical orbits using Orvara's method.

    Also returns the sine and cosine of the eccentric anomaly.

    Args:
        M (jnp.ndarray): Mean anomalies (rad). Shape: (n,).
        e (float): Eccentricity. Must satisfy 0 <= e < 1.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            - E (jnp.ndarray): Eccentric anomalies (rad). Shape: (n,).
            - sinE (jnp.ndarray): Sine of eccentric anomalies (rad). Shape: (n,).
            - cosE (jnp.ndarray): Cosine of eccentric anomalies (rad). Shape: (n,).
    """
    # Get bounds and coeffs
    bounds, coeffs = getbounds(e)
    # Cut M to be between 0 and pi
    Esigns, _M = cut_M(M)

    # Get initial guess
    init_E = init_E_coeffs(_M, bounds, coeffs)
    sinE, cosE = fast_sinE_cosE(init_E)
    dE = dE_2nd(_M, init_E, 1.0 / e, sinE, cosE)
    E = jnp.fmod(Esigns * (init_E + dE) + two_pi, two_pi)
    sinE = Esigns * (sinE * (1.0 - 0.5 * dE * dE) + dE * cosE)
    cosE = cosE * (1.0 - 0.5 * dE * dE) - dE * sinE
    return E, sinE, cosE


def he_E(M: jnp.ndarray, e: float):
    """Inverts Kepler's time equation for elliptical orbits with e > 0.78.

    Args:
        M (jnp.ndarray): Mean anomalies (rad). Shape: (n,).
        e (float): Eccentricity. Must satisfy 0 <= e < 1.

    Returns:
        - E (jnp.ndarray): Eccentric anomalies (rad). Shape: (n,).
    """
    bounds, coeffs = getbounds(e)
    e_inv = 1.0 / e
    # Cut M to be between 0 and pi
    Esigns, _M = cut_M(M)

    # Get initial guess
    # TODO: Come up with a way to do this without evaluating both functions
    cond1 = (2 * _M + (1 - e)) > 0.2
    init_E = jnp.where(cond1, init_E_coeffs(_M, bounds, coeffs), init_E_poly(_M, e))

    sinE, cosE = fast_sinE_cosE(init_E)
    dE = compute_dE_vectorized(_M, init_E, e_inv, sinE, cosE)
    E = jnp.fmod(Esigns * (init_E + dE) + two_pi, two_pi)
    return E


def he_E_trig(M: jnp.ndarray, e: float):
    """Inverts Kepler's time equation for elliptical orbits with e > 0.78.

    Args:
        M (jnp.ndarray): Mean anomalies (rad). Shape: (n,).
        e (float): Eccentricity. Must satisfy 0 <= e < 1.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            - E (jnp.ndarray): Eccentric anomalies (rad). Shape: (n,).
            - sinE (jnp.ndarray): Sine of eccentric anomalies (rad). Shape: (n,).
            - cosE (jnp.ndarray): Cosine of eccentric anomalies (rad). Shape: (n,).
    """
    bounds, coeffs = getbounds(e)
    e_inv = 1.0 / e
    # Cut M to be between 0 and pi
    Esigns, _M = cut_M(M)

    # Get initial guess
    # TODO: Come up with a way to do this without evaluating both functions
    cond1 = (2 * _M + (1 - e)) > 0.2
    init_E = jnp.where(cond1, init_E_coeffs(_M, bounds, coeffs), init_E_poly(_M, e))

    sinE, cosE = fast_sinE_cosE(init_E)
    dE = compute_dE_vectorized(_M, init_E, e_inv, sinE, cosE)
    dEsq_d6 = dE**2 * if3
    E = jnp.fmod(Esigns * (init_E + dE) + two_pi, two_pi)
    sinE = Esigns * (sinE * (1 - 3 * dEsq_d6) + dE * (1 - dEsq_d6) * cosE)
    cosE = cosE * (1 - 3 * dEsq_d6) - dE * (1 - dEsq_d6) * sinE
    return E, sinE, cosE


def Etrig_1(E):
    """When E <= pi_d_4."""
    sinE = shortsin(E)
    cosE = lax.sqrt(1.0 - sinE**2)
    return sinE, cosE


def Etrig_2(E):
    """When E > pi_d_4 and E < three_pi_d_4."""
    cosE = shortsin(pi_d_2 - E)
    sinE = lax.sqrt(1.0 - cosE**2)
    return sinE, cosE


def Etrig_3(E):
    """When E > pi_d_2 and E > three_pi_d_4."""
    sinE = shortsin(pi - E)
    cosE = -lax.sqrt(1.0 - sinE**2)
    return sinE, cosE


def Etrig(i, E):
    """Apply the correct trigonometric function based on the index."""
    return lax.switch(i, [Etrig_1, Etrig_2, Etrig_3], E)


def fast_sinE_cosE(E):
    """Compute the sine and cosine of the eccentric anomaly using shortsin."""
    # Vectorize the computation across all elements
    Ei = jnp.select([E <= pi_d_4, E < threepi_d_4], [0, 1], default=2)
    sinE, cosE = jax.vmap(Etrig, in_axes=(0, 0))(Ei, E)
    return sinE, cosE


def identity_solver(M, e):
    """Returns M as E when e is 0."""
    return M
