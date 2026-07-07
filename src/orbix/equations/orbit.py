"""Common equations for orbital mechanics."""

import jax.numpy as jnp
from hwoutils.constants import G, two_pi

two_pi_G = two_pi * G


def period_a(a, mu):
    """Orbital period from semi-major axis and standard gravitational parameter.

    Args:
        a: Array
            Semi-major axis
        mu: Array
            Standard gravitational parameter
    Returns:
        T: Array
            Orbital period
    """
    return 2 * jnp.pi * jnp.sqrt(a**3 / mu)


def period_n(n):
    """Orbital period from mean motion.

    Args:
        n: Array
            Mean motion

    Returns:
        T: Array
            Orbital period
    """
    return 2 * jnp.pi / n


def mean_motion(a, mu):
    """Mean motion from semi-major axis and standard gravitational parameter.

    Args:
        a: Array
            Semi-major axis
        mu: Array
            Standard gravitational parameter
    Returns:
        n: Array
            Mean motion
    """
    return jnp.sqrt(mu / a**3)


def period_to_sma(T, Ms):
    """Semi-major axis from orbital period via Kepler's third law.

    Args:
        T: Orbital period (days). Scalar or array.
        Ms: Stellar mass (kg). Scalar or array.

    Returns:
        a: Semi-major axis (AU). Scalar or array.
    """
    mu = G * Ms  # AU^3 / day^2
    return (mu * (T / two_pi) ** 2) ** (1.0 / 3.0)


def semi_amplitude(T, Ms, Mp, e, i):
    """Semi-amplitude of the radial velocity curve from base quantities.

    Args:
        T: Array
            Orbital period
        Ms: Array
            Mass of the star
        Mp: Array
            Mass of the planet
        e: Array
            Eccentricity
        i: Array
            Inclination
    Returns:
        K: Array
            Semi-amplitude of the radial velocity curve
    """
    return (
        (two_pi_G / T) ** (1 / 3.0)
        * (Mp * jnp.sin(i) / (Ms ** (2 / 3.0)))
        / jnp.sqrt(1 - e**2)
    )


def semi_amplitude_reduced(T, Ms, minimum_mass, sqrt_one_minus_e2):
    """Semi-amplitude of the radial velocity curve from pre-calculated quantities.

    Args:
        T: Array
            Orbital period
        Ms: Array
            Mass of the star
        minimum_mass: Array
            Mass of the planet multiplied by sin(i)
        sqrt_one_minus_e2: Array
            Square root of (1 - eccentricity^2)

    Returns:
        K: Array
            Semi-amplitude of the radial velocity curve
    """
    return (
        (two_pi_G / T) ** (1 / 3.0)
        * (minimum_mass / (Ms ** (2 / 3.0)))
        / sqrt_one_minus_e2
    )


def mean_anomaly_t0(t, n, M0, t0):
    """Mean anomaly at time t (can be vector) from epoch.

    Requires that all units are consistent and does NOT clip the mean anomaly
    to the range [0, 2pi).

    Args:
        t: Array
            Time
        n: Array
            Mean motion
        M0: Array
            Mean anomaly at epoch
        t0: Array
            Epoch
    Returns:
        M: Array
            Mean anomaly at time t
    """
    return n * (t - t0) + M0


def mean_anomaly_tp(t, n, tp):
    """Mean anomaly at time t (can be vector) from periapsis passage.

    Args:
        t: Array
            Time
        n: Array
            Mean motion
        tp: Array
            Time of periapsis passage
    Returns:
        M: Array
            Mean anomaly at time t
    """
    return n * (t - tp)


def AB_matrices(a, e, i, W, w):
    """Compute the A and B matrices for a given set of orbital elements.

    In keplertools Dmitry defines these as:
    "inertial frame components of perifocal frame unit vectors scaled
    by orbit semi-major and semi-minor axes."
    and I wouldn't dare disagree with him on this.

    Args:
        a: Array
            Semi-major axis
        e: Array
            Eccentricity
        i: Array
            Inclination
        W: Array
            Longitude of the ascending node
        w: Array
            Argument of periapsis
    Returns:
        A: jnp.ndarray
            A matrix
        B: jnp.ndarray
            B matrix
    """
    # Get the trig values
    sini, cosi = jnp.sin(i), jnp.cos(i)
    sinW, cosW = jnp.sin(W), jnp.cos(W)
    sinw, cosw = jnp.sin(w), jnp.cos(w)

    sqrt_one_minus_e2 = jnp.sqrt(1 - e**2)
    return AB_matrices_reduced(a, sqrt_one_minus_e2, sini, cosi, sinW, cosW, sinw, cosw)


def AB_matrices_reduced(a, sqrt_one_minus_e2, sini, cosi, sinW, cosW, sinw, cosw):
    """Compute the A and B matrices from the trig values of the orbital elements.

    Args:
        a: Semi-major axis
        sqrt_one_minus_e2: Square root of (1 - eccentricity^2)
        sini: Sine of the inclination
        cosi: Cosine of the inclination
        sinW: Sine of the longitude of the ascending node
        cosW: Cosine of the longitude of the ascending node
        sinw: Sine of the argument of periapsis
        cosw: Cosine of the argument of periapsis
    Returns:
        A: jnp.ndarray
            A matrix
        B: jnp.ndarray
            B matrix
    """
    sinwcosi = sinw * cosi
    coswcosi = cosw * cosi
    # Compute the A and B matrices as 3x1 JAX arrays
    A = a * jnp.asarray(
        [
            cosW * cosw - sinW * sinwcosi,
            sinW * cosw + cosW * sinwcosi,
            sinw * sini,
        ]
    )
    B = (
        a
        * sqrt_one_minus_e2
        * jnp.asarray(
            [
                -cosW * sinw - sinW * coswcosi,
                -sinW * sinw + cosW * coswcosi,
                cosw * sini,
            ]
        )
    )

    return A, B


def thiele_innes_constants(W, i, w):
    """Compute the Thiele-Innes constants from the orbital angles.

    Args:
        W: Longitude of the ascending node
        i: Inclination
        w: Argument of periapsis
    Returns:
        A: A constant
        B: B constant
        F: F constant
        G: G constant
    """
    cosi = jnp.cos(i)
    sinW, cosW = jnp.sin(W), jnp.cos(W)
    sinw, cosw = jnp.sin(w), jnp.cos(w)
    sinwcosi = sinw * cosi
    coswcosi = cosw * cosi
    return thiele_innes_constants_reduced(sinW, cosW, sinw, cosw, sinwcosi, coswcosi)


def thiele_innes_constants_reduced(sinW, cosW, sinw, cosw, sinwcosi, coswcosi):
    """Compute the Thiele-Innes constants from the orbital angles.

    Args:
        sinW: Sine of the longitude of the ascending node
        cosW: Cosine of the longitude of the ascending node
        sinw: Sine of the argument of periapsis
        cosw: Cosine of the argument of periapsis
        sinwcosi: Sine of the argument of periapsis times cosine of the inclination
        coswcosi: Cosine of the argument of periapsis times cosine of the inclination
    Returns:
        A: A constant
        B: B constant
        F: F constant
        G: G constant
    """
    A = cosW * cosw - sinW * sinwcosi
    B = sinW * cosw + cosW * sinwcosi
    F = -cosW * sinw - sinW * coswcosi
    G = -sinW * sinw + cosW * coswcosi
    return A, B, F, G


_TOL_E = 1e-9
_TOL_N = 1e-9
_ARC_EPS = 1e-7


def _safe_norm(x):
    """Vector norm with a finite (zero) gradient at ``x == 0``.

    ``jnp.linalg.norm`` has a NaN gradient at exactly the zero vector
    (``x / norm(x)`` is ``0 / 0`` there). This computes the same value
    but evaluates the norm on a nonzero stand-in whenever ``x`` is zero,
    so the local derivative is finite; the ``jnp.where`` then zeroes the
    corresponding cotangent, giving a well-defined (zero) gradient.
    """
    is_zero = jnp.all(x == 0.0)
    x_safe = jnp.where(is_zero, jnp.ones_like(x), x)
    return jnp.where(is_zero, 0.0, jnp.linalg.norm(x_safe))


def state_vector_to_keplerian(r, v, mu):
    """Convert state vectors (r, v) to Keplerian elements using JAX.

    Robust implementation handling edge cases (circular, equatorial,
    and non-bound orbits) using ``jnp.where`` for JIT compatibility.

    Unit-agnostic: ``r``, ``v``, and ``mu`` must be expressed in one
    consistent unit system (e.g. meters / m/s / m^3 s^-2, or the AU / day
    units used elsewhere in this library); the function does not enforce
    or convert any particular convention, and ``a`` is returned in the
    same length unit as ``r``.

    Args:
        r: Stellar-centric position vector ``(3,)``.
        v: Stellar-centric velocity vector ``(3,)``.
        mu: Gravitational parameter ``G * M_total``.

    Returns:
        tuple: ``(a, e, i, W, w, M)`` -- semi-major axis (same length
            unit as ``r``), eccentricity, inclination [rad], longitude of
            ascending node [rad], argument of periapsis [rad], mean
            anomaly [rad].
    """
    r_mag = jnp.linalg.norm(r)
    v_mag = jnp.linalg.norm(v)

    h = jnp.cross(r, v)
    h_mag = _safe_norm(h)
    h_mag_safe = jnp.where(h_mag > _TOL_N, h_mag, 1.0)

    i = jnp.arccos(jnp.clip(h[2] / h_mag_safe, -1.0 + _ARC_EPS, 1.0 - _ARC_EPS))

    k = jnp.array([0.0, 0.0, 1.0])
    n = jnp.cross(k, h)
    n_mag = _safe_norm(n)

    e_vec = (1 / mu) * ((v_mag**2 - mu / r_mag) * r - jnp.dot(r, v) * v)
    e = _safe_norm(e_vec)

    E_energy = 0.5 * v_mag**2 - mu / r_mag
    E_energy_safe = jnp.where(jnp.abs(E_energy) > 1e-10, E_energy, 1.0)
    a = jnp.where(jnp.abs(E_energy) > 1e-10, -mu / (2 * E_energy_safe), jnp.inf)

    TOL_E = _TOL_E
    TOL_I = _TOL_N
    is_circular = e < TOL_E
    is_inclined = n_mag > TOL_I

    n0_safe = jnp.where(is_inclined, n[0], 1.0)
    n1_safe = jnp.where(is_inclined, n[1], 0.0)
    W = jnp.where(is_inclined, jnp.arctan2(n1_safe, n0_safe), 0.0)

    e_safe = jnp.where(e > _TOL_E, e, 1.0)
    n_mag_safe = jnp.where(n_mag > _TOL_N, n_mag, 1.0)

    cos_w = jnp.where(
        is_circular | ~is_inclined,
        0.0,
        jnp.dot(n, e_vec) / (n_mag_safe * e_safe),
    )
    w_inclined = jnp.arccos(jnp.clip(cos_w, -1.0 + _ARC_EPS, 1.0 - _ARC_EPS))
    w_inclined = jnp.where(e_vec[2] < 0, 2 * jnp.pi - w_inclined, w_inclined)

    e_xy_mag = jnp.sqrt(e_vec[0] ** 2 + e_vec[1] ** 2)
    is_e_xy_safe = e_xy_mag > _TOL_E
    ex_safe = jnp.where(is_e_xy_safe, e_vec[0], 1.0)
    ey_safe = jnp.where(is_e_xy_safe, e_vec[1], 0.0)
    w_equatorial = jnp.arctan2(ey_safe, ex_safe)
    w_equatorial = w_equatorial * jnp.sign(h[2])

    w = jnp.where(
        is_circular,
        0.0,
        jnp.where(is_inclined, w_inclined, w_equatorial),
    )

    cos_nu = jnp.where(
        is_circular,
        0.0,
        jnp.dot(e_vec, r) / (e_safe * r_mag),
    )
    nu_elliptical = jnp.arccos(jnp.clip(cos_nu, -1.0 + _ARC_EPS, 1.0 - _ARC_EPS))
    nu_elliptical = jnp.where(
        jnp.dot(r, v) < 0,
        2 * jnp.pi - nu_elliptical,
        nu_elliptical,
    )

    cos_u = jnp.where(
        ~is_inclined,
        0.0,
        jnp.dot(n, r) / (n_mag_safe * r_mag),
    )
    u_inclined = jnp.arccos(jnp.clip(cos_u, -1.0 + _ARC_EPS, 1.0 - _ARC_EPS))
    u_inclined = jnp.where(r[2] < 0, 2 * jnp.pi - u_inclined, u_inclined)

    r_xy_mag = jnp.sqrt(r[0] ** 2 + r[1] ** 2)
    is_r_xy_safe = r_xy_mag > _TOL_N
    rx_safe = jnp.where(is_r_xy_safe, r[0], 1.0)
    ry_safe = jnp.where(is_r_xy_safe, r[1], 0.0)
    nu_equatorial = jnp.arctan2(ry_safe, rx_safe)
    nu_equatorial = nu_equatorial * jnp.sign(h[2])

    nu = jnp.where(
        is_circular,
        jnp.where(is_inclined, u_inclined, nu_equatorial),
        nu_elliptical,
    )

    W = W % (2 * jnp.pi)
    w = w % (2 * jnp.pi)
    nu = nu % (2 * jnp.pi)

    E = jnp.arctan2(jnp.sqrt(1 - e**2) * jnp.sin(nu), e + jnp.cos(nu))
    M = E - e * jnp.sin(E)
    M = M % (2 * jnp.pi)
    M = jnp.where(e < 1.0, M, jnp.nan)

    return a, e, i, W, w, M
