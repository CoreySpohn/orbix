"""Common equations for orbital mechanics."""

import jax.numpy as jnp

from orbix.constants import G, two_pi

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
        sinwcosi: Sine of the argument of periapsis times cosine of the inclination
        coswcosi: Cosine of the argument of periapsis times cosine of the inclination
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
