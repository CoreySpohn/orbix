"""Common equations for orbital mechanics."""

import jax.numpy as jnp
from hwoutils.constants import G, two_pi
from hwoutils.numerics import COARSEST, eps, roundoff
from hwoutils.precision import like
from jax import lax

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


def thiele_innes_constants(W, i, w):
    """Compute the Thiele-Innes constants from the orbital angles.

    Keep the classical letters: A and F describe north, B and G east.
    These scalar names are independent of the east-first public vector order.

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


def thiele_innes_to_elements(A, B, F, G):
    """Invert classical Thiele-Innes constants to reduced Campbell elements.

    Inputs broadcast over arbitrary batch shapes. A,F are north and B,G east,
    with all four constants multiplied by the same positive semi-major axis.
    F,G must NOT include sqrt(1-e^2); divide it out before calling if needed.
    The output is (a, cos_i, W, w), retaining cos_i to avoid the singular
    derivative of arccos at face-on inclinations.

    Angles are radians: W is the ascending (receding) node position angle,
    north through east, and w runs from that node in the direction of motion.
    Both are returned in [0, 2*pi). The principal atan2 values of W+w and W-w
    select one of the two sky-identical mirrors; adding pi to both W and w
    selects the other. This branch has angular seams.

    At i=0 only W+w is identifiable; at i=pi only W-w is identifiable.
    Choose W=0 and put the identifiable angle in w in either case. Components
    within 8 eps times a of these singularities use that gauge. At e=0 the
    input constants still fix a reference direction, but w is not a physical
    periapsis; preserve that direction and its caller-supplied phase together.
    A zero block returns (0, 0, 0, 0), with orientation unidentified.
    JIT and autodiff are supported away from singularities and branch seams;
    derivatives of the unidentified angles at singularities are not physical.

    Args:
        A: North coefficient of the first in-plane axis, in length units.
        B: East coefficient of the first in-plane axis, in length units.
        F: North coefficient of the second in-plane axis, in length units.
        G: East coefficient of the second in-plane axis, in length units.

    Returns:
        Tuple (a, cos_i, W, w), each with the broadcast input shape.
        a has the same length units as the constants, cos_i is in [-1, 1].
    """
    A, B, F, G = jnp.broadcast_arrays(*map(jnp.asarray, (A, B, F, G)))
    plus_x, plus_y = A + G, B - F
    minus_x, minus_y = A - G, B + F
    r_plus = jnp.hypot(plus_x, plus_y)
    r_minus = jnp.hypot(minus_x, minus_y)
    total = r_plus + r_minus
    a = total / 2.0
    one, zero = jnp.ones_like(total), jnp.zeros_like(total)
    cos_i = (r_plus - r_minus) / jnp.where(total > 0.0, total, one)
    tol = 8 * jnp.finfo(a.dtype).eps * a
    prograde = r_minus <= tol
    retrograde = r_plus <= tol
    sigma = jnp.arctan2(
        jnp.where(retrograde, zero, plus_y),
        jnp.where(retrograde, one, plus_x),
    )
    phi = jnp.arctan2(
        jnp.where(prograde, zero, minus_y),
        jnp.where(prograde, one, minus_x),
    )
    face_on = prograde | retrograde
    turn = like(two_pi, sigma)
    W = jnp.where(face_on, zero, (sigma + phi) / 2.0) % turn
    w = jnp.where(face_on, jnp.where(prograde, sigma, -phi), (sigma - phi) / 2.0) % turn
    return a, cos_i, W, w


def thiele_innes_constants_reduced(sinW, cosW, sinw, cosw, sinwcosi, coswcosi):
    """Compute the Thiele-Innes constants from the orbital angles.

    Keep the classical letters: A and F describe north, B and G east.
    These scalar names are independent of the east-first public vector order.

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


# Eight roundings in the normalized cross product and norms define an equatorial node.
_NODE_RESOLUTION = 8 * eps(COARSEST)


def _safe_norm(x):
    """Vector norm with a finite (zero) gradient at ``x == 0``.

    ``jnp.linalg.norm`` has a NaN gradient at exactly the zero vector
    (``x / norm(x)`` is ``0 / 0`` there). This computes the same value
    but evaluates the norm on a nonzero stand-in whenever ``x`` is zero,
    so the local derivative is finite; the ``jnp.where`` then zeroes the
    corresponding cotangent, giving a well-defined (zero) gradient.
    """
    scale = lax.stop_gradient(jnp.max(jnp.abs(x)))
    is_zero = scale == 0.0
    scaled = x / jnp.where(is_zero, jnp.ones_like(scale), scale)
    norm = jnp.linalg.norm(jnp.where(is_zero, jnp.ones_like(x), scaled))
    return jnp.where(is_zero, jnp.zeros_like(scale), scale * norm)


def _plane_decision_vector(x):
    """Put a Cartesian vector on the common grid without changing its direction."""
    exponent = jnp.frexp(jnp.max(jnp.abs(x)))[1]
    return jnp.ldexp(x, -exponent).astype(COARSEST)


def _oriented_angle(cosine, sine):
    """Angle from its cosine and sine through a guarded atan2.

    Recovering the angle with arccos amplifies rounding in the cosine near
    the axes, where its derivative diverges; atan2 uses both components and
    keeps the derivative well conditioned in float32. Scale the pair to
    avoid underflow in the atan2 derivative of a tiny eccentricity vector.
    Only a zero pair takes the zero-angle convention.
    """
    scale = lax.stop_gradient(jnp.maximum(jnp.abs(cosine), jnp.abs(sine)))
    is_zero = scale == 0.0
    one = jnp.ones_like(scale)
    scale = jnp.where(is_zero, one, scale)
    return jnp.arctan2(
        jnp.where(is_zero, jnp.zeros_like(scale), sine / scale),
        jnp.where(is_zero, one, cosine / scale),
    )


def state_vector_to_keplerian(r, v, mu):
    """Convert state vectors (r, v) to Keplerian elements using JAX.

    Both vectors use the right-handed (east, north, away) frame. Positive
    dz/dt is the body's radial velocity relative to the star (receding).
    Omega is the PA of the ascending (receding) node, north through east;
    i < 90 deg gives increasing PA, and omega is measured from that node
    in the direction of motion.

    Robust implementation handling edge cases (circular, equatorial,
    and non-bound orbits) using ``jnp.where`` for JIT compatibility.

    The direction of a nonzero eccentricity vector sets periapsis. At
    zero eccentricity, periapsis is zero and the anomaly is the argument
    of latitude from the node, or true longitude if also equatorial.
    A state is radial when every angular momentum component is within
    roundoff on a shared float32 grid. The node-length fraction is compared
    with eight float32 epsilons on that grid to identify an equatorial plane.
    Equatorial orbits use ``W = 0``; ``w`` (or the circular true longitude)
    is measured from north toward east and multiplied by the sign of
    ``h_z`` in the classical (north, east, away) component order.

    Unit-agnostic: ``r``, ``v``, and ``mu`` must be expressed in one
    consistent unit system (e.g. meters / m/s / m^3 s^-2, or the AU / day
    units used elsewhere in this library); the function does not enforce
    or convert any particular unit system, and ``a`` is returned in the
    same length unit as ``r``. The degeneracy thresholds scale with the
    terms that produce the energy, angular momentum, and node vector, so
    the elements do not depend on the choice of units. The semi-major axis
    is infinite when the specific orbital energy is within roundoff of zero.

    Args:
        r: Star-centered (east, north, away) position vector ``(3,)``,
            typically AU.
        v: Velocity in the same frame, ``(3,)``, typically AU/day.
        mu: Gravitational parameter ``G * M_total``.

    Returns:
        tuple: ``(a, e, i, W, w, M)`` -- semi-major axis (same length
            unit as ``r``), eccentricity, inclination [rad], longitude of
            ascending node [rad], argument of periapsis [rad], mean
            anomaly [rad].
    """
    # The existing element formulas use classical (north, east, away)
    # components, including the cross products and atan2 for the node PA.
    classical_order = jnp.array([1, 0, 2])
    r = jnp.asarray(r)[classical_order]
    v = jnp.asarray(v)[classical_order]
    # Exact power-of-two rescaling keeps the common-grid cross product in range.
    coarse_r, coarse_v = _plane_decision_vector(r), _plane_decision_vector(v)
    coarse_h = jnp.cross(coarse_r, coarse_v)
    # The two input casts, product, and subtraction can round each term.
    # Use the terms of each component so a small transverse velocity can
    # still define the plane when there is no cancellation in that component.
    rx, ry, rz = jnp.abs(coarse_r)
    vx, vy, vz = jnp.abs(coarse_v)
    cross_scale = jnp.stack([ry * vz + rz * vy, rz * vx + rx * vz, rx * vy + ry * vx])
    has_h = jnp.any(jnp.abs(coarse_h) > roundoff(cross_scale, n_ops=4))
    coarse_h_mag = _safe_norm(coarse_h)
    is_inclined = has_h & (_safe_norm(coarse_h[:2]) > _NODE_RESOLUTION * coarse_h_mag)

    # Work in units of |r| and |v|: every element is homogeneous in these two
    # scales, so constant (stop-gradient) scales leave the derivatives exact
    # while keeping float32 SI products and their squares in range.
    length = lax.stop_gradient(_safe_norm(r))
    length = jnp.where(length > 0, length, jnp.ones_like(length))
    speed = lax.stop_gradient(_safe_norm(v))
    speed = jnp.where(speed > 0, speed, jnp.ones_like(speed))
    r = r / length
    v = v / speed
    mu = mu / (length * speed**2)
    r_mag = jnp.linalg.norm(r)
    v_mag = jnp.linalg.norm(v)

    h = jnp.cross(r, v)
    h_mag = _safe_norm(h)
    one, zero = jnp.ones_like(h_mag), jnp.zeros_like(h_mag)
    h_mag_safe = jnp.where(has_h, h_mag, one)

    # A radial state (no angular momentum) keeps the i = pi/2 convention.
    i = jnp.arctan2(
        jnp.where(has_h, _safe_norm(h[:2]), one), jnp.where(has_h, h[2], zero)
    )

    k = jnp.array([0.0, 0.0, 1.0], dtype=h.dtype)
    n = jnp.cross(k, h)
    n_mag = _safe_norm(n)

    e_vec = (1 / mu) * ((v_mag**2 - mu / r_mag) * r - jnp.dot(r, v) * v)
    e = _safe_norm(e_vec)

    kinetic = 0.5 * v_mag**2
    binding = mu / r_mag
    E_energy = kinetic - binding
    # Allow for roundoff from the norms, normalization, and energy subtraction.
    energy_roundoff = roundoff(kinetic + jnp.abs(binding), n_ops=8)
    non_parabolic = jnp.abs(E_energy) > energy_roundoff
    E_energy_safe = jnp.where(non_parabolic, E_energy, jnp.ones_like(E_energy))
    a = jnp.where(non_parabolic, -mu / (2 * E_energy_safe), like(jnp.inf, E_energy))

    # A single longitude frame covers ellipses and circles. The same
    # projection measures the body's longitude and its periapsis, so their
    # difference retains phase even when the eccentricity direction is noisy.
    node = n / jnp.where(is_inclined, n_mag, one)
    node = jnp.where(is_inclined, node, jnp.array([1.0, 0.0, 0.0], dtype=r.dtype))
    tangent = jnp.cross(h / h_mag_safe, node)
    W = jnp.arctan2(node[1], node[0])
    w = _oriented_angle(jnp.dot(e_vec, node), jnp.dot(e_vec, tangent))
    longitude = _oriented_angle(jnp.dot(r, node), jnp.dot(r, tangent))
    nu = longitude - w

    turn = like(two_pi, nu)
    W = W % turn
    w = w % turn
    nu = nu % turn

    E = jnp.arctan2(jnp.sqrt(1 - e**2) * jnp.sin(nu), e + jnp.cos(nu))
    M = E - e * jnp.sin(E)
    M = M % turn
    M = jnp.where(e < 1.0, M, like(jnp.nan, M))

    return a * length, e, i, W, w, M
