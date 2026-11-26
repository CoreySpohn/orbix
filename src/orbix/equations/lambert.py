"""Lambert boundary-value problem: elliptic, multi-revolution, differentiable.

Given two position vectors and the time of flight between them, solve for
the terminal velocities of the connecting Keplerian orbit. This is the
classical Lambert *boundary-value* problem (Lancaster & Blanchard 1969;
Battin 1999, ch. 7; Izzo 2015), distinct from the Lambertian scattering
phase function in ``orbix.equations.phase``.

The transfer is parameterized by ``x = cos(alpha/2)`` on the open interval
``(-1, 1)``, where ``alpha`` is the Lagrange angle: the semi-major axis is
``a = s / (2 (1 - x^2))``, both classical alpha branches collapse into the
sign of ``x``, and the time of flight ``T(x)`` is strictly decreasing for
``N = 0`` and U-shaped (two roots per achievable TOF) for ``N >= 1``
revolutions. Roots are bracketed by a fixed-iteration ternary search for
the TOF minimum, then bisected, then polished with Newton steps that carry
implicit-function-theorem gradients (the bracketing itself runs under
``stop_gradient``).

Every solution family is indexed by three discrete choices:

- ``N``: number of complete revolutions on the arc.
- ``long_way``: transfer angle above pi (opposite orbit normal).
- ``high_branch``: for ``N >= 1``, the larger-x (larger period) of the two
  roots; flagged invalid for ``N = 0``.

Units are any consistent set (orbix convention: AU, days, and
``mu = G * Ms_kg`` in AU^3/day^2). All functions are scalar-core: ``vmap``
over batches of positions, times, ``N``, or branch flags.

Degenerate geometries: transfer angles of exactly 0 or pi leave the orbit
plane undefined and the returned velocities blow up there; callers sample
past these measure-zero configurations.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from hwoutils.constants import two_pi
from hwoutils.precision import like
from jaxtyping import Array

# Interior clip for the x domain: keeps arccos/arcsin gradients finite and
# bounds the semi-major axis at a ~ s / (4 * _X_EPS).
_X_EPS = 1e-9
# Sign-preserving floor for Newton denominators (dT/dx -> 0 only at the
# multi-rev double root, where sensitivity is genuinely unbounded).
_DTOF_FLOOR = 1e-30


def _geometry(r1: Array, r2: Array, long_way) -> tuple:
    """Chord/semi-perimeter geometry shared by every Lambert routine.

    Args:
        r1: First position vector, shape ``(3,)``.
        r2: Second position vector, shape ``(3,)``.
        long_way: Whether the transfer angle exceeds pi.

    Returns:
        Tuple ``(r1n, r2n, c, s, lam)``: the two radii, chord length,
        semi-perimeter, and the signed Lambert parameter
        ``lam = +/- sqrt((s - c) / s)`` (negative on the long way).
    """
    r1n = jnp.linalg.norm(r1)
    r2n = jnp.linalg.norm(r2)
    c = jnp.linalg.norm(r2 - r1)
    s = 0.5 * (r1n + r2n + c)
    # sqrt((s-c)/s) loses the small half-angle near antipodal endpoints.
    half_angle = jnp.linalg.norm(r1 / r1n + r2 / r2n) / 2
    lam_mag = jnp.minimum(jnp.sqrt(r1n * r2n) * half_angle / s, 1)
    lam = jnp.where(long_way, -lam_mag, lam_mag)
    return r1n, r2n, c, s, lam


def _tof_of_x(x, s, lam, mu, N) -> Array:
    """Elliptic Lagrange time of flight at ``x = cos(alpha/2)``.

    ``alpha = 2 arccos(x)`` spans ``(0, 2 pi)`` as ``x`` spans ``(-1, 1)``,
    so both classical alpha branches are covered without a flag;
    ``beta = 2 arcsin(lam sqrt(1 - x^2))`` carries the transfer-way sign
    through ``lam``.
    """
    z2 = (1 - x) * (1 + x)

    def near_parabolic():
        # asin(z) - z*sqrt(1-z**2) is the integral of 2*z**2/sqrt(1-z**2).
        # Divide its series by z**3 before subtracting the beta contribution.
        # At z**2 <= 1/64, the omitted tail is a few float64 epsilons relative.
        lam2 = lam * lam
        gap = (1 - lam) * (1 + lam)
        diff = _parabolic_factor(lam)
        polynomial = (2 / 3) * diff
        power = z2
        for coefficient in (
            1 / 5,
            3 / 28,
            5 / 72,
            35 / 704,
            63 / 1664,
            77 / 2560,
            429 / 17408,
        ):
            diff = gap + lam2 * diff
            polynomial = polynomial + coefficient * diff * power
            power = power * z2
        return jnp.sqrt(s / (2 * mu)) * s * polynomial

    def elliptic():
        a = s / (2 * z2)
        alpha = 2 * jnp.arccos(x)
        beta = 2 * jnp.arcsin(lam * jnp.sqrt(z2))
        turns = two_pi * jnp.asarray(N, dtype=x.dtype)
        return jnp.sqrt(a**3 / mu) * (
            (alpha - jnp.sin(alpha)) - (beta - jnp.sin(beta)) + turns
        )

    return jax.lax.cond((N == 0) & (x > 0) & (z2 <= 1 / 64), near_parabolic, elliptic)


def _parabolic_factor(lam):
    """Evaluate 1 - lam**3 without losing the gap near lam=1."""
    return (1 - lam) * (1 + lam + lam * lam)


def _minimum_tof(s, lam, mu, N, edge, iters):
    """Return the minimum time and its corresponding elliptic bracket endpoint."""

    def multirev():
        x = jax.lax.stop_gradient(_tof_argmin_x(s, lam, mu, N, edge, iters))
        return _tof_of_x(x, s, lam, mu, N), x

    def zero_rev():
        return jnp.sqrt(2 * s / mu) * s * _parabolic_factor(lam) / 3, edge

    return jax.lax.cond(N == 0, zero_rev, multirev)


def _x_edge(s):
    """Keep the bracket strictly inside (-1, 1) in either precision."""
    return like(1 - max(_X_EPS, jnp.finfo(s.dtype).eps / 2), s)


def _tof_argmin_x(s, lam, mu, N, edge, iters: int) -> Array:
    """Locate the x minimizing ``T(x)`` via fixed-iteration ternary search.

    For ``N >= 1`` the minimum is the interior double-root point.
    This search accepts values only. Callers wrap it in ``stop_gradient``.
    """

    def body(_, bounds):
        lo, hi = bounds
        third = (hi - lo) / 3.0
        m1 = lo + third
        m2 = hi - third
        shrink_lo = _tof_of_x(m1, s, lam, mu, N) > _tof_of_x(m2, s, lam, mu, N)
        return (jnp.where(shrink_lo, m1, lo), jnp.where(shrink_lo, hi, m2))

    lo, hi = jax.lax.fori_loop(0, iters, body, (-edge, edge))
    return 0.5 * (lo + hi)


def _bisect_x(tof, s, lam, mu, N, lo, hi, iters: int) -> Array:
    """Bisection root of ``T(x) = tof`` on a monotone bracket ``[lo, hi]``."""

    def body(_, bounds):
        lo_i, hi_i = bounds
        mid = 0.5 * (lo_i + hi_i)
        f_mid = _tof_of_x(mid, s, lam, mu, N) - tof
        f_lo = _tof_of_x(lo_i, s, lam, mu, N) - tof
        same = jnp.sign(f_mid) == jnp.sign(f_lo)
        return (jnp.where(same, mid, lo_i), jnp.where(same, hi_i, mid))

    lo_f, hi_f = jax.lax.fori_loop(0, iters, body, (lo, hi))
    return 0.5 * (lo_f + hi_f)


def _newton_polish(x, tof, s, lam, mu, N, edge, steps: int) -> Array:
    """Differentiable Newton refinement of a gradient-stopped root.

    Refines ``T(x) = tof`` and, because the incoming ``x`` carries no
    gradients, attaches the implicit-function gradients of the root to
    every upstream input.
    """
    dtof_dx = jax.grad(_tof_of_x, argnums=0)
    floor = like(_DTOF_FLOOR, x)
    for _ in range(steps):
        f = _tof_of_x(x, s, lam, mu, N) - tof
        d = dtof_dx(x, s, lam, mu, N)
        d_safe = jnp.where(d >= 0.0, jnp.maximum(d, floor), jnp.minimum(d, -floor))
        x = jnp.clip(x - f / d_safe, -edge, edge)
    return x


def _terminal_velocities(r1, r2, x, s, lam, mu, long_way, r1n, r2n, c) -> tuple:
    """Terminal velocities from the converged ``x`` via Lagrange f and g."""
    one_m_x2 = (1 - x) * (1 + x)
    a = s / (2.0 * one_m_x2)
    alpha = 2.0 * jnp.arccos(x)
    beta = 2.0 * jnp.arcsin(lam * jnp.sqrt(one_m_x2))
    p = 4.0 * a * (s - r1n) * (s - r2n) / c**2 * jnp.sin((alpha + beta) / 2.0) ** 2
    u1, u2 = r1 / r1n, r2 / r2n
    cosdth = jnp.clip(jnp.dot(u1, u2), -1, 1)
    sin_mag = jnp.linalg.norm(jnp.cross(u1, u2))
    sindth = jnp.where(long_way, -sin_mag, sin_mag)
    f = 1.0 - r2n / p * (1.0 - cosdth)
    g = sindth * r1n * r2n / jnp.sqrt(mu * p)
    gdot = 1.0 - r1n / p * (1.0 - cosdth)
    v1 = (r2 - f * r1) / g
    v2 = (gdot * r2 - r1) / g
    return v1, v2


def lambert_solve(
    r1: Array,
    r2: Array,
    tof: Array,
    mu: Array,
    N=0,
    long_way=False,
    high_branch=False,
    *,
    bisect_iters: int = 64,
    ternary_iters: int = 104,
    polish_steps: int = 2,
) -> tuple[Array, Array, Array]:
    """Solve the elliptic Lambert problem for one ``(N, way, branch)`` family.

    Args:
        r1: Position at the first epoch, shape ``(3,)``.
        r2: Position at the second epoch, shape ``(3,)``.
        tof: Time of flight between the epochs (same units as ``mu``).
        mu: Gravitational parameter ``G * M``.
        N: Complete revolutions on the arc (int, traceable).
        long_way: Transfer angle above pi (flips the orbit normal).
        high_branch: For ``N >= 1``, select the larger-x of the two roots.
            No high branch exists for ``N = 0`` (flagged invalid).
        bisect_iters: Fixed bisection iterations (static).
        ternary_iters: Fixed ternary-search iterations for the TOF
            minimum used to bracket and to test existence (static).
        polish_steps: Differentiable Newton refinements (static); these
            carry the implicit-function gradients of the solution.

    Returns:
        tuple[Array, Array, Array]: Terminal velocities and validity flag:

            - ``v1``: Velocity at ``r1``, shape ``(3,)``.
            - ``v2``: Velocity at ``r2``, shape ``(3,)``.
            - ``valid``: Boolean; False when no elliptic solution exists for
              this ``(N, long_way, high_branch)`` family (outputs are then
              meaningless and must be masked by the caller).
    """
    dtype = jnp.result_type(*jax.tree.leaves((r1, r2, tof, mu)), 0.0)
    r1, r2, tof, mu = (jnp.asarray(value, dtype=dtype) for value in (r1, r2, tof, mu))
    r1n, r2n, c, s, lam = _geometry(r1, r2, long_way)

    edge = _x_edge(s)
    tof_min, x_min = _minimum_tof(s, lam, mu, N, edge, ternary_iters)

    x_lo = _bisect_x(tof, s, lam, mu, N, -edge, x_min, bisect_iters)
    x_hi = _bisect_x(tof, s, lam, mu, N, x_min, edge, bisect_iters)
    x_raw = jax.lax.stop_gradient(jnp.where(high_branch, x_hi, x_lo))
    x = _newton_polish(x_raw, tof, s, lam, mu, N, edge, polish_steps)

    v1, v2 = _terminal_velocities(r1, r2, x, s, lam, mu, long_way, r1n, r2n, c)
    is_multirev = jnp.asarray(N) >= 1
    reaches_minimum = jnp.where(is_multirev, tof >= tof_min, tof > tof_min)
    valid = reaches_minimum & (is_multirev | ~jnp.asarray(high_branch))
    return v1, v2, valid


def lambert_tof_min(
    r1: Array,
    r2: Array,
    mu: Array,
    N=0,
    long_way=False,
    *,
    ternary_iters: int = 104,
) -> Array:
    """Minimum elliptic time of flight for an ``N``-revolution transfer.

    For ``N >= 1`` this is the TOF at the double-root point (solutions
    exist iff ``tof >= lambert_tof_min``); for ``N = 0`` it is the
    parabolic limit, which bounds the elliptic family from below. Gradients
    are correct at interior minima by the envelope theorem.

    Args:
        r1: Position at the first epoch, shape ``(3,)``.
        r2: Position at the second epoch, shape ``(3,)``.
        mu: Gravitational parameter ``G * M``.
        N: Complete revolutions on the arc (int, traceable).
        long_way: Transfer angle above pi.
        ternary_iters: Fixed ternary-search iterations (static).

    Returns:
        The minimum time of flight (same units as ``mu``).
    """
    dtype = jnp.result_type(*jax.tree.leaves((r1, r2, mu)), 0.0)
    r1, r2, mu = (jnp.asarray(value, dtype=dtype) for value in (r1, r2, mu))
    _, _, _, s, lam = _geometry(r1, r2, long_way)
    minimum, _ = _minimum_tof(s, lam, mu, N, _x_edge(s), ternary_iters)
    return minimum
