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
    lam_mag = jnp.sqrt(jnp.clip((s - c) / s, 0.0, 1.0))
    lam = jnp.where(long_way, -lam_mag, lam_mag)
    return r1n, r2n, c, s, lam


def _tof_of_x(x, s, lam, mu, N) -> Array:
    """Elliptic Lagrange time of flight at ``x = cos(alpha/2)``.

    ``alpha = 2 arccos(x)`` spans ``(0, 2 pi)`` as ``x`` spans ``(-1, 1)``,
    so both classical alpha branches are covered without a flag;
    ``beta = 2 arcsin(lam sqrt(1 - x^2))`` carries the transfer-way sign
    through ``lam``.
    """
    one_m_x2 = 1.0 - x * x
    a = s / (2.0 * one_m_x2)
    alpha = 2.0 * jnp.arccos(x)
    beta = 2.0 * jnp.arcsin(lam * jnp.sqrt(one_m_x2))
    return jnp.sqrt(a**3 / mu) * (
        (alpha - jnp.sin(alpha)) - (beta - jnp.sin(beta)) + 2.0 * jnp.pi * N
    )


def _tof_argmin_x(s, lam, mu, N, iters: int) -> Array:
    """Locate the x minimizing ``T(x)`` via fixed-iteration ternary search.

    For ``N >= 1`` the minimum is interior (the double-root point); for
    ``N = 0``, where ``T`` is strictly decreasing, the search converges to
    the upper domain edge, whose TOF is the near-parabolic elliptic
    infimum. Runs on values only -- callers wrap in ``stop_gradient``.
    """

    def body(_, bounds):
        lo, hi = bounds
        third = (hi - lo) / 3.0
        m1 = lo + third
        m2 = hi - third
        shrink_lo = _tof_of_x(m1, s, lam, mu, N) > _tof_of_x(m2, s, lam, mu, N)
        return (jnp.where(shrink_lo, m1, lo), jnp.where(shrink_lo, hi, m2))

    lo, hi = jax.lax.fori_loop(
        0, iters, body, (-1.0 + _X_EPS, jnp.asarray(1.0 - _X_EPS))
    )
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


def _newton_polish(x, tof, s, lam, mu, N, steps: int) -> Array:
    """Differentiable Newton refinement of a gradient-stopped root.

    Refines ``T(x) = tof`` and, because the incoming ``x`` carries no
    gradients, attaches the implicit-function gradients of the root to
    every upstream input.
    """
    dtof_dx = jax.grad(_tof_of_x, argnums=0)
    for _ in range(steps):
        f = _tof_of_x(x, s, lam, mu, N) - tof
        d = dtof_dx(x, s, lam, mu, N)
        d_safe = jnp.where(
            d >= 0.0, jnp.maximum(d, _DTOF_FLOOR), jnp.minimum(d, -_DTOF_FLOOR)
        )
        x = jnp.clip(x - f / d_safe, -1.0 + _X_EPS, 1.0 - _X_EPS)
    return x


def _terminal_velocities(r1, r2, x, s, lam, mu, long_way, r1n, r2n, c) -> tuple:
    """Terminal velocities from the converged ``x`` via Lagrange f and g."""
    one_m_x2 = 1.0 - x * x
    a = s / (2.0 * one_m_x2)
    alpha = 2.0 * jnp.arccos(x)
    beta = 2.0 * jnp.arcsin(lam * jnp.sqrt(one_m_x2))
    p = 4.0 * a * (s - r1n) * (s - r2n) / c**2 * jnp.sin((alpha + beta) / 2.0) ** 2
    cosdth = jnp.clip(jnp.dot(r1, r2) / (r1n * r2n), -1.0, 1.0)
    sindth = jnp.where(long_way, -1.0, 1.0) * jnp.sqrt(1.0 - cosdth**2)
    f = 1.0 - r2n / p * (1.0 - cosdth)
    g = r1n * r2n * sindth / jnp.sqrt(mu * p)
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
        v1: Velocity at ``r1``, shape ``(3,)``.
        v2: Velocity at ``r2``, shape ``(3,)``.
        valid: Boolean; False when no elliptic solution exists for this
            ``(N, long_way, high_branch)`` family (outputs are then
            meaningless and must be masked by the caller).
    """
    r1n, r2n, c, s, lam = _geometry(r1, r2, long_way)

    x_min = jax.lax.stop_gradient(_tof_argmin_x(s, lam, mu, N, ternary_iters))
    tof_min = _tof_of_x(x_min, s, lam, mu, N)

    x_lo = _bisect_x(tof, s, lam, mu, N, -1.0 + _X_EPS, x_min, bisect_iters)
    x_hi = _bisect_x(tof, s, lam, mu, N, x_min, 1.0 - _X_EPS, bisect_iters)
    x_raw = jax.lax.stop_gradient(jnp.where(high_branch, x_hi, x_lo))
    x = _newton_polish(x_raw, tof, s, lam, mu, N, polish_steps)

    v1, v2 = _terminal_velocities(r1, r2, x, s, lam, mu, long_way, r1n, r2n, c)
    is_multirev = jnp.asarray(N) >= 1
    valid = (tof >= tof_min) & (is_multirev | ~jnp.asarray(high_branch))
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
    near-parabolic infimum of the elliptic family. Gradients are correct
    at interior minima by the envelope theorem.

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
    _, _, _, s, lam = _geometry(r1, r2, long_way)
    x_min = jax.lax.stop_gradient(_tof_argmin_x(s, lam, mu, N, ternary_iters))
    return _tof_of_x(x_min, s, lam, mu, N)
