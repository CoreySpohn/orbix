"""Sky projection of orbit frames and conditional relative-astrometry fits."""

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.numerics import COARSEST, eps
from hwoutils.precision import like, require_float64
from jax.scipy.linalg import solve_triangular
from jaxtyping import Array

from orbix.system.frame_orbit import FrameOrbit
from orbix.system.rotation import from_basis, rotate


def _hypot_with_zero_gradient(x, y):
    # Select the zero subgradient at the origin. Guard the inputs to hypot
    # as well as its output so a masked derivative cannot propagate NaNs.
    zero = (x == 0) & (y == 0)
    safe_x = jnp.where(zero, jnp.ones_like(x), x)
    return jnp.where(zero, jnp.zeros_like(safe_x), jnp.hypot(safe_x, y))


def sky_block_geometry(A, B, F, G) -> tuple[Array, Array]:
    """Return the largest singular value and the squared-singular-value gap.

    The sky matrix is [[B,G],[A,F]], with A,F north and B,G east. Inputs
    broadcast. For a projected orbit of semimajor axis a and inclination i,
    the results are a and a^2*(1 - cos(i)^2) in the input length units and
    their square. At face-on, the vanishing norm uses a zero subgradient.

    Args:
        A: North component of the first plane axis, in length units.
        B: East component of the first plane axis, in the same units.
        F: North component of the second plane axis, in the same units.
        G: East component of the second plane axis, in the same units.

    Returns:
        Tuple of the largest singular value of the sky block, in the input length unit,
        and the gap of squared singular values, in its square.
    """
    A, B, F, G = map(jnp.asarray, (A, B, F, G))
    # Sum/difference norms avoid subtracting nearly equal squared singular
    # values. For a projected frame they are a*(1+cos i), a*(1-cos i).
    plus = _hypot_with_zero_gradient(A + G, B - F)
    minus = _hypot_with_zero_gradient(A - G, B + F)
    return (plus + minus) / 2, plus * minus


def _quaternion_sheet(quat):
    """Read the represented away row on a common relative-magnitude grid."""
    # Both the rotation and the pivot comparison use the decision precision.
    # R.T @ away gives the away components of all three rotated axes.
    quat = quat.astype(COARSEST)
    conjugate = quat * jnp.array([1, -1, -1, -1], dtype=quat.dtype)
    away = rotate(conjugate, jnp.array([0, 0, 1], dtype=quat.dtype))[..., :2]
    scale = jnp.max(jnp.abs(away), axis=-1, keepdims=True)
    relative = away / jnp.where(scale > 0, scale, 1)
    c, h = jnp.moveaxis(relative, -1, 0)
    first = jnp.abs(h) - jnp.abs(c) <= 4 * eps(COARSEST)
    pivot = jnp.where(first, c, h)
    return jnp.where(pivot < 0, -1, 1)


def _reflected_quaternion(quat):
    # Reflecting both coordinate frames through their xy planes negates
    # the away row of the plane axes and preserves a proper rotation.
    return quat * jnp.array([1, -1, -1, 1], dtype=quat.dtype)


def line_of_sight_sheet(orbit: FrameOrbit) -> Array:
    """Return the line-of-sight sheet (+1 or -1) of orbits.

    The sheet is the one ``sky_block_to_frame`` defines: the sign of the
    largest-magnitude entry of the away row of
    ``orbit.plane_basis()``. After scaling the larger magnitude to one, the
    first entry wins a tie within four float32 epsilons. A face-on orbit gives +1.
    Reflected orbits share the sky block and differ only in this sign. The result
    has the orbit's batch shape, dtype int.

    Args:
        orbit: Orbits with batch shape B.

    Returns:
        Integer array of shape B holding +1 or -1.
    """
    return _quaternion_sheet(orbit.quat)


def reflect_line_of_sight(orbits: FrameOrbit, planet: int) -> FrameOrbit:
    """Return stacked ``orbits`` with ``planet`` reflected through the sky plane.

    The in-plane axes of the planet have their away component negated and the plane
    normal follows from them, so the reflected orbit has the same sky track (east and
    north positions at every time) and the opposite line-of-sight coordinate and
    velocity: ``frame_to_sky_block`` is unchanged and ``line_of_sight_sheet``
    flips. ``planet`` indexes the last batch axis of ``orbits``; the other planets are
    unchanged.

    A likelihood that tells the two sheets apart (radial velocity, phase-dependent
    photometry) is summed over them as ``logaddexp(L(orbit), L(reflected)) - log 2``.

    Args:
        orbits: Stacked orbits, batch shape (..., K), with planets on the last batch
            axis.
        planet: Index of the planet to reflect along the last batch axis.

    Returns:
        FrameOrbit with the same batch shape in which the selected planet is reflected
        through the sky plane.
    """
    reflected = _reflected_quaternion(orbits.quat[..., planet, :])
    return eqx.tree_at(
        lambda o: o.quat,
        orbits,
        orbits.quat.at[..., planet, :].set(reflected),
    )


def sky_block_to_frame(A, B, F, G, *, sheet=1) -> tuple[Array, Array]:
    """Complete a sky block to (a_AU, quat), with broadcast batch shape B.

    A,F are north and B,G east, all in AU: the sky matrix is [[B,G],[A,F]].
    The columns can be any orthogonal in-plane axes, including the stored
    epoch axes used by FrameOrbit.linear_map. Output shapes are B and
    B+(4,); the scalar-first quaternion maps those axes to east,north,away.

    Every finite nonzero real 2x2 block has a completion. Its largest
    singular value is a; a^2 I - S.T S = z z.T determines the away row
    z=(C,H). Static sheet must be +1 or -1 and sets the sign of z's
    largest-magnitude entry (first wins ties at float32 resolution). The
    other entry follows from C*H=-(A*F+B*G). The chart has a sign seam
    where equally large, opposite-sign entries exchange pivot. The two sheets
    coincide at face-on, where z=0; zero size is excluded.

    Values are finite at both face-on orientations and independent of
    eccentricity. The inverse is intrinsically non-differentiable at
    face-on (tilt is a square root of projected flattening); no smooth
    inverse there is claimed. All numerical work is host-free.

    Args:
        A: North component of the first plane axis, in AU.
        B: East component of the first plane axis, in AU.
        F: North component of the second plane axis, in AU.
        G: East component of the second plane axis, in AU.
        sheet: Line-of-sight sheet, +1 or -1. Static under JIT.

    Returns:
        Tuple of the semi-major axis in AU, shape B, and the scalar-first unit
        quaternion, shape B + (4,).
    """
    if sheet not in (-1, 1):
        raise ValueError("sheet must be +1 or -1")
    A, B, F, G = jnp.broadcast_arrays(*map(jnp.asarray, (A, B, F, G)))
    # Normalize first to avoid squaring a large or small physical scale.
    scale = jnp.maximum(
        jnp.maximum(jnp.abs(A), jnp.abs(B)), jnp.maximum(jnp.abs(F), jnp.abs(G))
    )
    A, B, F, G = (value / scale for value in (A, B, F, G))
    a, gap = sky_block_geometry(A, B, F, G)
    # Products of conformal/anticonformal parts vanish exactly at face-on;
    # subtracting squared column norms would manufacture an O(sqrt(eps)) tilt.
    plus_a, plus_b, minus_a, minus_b = A + G, B - F, A - G, B + F
    difference = -(plus_a * minus_a + plus_b * minus_b)
    # Use the larger diagonal for conditioning. The sheet is assigned after
    # quaternion conversion, which can exchange nearly equal away magnitudes.
    first = difference >= 0
    pivot2 = (gap + jnp.abs(difference)) / 2
    nonzero = pivot2 > 0
    pivot = jnp.sqrt(jnp.where(nonzero, pivot2, jnp.ones_like(pivot2)))
    other = -(plus_a * minus_b - minus_a * plus_b) / (2 * pivot)
    pivot = jnp.where(nonzero, pivot, jnp.zeros_like(pivot))
    other = jnp.where(nonzero, other, jnp.zeros_like(other))
    C = jnp.where(first, pivot, other)
    H = jnp.where(first, other, pivot)
    f = jnp.stack([B, A, C], -1) / a[..., None]
    g = jnp.stack([G, F, H], -1) / a[..., None]
    quat = from_basis(f, g, jnp.cross(f, g))
    same_sheet = _quaternion_sheet(quat) == sheet
    return scale * a, jnp.where(
        same_sheet[..., None], quat, _reflected_quaternion(quat)
    )


def frame_to_sky_block(a_AU, quat) -> Array:
    """Project a frame to (A,B,F,G) in AU, shape broadcast(B) + (4,).

    quat is a scalar-first unit quaternion, shape B+(4,), rotating the
    chosen plane axes into east,north,away. A,F are north; B,G are east.
    For FrameOrbit pass its a_AU and quat to obtain the epoch-axis block.
    sky_block_to_frame inverts this projection on its declared sheet.
    No eccentricity or inclination angle is used, including at e=0 or
    either face-on orientation.

    Args:
        a_AU: Semi-major axis in AU, shape B.
        quat: Scalar-first unit quaternion, shape B + (4,).

    Returns:
        Sky constants (A, B, F, G) in AU, shape B + (4,).
    """
    quat = jnp.asarray(quat)
    w, x, y, z = jnp.moveaxis(quat, -1, 0)
    # Symmetric quadratic products preserve exact conformality at face-on.
    # Using 1-2*q^2 independently for the diagonals can inject flattening
    # at roundoff level, which the inverse amplifies to sqrt(eps) tilt.
    cp, sp = w * w - z * z, 2 * w * z
    cm, sm = x * x - y * y, 2 * x * y
    scale = jnp.asarray(a_AU) / jnp.sum(quat * quat, axis=-1)
    return scale[..., None] * jnp.stack([sp + sm, cp + cm, cp - cm, sm - sp], -1)


def sky_design(orbit_shape: FrameOrbit, time_d) -> Array:
    """Design matrix of the sky constants at fixed nonlinear shape, shape ``(T, 2, 4)``.

    The predicted sky offset of a row is linear in the constants ``c = (A, B, F, G)``
    (AU, ``A``, ``F`` north and ``B``, ``G`` east): ``(x, y) = design @ c`` with
    ``x`` east and ``y`` north, so each time gives ``[[0, X, 0, Y], [X, 0, Y, 0]]``
    with ``(X, Y)`` the in-plane position at that time on the epoch mean-longitude
    axes. ``orbit_shape`` is an unbatched :class:`FrameOrbit`; only ``n_rad_per_d``,
    ``ex``, ``ey`` and ``epoch_d`` enter. ``time_d`` has shape ``(T,)`` in days.

    Args:
        orbit_shape: Unbatched orbit supplying the nonlinear shape.
        time_d: Observation times in days, shape (T,).

    Returns:
        Dimensionless design matrix mapping the constants (A, B, F, G) in AU to east and
        north offsets in AU, shape (T, 2, 4).
    """
    time = jnp.asarray(time_d)
    if orbit_shape.a_AU.ndim != 0 or time.ndim != 1:
        raise ValueError("Use an unbatched orbit and a (T,) time")
    plane = orbit_shape.in_plane(time)[0]
    X, Y = plane[:, 0], plane[:, 1]
    zero = jnp.zeros_like(X)
    return jnp.stack(
        [jnp.stack([zero, X, zero, Y], -1), jnp.stack([X, zero, Y, zero], -1)], -2
    )


def linear_gaussian_conditional(
    design, target, precision, prior_precision
) -> tuple[Array, Array, Array]:
    """Gaussian conditional of linear coefficients and the log marginal likelihood.

    The model is ``target_t = design_t @ c + noise_t`` for ``T`` records of ``k``
    values, with noise precision ``precision_t`` (``k x k``) and prior
    ``c ~ N(0, prior_precision^-1)`` on ``p`` coefficients. With
    ``P = prior_precision + sum_t design_t^T precision_t design_t`` and
    ``b = sum_t design_t^T precision_t target_t`` the conditional is
    ``N(P^-1 b, P^-1)``, and the data density with ``c`` integrated out is

    ``log Z = -(n/2) log(2 pi) + (1/2) sum_t log det precision_t
    - (1/2) sum_t target_t^T precision_t target_t
    + (1/2) log det prior_precision - (1/2) log det P + (1/2) b^T P^-1 b``,

    with ``n`` the number of values of the active records. A record whose precision
    matrix is zero is inactive: it adds nothing to ``P`` or ``b`` and nothing to the
    first three terms of ``log Z``, whatever its target (mask a record by zeroing
    its precision). The design must have float64 precision. Observations and
    noise or prior precisions widen before any arithmetic.

    Args:
        design: Shape ``(T, k, p)``, for example :func:`sky_design` with ``k = 2``,
            ``p = 4``.
        target: Shape ``(T, k)``.
        precision: Shape ``(T, k, k)``, symmetric positive semi-definite per record.
        prior_precision: Shape ``(p, p)``, ``(p,)`` (diagonal) or a scalar (times the
            identity); positive definite.

    Returns:
        ``(mean, chol, log_marginal)``: the mean ``(p,)``, the lower Cholesky factor
        of ``P`` ``(p, p)`` (draw ``c = mean + solve_triangular(chol.T, z)`` with
        ``z`` standard normal) and the scalar ``log Z``.
    """
    design = jnp.asarray(design)
    require_float64("linear_gaussian_conditional", design)
    target, precision, prior = (
        like(value, design) for value in (target, precision, prior_precision)
    )
    p = design.shape[-1]
    prior = jnp.diag(jnp.broadcast_to(prior, (p,))) if prior.ndim < 2 else prior
    weighted = jnp.einsum("tkl,tlp->tkp", precision, design)
    information = prior + jnp.einsum("tkp,tkq->pq", design, weighted)
    drive = jnp.einsum("tkp,tk->p", weighted, target)
    chol = jnp.linalg.cholesky(information)
    mean = jax.scipy.linalg.cho_solve((chol, True), drive)
    determinant = jnp.linalg.det(precision)
    active = determinant > 0
    n_active = jnp.sum(active) * design.shape[-2]
    data = 0.5 * jnp.sum(
        jnp.where(active, jnp.log(jnp.where(active, determinant, 1.0)), 0.0)
    ) - 0.5 * jnp.einsum("tk,tkl,tl->", target, precision, target)
    prior_logdet = 2 * jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky(prior))))
    log_marginal = (
        -0.5 * n_active * jnp.log(2 * jnp.pi)
        + data
        + 0.5 * prior_logdet
        - jnp.sum(jnp.log(jnp.diag(chol)))
        + 0.5 * drive @ mean
    )
    return mean, chol, log_marginal


def fit_linear_map(orbit_shape: FrameOrbit, time_d, x, y, cov_or_errors, *, mask=None):
    """Fit the four sky constants at fixed nonlinear shape, with fixed shapes.

    orbit_shape is an unbatched float64 FrameOrbit; only n_rad_per_d, ex, ey and
    epoch_d enter. time_d, x, y have shape (T,), in days and east/north AU
    (the SkyFrame observable convention). Convert angular offsets and their
    uncertainties to AU using distance before calling. Size/orientation of
    orbit_shape are ignored. Times, observations and noise widen to the orbit
    precision. External vmap handles batches of independent fits.

    cov_or_errors is either a covariance array (2,2) or (T,2,2), in AU^2
    ordered east,north, or a tuple (x_err,y_err,corr), each scalar or (T,).
    Errors are positive AU standard deviations; abs(corr)<1. mask is a
    boolean (T,) array: False rows have exactly zero weight. Masked data
    and covariance may be nonfinite; active covariance must be positive
    definite and active data finite. No host-side value checks are made.

    Returns (sky_block, covariance, chi_square, rank): shapes (2,2), (4,4),
    (), (). sky_block=[[B,G],[A,F]] is in AU, covariance is in AU^2 in
    (A,B,F,G) order, and rank is the integer numerical rank (full rank=4).
    Whitening uses Cholesky; QR followed by a small SVD solves the normal
    equations without squaring their condition number. Singular values
    <= eps*max(2*T,4)*s_max are discarded. Fewer than two independent
    epochs, repeated phases, or masked rows can yield rank<4: the result
    is minimum norm and covariance is the pseudoinverse information on
    the estimable subspace, NOT finite uncertainty on unidentified modes.
    Zero usable rows return zeros, including chi-square and rank.

    Args:
        orbit_shape: Unbatched orbit supplying the nonlinear shape.
        time_d: Observation times in days, shape (T,).
        x: East offsets in AU, shape (T,).
        y: North offsets in AU, shape (T,).
        cov_or_errors: Covariance in AU^2 ordered east, north, shape (2, 2) or (T, 2,
            2), or a tuple (x_err, y_err, corr) of standard deviations in AU and a
            correlation coefficient, each scalar or shape (T,).
        mask: Boolean array of shape (T,), False for rows to drop. Defaults to using
            every row.

    Returns:
        Tuple (sky_block, covariance, chi_square, rank): the block [[B, G], [A, F]] in
        AU, shape (2, 2); the covariance of (A, B, F, G) in AU^2, shape (4, 4); the
        chi-square of the fit; and the integer numerical rank.
    """
    require_float64("fit_linear_map", orbit_shape)
    time, x, y = (like(value, orbit_shape) for value in (time_d, x, y))
    if (
        orbit_shape.a_AU.ndim != 0
        or time.ndim != 1
        or x.shape != time.shape
        or y.shape != time.shape
    ):
        raise ValueError("Use an unbatched orbit and matching (T,) time, x, y")
    count = time.size
    if count == 0:
        dtype = jnp.result_type(x, y, orbit_shape.a_AU)
        return (
            jnp.zeros((2, 2), dtype),
            jnp.zeros((4, 4), dtype),
            jnp.zeros((), dtype),
            jnp.array(0),
        )
    active = (
        jnp.ones(count, dtype=bool) if mask is None else jnp.asarray(mask, dtype=bool)
    )
    if active.shape != time.shape:
        raise ValueError("mask must have shape (T,)")
    if isinstance(cov_or_errors, tuple):
        sx, sy, corr = (like(value, orbit_shape) for value in cov_or_errors)
        sx, sy, corr = jnp.broadcast_arrays(sx, sy, corr)
        cross = sx * sy * corr
        covariance = jnp.stack(
            [jnp.stack([sx * sx, cross], -1), jnp.stack([cross, sy * sy], -1)], -2
        )
    else:
        covariance = like(cov_or_errors, orbit_shape)
    covariance = jnp.broadcast_to(covariance, (count, 2, 2))
    covariance = jnp.where(active[:, None, None], covariance, jnp.eye(2))
    root = jnp.linalg.cholesky(covariance)
    # Sanitizing times as well as data keeps masked NaNs out of the solve.
    design = sky_design(orbit_shape, jnp.where(active, time, orbit_shape.epoch_d))
    design = jnp.where(active[:, None, None], design, 0.0)
    data = jnp.where(active[:, None], jnp.stack([x, y], -1), 0.0)
    design = solve_triangular(root, design, lower=True).reshape(2 * count, 4)
    data = solve_triangular(root, data[..., None], lower=True).reshape(2 * count)
    q, r = jnp.linalg.qr(design, mode="reduced")
    u, singular, vh = jnp.linalg.svd(r, full_matrices=False)
    keep = singular > jnp.finfo(design.dtype).eps * max(2 * count, 4) * singular[0]
    inverse = jnp.where(keep, 1 / jnp.where(keep, singular, 1.0), 0.0)
    pinv = (vh.T * inverse) @ u.T
    constants = pinv @ (q.T @ data)
    covariance = pinv @ pinv.T
    residual = data - design @ constants
    block = constants[jnp.array([[1, 3], [0, 2]])]
    return block, covariance, jnp.sum(residual**2), jnp.sum(keep)
