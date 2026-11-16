"""Sky projection of orbit frames and conditional relative-astrometry fits."""

import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from jaxtyping import Array

from orbix.system.frame_orbit import FrameOrbit
from orbix.system.rotation import from_basis


def _sky_geometry(A, B, F, G):
    """Return scale and squared-singular-value gap of a nonzero sky block."""
    # Sum/difference norms avoid subtracting nearly equal squared singular
    # values. For a projected frame they are a*(1+cos i), a*(1-cos i).
    plus = jnp.hypot(A + G, B - F)
    minus = jnp.hypot(A - G, B + F)
    return (plus + minus) / 2, plus * minus


def sky_block_to_frame(A, B, F, G, *, branch=1) -> tuple[Array, Array]:
    """Complete a sky block to (a_AU, quat), with broadcast batch shape B.

    A,F are north and B,G east, all in AU: the sky matrix is [[B,G],[A,F]].
    The columns can be any orthogonal in-plane axes, including the stored
    epoch axes used by FrameOrbit.linear_map. Output shapes are B and
    B+(4,); the scalar-first quaternion maps those axes to east,north,away.

    Every finite nonzero real 2x2 block has a completion. Its largest
    singular value is a; a^2 I - S.T S = z z.T determines the away row
    z=(C,H). Static branch must be +1 or -1 and sets the sign of z's
    largest-magnitude entry (first wins ties). The other entry follows
    from C*H=-(A*F+B*G). This is a local chart with a sign seam where
    equally large, opposite-sign entries exchange pivot. The two branches
    coincide at face-on, where z=0; zero size is excluded.

    Values are finite at both face-on orientations and independent of
    eccentricity. The inverse is intrinsically non-differentiable at
    face-on (tilt is a square root of projected flattening); no smooth
    inverse there is claimed. All numerical work is host-free.
    """
    if branch not in (-1, 1):
        raise ValueError("branch must be +1 or -1")
    A, B, F, G = jnp.broadcast_arrays(*map(jnp.asarray, (A, B, F, G)))
    # Normalize first to avoid squaring a large or small physical scale.
    scale = jnp.maximum(
        jnp.maximum(jnp.abs(A), jnp.abs(B)), jnp.maximum(jnp.abs(F), jnp.abs(G))
    )
    A, B, F, G = (value / scale for value in (A, B, F, G))
    a, gap = _sky_geometry(A, B, F, G)
    # Products of conformal/anticonformal parts vanish exactly at face-on;
    # subtracting squared column norms would manufacture an O(sqrt(eps)) tilt.
    plus_a, plus_b, minus_a, minus_b = A + G, B - F, A - G, B + F
    difference = -(plus_a * minus_a + plus_b * minus_b)
    first = difference >= 0
    # The larger diagonal of z z.T is at least gap/2; no cancellation.
    pivot2 = (gap + jnp.abs(difference)) / 2
    nonzero = pivot2 > 0
    pivot = jnp.sqrt(jnp.where(nonzero, pivot2, 1.0))
    other = -(plus_a * minus_b - minus_a * plus_b) / (2 * pivot)
    pivot = jnp.where(nonzero, pivot, 0.0)
    other = jnp.where(nonzero, other, 0.0)
    C = branch * jnp.where(first, pivot, other)
    H = branch * jnp.where(first, other, pivot)
    f = jnp.stack([B, A, C], -1) / a[..., None]
    g = jnp.stack([G, F, H], -1) / a[..., None]
    return scale * a, from_basis(f, g, jnp.cross(f, g))


def frame_to_sky_block(a_AU, quat) -> Array:
    """Project a frame to (A,B,F,G) in AU, shape broadcast(B) + (4,).

    quat is a scalar-first unit quaternion, shape B+(4,), rotating the
    chosen plane axes into east,north,away. A,F are north; B,G are east.
    For FrameOrbit pass its a_AU and quat to obtain the epoch-axis block.
    sky_block_to_frame inverts this projection on its declared branch.
    No eccentricity or inclination angle is used, including at e=0 or
    either face-on orientation.
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


def fit_linear_map(orbit_shape: FrameOrbit, time_d, x, y, cov_or_errors, *, mask=None):
    """Fit the four sky constants at fixed nonlinear shape, with fixed shapes.

    orbit_shape is an unbatched FrameOrbit; only n_rad_per_d, ex, ey and
    epoch_d enter. time_d, x, y have shape (T,), in days and east/north AU
    (the SkyFrame observable convention). Convert angular offsets and their
    uncertainties to AU using distance before calling. Size/orientation of
    orbit_shape are ignored. External vmap handles batches of independent fits.

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
    """
    time, x, y = map(jnp.asarray, (time_d, x, y))
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
        sx, sy, corr = map(jnp.asarray, cov_or_errors)
        sx, sy, corr = jnp.broadcast_arrays(sx, sy, corr)
        cross = sx * sy * corr
        covariance = jnp.stack(
            [jnp.stack([sx * sx, cross], -1), jnp.stack([cross, sy * sy], -1)], -2
        )
    else:
        covariance = jnp.asarray(cov_or_errors)
    covariance = jnp.broadcast_to(covariance, (count, 2, 2))
    covariance = jnp.where(active[:, None, None], covariance, jnp.eye(2))
    root = jnp.linalg.cholesky(covariance)
    # Sanitizing times as well as data keeps masked NaNs out of the solve.
    plane = orbit_shape.in_plane(jnp.where(active, time, orbit_shape.epoch_d))[0]
    X, Y = plane[:, 0], plane[:, 1]
    zero = jnp.zeros_like(X)
    design = jnp.stack(
        [jnp.stack([zero, X, zero, Y], -1), jnp.stack([X, zero, Y, zero], -1)], -2
    )
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
