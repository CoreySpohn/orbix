"""Methods to propagate the positions of planets."""

import jax.numpy as jnp


def calculate_r_v(A_mat_b, B_mat_b, e_vec_b, sinE_mat, cosE_mat, n_orb_vec_b):
    """Calculate position and velocity vectors for n planets over m time steps.

    Propagation is computed as:
    r = A * (cosE - e) + B * sinE
    v = n_orb / (1 - e * cosE) * (-A * sinE + B * cosE)
    where A, B, e, n_orb are pre-broadcasted outside the function. The sinE
    and cosE are managed inside the function for vectorization reasons.

    Some effort has been made to ensure there isn't division by zero.

    Args:
        A_mat_b (jax.Array): Pre-broadcasted A vectors. Shape (3, n, 1).
        B_mat_b (jax.Array): Pre-broadcasted B vectors. Shape (3, n, 1).
        e_vec_b (jax.Array): Pre-broadcasted eccentricity. Shape (n, 1).
        sinE_mat (jax.Array): Sine(Eccentric Anomaly). Shape (n, m).
        cosE_mat (jax.Array): Cosine(Eccentric Anomaly). Shape (n, m).
        n_orb_vec_b (jax.Array): Pre-broadcasted mean orbital motion. Shape (n, 1).

    Returns:
        tuple[jax.Array, jax.Array]: A tuple containing:
            - r (jax.Array): Position vectors. Shape (3, n, m).
            - v (jax.Array): Velocity vectors. Shape (3, n, m).
    """
    # A, B, e, n_orb are assumed to be pre-broadcasted outside the function.
    sinE_broadcast = sinE_mat[jnp.newaxis, :, :]  # Shape (1, n, m)
    cosE_broadcast = cosE_mat[jnp.newaxis, :, :]  # Shape (1, n, m)

    # calculate position
    # Broadcasting:
    # (cosE_mat - e_vec_b): (n, m) - (n, 1) -> (n, m)
    # A_mat_b * (result): (3, n, 1) * (n, m) [broadcast to (1,n,m)] -> (3, n, m)
    term1_r = A_mat_b * (cosE_mat - e_vec_b)[jnp.newaxis, :, :]
    # B_mat_b * sinE_broadcast: (3, n, 1) * (1, n, m) -> (3, n, m)
    term2_r = B_mat_b * sinE_broadcast
    r = term1_r + term2_r  # Shape (3, n, m)

    # calculate velocity
    # denominator: 1.0 - (n, 1) * (n, m) -> (n, m)
    denominator = 1.0 - e_vec_b * cosE_mat
    denominator_safe = jnp.where(
        denominator == 0, jnp.finfo(denominator.dtype).eps, denominator
    )
    # scalar_part: (n, 1) / (n, m) -> (n, m)
    scalar_part = n_orb_vec_b / denominator_safe

    # vector_part:
    # -A_mat_b * sinE_broadcast: (3, n, 1) * (1, n, m) -> (3, n, m)
    # B_mat_b * cosE_broadcast: (3, n, 1) * (1, n, m) -> (3, n, m)
    term1_v = -A_mat_b * sinE_broadcast
    term2_v = B_mat_b * cosE_broadcast
    vector_part = term1_v + term2_v  # Shape (3, n, m)

    # v: scalar_part[broadcast to (1,n,m)] * vector_part
    # (1, n, m) * (3, n, m) -> (3, n, m)
    v = scalar_part[jnp.newaxis, :, :] * vector_part

    return r, v


def calculate_r(A_mat_b, B_mat_b, e_vec_b, sinE_mat, cosE_mat):
    """Calculate position vectors for n planets over m time steps.

    Propagation is computed as:
    r = A * (cosE - e) + B * sinE
    where A, B, e, n_orb are pre-broadcasted outside the function. The sinE
    and cosE are managed inside the function for vectorization reasons.

    Some effort has been made to ensure there isn't division by zero.

    Args:
        A_mat_b (jax.Array): Pre-broadcasted A vectors. Shape (3, n, 1).
        B_mat_b (jax.Array): Pre-broadcasted B vectors. Shape (3, n, 1).
        e_vec_b (jax.Array): Pre-broadcasted eccentricity. Shape (n, 1).
        sinE_mat (jax.Array): Sine(Eccentric Anomaly). Shape (n, m).
        cosE_mat (jax.Array): Cosine(Eccentric Anomaly). Shape (n, m).

    Returns:
        r (jax.Array): Position vectors. Shape (3, n, m).
    """
    # A, B, e, n_orb are assumed to be pre-broadcasted outside the function.
    sinE_broadcast = sinE_mat[jnp.newaxis, :, :]  # Shape (1, n, m)

    # calculate position
    # Broadcasting:
    # (cosE_mat - e_vec_b): (n, m) - (n, 1) -> (n, m)
    # A_mat_b * (result): (3, n, 1) * (n, m) [broadcast to (1,n,m)] -> (3, n, m)
    term1_r = A_mat_b * (cosE_mat - e_vec_b)[jnp.newaxis, :, :]
    # B_mat_b * sinE_broadcast: (3, n, 1) * (1, n, m) -> (3, n, m)
    term2_r = B_mat_b * sinE_broadcast
    r = term1_r + term2_r  # Shape (3, n, m)

    return r
