"""Scalar-first unit quaternions mapping plane vectors into the sky frame."""

import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.leaves import canonical


def _cross(left, right):
    """Cross product with shape-independent rounding of each product."""
    first = left[..., [1, 2, 0]] * right[..., [2, 0, 1]]
    second = left[..., [2, 0, 1]] * right[..., [1, 2, 0]]
    return jax.lax.optimization_barrier(first) - jax.lax.optimization_barrier(second)


def from_axis_angle(axis, angle_rad):
    """Return a scalar-first unit quaternion about a supplied unit axis.

    axis has final dimension 3; angle_rad broadcasts over its leading shape.
    The positive rotation follows the right-hand rule.

    Args:
        axis: Unit rotation axis, last axis of length 3.
        angle_rad: Rotation angle in radians, broadcast against the leading shape of
            axis.

    Returns:
        Scalar-first unit quaternion (w, x, y, z), last axis of length 4.
    """
    dtype = jnp.result_type(*jax.tree.leaves((axis, angle_rad)), 0.0)
    if any(isinstance(x, jax.Array) for x in jax.tree.leaves(angle_rad)):
        half = jnp.asarray(angle_rad, dtype=dtype) / 2
        sine, cosine = jnp.sin(half), jnp.cos(half)
    else:
        # Near a full turn, rounding the angle can change the sign of sin(half).
        half = np.asarray(angle_rad, dtype=np.float64) / 2
        with jax.ensure_compile_time_eval():
            sine, cosine = canonical((np.sin(half), np.cos(half)), precision=dtype)
    imaginary = sine[..., None] * jnp.asarray(axis, dtype=dtype)
    real = jnp.broadcast_to(cosine[..., None], (*imaginary.shape[:-1], 1))
    return jnp.concatenate([real, imaginary], axis=-1)


def rotate(quat, vector):
    """Rotate last-axis vectors by broadcast unit quaternions (w, x, y, z).

    Round products before adding so CPU fusion cannot depend on batch length.

    Args:
        quat: Scalar-first unit quaternions (w, x, y, z), last axis of length 4.
        vector: Vectors to rotate, last axis of length 3.

    Returns:
        Rotated vectors, with the broadcast leading shape of the inputs and a last axis
        of length 3.
    """
    dtype = jnp.result_type(quat, *jax.tree.leaves(vector), 0.0)
    vector = jnp.asarray(vector, dtype=dtype)
    imaginary = quat[..., 1:]
    twice_cross = 2 * _cross(imaginary, vector)
    first = jax.lax.optimization_barrier(quat[..., :1] * twice_cross)
    second = _cross(imaginary, twice_cross)
    return vector + first + second


def from_basis(f, g, normal):
    """Convert orthonormal columns to a unit quaternion using a local chart.

    Select the largest quaternion component, guarding the unused square roots
    and denominators before evaluation. The quaternion double cover requires
    chart switches, but neither face-on orientation is intrinsically singular.

    Args:
        f: First orthonormal column, last axis of length 3.
        g: Second orthonormal column, last axis of length 3.
        normal: Third orthonormal column, completing a right-handed basis, last axis of
            length 3.

    Returns:
        Scalar-first unit quaternion (w, x, y, z), last axis of length 4, that rotates
        the standard basis into (f, g, normal).
    """
    m00, m10, m20 = (f[..., j] for j in range(3))
    m01, m11, m21 = (g[..., j] for j in range(3))
    m02, m12, m22 = (normal[..., j] for j in range(3))
    squared = jnp.stack(
        [
            1 + m00 + m11 + m22,
            1 + m00 - m11 - m22,
            1 - m00 + m11 - m22,
            1 - m00 - m11 + m22,
        ],
        axis=-1,
    )
    rows = jnp.stack(
        [
            jnp.stack([squared[..., 0], m21 - m12, m02 - m20, m10 - m01], -1),
            jnp.stack([m21 - m12, squared[..., 1], m01 + m10, m02 + m20], -1),
            jnp.stack([m02 - m20, m01 + m10, squared[..., 2], m12 + m21], -1),
            jnp.stack([m10 - m01, m02 + m20, m12 + m21, squared[..., 3]], -1),
        ],
        axis=-2,
    )
    divisor = 2 * jnp.sqrt(jnp.where(squared > 0, squared, jnp.ones_like(squared)))
    candidates = rows / divisor[..., :, None]
    index = jnp.argmax(squared, axis=-1)
    quat = jnp.take_along_axis(candidates, index[..., None, None], axis=-2)[..., 0, :]
    return quat / jnp.linalg.norm(quat, axis=-1, keepdims=True)
