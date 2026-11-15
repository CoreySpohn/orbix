"""Scalar-first unit quaternions mapping plane vectors into the sky frame."""

import jax.numpy as jnp


def rotate(quat, vector):
    """Rotate last-axis vectors by broadcast unit quaternions (w, x, y, z)."""
    imaginary = quat[..., 1:]
    twice_cross = 2 * jnp.cross(imaginary, vector)
    return vector + quat[..., :1] * twice_cross + jnp.cross(imaginary, twice_cross)


def from_basis(f, g, normal):
    """Convert orthonormal columns to a unit quaternion using a local chart.

    Select the largest quaternion component, guarding the unused square roots
    and denominators before evaluation. The quaternion double cover requires
    chart switches, but neither face-on orientation is intrinsically singular.
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
    divisor = 2 * jnp.sqrt(jnp.where(squared > 0, squared, 1.0))
    candidates = rows / divisor[..., :, None]
    index = jnp.argmax(squared, axis=-1)
    quat = jnp.take_along_axis(candidates, index[..., None, None], axis=-2)[..., 0, :]
    return quat / jnp.linalg.norm(quat, axis=-1, keepdims=True)
