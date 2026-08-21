# orbix

> Orbits, differentiably.

orbix is a JAX-native library for Keplerian orbit propagation and the
geometry of direct-imaging observations. It owns the orbital elements, the
Kepler solvers (including grid-accelerated and custom-gradient forms), the
projection onto the sky plane, and the observatory-side geometry of an L2
halo orbit with its keepout constraints. Everything is an Equinox module or
a pure function, so propagation composes with `jit`, `vmap`, and `grad`.

```python
import jax.numpy as jnp
from orbix import KeplerianOrbit

orbit = KeplerianOrbit(
    a_AU=1.0, e=0.2, W_rad=0.5, i_rad=1.0, w_rad=0.3, M0_rad=0.0, t0_d=0.0
)
r_AU, phase_angle_rad, dist_AU = orbit.propagate(
    t_jd=jnp.linspace(0.0, 365.0, 100), Ms_kg=1.99e30
)
```

Plotting lives in the optional `orbix.viz` package, built on
[eyepiece](https://eyepiece.readthedocs.io) and installed with
`pip install 'orbix[viz]'`; the base install carries no plotting stack.
Start with {doc}`examples/orbits`.

```{toctree}
:maxdepth: 1
:caption: Examples

examples/orbits
```
