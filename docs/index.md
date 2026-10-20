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

## Sky-coordinate convention

Star-centered orbit vectors use the right-handed **(east, north, away)**
frame: `r_AU[:, 0]` is east, `r_AU[:, 1]` is north, and `r_AU[:, 2]` is
positive away from the observer. Positions are in AU; velocities share the
frame in AU/day. The component dz/dt is the radial velocity of the body
relative to the star, positive when receding. `state_vector_to_keplerian`
accepts position and velocity in this same order.

`position_arcsec` returns `(raoff, decoff)`, each shaped `(K, T)`, where
raoff = Delta alpha cos delta (east) and decoff = Delta delta (north), in
arcsec. The first component includes cos delta, rather than being the RA
angle itself. Sky plots display east left and north up.

Omega (`W_rad`) is the PA of the ascending, receding node, measured north
through east. At that node z crosses from negative to positive. For i < 90
deg, PA increases with time; omega (`w_rad`) is measured from the ascending
node in the direction of motion. The classical Thiele-Innes letters remain
A, F for north and B, G for east; these names do not set the vector order.
This frame follows Perryman (2018), Fig. 2.2; the classical scalar notation
follows Green (1985), *Spherical Astronomy*, ch. 17.

The star-planet-observer phase angle is beta, with cos(beta) = z/r: zero is
full phase. Alpha denotes angular separation. The observatory's named
ecliptic and equatorial APIs retain their respective solar-system frames.

Plotting lives in the optional `orbix.viz` package, built on
[eyepiece](https://eyepiece.readthedocs.io) and installed with
`pip install 'orbix[viz]'`; the base install carries no plotting stack.
Start with {doc}`examples/orbits`; {doc}`examples/prepared` prepares sky
tracks once for a still, a strip, a movie, and a Manim clip; {doc}`gallery`
shows every function it exports.

```{toctree}
:maxdepth: 1
:caption: Examples

examples/orbits
examples/prepared
```

```{toctree}
:maxdepth: 1
:caption: Reference

gallery
```
