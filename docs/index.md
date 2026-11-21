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
from hwoutils.constants import G, Msun2kg
from orbix.system import Body, FrameOrbit, SkyFrame, System

orbit = FrameOrbit.from_classical(
    a_AU=[1.0],
    e=0.2,
    W_rad=0.5,
    i_rad=1.0,
    w_rad=0.3,
    M0_rad=0.0,
    mu_AU3_per_d2=G * Msun2kg,
)
system = System(Body(1.0), Body([0.0]), orbit, SkyFrame(10.0))
ephemeris = system.solve(jnp.linspace(0.0, 365.0, 100))  # days from system.frame.epoch
raoff, decoff = ephemeris.offset_arcsec(1)
```

## Orbits as fit parameters

`orbix.system` also describes an orbit as the free parameter of a fit. Three pieces work
together:

- A chart maps an orbit to a vector of real coordinates and back, with the exact volume of
  the map: `ClassicalChart`, `EccentricityVectorChart`, `FrameChart`, `CartesianChart` and
  `ThieleInnesChart`.
- An orbit law is the prior density of an orbit, independent of the chart. `OrbitLaw`
  takes a law on the semi-major axis or the period and one on the eccentricity, with
  isotropic orientation and uniform phase.
- A block combines a chart, a law and the rows of a `System` it frees. `OrbitBlock` works
  with every chart except the Thiele-Innes one, and `ThieleInnesBlock` frees one row
  through the Thiele-Innes chart and writes the star mass its coordinates imply. Both
  follow the `hwoutils.blocks.AbstractBlock` contract, which is how photomancy fits use
  them.

```python
import jax
from hwoutils.constants import G, Msun2kg
from orbix.system import (
    Body,
    EccentricityVectorChart,
    FrameOrbit,
    OrbitBlock,
    OrbitLaw,
    SkyFrame,
    System,
)
from priorities import LogUniform, Uniform

jax.config.update("jax_enable_x64", True)

orbit = FrameOrbit.from_classical(
    a_AU=[1.0],
    e=0.2,
    W_rad=0.5,
    i_rad=1.0,
    w_rad=0.3,
    M0_rad=0.0,
    mu_AU3_per_d2=G * Msun2kg,
)
system = System(Body(1.0), Body([0.0]), orbit, SkyFrame(10.0))

law = OrbitLaw(period_d=LogUniform(100.0, 10_000.0), eccentricity=Uniform(0.0, 0.9))
block = OrbitBlock(EccentricityVectorChart(), law, rows=(0,), system="")

x = block.sample(jax.random.key(0), system)  # {name: (1,)} per coordinate
orbits = block.forward(x, system)["orbits"]  # the freed row written into the orbits
log_p = block.log_prob(x, system)  # normalized density of the coordinates
print(block.coordinate_names)
```

Every coordinate takes any real value. A coordinate that the law bounds is mapped onto the
real line with a logit and named with the prefix `logit_` (`logit_ln_a_AU` above), and the
density includes the log derivative of that map. In the Thiele-Innes chart the sky
position of a planet is linear in the four constants `(A, B, F, G)`: `sky_design` gives the
design matrix at a fixed orbit shape and `linear_gaussian_conditional` the exact Gaussian
conditional of the constants under Gaussian astrometric errors.

A radial-velocity or phase-dependent likelihood can tell apart two orbits that share one
sky track and differ only in the sign of the line-of-sight coordinate. `line_of_sight_sheet`
reads that sign (+1 or -1) of an orbit, and `reflect_line_of_sight(orbits, planet)` returns
the orbits with one planet mirrored through the sky plane, which has the same east and
north positions at every time and the opposite away coordinate.

## Sky-coordinate convention

Star-centered orbit vectors use the right-handed **(east, north, away)**
frame: `r_AU[..., 0]` is east, `r_AU[..., 1]` is north, and `r_AU[..., 2]` is
positive away from the observer. Positions are in AU; velocities share the
frame in AU/day. The component dz/dt is the radial velocity of the body
relative to the star, positive when receding. `state_vector_to_keplerian`
accepts position and velocity in this same order.

`ephemeris.offset_arcsec(target)` returns `(raoff, decoff)`, each shaped
like the sample times for one target, where
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
Start with {doc}`examples/orbits`; {doc}`examples/observatory` checks keepout and
computes the zodiacal-light pointing angles of a target; {doc}`examples/prepared` prepares sky
tracks once for a still, a strip, a movie, and a Manim clip; {doc}`gallery`
shows every function it exports.

```{toctree}
:maxdepth: 1
:caption: Examples

examples/orbits
examples/prepared
examples/observatory
```

```{toctree}
:maxdepth: 1
:caption: Reference

gallery
migration
```
