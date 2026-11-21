# orbix

<p align="center">
  <a href="https://pypi.org/project/orbix/"><img src="https://img.shields.io/pypi/v/orbix.svg?style=flat-square&logo=pypi" alt="PyPI"/></a>
  <a href="https://orbix.readthedocs.io"><img src="https://readthedocs.org/projects/orbix/badge/?version=latest&style=flat-square" alt="Documentation Status"/></a>
  <a href="https://github.com/CoreySpohn/orbix/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/CoreySpohn/orbix/ci.yml?branch=main&style=flat-square&logo=github&label=tests" alt="Tests"/></a>
  <a href="https://github.com/CoreySpohn/orbix/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="License"/></a>
  <a href="https://github.com/CoreySpohn/orbix/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/python-3.12%2B-blue.svg?style=flat-square&logo=python" alt="Python 3.12+"/></a>
  <a href="https://doi.org/10.5281/zenodo.20819376"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20819376-blue?style=flat-square" alt="DOI"/></a>
</p>

A JAX library of functions for exoplanet simulations: differentiable Keplerian
orbit propagation and Kepler-equation solvers, plus solar-system geometry
(observatory position, keepout) for the Habitable Worlds Observatory direct
imaging simulation suite. Scene composition (planet catalogs, EXOSIMS
integration, zodiacal photometry) lives in other libraries; see below.

## Install

```
pip install orbix
```

## Quickstart

```python
import jax.numpy as jnp
from hwoutils.constants import G, Msun2kg
from orbix.system import FrameOrbit, offset_arcsec, phase_angle_rad, distance_AU

orbit = FrameOrbit.from_classical(
    a_AU=1.0,
    e=0.0167,
    W_rad=0.0,
    i_rad=0.0,
    w_rad=0.0,
    M0_rad=0.0,
    epoch_d=0.0,
    mu_AU3_per_d2=G * Msun2kg,
)
time_d = jnp.linspace(0.0, 365.25, 5)
r_AU, v_AU_per_d = orbit.at(time_d)
beta_rad = phase_angle_rad(r_AU)
radius_AU = distance_AU(r_AU)
raoff, decoff = offset_arcsec(r_AU, 10.0)
print(r_AU.shape)  # (5, 3): (T times, xyz)
```

`FrameOrbit` stores a non-singular orbit frame and propagates through
`at(time_d)`. Times are days on the same origin as `epoch_d`; positions and
velocities have shape `(*batch, T, 3)`. `System` constructs Jacobi orbits for
multiple bodies and returns a solve-once `Ephemeris` through `solve(time_d)`.
The package root exports only `__version__`; import engine names from
`orbix.system` and solver names from `orbix.kepler`.

Raw orbit rows are **(east, north, away)** in AU, a right-handed frame.
Velocities share this order in AU/day; dz/dt is the body's radial velocity
relative to the star, positive receding. `state_vector_to_keplerian` takes
position and velocity in that frame. `offset_arcsec` returns `(raoff,
decoff)` = (Delta alpha cos delta, Delta delta), east and north offsets in
arcsec. Omega is the ascending (receding) node's PA measured north through
east; i < 90 deg gives increasing PA, and omega is measured from that node
in the direction of motion. Phase angle is beta, with cos(beta) = z/r;
alpha denotes angular separation. See the documentation's sky-coordinate
convention for details.

The default solver is the table-free `Markley()` strategy. Pass
`solver=Halley()` or `solver=Grid()` to `at`, `solve`, or the non-singular
`kepler(mean_longitude, ex, ey, solver=...)` function from `orbix.kepler`.
It returns `(sin(F), cos(F))` for `lambda = F - ex*sin(F) + ey*cos(F)`,
where `(ex, ey) = e*(cos(p), sin(p))` and `p` is longitude of periapsis.
All strategies support broadcasting and implicit forward/reverse derivatives,
including circular orbits and higher orders. Enable
`jax.config.update("jax_enable_x64", True)` for double-precision accuracy.

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

## Observatory geometry

`orbix.observatory` places the telescope on its L2 halo orbit and answers whether a target
is outside the keepout zones at a given time (`Observatory.is_observable_at`) or for what
fraction of an interval (`Observatory.observability`). `ecliptic_pointing` returns the
ecliptic latitude and helioecliptic longitude that the local zodiacal light models need.

## Scope

orbix provides Keplerian orbit geometry, Kepler-equation solvers, and the charts, laws
and blocks that make an orbit a fit parameter. Scene composition lives in skyscapes, and
inference lives in photomancy.

## Citation

If you use orbix, please cite it as described in
[CITATION.cff](CITATION.cff).
