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

## Scope

orbix provides Keplerian orbit geometry and Kepler-equation solvers; scene
composition lives in skyscapes.

## Citation

If you use orbix, please cite it as described in
[CITATION.cff](CITATION.cff).
