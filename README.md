# orbix

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
from orbix.orbit import KeplerianOrbit
from orbix.kepler.shortcuts.grid import get_grid_solver

orbit = KeplerianOrbit(
    a_AU=1.0, e=0.0167, W_rad=0.0, i_rad=0.0, w_rad=0.0, M0_rad=0.0, t0_d=0.0,
)
trig_solver = get_grid_solver(level="scalar", E=False, trig=True)
t_jd = jnp.linspace(2451545.0, 2451545.0 + 365.25, 5)
r_AU, phase_angle_rad, dist_AU = orbit.propagate(trig_solver, t_jd, Ms_kg=1.989e30)
print(r_AU.shape)  # (1, 3, 5): (K orbits, xyz, T times)
```

`get_grid_solver` builds a cached grid-interpolation Kepler solver with the
contract `trig_solver(M, e) -> (sinE, cosE)`; `KeplerianOrbit.propagate` uses
it to convert orbital elements and times into position vectors, phase angle,
and star-planet distance.

## Scope

orbix provides Keplerian orbit geometry and Kepler-equation solvers; scene
composition lives in skyscapes.

## Citation

If you use orbix, please cite it as described in
[CITATION.cff](CITATION.cff).
