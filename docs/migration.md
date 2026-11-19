# Migrating from 4.x to 5.0

orbix 5.0 changes the order of raw state rows, corrects the sky axes of
`position_arcsec`, and moves several modules to other libraries.

## Raw state row order

* Raw `propagate` rows are now (east, north, away), right-handed, instead
  of (north, east, away). Positions are in AU and velocities use the same
  order in AU/day. The same row order applies to the exported helpers
  `AB_matrices`, `AB_matrices_reduced`, `single_r`, `single_r_v`,
  `system_r` and `system_r_v`. `position_arcsec` keeps its (ra, dec) return
  order, but the two numbers swap relative to 2.x: the first now holds the
  east offset and the second the north offset, where 2.x put the north offset
  first (see Sky positions). `state_vector_to_keplerian` now expects both
  position and velocity in (east, north, away) order. Swap the first two
  components of stored raw states before passing them to the new API.
  Orbital elements and the z sign are unchanged: dz/dt is the body's
  radial velocity relative to the star, positive receding. Phase angle
  is denoted beta, with cos(beta) = z/r; alpha denotes angular separation.
  The 3D orbit view now agrees with the sky panel when viewed from -z.

## Sky positions

* `position_arcsec` now returns (Delta alpha*, Delta Dec) = (east, north),
  where Delta alpha* = Delta RA cos Dec. Earlier releases returned these
  swapped, reflecting the sky about the diagonal; earlier sky outputs and
  anything fitted against them followed that old convention. The phase-angle
  documentation now states that z points away from the observer and the
  returned angle is the standard star-planet-observer phase angle beta.
  The phase-angle numbers are unchanged; the raw row-order change is
  described above. Published elements (standard convention) now give the correct
  sky positions; elements fitted against earlier `position_arcsec` outputs
  convert by i' = 180 deg - i, Omega' = (90 deg - Omega) mod 360, with a, e,
  omega and the epoch terms unchanged. Rerun figures drawn from
  `position_arcsec`.

## Removed and moved

* Orbit fitting (the `fitting` module and its `numpyro` extra) moved to photomancy.
* `Planets`/`System`/`Star` and the EXOSIMS integration moved to exosims-plugins.
* Zodiacal photometry moved to skyscapes.
* `quasi_random` (the vendored Roberts quasi-random sequence) moved to photomancy.
* The broken predict/pdet functions and the `discrete_e` factory were removed
  (they did not work correctly and had no working callers).
