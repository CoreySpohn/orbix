# Changelog

## [0.5.0](https://github.com/CoreySpohn/orbix/compare/v0.4.0...v0.5.0) (2026-04-23)


### Features

* **equations:** port state_vector_to_keplerian from orbit-fitting ([13ddcdc](https://github.com/CoreySpohn/orbix/commit/13ddcdc3018bfa6354b8590af61639a70e2c0831))
* **observatory:** add Observatory composition wrapper ([4f9c72a](https://github.com/CoreySpohn/orbix/commit/4f9c72a37b8d60f3d79f8ad6274a9fac3242152f))
* **orbit:** KeplerianOrbit with cached AB matrices ([0ab29e0](https://github.com/CoreySpohn/orbix/commit/0ab29e0f99d527067d5b72066c6eee01b9d07d36))
* **orbit:** KeplerianOrbit.propagate returns (r_AU, phase_angle_rad, dist_AU) ([c72986b](https://github.com/CoreySpohn/orbix/commit/c72986bbba712404b2262c46b9fe53f7eb8030c4))
* **orbit:** position_arcsec and separation_arcsec fast paths ([69d27a4](https://github.com/CoreySpohn/orbix/commit/69d27a49e4e63a020962e7846d78b5b3b93b3c09))
* **orbit:** scaffold AbstractOrbit module ([28bb767](https://github.com/CoreySpohn/orbix/commit/28bb767591dfd4ecb71d6b536b0bbf396ef4824f))


### Bug Fixes

* **orbit:** import constants from hwoutils per migration plan ([d53582c](https://github.com/CoreySpohn/orbix/commit/d53582cbe5aeb3e4346f0c5efb620947c9d6e4c5))
* **planets:** convert K_mps from AU/d to m/s ([6ef2bd0](https://github.com/CoreySpohn/orbix/commit/6ef2bd0417a8696b39ba4584393758faed065368))
* **planets:** import constants from hwoutils per migration plan ([86e8660](https://github.com/CoreySpohn/orbix/commit/86e8660c723cc3022da1ff21cf82d9129bfe74b7))

## [0.4.0](https://github.com/CoreySpohn/orbix/compare/v0.3.0...v0.4.0) (2026-02-09)


### Features

* Adding observatory halo orbit and zodiacal light functions ([d853b5d](https://github.com/CoreySpohn/orbix/commit/d853b5d34a5ae4b2a3ecd148f4af8687881fbedc))

## [0.3.0](https://github.com/CoreySpohn/orbix/compare/v0.2.0...v0.3.0) (2025-12-20)


### Features

* Adding overhead time as an optional component of the dMag0 values. Fixing alpha log spacing issue ([2c0aeb8](https://github.com/CoreySpohn/orbix/commit/2c0aeb858cfd10fdcdd29ed678efe6d683a05603))

## [0.2.0](https://github.com/CoreySpohn/orbix/compare/v0.1.0...v0.2.0) (2025-11-20)


### Features

* Add solve_trig function for computing just sine and cosine of eccentric anomaly ([e7388c4](https://github.com/CoreySpohn/orbix/commit/e7388c413d5f8f95f0ace14df915c3c2f029a464))


### Bug Fixes

* update trig_solver field to static in System class ([24be3f0](https://github.com/CoreySpohn/orbix/commit/24be3f05723f1a62f4044f2de1fb921085b8b1e7))

## [0.1.0](https://github.com/CoreySpohn/orbix/compare/v0.0.2...v0.1.0) (2025-10-07)


### Features

* Add a position function that operates on a single orbit/position ([1e088d9](https://github.com/CoreySpohn/orbix/commit/1e088d9821c68012b577961aceb8955be30fe5d9))
* Add jit-compilation to the Planet class ([d0aac5e](https://github.com/CoreySpohn/orbix/commit/d0aac5e8f3c1269dbfb3957dfa57e030d3aa4bf6))
* Add Lambert phase function and polynomial approximation ([fe808d7](https://github.com/CoreySpohn/orbix/commit/fe808d7c8507f321292e8d06ac6e0b4c2af37d58))
* Add orbital mechanics equations ([84992df](https://github.com/CoreySpohn/orbix/commit/84992df3cefab98c78161574b334d36e473e939b))
* Add pos/velocity vector calculations and rework jits in Planet ([032064a](https://github.com/CoreySpohn/orbix/commit/032064ab99ab7ed65e8cd49a310cbc2ddc9ad693))
* Adding basic constants ([0c5c1cc](https://github.com/CoreySpohn/orbix/commit/0c5c1ccd015d422cb939714c767ed373af4f1a44))
* Adding initial planet/system/star objects ([38c7209](https://github.com/CoreySpohn/orbix/commit/38c72098ed2ce304fc2aceb33f549ce68afdd255))
* Basic propagation ([8ddad40](https://github.com/CoreySpohn/orbix/commit/8ddad409daa6b0988dba5c7411af281f7dcc8f35))
* Calculate probability of detection for EXOSIMS ([c9225f8](https://github.com/CoreySpohn/orbix/commit/c9225f8594f46fedf37b7bca20ee7709ef9bab9c))
* Fully grid based eccentric anomaly calculations with new fitting grids for joint RV/astrometric data ([c259fee](https://github.com/CoreySpohn/orbix/commit/c259fee3fb249005ffc32e5b9d29e6467fa03db3))
* Indexing improvement to bilinear E grid solver, add `get_grid_solver` function to easily select from the available ones ([d0781bc](https://github.com/CoreySpohn/orbix/commit/d0781bc4c2ffe772d929cd556ecb1cffec49b197))
* Introduce Planets class and update system structure ([bb7e467](https://github.com/CoreySpohn/orbix/commit/bb7e4679cd91c5978f9ee9ffc4d561f343bfbad3))
* Refactor Lambert phase function ([6e89e59](https://github.com/CoreySpohn/orbix/commit/6e89e59438b1f7dd328fb61683b6910d0d4f0dc2))
* Update to kEZ formulation ([8413e27](https://github.com/CoreySpohn/orbix/commit/8413e279b3495726b2674c91dcd4ce272a636dc6))

## [0.0.2](https://github.com/CoreySpohn/orbix/compare/v0.0.1...v0.0.2) (2024-11-02)


### Bug Fixes

* Removing unecessary jits ([ce37254](https://github.com/CoreySpohn/orbix/commit/ce3725419a5b948e5b9a3ce670fbe9f1468d301e))

## 0.0.1 (2024-11-02)


### Features

* Add eccentric anomaly solver ([cf44bb0](https://github.com/CoreySpohn/orbix/commit/cf44bb0133759fc5542861c769eace9457f8a3d7))


### Miscellaneous Chores

* release 0.0.1 ([d5626e9](https://github.com/CoreySpohn/orbix/commit/d5626e903416ba00d737a6ead04c13d3d1ccb844))
