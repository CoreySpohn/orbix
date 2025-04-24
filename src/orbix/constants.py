"""Constants."""

import jax.numpy as jnp

# Mathematical constants
two_pi = 2 * jnp.pi
pi_over_2 = jnp.pi / 2
eps = jnp.finfo(jnp.float32).eps

# Gravitational constant in AU^3 / (kg d^2)
G = 1.488185170234519e-34

# Mass conversion factors
Msun2kg = 1.988409870698051e30  # kg
Mearth2kg = 5.972167867791379e24  # kg

# Distance conversion factors
m2AU = 6.684587122268445e-12  # AU
Rearth2AU = 4.263496512454037e-05  # AU
AU2m = 1.495978707e11  # m
pc2AU = 2.062648062470964e05  # AU

# Time conversion factors
d2s = 86400.0  # s
s2d = 1.157407407407407e-05  # d

# Angular conversion factors
rad2arcsec = 206264.80624709636  # arcsec
arcsec2rad = 4.84813681109536e-06  # rad
