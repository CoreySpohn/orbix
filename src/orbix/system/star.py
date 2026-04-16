"""Star class."""

import equinox as eqx
from hwoutils.constants import G, Msun2kg, pc2AU


class Star(eqx.Module):
    """Star class representing a central body in a system.

    Attributes:
        Ms_kg: Mass of the star in kg.
        dist_pc: Distance to the star in pc.
        dist_AU: Distance to the star in AU (derived).
        mu: Gravitational parameter G * Ms (AU^3 / (kg * d^2)).
    """

    Ms_kg: float
    dist_pc: float
    dist_AU: float
    mu: float

    def __init__(self, Ms_Msun, dist_pc):
        """Create a Star.

        Args:
            Ms_Msun: Mass of the star in solar masses.
            dist_pc: Distance to the star in pc.
        """
        self.Ms_kg = Ms_Msun * Msun2kg
        self.dist_pc = dist_pc
        self.dist_AU = dist_pc * pc2AU
        self.mu = G * self.Ms_kg
