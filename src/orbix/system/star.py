"""Star class."""

import equinox as eqx

from orbix.constants import G, Msun2kg, pc2AU


class Star(eqx.Module):
    """Star class representing a central body in a system.

    Attributes:
        Ms: Mass of the star in solar masses.
        dist: Distance to the star in pc.
        mu: Gravitational parameter (G * M) in AU^3/(kg*d^2).
    """

    Ms: float
    dist: float
    mu: float

    def __init__(self, Ms, dist):
        """Create necessary parameters for the star.

        Args:
            Ms: Mass of the star in solar masses.
            dist: Distance to the star in pc.
        """
        # Convert mass to kg internally
        self.Ms = Ms * Msun2kg
        self.mu = G * self.Ms
        self.dist = dist * pc2AU
