"""System of a star and a set of planets."""

from typing import Tuple

import equinox as eqx

from orbix.kepler.shortcuts.grid import get_grid_solver

from .planets import Planets
from .star import Star


class System(eqx.Module):
    """System of a star and a set of planets.

    Args:
        star:
            Star object
        planets:
            Tuple of Planet objects
        E_solver:
            Function to solve Kepler's equation (M, e) -> E.
            Defaults to `orbix.kepler.core.E_solve`.
        name:
            Name of the system
    """

    star: Star
    planets: Tuple[Planets, ...]
    trig_solver: callable = eqx.static_field()

    def __init__(
        self,
        star,
        planets=(),
        *,
        solver=None,
    ):
        """Initialize a system with a star and optional planets.

        Args:
            star: Star object
            planets: List of Planet objects (optional)
            E_solver: Function to solve Kepler's equation (M, e) -> E.
                      Defaults to `orbix.eccanom.solve_E`.
            name: Name of the system (optional)
        """
        self.star = star
        self.planets = tuple(planets or [])
        self.trig_solver = get_grid_solver() if solver is None else solver

    def add_planet(self, **kwargs) -> "System":
        """Add a planet to the system and return a new system.

        Args:
            **kwargs: Keyword arguments for Planet constructor.
                     Should not include 'star_mass' as it will be taken from the star.

        Returns:
            A new System object with the added planet
        """
        # Create a new planet with the star's mass
        new_planet = Planets(Ms=self.star.Ms, dist=self.star.dist, **kwargs)

        # Create a new system with the new planet added, preserving the E_solver
        return System(
            star=self.star,
            planets=self.planets + (new_planet,),
            E_solver=self.E_solver,
            name=self.name,
        )
