"""Tests for orbix.system.orbit (AbstractOrbit, KeplerianOrbit)."""

import pytest


def test_abstract_orbit_cannot_instantiate():
    """AbstractOrbit is abstract and must not be directly instantiable."""
    from orbix.system.orbit import AbstractOrbit

    with pytest.raises(TypeError):
        AbstractOrbit()
