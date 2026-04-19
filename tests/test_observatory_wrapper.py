"""Tests for orbix.Observatory composition wrapper."""

import jax
import jax.numpy as jnp

from orbix.observatory import Observatory, ObservatoryL2Halo


def test_observatory_defaults():
    """Bare Observatory(orbit=...) uses documented scalar defaults."""
    orbit = ObservatoryL2Halo.from_default()
    obs = Observatory(orbit=orbit)

    assert obs.orbit is orbit
    assert obs.temperature_K == 270.0
    assert obs.settling_time_d == 1.0
    assert obs.overhead_multi == 1.0
    assert obs.overhead_fixed_s == 0.0
    assert obs.stability_fact == 1.0
    assert obs.keepout_min_deg == 0.0
    assert obs.keepout_max_deg == 180.0


def test_observatory_custom_fields():
    """Scalar fields accept overrides via keyword arguments."""
    orbit = ObservatoryL2Halo.from_default()
    obs = Observatory(
        orbit=orbit,
        temperature_K=265.0,
        overhead_multi=1.1,
        overhead_fixed_s=100.0,
        stability_fact=0.95,
    )
    assert obs.temperature_K == 265.0
    assert obs.overhead_multi == 1.1
    assert obs.overhead_fixed_s == 100.0
    assert obs.stability_fact == 0.95


def test_observatory_is_pytree():
    """Observatory flattens and unflattens cleanly as a JAX pytree."""
    orbit = ObservatoryL2Halo.from_default()
    obs = Observatory(orbit=orbit, temperature_K=265.0)

    leaves, treedef = jax.tree_util.tree_flatten(obs)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.temperature_K == 265.0


def test_observatory_delegates_orbit_method():
    """Observatory.orbit retains the full ObservatoryL2Halo API surface."""
    orbit = ObservatoryL2Halo.from_default()
    obs = Observatory(orbit=orbit)

    pos = obs.orbit.position_ecliptic(jnp.asarray(60000.0))
    assert pos.shape == (3,)
