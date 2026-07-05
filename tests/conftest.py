"""Shared test configuration: the whole suite runs in float64."""

import jax

jax.config.update("jax_enable_x64", True)
