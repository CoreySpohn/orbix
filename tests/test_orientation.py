"""The isotropic orientation records agree with the chart reference measure."""

import jax
import jax.numpy as jnp
import numpy as np
import priorities.testing

from orbix.system import ClassicalChart, IsotropicInclination, isotropic_orientation

jax.config.update("jax_enable_x64", True)


def test_distribution_contract():
    """The record meets the priorities distribution contract."""
    priorities.testing.check_distribution(
        IsotropicInclination(), jax.random.key(0), n_draws=50_000
    )


def test_matches_classical_chart_reference_measure():
    """Density in i times |di/dz| is the chart density, up to the 1/2 normalization."""
    dist, chart = IsotropicInclination(), ClassicalChart()
    inc = jnp.linspace(0.1, 3.0, 9)
    z = -jnp.log(jnp.tan(inc / 2))
    x = jnp.zeros((9, 6)).at[:, 1].set(0.3).at[:, 2].set(z)
    base = jnp.zeros((9, 6)).at[:, 1].set(0.3)
    chart_inc = chart.log_volume(x) - chart.log_volume(base)
    # The chart density at z=0 is log(sech(0)^2)=0, so subtract nothing else.
    ours = dist.log_prob(inc) + jnp.log(jnp.sin(inc)) + jnp.log(2.0)
    np.testing.assert_allclose(ours, chart_inc, atol=1e-12)


def test_orientation_set():
    """The helper returns the inclination record and three uniform angles."""
    inc, node, peri, anomaly = isotropic_orientation()
    assert isinstance(inc, IsotropicInclination)
    for angle in (node, peri, anomaly):
        low, high = angle.support()
        assert (float(low), float(high)) == (0.0, 2 * np.pi)
