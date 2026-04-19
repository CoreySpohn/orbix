"""Observatory composition wrapper around ObservatoryL2Halo.

Bundles an orbit model with platform-level operational scalars
(temperature, overheads, stability, keepout limits) that jaxedith
and other downstream consumers read when assembling ETC count rates.

Follows equinox-best-practices Style A: pure field declarations with
defaults, no custom __init__.
"""

import equinox as eqx

from orbix.observatory.orbit import ObservatoryL2Halo


class Observatory(eqx.Module):
    """Space telescope observatory platform.

    Composes an :class:`ObservatoryL2Halo` orbit model with operational
    scalars (thermal state, overheads, wavefront stability, keepout
    limits). All scalar fields have sensible defaults so a bare
    ``Observatory(orbit=orbit)`` constructs a reasonable HWO-like
    platform.

    Args:
        orbit: L2 halo orbit model.
        temperature_K: Telescope equilibrium temperature in Kelvin.
        settling_time_d: Slew-settling time in days.
        overhead_multi: Multiplicative overhead factor on science time.
        overhead_fixed_s: Fixed overhead in seconds per observation.
        stability_fact: Wavefront stability factor (EXOSIMS speckle model).
        keepout_min_deg: Minimum allowed solar elongation in degrees.
        keepout_max_deg: Maximum allowed solar elongation in degrees.
    """

    orbit: ObservatoryL2Halo
    temperature_K: float = 270.0
    settling_time_d: float = 1.0
    overhead_multi: float = 1.0
    overhead_fixed_s: float = 0.0
    stability_fact: float = 1.0
    keepout_min_deg: float = 0.0
    keepout_max_deg: float = 180.0
