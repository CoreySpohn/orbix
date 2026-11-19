"""Observatory composition wrapper around ObservatoryL2Halo.

Bundles an orbit model with platform-level operational scalars
(temperature, overheads, stability, keepout limits) that jaxedith
and other downstream consumers read when assembling ETC count rates.

Follows equinox-best-practices Style A: pure field declarations with
defaults, no custom __init__.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.epoch import Epoch
from jaxtyping import Array, ArrayLike

from orbix.observatory.keepout import is_observable
from orbix.observatory.orbit import ObservatoryL2Halo, epoch_mjd

_BISECTION_STEPS = 8


class Observability(eqx.Module):
    """Sampled visibility and refined interval summaries for one target.

    ``time_d`` is the closed uniform grid in days from the supplied epoch;
    ``is_observable`` contains the boolean keepout decisions on that grid.
    Both have shape ``(n_samples,)``. ``fraction``
    integrates visible duration using refined transition midpoints. ``first_d``
    and ``last_d`` are the earliest/latest visible boundary estimates, or
    ``inf``/``-inf`` if no grid sample is visible. ``any`` and ``all`` describe
    the grid decisions. All summaries are scalar arrays.

    See :meth:`Observatory.observability` for resolution and missed-window
    limitations; boundary estimates need not themselves be observable.
    """

    time_d: Array
    is_observable: Array
    fraction: Array
    first_d: Array
    last_d: Array
    any: Array
    all: Array


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

    def _is_observable_mjd(self, ra_deg, dec_deg, mjd) -> Array:
        """Broadcast inputs and map the scalar keepout geometry over them."""
        ra, dec, time = jnp.broadcast_arrays(ra_deg, dec_deg, mjd)

        def evaluate(r, d, m):
            return is_observable(
                self.orbit.position_ecliptic(m),
                r,
                d,
                m,
                ko_sun_min_deg=self.keepout_min_deg,
                ko_sun_max_deg=self.keepout_max_deg,
            )

        return jax.vmap(evaluate)(ra.ravel(), dec.ravel(), time.ravel()).reshape(
            time.shape
        )

    def is_observable_at(
        self, ra_deg: ArrayLike, dec_deg: ArrayLike, epoch: Epoch, time_d: ArrayLike
    ) -> Array:
        """Check keepout at day offsets from a static ``epoch``.

        RA, Dec (degrees, J2000), and ``time_d`` broadcast to one common output shape.
        Uses this observatory's orbit and solar keepout limits; Earth limits
        retain the keepout function's permissive defaults. Conversion to MJD
        preserves the epoch's time scale, with no time-scale conversion.
        Close over ``epoch`` or mark it static when using ``jax.jit``.
        """
        return self._is_observable_mjd(ra_deg, dec_deg, epoch_mjd(epoch, time_d))

    def observability(
        self,
        ra_deg: ArrayLike,
        dec_deg: ArrayLike,
        epoch: Epoch,
        start_d: ArrayLike,
        stop_d: ArrayLike,
        *,
        n_samples: int = 64,
    ) -> Observability:
        """Summarize one target over the closed interval in days from ``epoch``.

        Inputs are scalar, with ``stop_d >= start_d``. Use ``vmap`` over targets
        or intervals. ``epoch`` and ``n_samples >= 2`` must be static under JIT.
        The uniform grid includes both endpoints. A zero-duration interval
        has fraction 1 or 0 according to its single-time keepout decision.

        Each pair of opposite grid decisions brackets a transition. Eight
        bisections give a bracket width ``delta_d = (stop_d-start_d) /
        ((n_samples-1) * 256)``. The midpoint estimates ``first_d``, ``last_d``
        and each transition used for ``fraction`` to within ``delta_d/2``
        (plus floating-point roundoff), assuming one transition per bracket.
        Fraction integrates constant visible pieces and a trapezoidal step
        across each final bracket: its error is at most ``N * delta_d/2``
        divided by interval duration for N detected transitions.

        Windows or keepouts narrower than the grid spacing
        ``h_d = (stop_d-start_d)/(n_samples-1)`` can be missed, including pairs
        of transitions between equal decisions. Choose ``h_d`` smaller than
        the shortest window or keepout to resolve; for width ``w_d``, use
        ``n_samples > 1 + (stop_d-start_d)/w_d``. ``any`` and ``all`` describe
        sampled decisions and share this limitation.
        """
        if n_samples < 2:
            raise ValueError("n_samples must be at least 2")
        time = jnp.linspace(start_d, stop_d, n_samples)

        def evaluate(t):
            return self._is_observable_mjd(ra_deg, dec_deg, epoch_mjd(epoch, t))

        visible = evaluate(time)
        left, right = time[:-1], time[1:]
        left_visible, right_visible = visible[:-1], visible[1:]
        transition = left_visible != right_visible

        def bisect(_, bracket):
            lo, hi = bracket
            mid = lo + (hi - lo) / 2
            same = evaluate(mid) == left_visible
            return jnp.where(same, mid, lo), jnp.where(same, hi, mid)

        lo, hi = jax.lax.fori_loop(0, _BISECTION_STEPS, bisect, (left, right))
        boundary = lo + (hi - lo) / 2
        duration = jnp.where(
            transition,
            jnp.where(left_visible, boundary - left, right - boundary),
            jnp.where(left_visible, right - left, 0.0),
        )
        span = jnp.asarray(stop_d) - jnp.asarray(start_d)
        fraction = jnp.where(
            span == 0,
            visible[0].astype(time.dtype),
            jnp.sum(duration) / jnp.where(span == 0, 1.0, span),
        )
        fraction = jnp.where(jnp.all(visible), 1.0, fraction)
        first = jnp.minimum(
            jnp.min(jnp.where(visible, time, jnp.inf)),
            jnp.min(jnp.where(transition & right_visible, boundary, jnp.inf)),
        )
        last = jnp.maximum(
            jnp.max(jnp.where(visible, time, -jnp.inf)),
            jnp.max(jnp.where(transition & left_visible, boundary, -jnp.inf)),
        )
        return Observability(
            time_d=time,
            is_observable=visible,
            fraction=fraction,
            first_d=first,
            last_d=last,
            any=jnp.any(visible),
            all=jnp.all(visible),
        )
