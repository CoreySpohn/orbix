---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# When a target can be observed

A telescope at the Sun-Earth L2 point cannot point at every star on every day.
The Sun and the Earth each exclude a cone around themselves, and the angle
between a target and the Sun changes through the year. `orbix.observatory`
answers two questions about one target: is it outside the keepout zones at a
given time, and for what fraction of an interval is it. The same module also
gives the two angles that the local zodiacal light depends on.

Times are days from a static `Epoch`, not absolute dates: a float32 number
cannot hold a Julian date to better than about a quarter of a day (a modified
Julian date, about 0.004 d), while a few hundred days from a fixed epoch it
keeps a few seconds. The target is given by its right ascension and declination
in degrees, J2000.

```{code-cell} ipython3
import jax
import jax.numpy as jnp
from hwoutils.epoch import Epoch

from orbix.observatory import Spacecraft, ObservatoryL2Halo

jax.config.update("jax_enable_x64", True)

epoch = Epoch(2462502.5)  # 2030-01-01, the day that time_d counts from
ra_deg, dec_deg = 101.287, -16.716  # Sirius

spacecraft = Spacecraft(
    orbit=ObservatoryL2Halo.from_default(),
    keepout_min_deg=45.0,
    keepout_max_deg=180.0,
)
time_d = jnp.array([0.0, 91.0, 182.0, 273.0])
print(spacecraft.is_observable_at(ra_deg, dec_deg, epoch, time_d))
```

The telescope position on its halo orbit takes the same two arguments, an epoch
and days from it, and returns the heliocentric ecliptic position in AU.

```{code-cell} ipython3
print(spacecraft.orbit.position_ecliptic(epoch, 91.0))
```

The keepout limits are solar elongations: the target must lie at least
`keepout_min_deg` and at most `keepout_max_deg` from the Sun, as seen from the
spacecraft on its halo orbit. A single time answers "now". To ask how long the
target stays visible, give `observability` an interval.

```{code-cell} ipython3
window = spacecraft.observability(
    ra_deg, dec_deg, epoch, 0.0, 365.25, n_samples=73
)
print("visible fraction:", float(window.fraction))
print("ever visible:", bool(window.any), " always visible:", bool(window.all))
```

`observability` samples the interval on a uniform grid and refines every change
between visible and hidden by bisection, so the fraction is accurate to a small
part of the grid spacing. A keepout window narrower than the grid spacing can
fall between two samples, so choose `n_samples` so that the spacing is smaller
than the shortest window you care about. Both functions are pure, so `vmap` over
targets to score a whole list.

## The zodiacal light angles

The local zodiacal light depends on where the target sits relative to the Sun.
`ecliptic_pointing` returns the ecliptic latitude and the helioecliptic
longitude, the target longitude minus the Sun longitude, which are the two angles
the skyscapes zodi models take.

```{code-cell} ipython3
from orbix.observatory import ecliptic_pointing

latitude_deg, helioecliptic_longitude_deg = ecliptic_pointing(
    ra_deg, dec_deg, epoch, time_d
)
print("ecliptic latitude [deg]:", latitude_deg)
print("helioecliptic longitude [deg]:", helioecliptic_longitude_deg)
```

The latitude of a star does not change, while the longitude follows the Sun
around the year. Where the longitude is near plus or minus 180 degrees the
target is opposite the Sun.
