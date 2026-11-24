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

# Sky tracks, prepared once

A sky-plane figure of a planet's orbit is needed in several forms: a paper
still of a posterior fan, a strip of chosen epochs, a movie, and a clip in a
talk. Each form needs the same science: the orbits propagated over the time
grid, projected to arcseconds, and moved from the device to the host. The
preparation functions in `orbix.viz` do that work once and return
[eyepiece](https://eyepiece.readthedocs.io) prepared records, which
eyepiece's renderers draw. None of the outputs below propagates an orbit
again, so they cannot disagree about where the planet is.

- `prepare_sky_view` returns a static `TrackView` of full tracks, with no
  time axis.
- `prepare_sky_track` returns a `Sequence` whose samples are the propagated
  epochs, with a head per track, an optional trail, and a clock.

Their module imports NumPy, JAX, the orbit types, and `eyepiece.prepared`,
and no renderer. Drawing belongs to
`eyepiece.mpl` for figures and movies and `eyepiece.manim` for native Manim
clips. The records, scales, and sample-and-hold time rules are described in
eyepiece's
[prepared views guide](https://eyepiece.readthedocs.io/en/latest/prepared-views.html).

```{code-cell} ipython3
import jax

# Use double precision for propagation and the sampled-time clock.
jax.config.update("jax_enable_x64", True)

import hwostyle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML

import eyepiece.mpl as mpl
from eyepiece.prepared import find_element
from eyepiece.style import SourceCast, snapshot_profile
from orbix.system import FrameOrbit, offset_arcsec
from hwoutils.constants import G, Msun2kg
from orbix.viz import prepare_sky_track, prepare_sky_view

hwostyle.use("dark")

# Docs-builder concerns, not lines to copy into your own scripts: a fallback
# face for builders without the style's fonts, and small page images.
matplotlib.rcParams["font.sans-serif"] = list(
    matplotlib.rcParams["font.sans-serif"]
) + ["DejaVu Sans"]
matplotlib.rcParams["savefig.dpi"] = 120

# One cast and one profile for every output on this page, sized from the
# active style's rc settings.
CAST = SourceCast(["planet b"])
PROFILE = snapshot_profile(
    text_size_pt=plt.rcParams["font.size"],
    stroke_width_pt=plt.rcParams["lines.linewidth"],
)
```

## A posterior and its data

The planet below is synthetic: an eccentric, inclined orbit around a solar
mass star at 10 pc, three measured positions drawn from it with 8 mas
errors, and forty posterior draws scattered around it, each with a weight.

```{code-cell} ipython3
DIST_PC = 10.0
MU = G * Msun2kg
A_AU, ECC, INC_RAD = 1.3, 0.31, 1.05
BIG_OMEGA_RAD, SMALL_OMEGA_RAD, M0_RAD = 2.3, -0.7, 0.4
T0_D = 0.0
# Float days from the chosen epoch.
PERIOD_D = 2 * np.pi * np.sqrt(A_AU**3 / MU)
PERIAPSIS_D = T0_D - M0_RAD * PERIOD_D / (2.0 * np.pi)

orbit = FrameOrbit.from_classical(
    a_AU=A_AU,
    e=ECC,
    W_rad=BIG_OMEGA_RAD,
    i_rad=INC_RAD,
    w_rad=SMALL_OMEGA_RAD,
    M0_rad=M0_RAD,
    mu_AU3_per_d2=MU,
    epoch_d=T0_D,
)
rng = np.random.default_rng(0)

# Observed positions: the true orbit at three epochs, perturbed by the error.
SIGMA_ARCSEC = 0.008
t_obs = T0_D + np.array([70.0, 250.0, 430.0])
ra, dec = offset_arcsec(orbit.at(t_obs)[0], DIST_PC)
xy_obs = np.column_stack([np.asarray(ra), np.asarray(dec)])
xy_obs = xy_obs + SIGMA_ARCSEC * rng.standard_normal(xy_obs.shape)
sigma = np.full(len(t_obs), SIGMA_ARCSEC)
observations = (xy_obs, sigma, sigma)

# Forty posterior draws around the same orbit, and a weight for each.
K = 40
periods_d = PERIOD_D * (1.0 + 0.04 * rng.standard_normal(K))
periapsis_d = PERIAPSIS_D + 12.0 * rng.standard_normal(K)
fan = FrameOrbit.from_classical(
    a_AU=(MU * (periods_d / (2 * np.pi)) ** 2) ** (1 / 3),
    e=np.clip(ECC + 0.04 * rng.standard_normal(K), 0.0, 0.9),
    i_rad=np.arccos(np.clip(np.cos(INC_RAD) + 0.06 * rng.standard_normal(K), -1, 1)),
    W_rad=BIG_OMEGA_RAD + 0.08 * rng.standard_normal(K),
    w_rad=SMALL_OMEGA_RAD + 0.15 * rng.standard_normal(K),
    M0_rad=2 * np.pi * (T0_D - periapsis_d) / periods_d,
    mu_AU3_per_d2=MU,
    epoch_d=T0_D,
)
weights = rng.dirichlet(np.full(K, 3.0))

# One orbital period in 180 steps of about three days.
time_d = np.linspace(T0_D, T0_D + PERIOD_D, 181)
```

## A still of the fan

`prepare_sky_view` propagates every draw over `time_d` and returns one
`TrackView`. Each draw is its own `Path` (`path/0` to `path/39`), so every
draw stays addressable, while `source_ids="planet b"` names the one planet
they all describe, so they share one color from the cast. The weights become
relative opacity. The observations carry their error bars, and
`iwa_arcsec` adds the inner working angle as a shaded disk about the star.
RA offset increases to the left and the two axes share one scale; both are
part of the view, not options.

```{code-cell} ipython3
view = prepare_sky_view(
    fan,
    time_d=time_d,
    dist_pc=DIST_PC,
    source_ids="planet b",
    weights=weights,
    observations=observations,
    iwa_arcsec=0.06,
)
fig, ax = plt.subplots(figsize=(4.2, 4.0), layout="constrained")
fan_still = mpl.render(view, ax=ax, cast=CAST, profile=PROFILE)
print(len(fan_still.parts), "parts, including", sorted(fan_still.parts)[:4])
```

`fan_still.parts` maps each element ID to its Matplotlib artist, so one
draw, the star, or the IWA disk is adjusted through its handle. The
Matplotlib convenience `plot_sky_track` makes this same view and render in
one call.

## A sequence of epochs

`prepare_sky_track` takes the same arguments and returns a `Sequence` over
the propagated epochs. Its template adds a head per draw (`head/<k>`), which
holds the propagated position at the current sample, and a clock. `history`
adds a trail per draw: `"all"` grows from the first sample, and an integer
keeps a sliding window of that many samples, here about ninety days.

```{code-cell} ipython3
sequence = prepare_sky_track(
    fan,
    time_d=time_d,
    dist_pc=DIST_PC,
    source_ids="planet b",
    weights=weights,
    observations=observations,
    iwa_arcsec=0.06,
    history=30,
)
print(
    f"{len(sequence.times)} samples in {sequence.time_unit!r} after day {sequence.epoch}"
)
```

Times are days since the first sample, and the first sample's epoch-day value is
the sequence `epoch`. A head never moves along the chord between two
samples. Asked for a time between samples, the sequence holds the earlier
one, so a head is always a propagated position. On a sparsely sampled orbit
an interpolated head could cut across the inner working angle while the
orbit itself never enters it; a held head cannot.

```{code-cell} ipython3
between = 0.5 * (sequence.times[60] + sequence.times[61])
sample = sequence.at(between)
held = find_element(sample.view, "head/0").xy[0]
propagated = find_element(sequence.template, "path/0").xy[60]
print(
    f"t = {between:.1f} d holds sample {sample.index}:",
    np.array_equal(held, propagated),
)
```

## A still and a strip

`sequence.frame(i)` is the complete state at sample `i`, and
`sequence.strip(indices)` lays chosen samples side by side. Each strip slot's
IDs are prefixed with its position, and each slot's clock is its time label,
restamped with that slot's time. The heads show where the planet could be at each epoch: the spread of
the forty heads is the posterior uncertainty in its position at that time.

```{code-cell} ipython3
picks = [0, 60, 120]
fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.6), layout="constrained")
strip = mpl.render(sequence.strip(picks), axes=axes, cast=CAST, profile=PROFILE)
```

## A movie

`eyepiece.mpl.animate` plays the sequence in physical time: three seconds at
ten frames per second is thirty output frames, spaced evenly from the first
sample to the last. Each output frame holds the latest sample acquired at or
before its time, so the heads move at their projected speed on the sky:
quickly near periapsis and slowly near apoapsis.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(3.6, 3.4), layout="constrained")
movie = mpl.animate(sequence, run_time=3.0, fps=10, ax=ax, cast=CAST, profile=PROFILE)
html = movie.jshtml(dpi=100)
plt.close(fig)
print("output frames:", movie.n_frames)
HTML(html)
```

`animate_orbit(..., kind="sky")` wraps preparation and this call in one step,
for a movie that needs no other output.

## Bare tracks

Tracks that were propagated elsewhere, or loaded from a file, enter through
the same functions: an array of arcsecond offsets shaped `(K, T, 2)`, or an
`(ra, dec)` pair of `(T,)` or `(K, T)` arrays. A bare array is borrowed as
given, without a copy. `prepare_sky_view` draws bare tracks with no times at
all, and `prepare_sky_track` takes the times they were sampled at. A row with
a nonfinite coordinate is a gap in the track.

```{code-cell} ipython3
tracks = np.stack(
    [find_element(view, f"path/{k}").xy for k in range(K)]
)  # (K, T, 2) arcsec
bare = prepare_sky_view(tracks, source_ids="planet b", iwa_arcsec=0.06)
print(np.shares_memory(find_element(bare, "path/0").xy, tracks))
```

## In a talk

The same sequence plays as a native Manim clip through `eyepiece.manim`,
with a profile sized for slides and the same cast, so the planet keeps its
color from paper to talk. The code below is shown rather than run, because
rendering Manim media needs system libraries a documentation builder does
not have.

This block needs Manim and its system rendering libraries.

```python notest
import manim
import eyepiece.manim as em
from matplotlib.colors import to_hex

TALK = snapshot_profile()  # 24 pt text and 1.5 pt strokes


class Orbit(manim.Scene):
    def construct(self):
        self.camera.background_color = manim.ManimColor(to_hex(TALK.background_color))
        clip = em.animate(sequence, cast=CAST, profile=TALK)
        self.add(clip.mobject)
        self.play(clip.playback(run_time=8))
```

Render it with Manim's caching disabled (`manim --disable_caching`). The
playback follows the same physical-time schedule as the Matplotlib movie.
Eyepiece's [Manim guide](https://eyepiece.readthedocs.io/en/latest/manim.html)
covers the parts, reveals, and Manim Slides.

## Migrating from the earlier sky-track functions

Earlier releases drew sky tracks directly through `eyepiece.plot.sky_fan`.
`plot_sky_track` and `animate_orbit(kind="sky")` now prepare the tracks
first and render them through eyepiece, which changes the following
behavior.

- Bare tracks are shaped `(K, T, 2)`, (RA, Dec) last, or given as an
  `(ra, dec)` pair of `(T,)` or `(K, T)` arrays. The earlier `(2, T)` and
  `(K, 2, T)` arrays are no longer read: pass a `(2, T)` array as the pair
  `tuple(tracks)`, and a `(K, 2, T)` array as `np.moveaxis(tracks, 1, 2)`.
- `invert_ra=` is removed. RA always increases to the left.
- `plot_sky_track` returns an `eyepiece.mpl.MplResult`, not a `PlotResult`.
  Track `k`'s line is `result.parts["path/k"]`, and the IWA disk and the
  observations are `result.parts["iwa"]` and `result.parts["observations"]`.
- Bare tracks are in arcsec and get the same RA and Dec offset axis labels
  as propagated orbits.
- A lone track is drawn at opacity 0.80, the top of the shared weight
  mapping, rather than 0.75. Heads carry their track's weight and fade with
  it; the star and the observations carry the top track weight, so they
  stay at full opacity.
- `plot_sky_track` and `animate_orbit(kind="sky")` take the measured offsets as
  `observations=(xy, xerr, yerr)` and the inner working angle as `iwa_arcsec=`;
  the earlier `data=` and `iwa=` keywords are gone. Convert `(ra, dec, err)` by
  hand:

```python
import matplotlib

matplotlib.use("Agg")
import numpy as np
from orbix.viz import plot_sky_track

t = np.linspace(0.0, 6.0, 10)
ra, dec, err = np.cos(t), np.sin(t), np.full(10, 0.05)
result = plot_sky_track(
    (ra, dec),
    observations=(np.column_stack([ra, dec]), err, err),
    iwa_arcsec=0.1,
)
```

- `animate_orbit(kind="sky")` refuses `style=`, `base_ms=`, and `marks=`,
  which apply to `kind="3d"` only; sky colors come from the cast. It needs
  `time_d=` for bare tracks too, because the sample times drive its clock and
  its physical-time playback.
