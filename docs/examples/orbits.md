---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Plotting orbits

`orbix.viz` turns an orbit into a figure in one call. It is installed with
the `viz` extra (`pip install 'orbix[viz]'`) and is built on eyepiece, so
every figure here follows the same conventions as any other figure in the
eyepiece fleet: stateless ax-first functions, a `PlotResult` back, and
style resolved from hwostyle at call time.

Every page starts the same way: activate a style mode, import the plotting
module, and declare the document's cast.

```{code-cell} ipython3
import hwostyle
import matplotlib
import jax.numpy as jnp
import numpy as np
import eyepiece as ep

from orbix import KeplerianOrbit
from orbix import viz

hwostyle.use("dark")

# Docs-builder concerns, not lines to copy into your own scripts. hwostyle asks
# for Inter/Helvetica/Arial and a CI builder has none of them, so name the face
# matplotlib always ships as a last resort -- otherwise every figure emits a
# findfont warning. And render at a resolution that holds up on a high-DPI
# screen; the notebook default of 100 dpi does not.
matplotlib.rcParams["font.sans-serif"] = list(
    matplotlib.rcParams["font.sans-serif"]
) + ["DejaVu Sans"]
matplotlib.rcParams["figure.dpi"] = 160

styles = ep.SourceStyles(["planet b"])
```

## One orbit on the sky

`plot_sky_track` propagates an orbit and draws its sky-plane track in
arcseconds: RA offset on the x axis (increasing to the left, per the
astronomical convention), Dec offset on the y axis, the star at the
origin, and an equal aspect so the ellipse is not distorted.

```{code-cell} ipython3
MSUN_KG = 1.988409870698051e30

# The elements are named because the posterior fan further down is built
# around this same orbit.
A_AU, ECC, INC_RAD = 1.3, 0.31, 1.05
BIG_OMEGA_RAD, SMALL_OMEGA_RAD, M0_RAD = 2.3, -0.7, 0.4
T0_D = 2460000.0

# Kepler's third law for a solar-mass star, so the track spans exactly one
# period and the ellipse closes instead of stopping in mid-air.
PERIOD_D = A_AU**1.5 * 365.25
PERIAPSIS_D = T0_D - M0_RAD * PERIOD_D / (2.0 * np.pi)

orbit = KeplerianOrbit(
    a_AU=A_AU, e=ECC, W_rad=BIG_OMEGA_RAD, i_rad=INC_RAD,
    w_rad=SMALL_OMEGA_RAD, M0_rad=M0_RAD, t0_d=T0_D,
)
t_jd = jnp.linspace(T0_D, T0_D + PERIOD_D, 240)

# A second cast for the 3D views: three near-coplanar planets, the way a
# planetary system actually sits. Keeping the mutual inclinations small (7 to
# 16 degrees) keeps the z excursion to about a quarter of the in-plane extent,
# so the orbits read as nested ellipses seen from above rather than as thin
# slivers pointed at the camera.
system = KeplerianOrbit(
    a_AU=np.array([0.72, 1.30, 2.35]),
    e=np.array([0.05, 0.21, 0.40]),
    W_rad=np.array([2.20, 2.35, 2.50]),
    i_rad=np.array([0.12, 0.20, 0.28]),
    w_rad=np.array([-0.40, -0.70, 0.90]),
    M0_rad=np.array([0.00, 2.10, 4.30]),
    t0_d=np.full(3, T0_D),
)
OUTER_PERIOD_D = 2.35**1.5 * 365.25
t_system = jnp.linspace(T0_D, T0_D + OUTER_PERIOD_D, 240)

result = viz.plot_sky_track(
    orbit, t_jd, Ms_kg=MSUN_KG, dist_pc=10.0,
    style=styles["planet b"], iwa=0.06,
)
```

The shaded disk is an inner working angle: the region a coronagraph
cannot see into, in the same arcsecond units as the track.

## A posterior fan

A `(K,)`-batched orbit draws as a fan of candidate tracks, faded by
per-track weights. `KeplerianOrbit.from_period` builds the batch directly
from the parameterization orbit-fitting code samples, so posterior draws
become a fan in two calls.

```{code-cell} ipython3
rng = np.random.default_rng(7)

# The measured epochs are sampled from the orbit above and perturbed by the
# measurement error, so the fan and the data describe the same system.
SIGMA_ARCSEC = 0.008
t_obs = T0_D + np.array([70.0, 250.0, 430.0])
ra_true, dec_true = orbit.position_arcsec(
    t_jd=jnp.asarray(t_obs), Ms_kg=MSUN_KG, dist_pc=10.0
)
epochs = (
    np.asarray(ra_true)[0] + SIGMA_ARCSEC * rng.standard_normal(3),
    np.asarray(dec_true)[0] + SIGMA_ARCSEC * rng.standard_normal(3),
    np.full(3, SIGMA_ARCSEC),
)

K = 40
draws = dict(
    T_d=PERIOD_D * (1.0 + 0.04 * rng.standard_normal(K)),
    e=np.clip(ECC + 0.04 * rng.standard_normal(K), 0.0, 0.9),
    cos_i=np.clip(np.cos(INC_RAD) + 0.06 * rng.standard_normal(K), -1.0, 1.0),
    W_rad=BIG_OMEGA_RAD + 0.08 * rng.standard_normal(K),
    cos_w=np.cos(SMALL_OMEGA_RAD + 0.15 * rng.standard_normal(K)),
    sin_w=np.sin(SMALL_OMEGA_RAD + 0.15 * rng.standard_normal(K)),
    tp_d=PERIAPSIS_D + 12.0 * rng.standard_normal(K),
)
fan = KeplerianOrbit.from_period(**draws, Ms_kg=MSUN_KG)
result = viz.plot_sky_track(
    fan, t_jd, Ms_kg=MSUN_KG, dist_pc=10.0,
    style=styles["planet b"], iwa=0.06, data=epochs,
)
```

Every function that accepts an orbit also accepts bare arrays, so tracks
loaded from a file draw through the same door:
`viz.plot_sky_track((ra, dec))`.

## The orbit in three dimensions

`plot_orbit` draws the star-centric orbit in AU through `eyepiece.trail`.
With no `style` it uses the star-chart look: markers in the mode's text
color (white dots on this dark background) over a transparent dashed gray
path, with the 3D panes painted in the background color so the scene
reads as space. Passing `style=` opts into that source's solid color
instead. Set the camera before calling it; the per-point depth cue on a
still is baked from the camera at call time. `marks` adds the exact
periapsis (diamond) and the line of nodes (triangles mark where each
orbit pierces the sky plane).

This draws the three-planet system rather than the single track above: a
set of near-coplanar orbits is what a 3D view is for, since they nest and
the outer planet's eccentricity shows as the star sitting off centre. Point
the camera down on the system's plane rather than at its edge -- the default
3D view sits 79 degrees off an orbit normal like this one, which squashes an
ellipse to a fifth of its width and reads as a sliver aimed at the viewer.

```{code-cell} ipython3
import matplotlib.pyplot as plt

fig = plt.figure(layout="constrained")
ax = fig.add_subplot(projection="3d")
ax.view_init(elev=46.0, azim=-130.0)

result = viz.plot_orbit(
    system, t_system, Ms_kg=MSUN_KG, ax=ax, marks={"periapsis", "nodes"},
)
```

## Animation

`animate_orbit` returns a lazy `eyepiece.Animation`: nothing renders until
a sink is asked for, and one animation can go to several sinks in one
pass (`.save("orbit.mp4", "orbit.gif")`). The 3D defaults do the whole
star-chart presentation in one call: dashed gray paths, a moving dot per
orbit that swells gently on the near side of the trajectory, and a slow
single-axis azimuth sweep, with elevation held. `rotate` takes a dict of
`(start_deg, stop_deg)` pairs and holds any axis you leave out, so the call
below sweeps azimuth alone; `rotate="auto"` instead travels a cone about
the orbit normal, and `None` holds the camera still.

Place the sweep with care. The projected area of a planar ellipse goes as
`cos(tilt)` to the orbit normal, so an azimuth window that changes the tilt
changes the drawn size and the orbits read as inflating rather than turning.
A near-coplanar system makes this easy: its normal is only 14 degrees off
the rotation axis, so a window centred on that normal's azimuth holds the
drawn size to 1.02x, where one placed a quarter turn away gives 1.18x.
Elevation 46 leaves the camera 58 degrees off the normal, far enough from
face-on that the depth cue still reads.

The `history` argument controls the trail: `"all"` accumulates from the first epoch, an integer
keeps a trailing window, `"none"` moves the head markers alone. The base
marker size is the anchor the depth cue swells around, so it is where
physical meaning lives: `size_by_radius` maps planet radii onto marker
diameters. The same call with `kind="sky"` animates the sky-plane track
instead.

```{code-cell} ipython3
from IPython.display import HTML

t_anim = jnp.linspace(T0_D, T0_D + OUTER_PERIOD_D, 72)

anim = viz.animate_orbit(
    system, t_anim, Ms_kg=MSUN_KG, kind="3d", history="none",
    base_ms=viz.size_by_radius([1.0, 3.9, 11.2]), fps=15,
    rotate={"azim": (-150.0, -110.0), "elev": (46.0, 46.0)},
)
HTML(anim.jshtml(dpi=130))
```
