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

# The viz reference

Every function `orbix.viz` exports, with the smallest call that produces a
real picture. This page is a reference rather than a walkthrough: it is
organized by function, not by task, so it answers "what does this one draw?"
For the narrative version, which builds a figure up argument by argument,
see {doc}`examples/orbits`.

Every figure below is executed when these docs are built, on seeded
synthetic data and nothing else, so a function that stops rendering fails
the build rather than the documentation host.

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

MSUN_KG = 1.988409870698051e30
styles = ep.SourceStyles(["planet b", "planet c", "planet d"])

# One eccentric, inclined orbit carries most of this page. Its elements are
# named here because the posterior fan further down is built around them.
A_AU, ECC, INC_RAD = 1.3, 0.31, 1.05
BIG_OMEGA_RAD, SMALL_OMEGA_RAD, M0_RAD = 2.3, -0.7, 0.4
T0_D = 2460000.0

# Kepler's third law for a solar-mass star, so the track below can span
# exactly one period and close on itself.
PERIOD_D = A_AU**1.5 * 365.25
PERIAPSIS_D = T0_D - M0_RAD * PERIOD_D / (2.0 * np.pi)

orbit = KeplerianOrbit(
    a_AU=A_AU, e=ECC, W_rad=BIG_OMEGA_RAD, i_rad=INC_RAD,
    w_rad=SMALL_OMEGA_RAD, M0_rad=M0_RAD, t0_d=T0_D,
)
t_jd = jnp.linspace(T0_D, T0_D + PERIOD_D, 240)
```

## `plot_sky_track`

The sky-plane track in arcseconds: RA offset increasing to the left per the
astronomical convention, Dec offset up, the star at the origin, equal
aspect. `iwa` shades the inner working angle a coronagraph cannot see into.

```{code-cell} ipython3
result = viz.plot_sky_track(
    orbit, t_jd, Ms_kg=MSUN_KG, dist_pc=10.0,
    style=styles["planet b"], iwa=0.06,
)
```

The same function draws a `(K,)`-batched orbit as a fan of candidates faded
by weight, which is what a posterior looks like. `data` overlays measured
epochs as points with error bars.

```{code-cell} ipython3
rng = np.random.default_rng(11)
# The measured epochs come FROM the orbit above, perturbed by the measurement
# error -- otherwise the figure shows a posterior next to data it disagrees
# with, which is exactly the thing a fan is supposed to rule out.
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

# ... and the fan is a posterior scattered around that same orbit.
K = 60
fan = KeplerianOrbit.from_period(
    T_d=PERIOD_D * (1.0 + 0.04 * rng.standard_normal(K)),
    e=np.clip(ECC + 0.04 * rng.standard_normal(K), 0.0, 0.9),
    cos_i=np.clip(np.cos(INC_RAD) + 0.06 * rng.standard_normal(K), -1.0, 1.0),
    W_rad=BIG_OMEGA_RAD + 0.08 * rng.standard_normal(K),
    cos_w=np.cos(SMALL_OMEGA_RAD + 0.15 * rng.standard_normal(K)),
    sin_w=np.sin(SMALL_OMEGA_RAD + 0.15 * rng.standard_normal(K)),
    tp_d=PERIAPSIS_D + 12.0 * rng.standard_normal(K),
    Ms_kg=MSUN_KG,
)
weights = rng.dirichlet(np.full(K, 3.0))
result = viz.plot_sky_track(
    fan, t_jd, Ms_kg=MSUN_KG, dist_pc=10.0,
    style=styles["planet b"], weights=weights, iwa=0.06, data=epochs,
)
```

Bare arrays work anywhere an orbit does, so tracks loaded from a file draw
through the same door: `viz.plot_sky_track((ra, dec))`.

## `plot_orbit`

The star-centric orbit in three dimensions, in AU, through
`eyepiece.trail`. `marks` adds the exact periapsis (a diamond) and the line
of nodes (triangles, where the orbit pierces the sky plane).

Set the camera *before* calling: the per-point depth cue on a still is baked
from the camera at call time, so a `view_init` afterwards moves the scene
without moving the cue. That is also why this page creates the 3D axes by
hand here rather than letting the function do it.

```{code-cell} ipython3
import matplotlib.pyplot as plt

fig = plt.figure(layout="constrained")
ax = fig.add_subplot(projection="3d")
ax.view_init(elev=25.0, azim=-50.0)

result = viz.plot_orbit(
    orbit, t_jd, Ms_kg=MSUN_KG, ax=ax, marks={"periapsis", "nodes"},
)
```

## `animate_orbit`

Returns a lazy `eyepiece.Animation`: nothing renders until a sink is asked
for, and one animation feeds several sinks in a single draw pass
(`.save("orbit.mp4", "orbit.gif")`). Here it goes to `jshtml`, which needs
no ffmpeg and so survives any documentation builder.

`kind="3d"` gives the star-chart presentation, `kind="sky"` the sky-plane
track. `history` controls the trail: `"all"` accumulates, an integer keeps a
trailing window, `"none"` moves the heads alone.

`rotate` controls the camera sweep. A dict of `(start_deg, stop_deg)` pairs
gives full manual control, and any axis left out is held at its current value,
so the call below is a single-axis azimuth sweep with elevation pinned.

Where you put that sweep matters more than how wide it is. The projected area
of a planar ellipse is `pi * a * b * cos(tilt)`, with `tilt` measured to the
orbit normal, so an azimuth sweep that changes the tilt changes the drawn size
and the orbit reads as inflating rather than turning. Swept from azimuth -50 to
-10 this system's drawn size swings by 5.7x; the window used here holds it to
1.09x while staying 56 degrees off the normal, which is far enough from face-on
to keep the depth cue alive (the head markers span 0.09 to 0.91 of full scale).
`rotate="auto"` avoids the question by travelling a cone about the orbit normal
at fixed tilt, which holds the size exactly but moves azimuth and elevation
together; `None` holds the camera still.

```{code-cell} ipython3
from IPython.display import HTML

system = KeplerianOrbit(
    a_AU=np.array([0.72, 1.30, 2.35]),
    e=np.array([0.05, 0.21, 0.40]),
    W_rad=np.array([2.20, 2.35, 2.50]),
    i_rad=np.array([1.00, 1.08, 0.95]),
    w_rad=np.array([-0.40, -0.70, 0.90]),
    M0_rad=np.array([0.00, 2.10, 4.30]),
    t0_d=np.full(3, 2460000.0),
)
t_anim = jnp.linspace(2460000.0, 2460000.0 + 365.25 * 2.35**1.5, 72)

anim = viz.animate_orbit(
    system, t_anim, Ms_kg=MSUN_KG, kind="3d", history="none",
    base_ms=viz.size_by_radius([1.0, 3.9, 11.2]), fps=15,
    rotate={"azim": (-150.0, -110.0), "elev": (20.0, 20.0)},
)
HTML(anim.jshtml(dpi=130))
```

## `size_by_radius`

Marker diameters from planet radii, interpolated geometrically between
Mercury and Jupiter by default. The resting marker size is the anchor the
depth cue swells around, so it is the place physical meaning belongs.

```{code-cell} ipython3
names = ["Mercury", "Earth", "Neptune", "Jupiter"]
radii = [0.38, 1.0, 3.9, 11.2]

# One vectorized call: the result is always an array of shape (K,), so a
# scalar radius comes back as a one-element array rather than a float.
for name, r_earth, ms in zip(names, radii, viz.size_by_radius(radii)):
    print(f"{name:8s} {r_earth:5.2f} R_earth -> {ms:5.2f} pt")
```

The result is a set of marker **diameters**, which is what
`animate_orbit(base_ms=...)` takes. Do not hand it to
`plot_orbit(marker_scale=...)`: that reaches `scatter(s=...)`, an area, so
the encoding would be silently square-rooted.

## `depth_scale` and `depth_size`

The two halves of the depth cue, public so a caller can reproduce it on
their own artist. `depth_scale` turns positions and a camera into a factor
in `[0, 1]` -- 0 at the far side of the orbit, 1 at the near side.
`depth_size` maps that factor onto a marker-diameter multiplier.

Positions are `(N, 3)`, one row per point, while `propagate` returns
`(K, 3, N)`; transpose one orbit out of the batch to feed it.

```{code-cell} ipython3
xyz = np.asarray(orbit.propagate(t_jd=t_jd, Ms_kg=MSUN_KG)[0][0]).T
depth = viz.depth_scale(xyz, azim_deg=-50.0, elev_deg=25.0)
mult = viz.depth_size(depth)

print(f"depth  in [{depth.min():.2f}, {depth.max():.2f}]")
print(f"marker in [{mult.min():.3f}, {mult.max():.3f}] x base")
```

Both return plain arrays rather than figures, so this last picture is drawn
with bare matplotlib. There is no eyepiece primitive for a mapping curve and
there should not be: a primitive that wrapped `ax.plot` would be
reimplementing matplotlib, not adding anything.

```{code-cell} ipython3
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.2), layout="constrained")

ax1.plot(depth, lw=1.5)
ax1.set_xlabel("epoch index along the orbit")
ax1.set_ylabel("depth factor")
ax1.set_title("depth_scale along the track")

grid = np.linspace(0.0, 1.0, 200)
ax2.plot(grid, viz.depth_size(grid), lw=1.5)
ax2.set_xlabel("depth factor")
ax2.set_ylabel("diameter multiplier")
ax2.set_title("depth_size, peaking at +22%")
```

The far side is the anchor at exactly the base size and the near side swells
by about a fifth, so depth reads as a brief swell rather than a shrink, and
the resting size stays free to mean something physical.
