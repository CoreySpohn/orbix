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
import jax.numpy as jnp
import numpy as np
import eyepiece as ep

from orbix import KeplerianOrbit
from orbix import viz

hwostyle.use("dark")

MSUN_KG = 1.988409870698051e30
styles = ep.SourceStyles(["planet b", "planet c", "planet d"])

# One eccentric, inclined orbit carries most of this page.
orbit = KeplerianOrbit(
    a_AU=1.3, e=0.31, W_rad=2.3, i_rad=1.05, w_rad=-0.7,
    M0_rad=0.4, t0_d=2460000.0,
)
t_jd = jnp.linspace(2460000.0, 2460500.0, 120)
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
K = 60
fan = KeplerianOrbit.from_period(
    T_d=431.7 * (1.0 + 0.06 * rng.standard_normal(K)),
    e=np.clip(0.31 + 0.05 * rng.standard_normal(K), 0.0, 0.9),
    cos_i=np.clip(0.5 + 0.08 * rng.standard_normal(K), -1.0, 1.0),
    W_rad=2.3 + 0.1 * rng.standard_normal(K),
    cos_w=np.cos(-0.7 + 0.2 * rng.standard_normal(K)),
    sin_w=np.sin(-0.7 + 0.2 * rng.standard_normal(K)),
    tp_d=2460000.0 + 15.0 * rng.standard_normal(K),
    Ms_kg=MSUN_KG,
)
weights = rng.dirichlet(np.full(K, 3.0))

epochs = (
    np.array([0.10, -0.05, -0.11]),      # RA offset, arcsec
    np.array([0.04, 0.11, -0.02]),       # Dec offset, arcsec
    np.array([0.008, 0.008, 0.008]),     # 1-sigma
)
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

Matplotlib's tight-bbox cropping, which the notebook backend applies by
default, does not account for a 3D axes' z label and slices it off. That is a
matplotlib limitation rather than something for `plot_orbit` to distort its
layout around, so a notebook embedding these figures turns the cropping off.

```{code-cell} ipython3
import matplotlib.pyplot as plt

%config InlineBackend.print_figure_kwargs = {"bbox_inches": None}

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
t_anim = jnp.linspace(2460000.0, 2460000.0 + 365.25 * 2.35**1.5, 30)

anim = viz.animate_orbit(
    system, t_anim, Ms_kg=MSUN_KG, kind="3d", history="none",
    base_ms=viz.size_by_radius([1.0, 3.9, 11.2]), fps=10,
)
HTML(anim.jshtml(dpi=100))
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
%config InlineBackend.print_figure_kwargs = {"bbox_inches": "tight"}

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
