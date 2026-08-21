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
import jax.numpy as jnp
import numpy as np
import eyepiece as ep

from orbix import KeplerianOrbit
from orbix import viz

hwostyle.use("dark")
styles = ep.SourceStyles(["planet b"])
```

## One orbit on the sky

`plot_sky_track` propagates an orbit and draws its sky-plane track in
arcseconds: RA offset on the x axis (increasing to the left, per the
astronomical convention), Dec offset on the y axis, the star at the
origin, and an equal aspect so the ellipse is not distorted.

```{code-cell} ipython3
MSUN_KG = 1.988409870698051e30

orbit = KeplerianOrbit(
    a_AU=1.3, e=0.31, W_rad=2.3, i_rad=1.05, w_rad=-0.7,
    M0_rad=0.4, t0_d=2460000.0,
)
t_jd = jnp.linspace(2460000.0, 2460500.0, 120)

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
K = 40
draws = dict(
    T_d=431.7 * (1.0 + 0.06 * rng.standard_normal(K)),
    e=np.clip(0.31 + 0.05 * rng.standard_normal(K), 0.0, 0.9),
    cos_i=np.clip(0.5 + 0.08 * rng.standard_normal(K), -1.0, 1.0),
    W_rad=2.3 + 0.1 * rng.standard_normal(K),
    cos_w=np.cos(-0.7 + 0.2 * rng.standard_normal(K)),
    sin_w=np.sin(-0.7 + 0.2 * rng.standard_normal(K)),
    tp_d=2460000.0 + 15.0 * rng.standard_normal(K),
)
fan = KeplerianOrbit.from_period(**draws, Ms_kg=MSUN_KG)

epochs = (
    np.array([0.10, -0.05, -0.11]),
    np.array([0.04, 0.11, -0.02]),
    np.array([0.008, 0.008, 0.008]),
)
result = viz.plot_sky_track(
    fan, t_jd, Ms_kg=MSUN_KG, dist_pc=10.0,
    style=styles["planet b"], iwa=0.06, data=epochs,
)
```

Every function that accepts an orbit also accepts bare arrays, so tracks
loaded from a file draw through the same door:
`viz.plot_sky_track((ra, dec))`.

## The orbit in three dimensions

`plot_orbit` draws the star-centric orbit in AU through `eyepiece.trail`,
whose marker sizes shrink on the far side of the trajectory to cue depth.
Set the camera before calling it; the depth cue is baked from the camera
at call time. `marks` adds the exact periapsis (diamond) and the line of
nodes (triangles mark where the orbit pierces the sky plane).

```{code-cell} ipython3
import matplotlib.pyplot as plt

fig = plt.figure(layout="constrained")
ax = fig.add_subplot(projection="3d")
ax.view_init(elev=25.0, azim=-50.0)

result = viz.plot_orbit(
    orbit, t_jd, Ms_kg=MSUN_KG,
    ax=ax, style=styles["planet b"], marks={"periapsis", "nodes"},
)
```

## Animation

`animate_orbit` returns a lazy `eyepiece.Animation`: nothing renders until
a sink is asked for, and one animation can go to several sinks in one
pass (`.save("orbit.mp4", "orbit.gif")`). The `history` argument controls
the trail: `"all"` accumulates from the first epoch, an integer keeps a
trailing window, `"none"` moves the head marker alone.

```{code-cell} ipython3
from IPython.display import HTML

t_anim = jnp.linspace(2460000.0, 2460430.0, 30)
anim = viz.animate_orbit(
    orbit, t_anim, Ms_kg=MSUN_KG, dist_pc=10.0,
    style=styles["planet b"], iwa=0.06, fps=10,
)
HTML(anim.jshtml(dpi=100))
```
