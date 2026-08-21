"""Orbit animation: a ghost of the full path, a growing trail, a moving head.

Built on ``eyepiece.animate`` in update mode: the figure is drawn once
(the full track as a faint ghost, via the static plot functions, so an
animated figure looks exactly like its still counterpart), the animated
artists are created empty, and each frame is a ``set_data`` -- nothing is
cleared and no artist is created inside the frame loop.

The ``history`` vocabulary carries over from the key-strategy idea in
earlier orbit-animation code: ``"all"`` accumulates the trail from the
first epoch, an int keeps a trailing window of that many frames, and
``"none"`` moves the head marker alone.
"""

import numpy as np

from orbix.orbit import AbstractOrbit
from orbix.viz._require import eyepiece
from orbix.viz.orbit import (
    _positions,
    _resolve_color,
    _sky_tracks,
    plot_orbit,
    plot_sky_track,
)


def _history_slice(i, history):
    """The [start, stop) trail slice for frame ``i`` under ``history``."""
    if history == "all":
        return 0, i + 1
    return max(0, i + 1 - history), i + 1


def _track_alphas(n_tracks, weights):
    """Per-track alpha for the animated artists: opaque, faded by weight."""
    if weights is None:
        return [1.0] * n_tracks
    return [min(1.0, 0.3 + 0.7 * float(w)) for w in weights]


def animate_orbit(
    orbit_or_tracks,
    t_jd=None,
    *,
    Ms_kg=None,
    dist_pc=None,
    trig_solver=None,
    kind="sky",
    history="all",
    fps=10,
    style=None,
    weights=None,
    data=None,
    iwa=None,
    marks=None,
):
    """Animate one or more orbits and return a lazy ``eyepiece.Animation``.

    Nothing renders until ``.save``, ``.jshtml``, or ``.video`` is called
    on the result, so one animation can go to several sinks in one pass
    and the test path never needs ffmpeg.

    Args:
        orbit_or_tracks: An ``AbstractOrbit``, or bare tracks in the same
            forms the static functions accept (``(ra, dec)`` arrays or
            ``(2, T)`` / ``(K, 2, T)`` for ``kind="sky"``; ``(T, 3)`` /
            ``(K, T, 3)`` for ``kind="3d"``).
        t_jd: Times in Julian Days, shape ``(T,)``. Required on the orbit
            door; on the bare-track door it is optional and used only for
            the elapsed-time label (length must match the track).
        Ms_kg: Stellar mass in kg. Orbit door only.
        dist_pc: Distance in parsecs. Orbit door with ``kind="sky"`` only.
        trig_solver: Optional Kepler solver; None uses orbix's default.
        kind: ``"sky"`` (sky-plane arcsec, via ``plot_sky_track``) or
            ``"3d"`` (star-centric AU, via ``plot_orbit``).
        history: ``"all"`` grows the trail from the first epoch, an int
            keeps a trailing window of that many frames, ``"none"`` moves
            the head marker alone.
        fps: Default playback rate carried by the returned animation.
        style: A color or ``SourceStyles`` entry for the tracks, forwarded
            to the static function and used for the animated artists.
        weights: Optional per-track weights (length K), fading both the
            ghost fan and the animated trails.
        data: Optional ``(ra, dec, err)`` observed epochs (sky only).
        iwa: Optional inner-working-angle disk radius (sky only).
        marks: Optional ``{"periapsis", "nodes"}`` (3d orbit door only).

    Returns:
        An ``eyepiece.Animation`` bound to the built figure, with one
        frame per epoch of the track.

    Raises:
        ValueError: If ``kind`` or ``history`` is not one of the
            documented values, or if a labeled ``t_jd`` does not match the
            track length.
    """
    ep = eyepiece()
    if kind not in ("sky", "3d"):
        raise ValueError(f'kind must be "sky" or "3d", got {kind!r}')
    if history != "all" and history != "none" and not isinstance(history, int):
        raise ValueError(
            f'history must be "all", "none", or an int window, got {history!r}'
        )

    from_orbit = isinstance(orbit_or_tracks, AbstractOrbit)
    base_t = t_jd if from_orbit else None

    if kind == "sky":
        base = plot_sky_track(
            orbit_or_tracks,
            base_t,
            Ms_kg=Ms_kg,
            dist_pc=dist_pc,
            trig_solver=trig_solver,
            style=style,
            weights=weights,
            data=data,
            iwa=iwa,
            fan_kw={"alpha": 0.3} if _single(orbit_or_tracks) else None,
        )
        ra, dec, _ = _sky_tracks(orbit_or_tracks, base_t, Ms_kg, dist_pc, trig_solver)
        tracks = np.stack([ra, dec], axis=-1)
    else:
        base = plot_orbit(
            orbit_or_tracks,
            base_t,
            Ms_kg=Ms_kg,
            trig_solver=trig_solver,
            style=style,
            marks=marks,
            marker_scale=10.0,
            trail_kw={"alpha": 0.3},
        )
        tracks, _ = _positions(orbit_or_tracks, base_t, Ms_kg, trig_solver)

    ax = base.ax
    n_tracks, n_frames = tracks.shape[0], tracks.shape[1]

    times = None
    if t_jd is not None:
        times = np.asarray(t_jd, float).reshape(-1)
        if times.shape[0] != n_frames:
            raise ValueError(
                f"t_jd has {times.shape[0]} entries for a {n_frames}-epoch track"
            )

    color = _resolve_color(ep, style)
    alphas = _track_alphas(n_tracks, weights)

    trails, heads = [], []
    for k in range(n_tracks):
        if history != "none":
            if kind == "sky":
                (trail_line,) = ax.plot([], [], color=color, lw=1.5, alpha=alphas[k])
            else:
                (trail_line,) = ax.plot(
                    [], [], [], color=color, lw=1.5, alpha=alphas[k]
                )
            trails.append(trail_line)
        if kind == "sky":
            (head,) = ax.plot(
                [], [], linestyle="", marker="o", ms=6, color=color, alpha=alphas[k]
            )
        else:
            (head,) = ax.plot(
                [],
                [],
                [],
                linestyle="",
                marker="o",
                ms=6,
                color=color,
                alpha=alphas[k],
            )
        heads.append(head)

    label = None
    if times is not None:
        text_kw = {"transform": ax.transAxes, "ha": "left", "va": "top"}
        if kind == "sky":
            label = ax.text(0.02, 0.98, "", **text_kw)
        else:
            label = ax.text2D(0.02, 0.98, "", **text_kw)

    def draw(fig, i):
        """Advance every animated artist to epoch ``i`` (a set_data pass)."""
        start, stop = _history_slice(i, history)
        for k in range(n_tracks):
            segment = tracks[k, start:stop]
            if history != "none":
                if kind == "sky":
                    trails[k].set_data(segment[:, 0], segment[:, 1])
                else:
                    trails[k].set_data_3d(segment[:, 0], segment[:, 1], segment[:, 2])
            point = tracks[k, i]
            if kind == "sky":
                heads[k].set_data([point[0]], [point[1]])
            else:
                heads[k].set_data_3d([point[0]], [point[1]], [point[2]])
        if label is not None:
            label.set_text(f"t = +{times[i] - times[0]:.0f} d")

    return ep.animate(base.fig, draw, n_frames, fps=fps)


def _single(orbit_or_tracks):
    """Whether the input is a single track (K == 1), cheaply and safely.

    Only tunes the ghost's alpha, so an exotic ``AbstractOrbit`` without
    element arrays to inspect just keeps the fan default.
    """
    if isinstance(orbit_or_tracks, AbstractOrbit):
        elements = getattr(orbit_or_tracks, "a_AU", None)
        return elements is not None and int(np.asarray(elements).shape[0]) == 1
    if isinstance(orbit_or_tracks, (tuple, list)) and len(orbit_or_tracks) == 2:
        return np.asarray(orbit_or_tracks[0]).ndim == 1
    arr = np.asarray(orbit_or_tracks)
    return arr.ndim == 2
