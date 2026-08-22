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
    _orbit_look,
    _per_track,
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


def depth_size(depth):
    """Map a [0, 1] depth factor onto a marker-diameter multiplier.

    The tuning is inherited from the original hand-tuned orbit renders:
    the marker's scatter AREA grew additively by at most half its base
    on the near side, which in diameter terms is ``sqrt(1 + 0.5 * d)``
    -- a swell that peaks at about 22 percent. The far side is the
    anchor at exactly the base size, so the resting size stays free to
    encode physical meaning such as a planet radius; depth reads as a
    brief near-side swell, not a shrink.

    Callers styling a scatter artist (whose ``s`` is an area) should
    square this factor to stay in area units.
    """
    return np.sqrt(1.0 + 0.5 * depth)


def depth_scale(positions, azim_deg, elev_deg):
    """Per-point head-marker scale in [0, 1] from the camera geometry.

    The same viewer-angle cue ``eyepiece.trail`` bakes into its per-point
    markers -- ``(1 + cos(angle)) / 2`` between each position vector and
    the position-to-viewer vector, with the viewer far along the camera
    direction -- but here it drives the one moving head marker instead of
    beads along the whole path: in an animation the head can carry the
    depth cue itself, so the path stays a clean line.
    """
    radius = max(float(np.max(np.linalg.norm(positions, axis=-1))), 1.0)
    elev_rad, azim_rad = np.deg2rad(elev_deg), np.deg2rad(azim_deg)
    r_v = (
        1.0e3
        * radius
        * np.array(
            [
                np.cos(elev_rad) * np.cos(azim_rad),
                np.cos(elev_rad) * np.sin(azim_rad),
                np.sin(elev_rad),
            ]
        )
    )
    r_ov = positions - r_v
    dot = -np.einsum("ij,ij->i", r_ov, positions)
    denom = np.linalg.norm(r_ov, axis=-1) * np.linalg.norm(positions, axis=-1)
    denom = np.where(denom == 0.0, np.finfo(float).eps, denom)
    cos_angle = np.clip(dot / denom, -1.0, 1.0)
    return (1.0 + cos_angle) / 2.0


def animate_orbit(
    orbit_or_tracks,
    t_jd=None,
    *,
    Ms_kg=None,
    dist_pc=None,
    trig_solver=None,
    kind="sky",
    history="all",
    rotate="auto",
    fps=10,
    style=None,
    base_ms=6.0,
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
        rotate: Camera sweep, ``kind="3d"`` only. The default ``"auto"``
            is a slow single-axis sweep about z -- 40 degrees of azimuth
            from the camera's starting position, elevation held -- since
            one-axis parallax is what makes the 3D geometry legible
            without disorienting the viewer. ``None`` holds the camera
            still. A dict maps any of ``"azim"``, ``"elev"``, ``"roll"``
            to a ``(start_deg, stop_deg)`` pair interpolated linearly
            across the frames, for full control. The head's depth cue
            tracks the moving camera frame by frame. Ignored for
            ``kind="sky"`` unless a dict is passed, which raises.
        fps: Default playback rate carried by the returned animation.
        style: A color or ``SourceStyles`` entry for the tracks, forwarded
            to the static function and used for the animated artists. For
            ``kind="3d"``, None gives the star-chart default: heads in the
            mode's text color (white dots on a dark background) over
            transparent dashed gray trails; a ``style`` opts into that
            source's solid color instead.
        base_ms: Head-marker diameter in points, a scalar or one value
            per track. The base size is the anchor the depth cue swells
            around, so it is where physical meaning lives -- pass
            ``size_by_radius(radii)`` to encode planet radii.
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
    if isinstance(rotate, dict):
        if kind != "3d":
            raise ValueError('rotate only applies to kind="3d"')
        unknown = set(rotate) - {"azim", "elev", "roll"}
        if unknown:
            raise ValueError(
                f"unknown rotate keys {sorted(unknown)}; expected azim/elev/roll"
            )
    elif rotate not in (None, "auto"):
        raise ValueError(
            f'rotate must be "auto", None, or an angle dict, got {rotate!r}'
        )
    if kind != "3d":
        rotate = None

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
        # marker_scale=0 keeps the ghost a bare line: the moving head
        # carries the depth cue in an animation, so per-point beads along
        # the whole path would only be clutter.
        base = plot_orbit(
            orbit_or_tracks,
            base_t,
            Ms_kg=Ms_kg,
            trig_solver=trig_solver,
            style=style,
            marks=marks,
            marker_scale=0.0,
            trail_kw={"alpha": 0.45},
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

    alphas = _track_alphas(n_tracks, weights)
    if kind == "3d":
        # the star-chart default: heads in the text color (white dots on a
        # dark mode) edged in the background color, trails dashed gray; a
        # style= opts into that source's solid color for both.
        head_color, path_kw = _orbit_look(style, ep)
        trail_color = path_kw.get("color", head_color)
        trail_ls = path_kw.get("linestyle", "-")
        trail_alpha = 0.75 if style is None else 1.0
        head_kw = {}
        if style is None:
            import matplotlib as mpl

            head_kw = {
                "markeredgecolor": mpl.rcParams["axes.facecolor"],
                "markeredgewidth": 0.8,
            }
    else:
        head_color = trail_color = _resolve_color(ep, style)
        trail_ls = "-"
        trail_alpha = 1.0
        head_kw = {}

    head_ms = _per_track(base_ms, n_tracks, "base_ms")
    head_scales = None
    camera = None
    if kind == "3d":
        if rotate == "auto":
            rotate = {"azim": (ax.azim, ax.azim + 40.0)}
        if rotate is not None:
            base_angles = {"azim": ax.azim, "elev": ax.elev, "roll": ax.roll}
            camera = {
                key: np.linspace(*rotate[key], n_frames)
                if key in rotate
                else np.full(n_frames, base_angles[key])
                for key in ("azim", "elev", "roll")
            }
            head_scales = np.stack(
                [
                    np.array(
                        [
                            depth_scale(
                                tracks[k, i : i + 1],
                                camera["azim"][i],
                                camera["elev"][i],
                            )[0]
                            for i in range(n_frames)
                        ]
                    )
                    for k in range(n_tracks)
                ]
            )
        else:
            head_scales = np.stack(
                [depth_scale(tracks[k], ax.azim, ax.elev) for k in range(n_tracks)]
            )

    trails, heads = [], []
    for k in range(n_tracks):
        if history != "none":
            if kind == "sky":
                (trail_line,) = ax.plot(
                    [], [], color=trail_color, lw=1.5, alpha=alphas[k]
                )
            else:
                (trail_line,) = ax.plot(
                    [],
                    [],
                    [],
                    color=trail_color,
                    linestyle=trail_ls,
                    lw=1.5,
                    alpha=trail_alpha * alphas[k],
                )
            trails.append(trail_line)
        if kind == "sky":
            (head,) = ax.plot(
                [],
                [],
                linestyle="",
                marker="o",
                ms=head_ms[k],
                color=head_color,
                alpha=alphas[k],
            )
        else:
            (head,) = ax.plot(
                [],
                [],
                [],
                linestyle="",
                marker="o",
                ms=head_ms[k],
                color=head_color,
                alpha=alphas[k],
                **head_kw,
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
        if camera is not None:
            ax.view_init(
                elev=camera["elev"][i],
                azim=camera["azim"][i],
                roll=camera["roll"][i],
            )
        if history != "none":
            start, stop = _history_slice(i, history)
        for k in range(n_tracks):
            if history != "none":
                segment = tracks[k, start:stop]
                if kind == "sky":
                    trails[k].set_data(segment[:, 0], segment[:, 1])
                else:
                    trails[k].set_data_3d(segment[:, 0], segment[:, 1], segment[:, 2])
            point = tracks[k, i]
            if kind == "sky":
                heads[k].set_data([point[0]], [point[1]])
            else:
                heads[k].set_data_3d([point[0]], [point[1]], [point[2]])
                # never vanish entirely: the far side reads as "small",
                # not "gone behind the star"
                heads[k].set_markersize(head_ms[k] * depth_size(head_scales[k, i]))
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
