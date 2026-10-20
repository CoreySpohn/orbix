"""Orbit animation: a ghost of the full path, a growing trail, a moving head.

The sky-plane animation is a prepared sequence (``prepare_sky_track``)
played by ``eyepiece.mpl.animate``: the orbit is propagated once, and each
output frame holds the propagated sample at its physical time. The 3D
animation is built on ``eyepiece.animate`` in update mode: the figure is
drawn once (the full track as a faint ghost, via the static plot function,
so an animated figure's ghost matches its still counterpart's path), the
animated artists are created empty, and each frame is a ``set_data`` --
nothing is cleared and no artist is created inside the frame loop.

Orbit tracks use right-handed (east, north, away) coordinates in AU.
The camera and depth cues operate in that same frame; an observer at -z
sees east left and north up with elev=-90, azim=90, roll=0.

The ``history`` vocabulary carries over from the key-strategy idea in
earlier orbit-animation code: ``"all"`` accumulates the trail from the
first epoch, an int keeps a trailing window of that many frames, and
``"none"`` moves the head marker alone.
"""

import numpy as np

from orbix.orbit import AbstractOrbit
from orbix.viz._prepare import prepare_sky_track
from orbix.viz._require import eyepiece
from orbix.viz.orbit import (
    _orbit_look,
    _per_track,
    _positions,
    plot_orbit,
)

_DEFAULT_BASE_MS = 6.0


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


def _auto_camera(tracks, n_frames, roll_deg, tilt_deg=55.0, sweep_deg=40.0):
    """A sweep that turns the orbit without changing how big it is drawn.

    Sweeping raw azimuth from wherever the axes happened to be pointing
    changes the projected area of a fixed orbit -- from matplotlib's default
    camera this orbit's drawn area swings by a factor of six, which a reader
    reads as a change in the orbit. It is the same failure as an axis that
    rescales between frames, with a rotation matrix in front of it.

    A camera on a cone about the orbit normal cannot do that: the projected
    area of a planar ellipse is ``pi * a * b * cos(tilt)``, so holding the
    tilt fixed holds the drawn size fixed exactly, and the sweep turns the
    ellipse in the page plane instead of inflating it. The parallax that
    makes the geometry legible is unchanged.

    Args:
        tracks: Positions, shape ``(K, T, 3)``, in the plot's own units.
        n_frames: Number of frames to generate angles for.
        roll_deg: Camera roll, held fixed across the sweep.
        tilt_deg: Angle between the camera and the orbit normal. Well away
            from 0 (face-on, where the depth cue flattens to nothing) and
            from 90 (edge-on, where the orbit collapses to a line).
        sweep_deg: How far around the cone to travel.

    Returns:
        Dict with ``"azim"``, ``"elev"`` and ``"roll"`` arrays.
    """
    points = np.asarray(tracks, float).reshape(-1, 3)
    centered = points - points.mean(axis=0)
    normal = np.linalg.svd(centered, full_matrices=False)[2][-1]
    normal = normal / np.linalg.norm(normal)

    seed = np.array([0.0, 0.0, 1.0])
    if abs(float(normal @ seed)) > 0.9:
        seed = np.array([1.0, 0.0, 0.0])
    u = np.cross(normal, seed)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    tilt = np.deg2rad(tilt_deg)
    phase = np.deg2rad(np.linspace(0.0, sweep_deg, n_frames))
    direction = np.cos(tilt) * normal[None, :] + np.sin(tilt) * (
        np.cos(phase)[:, None] * u + np.sin(phase)[:, None] * v
    )
    return {
        "elev": np.degrees(np.arcsin(np.clip(direction[:, 2], -1.0, 1.0))),
        "azim": np.degrees(np.arctan2(direction[:, 1], direction[:, 0])),
        "roll": np.full(n_frames, roll_deg),
    }


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
    base_ms=None,
    weights=None,
    marks=None,
    source_ids=None,
    observations=None,
    iwa_arcsec=None,
    run_time=None,
    cast=None,
    profile=None,
):
    """Animate one or more orbits and return a lazy ``eyepiece.Animation``.

    Nothing renders until ``.save``, ``.jshtml``, or ``.video`` is called
    on the result, so one animation can go to several sinks in one pass
    and the test path never needs ffmpeg.

    Args:
        orbit_or_tracks: An ``AbstractOrbit``, or bare tracks: for
            ``kind="sky"`` the arcsec forms ``prepare_sky_track`` accepts
            (``(K, T, 2)`` or an ``(ra, dec)`` pair); for ``kind="3d"``
            ``(T, 3)`` / ``(K, T, 3)`` positions.
        t_jd: Times in Julian Days, shape ``(T,)``. Required for
            ``kind="sky"``, where the samples' physical times drive the
            playback clock, and on the 3D orbit door; on the 3D bare-track
            door it is optional and used only for the elapsed-time label
            (length must match the track).
        Ms_kg: Stellar mass in kg. Orbit door only.
        dist_pc: Distance in parsecs. Orbit door with ``kind="sky"`` only.
        trig_solver: Optional Kepler solver; None uses orbix's default.
        kind: ``"sky"`` (sky-plane arcsec: ``prepare_sky_track`` played by
            ``eyepiece.mpl.animate``) or ``"3d"`` (star-centric AU, via
            ``plot_orbit``).
        history: ``"all"`` grows the trail from the first epoch, an int
            keeps a trailing window of that many frames, ``"none"`` moves
            the head marker alone.
        rotate: Camera sweep, ``kind="3d"`` only. The default ``"auto"``
            travels 40 degrees around a cone about the ORBIT NORMAL, at a
            fixed tilt to it, so azimuth and elevation both vary. That is
            deliberate: holding the tilt fixed holds the projected area
            ``pi * a * b * cos(tilt)`` fixed, where a plain azimuth sweep
            about z at held elevation changes the tilt and so swings the
            drawn size -- by 5.7x on the three-planet system in the docs,
            which a reader misreads as the orbit growing. ``None`` holds
            the camera still. A dict maps any of ``"azim"``, ``"elev"``, ``"roll"``
            to a ``(start_deg, stop_deg)`` pair interpolated linearly
            across the frames, for full control. The head's depth cue
            tracks the moving camera frame by frame. Ignored for
            ``kind="sky"`` unless a dict is passed, which raises.
        fps: Playback rate carried by the returned animation.
        style: ``kind="3d"`` only. A color or ``SourceStyles`` entry for
            the tracks, forwarded to the static function and used for the
            animated artists. None gives the star-chart default: heads in
            the mode's text color (white dots on a dark background) over
            transparent dashed gray trails; a ``style`` opts into that
            source's solid color instead. Sky colors come from
            ``source_ids`` and ``cast``.
        base_ms: ``kind="3d"`` only. Head-marker diameter in points, a
            scalar or one value per track (default 6). The base size is
            the anchor the depth cue swells around, so it is where
            physical meaning lives -- pass ``size_by_radius(radii)`` to
            encode planet radii.
        weights: Optional per-track weights (length K), fading the ghost
            fan and the animated trails.
        marks: Optional ``{"periapsis", "nodes"}`` (3d orbit door only).
        source_ids: ``kind="sky"`` only: the source of each track, as for
            ``prepare_sky_track``.
        observations: ``kind="sky"`` only: ``(xy_arcsec, xerr_arcsec,
            yerr_arcsec)`` observed positions.
        iwa_arcsec: ``kind="sky"`` only: inner-working-angle radius.
        run_time: ``kind="sky"`` only: presentation duration in seconds.
            None plays ``T / fps`` seconds, one output frame per sample of
            an evenly spaced grid. Output frames are evenly spaced in
            physical time and each holds the latest sample.
        cast: ``kind="sky"`` only: an ``eyepiece.style.SourceCast``.
        profile: ``kind="sky"`` only: an ``eyepiece.style.RenderProfile``.

    Returns:
        An ``eyepiece.Animation`` bound to the built figure. A 3D
        animation has one frame per epoch of the track.

    Raises:
        TypeError: If a sky animation has no ``t_jd``.
        ValueError: If ``kind`` or ``history`` is not one of the
            documented values, an argument belongs to the other kind, or
            a labeled ``t_jd`` does not match the track length.
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
    if kind == "sky":
        three_d_only = [
            name
            for name, value in (
                ("style", style),
                ("base_ms", base_ms),
                ("marks", marks),
            )
            if value is not None
        ]
        if three_d_only:
            raise ValueError(
                ", ".join(three_d_only) + " only apply to kind='3d'; sky colors "
                "come from source_ids and cast"
            )
        return _animate_sky(
            orbit_or_tracks,
            t_jd,
            Ms_kg=Ms_kg,
            dist_pc=dist_pc,
            trig_solver=trig_solver,
            history=history,
            fps=fps,
            weights=weights,
            source_ids=source_ids,
            observations=observations,
            iwa_arcsec=iwa_arcsec,
            run_time=run_time,
            cast=cast,
            profile=profile,
        )

    sky_only = [
        name
        for name, value in (
            ("source_ids", source_ids),
            ("observations", observations),
            ("iwa_arcsec", iwa_arcsec),
            ("run_time", run_time),
            ("cast", cast),
            ("profile", profile),
        )
        if value is not None
    ]
    if sky_only:
        raise ValueError(", ".join(sky_only) + " only apply to kind='sky'")
    if base_ms is None:
        base_ms = _DEFAULT_BASE_MS

    from_orbit = isinstance(orbit_or_tracks, AbstractOrbit)
    base_t = t_jd if from_orbit else None

    # marker_scale=0 keeps the ghost a bare line: the moving head carries
    # the depth cue in an animation, so per-point beads along the whole
    # path would only be clutter.
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
    # the star-chart default: heads in the text color (white dots on a
    # dark mode) edged in the background color, trails dashed gray; a
    # style= opts into that source's solid color for both.
    # The animation always wants the star-chart pairing: a dim dashed
    # ghost path with the depth cue carried by the moving head marker,
    # which is exactly what _orbit_look builds for "markers".
    head_color, path_kw = _orbit_look(style, ep, "markers")
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

    head_ms = _per_track(base_ms, n_tracks, "base_ms")
    camera = None
    if rotate == "auto":
        camera = _auto_camera(tracks, n_frames, ax.roll)
    elif rotate is not None:
        base_angles = {"azim": ax.azim, "elev": ax.elev, "roll": ax.roll}
        camera = {
            key: np.linspace(*rotate[key], n_frames)
            if key in rotate
            else np.full(n_frames, base_angles[key])
            for key in ("azim", "elev", "roll")
        }
    if camera is not None:
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
        label = ax.text2D(0.02, 0.98, "", transform=ax.transAxes, ha="left", va="top")

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
                trails[k].set_data_3d(segment[:, 0], segment[:, 1], segment[:, 2])
            point = tracks[k, i]
            heads[k].set_data_3d([point[0]], [point[1]], [point[2]])
            # never vanish entirely: the far side reads as "small",
            # not "gone behind the star"
            heads[k].set_markersize(head_ms[k] * depth_size(head_scales[k, i]))
        if label is not None:
            label.set_text(f"t = +{times[i] - times[0]:.0f} d")

    return ep.animate(base.fig, draw, n_frames, fps=fps)


def _animate_sky(
    orbit_or_tracks,
    t_jd,
    *,
    Ms_kg,
    dist_pc,
    trig_solver,
    history,
    fps,
    weights,
    source_ids,
    observations,
    iwa_arcsec,
    run_time,
    cast,
    profile,
):
    """Prepare the sky tracks once and play them through ``eyepiece.mpl``."""
    eyepiece()
    from eyepiece import mpl

    if t_jd is None:
        raise TypeError("a sky animation needs t_jd: the sample times drive its clock")
    sequence = prepare_sky_track(
        orbit_or_tracks,
        t_jd=t_jd,
        Ms_kg=Ms_kg,
        dist_pc=dist_pc,
        trig_solver=trig_solver,
        source_ids=source_ids,
        weights=weights,
        observations=observations,
        iwa_arcsec=iwa_arcsec,
        history=None if history == "none" else history,
    )
    if run_time is None:
        run_time = len(sequence.times) / fps
    return mpl.animate(sequence, run_time=run_time, fps=fps, cast=cast, profile=profile)
