"""Sky-plane and 3D orbit plots: extract, delegate to eyepiece, decorate.

orbix supplies only what eyepiece cannot know: how to turn an orbit into
tracks (propagation, projection, units), which axis is RA (and that
astronomers draw it increasing to the left), and where periapsis and the
nodes sit. The sky plane is prepared once (``orbix.viz._prepare``) and
drawn by ``eyepiece.mpl``; the 3D rendering is eyepiece's ``trail``, so
figures drawn here match every other figure built on eyepiece.

Every function that accepts an orbit also accepts bare track arrays, for
consumers holding precomputed coordinates (a CSV of posterior traces, a
cached ephemeris) rather than a live orbit. Bare sky tracks
are arcsec offsets; bare 3D positions are in the caller's units, so no
axis labels are set on that door.
"""

import dataclasses
from collections.abc import Mapping

import numpy as np

from orbix.system import System
from orbix.viz._prepare import (
    _orbit_arrays,
    prepare_sky_view,
)
from orbix.viz._require import eyepiece


def _resolve_color(ep, style):
    """Resolve ``style`` (color, SourceStyles entry, or None) to a color.

    None resolves to the active palette's first color through the public
    ``SourceStyles`` mechanism, at call time, so a mode switch between two
    calls is honored.
    """
    if isinstance(style, Mapping):
        return style.get("color")
    if style is not None:
        return style
    return ep.style.SourceStyles(["track"])["track"]["color"]


def _orbit_look(style, ep, depth):
    """The 3D orbit rendering defaults: marker color and path-line kwargs.

    With no ``style`` and ``depth="markers"``, the look is the classic
    star-chart one: markers in the mode's text color (white dots on a dark
    background) over a transparent dashed gray path. Under any other depth
    cue there are no markers, so the path itself has to be the visible
    element and takes the text color rather than a dim gray -- otherwise an
    unstyled orbit renders as a barely-there gray line. A ``style`` opts out
    into that source's color either way.
    """
    if style is None:
        marker_color = ep.style.neutral(1.0)
        if depth == "markers":
            return marker_color, {
                "color": ep.style.neutral(0.55),
                "linestyle": "--",
                "alpha": 0.5,
            }
        return marker_color, {"color": marker_color}
    return _resolve_color(ep, style), {}


def size_by_radius(
    radius_Rearth, *, ms_range=(3.0, 9.0), radius_range_Rearth=(0.38, 11.2)
):
    """Marker diameters encoding planet radii, geometrically interpolated.

    The base marker size is the anchor the depth cue swells around, so it
    is the place physical meaning lives. This maps radii onto diameters
    the same way the original hand-tuned renders mapped mass onto marker
    size: geometrically between ``ms_range`` across
    ``radius_range_Rearth`` (Mercury to Jupiter by default), so a
    super-Earth reads visibly larger than a sub-Earth without Jupiter
    dwarfing everything. Radii outside the range clip to its ends.

    The result is a set of marker DIAMETERS in points, which is what
    ``animate_orbit(base_ms=...)`` takes (matplotlib's ``ms``). Do NOT pass
    it to ``plot_orbit(marker_scale=...)``: that reaches ``scatter(s=...)``,
    an AREA in points squared, so the diameters would be read as areas and
    the encoding silently square-rooted -- an 11.2-Earth-radius planet drawn
    1.48x an Earth instead of 2.19x. Square the result first if you need an
    area.

    Args:
        radius_Rearth: Planet radii in Earth radii, scalar or ``(K,)``.
        ms_range: Marker diameters in points at the two ends of the
            radius range.
        radius_range_Rearth: The radii mapped onto ``ms_range``'s ends.

    Returns:
        Marker diameters in points, shape ``(K,)``.
    """
    lo, hi = radius_range_Rearth
    r = np.clip(np.atleast_1d(np.asarray(radius_Rearth, float)), lo, hi)
    frac = np.log(r / lo) / np.log(hi / lo)
    return ms_range[0] * (ms_range[1] / ms_range[0]) ** frac


def _per_track(values, n_tracks, name):
    """Broadcast a scalar or length-K sequence to one float per track."""
    arr = np.atleast_1d(np.asarray(values, float))
    if arr.shape[0] == 1:
        return np.full(n_tracks, arr[0])
    if arr.shape[0] != n_tracks:
        raise ValueError(f"{name} has {arr.shape[0]} entries for {n_tracks} tracks")
    return arr


def _widened(view, ax):
    """``view`` with axis limits grown to cover what ``ax`` already shows.

    Overplotting a second set of tracks must not crop the first, so a
    caller axes that already holds data widens the new view's fixed limits
    to include its current ones.
    """
    if not ax.has_data():
        return view
    x_low, x_high = sorted(ax.get_xlim())
    y_low, y_high = sorted(ax.get_ylim())
    axes = dataclasses.replace(
        view.axes,
        x_limits=(
            min(view.axes.x_limits[0], x_low),
            max(view.axes.x_limits[1], x_high),
        ),
        y_limits=(
            min(view.axes.y_limits[0], y_low),
            max(view.axes.y_limits[1], y_high),
        ),
    )
    return dataclasses.replace(view, axes=axes)


def plot_sky_track(
    orbit_or_radec,
    *,
    time_d=None,
    dist_pc=None,
    trig_solver=None,
    ax=None,
    source_ids=None,
    weights=None,
    observations=None,
    iwa_arcsec=None,
    cast=None,
    profile=None,
):
    """Draw sky-plane orbit tracks: one orbit, or a fan of K candidates.

    A ``(K,)``-batched orbit (posterior draws through
    ``FrameOrbit.from_classical``, for instance) becomes a fan of K
    tracks faded by ``weights``; a single orbit becomes one track. The
    tracks are prepared once with ``prepare_sky_view`` and drawn with
    ``eyepiece.mpl.render``, which owns the equal aspect, RA increasing to
    the left, the central star, the shaded inner-working-angle disk, and
    the observation error bars.

    Args:
        orbit_or_radec: A ``FrameOrbit`` (requiring ``dist_pc``), a ``System``
            (using its stored distance), or bare arcsec tracks: an
            array shaped ``(K, T, 2)``, or an ``(ra, dec)`` pair of
            ``(T,)`` or ``(K, T)`` arrays.
        time_d: Float days from the engine object's epoch, shape ``(T,)``.
        dist_pc: Distance in parsecs for a frame orbit. A system
            uses its stored distance.
        trig_solver: Engine solver strategy (Markley, Halley, or Grid).
            None uses the input object's default.
        ax: Axes to draw into. None creates a new figure and axes. An axes
            that already holds data keeps it in view: the new limits grow
            to cover the old ones.
        source_ids: The source of each track, as for ``prepare_sky_view``;
            renderers color by source, so K draws of one planet share a
            color and period-alias families can take one each.
        weights: Optional per-track weights (length K), fading each
            track's opacity; typically posterior mass per draw.
        observations: Optional ``(xy_arcsec, xerr_arcsec, yerr_arcsec)``
            observed positions, shaped ``(P, 2)``, ``(P,)``, ``(P,)``.
        iwa_arcsec: Optional inner-working-angle radius in arcsec.
        cast: ``eyepiece.style.SourceCast`` assigning colors to source
            IDs. None builds one in first-encounter order.
        profile: ``eyepiece.style.RenderProfile``; None snapshots the
            active style.

    Returns:
        The ``eyepiece.mpl.MplResult``; ``result.parts["path/<k>"]`` is
        track k's ``Line2D``.
    """
    eyepiece()
    from eyepiece import mpl

    view = prepare_sky_view(
        orbit_or_radec,
        time_d=time_d,
        dist_pc=dist_pc,
        trig_solver=trig_solver,
        source_ids=source_ids,
        weights=weights,
        observations=observations,
        iwa_arcsec=iwa_arcsec,
    )
    if ax is not None:
        view = _widened(view, ax)
    result = mpl.render(view, ax=ax, cast=cast, profile=profile)

    return result


def _positions(orbit_or_xyz, time_d, trig_solver):
    """Normalize the input to positions of shape ``(K, T, 3)``.

    Orbit rows already use right-handed (east, north, away); only the
    component axis moves. Bare xyz tracks retain the caller's frame.

    Returns:
        ``(positions, orbit_or_none)`` -- the orbit comes back so exact
        mark geometry can be computed from its elements rather than
        re-derived from the sampled track.
    """
    prepared = _orbit_arrays(orbit_or_xyz, time_d, None, trig_solver)
    if prepared is not None:
        return prepared

    if time_d is not None:
        raise TypeError(
            "time_d only applies when propagating an orbit; bare xyz tracks "
            "are drawn as given"
        )
    arr = np.asarray(orbit_or_xyz, float)
    if arr.ndim == 2 and arr.shape[-1] == 3:
        return arr[None, :, :], None
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr, None
    raise ValueError(
        "expected an orbit or an array shaped (T, 3) or (K, T, 3); "
        f"got shape {arr.shape}"
    )


def _nu_to_trig_E(nu, e):
    """Convert a true anomaly to ``(sinE, cosE)`` of the eccentric anomaly."""
    denom = 1.0 + e * np.cos(nu)
    cosE = (e + np.cos(nu)) / denom
    sinE = np.sqrt(1.0 - e**2) * np.sin(nu) / denom
    return sinE, cosE


def _mark_geometry(orbit):
    """Exact periapsis and node positions per orbit, from the elements.

    Positions come through the same ``A (cosE - e) + B sinE`` propagation
    form the orbit itself uses, so a mark sits exactly on the drawn track
    rather than on a re-derived approximation. The ascending node is
    identified analytically: z along the orbit is
    ``A_z (cosE - e) + B_z sinE``, so its derivative in E is
    ``-A_z sinE + B_z cosE`` and the node with a positive derivative is
    ascending (E increases monotonically with time).

    Returns:
        ``(periapsis, ascending, descending)``, each ``(K, 3)`` in AU.
    """
    A, B, e, w = orbit

    periapsis = (A * (1.0 - e)).T

    nodes = []
    for nu in (-w, np.pi - w):
        sinE, cosE = _nu_to_trig_E(nu, e)
        point = (A * (cosE - e) + B * sinE).T
        dz_dE = -A[2] * sinE + B[2] * cosE
        nodes.append((point, dz_dE))
    (point_a, dz_a), (point_b, _) = nodes
    rising_first = (dz_a > 0.0)[:, None]
    ascending = np.where(rising_first, point_a, point_b)
    descending = np.where(rising_first, point_b, point_a)
    return periapsis, ascending, descending


def plot_orbit(
    orbit_or_xyz,
    *,
    time_d=None,
    trig_solver=None,
    ax=None,
    style=None,
    marks=None,
    depth=None,
    marker_scale=25.0,
    trail_kw=None,
):
    """Draw one or more orbits in 3D, star-centric AU, via ``eyepiece.plot.trail``.

    Orbit vectors use right-handed (east, north, away) axes. For the
    observer's east-left, north-up view, set an orthographic camera with
    ``ax.view_init(elev=-90, azim=90, roll=0)`` before plotting. This looks
    from -z and agrees with the sky panel without reflecting any components.

    The trajectory rendering -- connected path, depth-cued marker sizes,
    camera-aware layering -- is ``trail``'s; orbix adds the propagation,
    the star at the origin, symmetric axis limits so the orbit is not
    distorted, AU labels, and the optional exact periapsis/node marks.

    ``trail`` bakes its depth cues from the camera at call time, so set
    the view first (``ax.view_init(...)`` before calling this function)
    and keep the camera well off the orbit normal -- a near-face-on view
    collapses the marker-size depth cue.

    Args:
        orbit_or_xyz: A ``FrameOrbit``, a ``System`` (planets relative to its
            star), or bare positions shaped ``(T, 3)``
            or ``(K, T, 3)``. Bare positions are in whatever units the
            caller made them, so no axis labels are set on that door.
        time_d: Float days from the engine object's epoch, shape ``(T,)``.
        trig_solver: Engine solver strategy (Markley, Halley, or Grid).
            None uses the input object's default.
        ax: A ``projection="3d"`` axes to draw into. None creates one.
        style: A color, or a ``SourceStyles`` entry (which also sets the
            track marker), applied to every track: solid path and markers
            in that color. None gives the star-chart default -- markers in
            the mode's text color (white dots on a dark background) over
            a transparent dashed gray path.
        marks: Optional set drawn from ``{"periapsis", "nodes"}``. Requires
            a Kepler ellipse: a frame orbit, a single-planet
            system, or a system with all planets massless. Bare xyz and
            massive multi-planet systems raise if marks are requested. Periapsis is a
            diamond in the track color; the nodes are up/down triangles
            joined by a dashed line of nodes through the origin.
        depth: Forwarded to ``eyepiece.plot.trail``: how the path shows which
            half faces the camera. ``None`` takes trail's own default,
            the hidden-line convention -- the whole orbit dashed and dim
            with the near half overdrawn solid. ``"markers"`` restores the
            older per-point markers, which also restores the star-chart
            look when no ``style`` is given. ``"none"`` drops the cue.
        marker_scale: Forwarded to ``trail``: marker AREA in points
            squared (matplotlib's ``scatter(s=)``) at full illumination,
            for the per-point depth cue a still figure needs. Pass ``0.0``
            for a bare line -- an animation does this, since its moving
            head carries the depth cue instead. Note the unit: this is an
            area, while ``size_by_radius`` returns diameters, so the two
            do not compose directly. Note also that ``trail``'s still
            depth law takes marker area to zero on the far side, so a
            per-track ``marker_scale`` encoding a physical quantity is
            unrecoverable there; encode physical size on an animation's
            ``base_ms`` instead, whose law is anchored at the base size.
        trail_kw: Extra kwargs for the connecting-line ``ax.plot`` call,
            forwarded to ``trail`` and applied last.

    Returns:
        An ``eyepiece.PlotResult``. For a single track the artists are
        ``trail``'s ``"line"`` and ``"scatter"``; for K tracks they are
        ``"lines"`` and ``"scatter"`` lists in track order. ``"scatter"``
        holds one depth-cue artist per track, whichever the mode drew: a
        ``PathCollection`` under ``depth="markers"``, the solid near-half
        ``Line3D`` under the default hidden-line cue, and nothing at all
        under ``depth="none"``. Mark artists are appended to ``"scatter"``
        (periapsis first, then nodes) and the line of nodes to ``"lines"``,
        after the per-track entries.
    """
    ep = eyepiece()
    positions, orbit = _positions(orbit_or_xyz, time_d, trig_solver)
    marks = set(marks or ())
    unknown = marks - {"periapsis", "nodes"}
    if unknown:
        raise ValueError(f"unknown marks {sorted(unknown)}; expected periapsis/nodes")
    if marks and orbit is None:
        raise ValueError(
            "marks need the orbital elements of a Kepler ellipse; "
            "pass a frame orbit, a single-planet system, or a massless system"
        )

    if positions.size == 0:
        if ax is None:
            import matplotlib.pyplot as plt

            ax = plt.figure().add_subplot(projection="3d")
        return ep.PlotResult(ax=ax, artists={})

    depth = "hidden" if depth is None else depth
    marker_color, path_kw = _orbit_look(style, ep, depth)
    lkw = {**path_kw, **(trail_kw or {})}
    scales = _per_track(marker_scale, positions.shape[0], "marker_scale")
    lines, scatters = [], []
    for k in range(positions.shape[0]):
        result = ep.plot.trail(
            positions[k],
            ax=ax,
            style=style if style is not None else marker_color,
            depth=depth,
            marker_scale=float(scales[k]),
            trail_kw=lkw,
        )
        ax = result.ax
        lines.append(result.artists["line"])
        # `"near"` under the hidden-line cue, `"scatter"` under markers, and
        # neither under "none": whichever the mode drew joins the same list.
        for key in ("near", "scatter"):
            if key in result.artists:
                scatters.append(result.artists[key])

    import matplotlib as mpl

    star_color = mpl.rcParams["text.color"]
    ax.scatter([0.0], [0.0], [0.0], marker="*", s=140, color=star_color, zorder=5)

    half = float(np.max(np.abs(positions))) * 1.05
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_zlim(-half, half)
    ax.set_box_aspect((1.0, 1.0, 1.0))

    # matplotlib's 3D panes are a fixed light gray that ignores the style
    # mode; take the axes facecolor instead, resolved at call time, so a
    # dark mode gets black space behind the orbit and a light mode gets
    # clean white panes.
    pane_color = ax.get_facecolor()
    for pane_axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane_axis.set_pane_color(pane_color)
    if (
        orbit is not None
        or isinstance(orbit_or_xyz, System)
        or hasattr(orbit_or_xyz, "propagate")
    ):
        ax.set_xlabel("East [AU]")
        ax.set_ylabel("North [AU]")
        ax.set_zlabel("Away [AU]")

    if marks:
        periapsis, ascending, descending = _mark_geometry(orbit)
        if "periapsis" in marks:
            scatters.append(
                ax.scatter(*periapsis.T, marker="D", s=30, color=marker_color, zorder=4)
            )
        if "nodes" in marks:
            scatters.append(
                ax.scatter(*ascending.T, marker="^", s=30, color=marker_color, zorder=4)
            )
            scatters.append(
                ax.scatter(
                    *descending.T, marker="v", s=30, color=marker_color, zorder=4
                )
            )
            for asc, desc in zip(ascending, descending, strict=False):
                (node_line,) = ax.plot(
                    *np.stack([asc, desc]).T,
                    linestyle="--",
                    lw=0.8,
                    color=star_color,
                    alpha=0.5,
                    zorder=1,
                )
                lines.append(node_line)

    if len(lines) == 1 and len(scatters) == 1:
        artists = {"line": lines[0], "scatter": scatters[0]}
    else:
        artists = {"lines": lines, "scatter": scatters}
    return ep.PlotResult(ax=ax, artists=artists)
