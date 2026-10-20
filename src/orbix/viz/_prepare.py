"""Sky tracks, prepared once: the orbit science behind every sky-plane view.

``prepare_sky_track`` does all of the scientific work a sky-plane track
needs -- propagating the orbit over the whole time grid, projecting it to
arcsec, and moving the result from the device to the host -- exactly once,
and returns an ``eyepiece`` prepared ``Sequence``. A still
(``sequence.frame(i)``), a strip of epochs (``sequence.strip([...])``), a
Matplotlib movie (``eyepiece.mpl.animate``), and a Manim clip
(``eyepiece.manim.animate``) are then all views of that one prepared
storage; none of them propagates again. ``prepare_sky_view`` builds the
same view without a time axis, for a static figure of bare tracks that
carry no timestamps.

This module imports NumPy, JAX (through the orbit), and
``eyepiece.prepared`` only. It never imports Matplotlib or a renderer.

**The head is always a propagated sample.** A sequence evaluated between
two sample times holds the earlier sample. It never draws a point along
the chord between samples: on a sparsely sampled orbit that chord can
pass inside the inner working angle while the orbit itself never does, so
an interpolated head would invent a crossing. Sample the orbit densely
enough for the figure instead; the preparation cost is paid once.

**One element per candidate, one source per planet.** Each of K tracks
(posterior draws, say) is its own ``Path`` (``path/<k>``), current-position
``Points`` (``head/<k>``), and optional revealed trail (``trail/<k>``), so
every draw stays addressable. The ``source_id`` names the planet a draw
belongs to and may repeat across draws; renderers color by source.
"""

import jax.numpy as jnp
import numpy as np
from eyepiece.prepared import (
    ArrayChannel,
    AxisSpec,
    Clock,
    Label,
    Path,
    PathWindow,
    Points,
    Region,
    Sequence,
    TrackView,
)

from orbix.orbit import AbstractOrbit

# Stable element IDs of a prepared sky view. Per-track IDs are these
# prefixes plus the track index (``path/0``, ``head/0``, ``trail/0``).
VIEW_ID = "sky"
IWA_ID = "iwa"
STAR_ID = "star"
OBSERVATIONS_ID = "observations"
CLOCK_ID = "clock"
PATH_PREFIX = "path"
HEAD_PREFIX = "head"
TRAIL_PREFIX = "trail"

# The source every track shares when the caller names none: a fan of
# candidates is one planet, so it takes one color rather than K.
DEFAULT_SOURCE = "track"

_AXIS_LABELS = ("RA offset [arcsec]", "Dec offset [arcsec]")
_TIME_UNIT = "d"
_CLOCK_XY = (0.02, 0.98)
_MARGIN = 0.05
_MAX_CLOCK_DECIMALS = 3
# Offsets within this many days of a printed value count as printed exactly.
_CLOCK_EXACT_D = 1e-6
# Largest rounding error, as a fraction of the smallest step, accepted when
# no clock precision prints every offset exactly. Under half a step keeps
# adjacent labels distinct and in order; a quarter leaves margin.
_CLOCK_STEP_FRACTION = 0.25


def _element_id(prefix, k):
    return f"{prefix}/{k}"


# --- inputs ------------------------------------------------------------------


def _sample_times(t_jd):
    """Validate a time grid for a sequence: 1D, finite, strictly increasing."""
    times = np.asarray(t_jd, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError(f"t_jd must be a nonempty 1D array, got shape {times.shape}")
    if not np.all(np.isfinite(times)):
        raise ValueError("t_jd must be finite")
    if times.size > 1 and not np.all(np.diff(times) > 0):
        raise ValueError("t_jd must be strictly increasing")
    return times


def _trail_window(history):
    """Map ``history`` to a ``PathWindow`` history, or None for no trail.

    Returns:
        ``(has_trail, window_history)``: ``"all"`` gives a growing trail
        (``PathWindow(history=None)``), a positive int a sliding one.
    """
    if history is None:
        return False, None
    if history == "all":
        return True, None
    if isinstance(history, (int, np.integer)) and not isinstance(history, bool):
        if history > 0:
            return True, int(history)
    raise ValueError(
        f'history must be None, "all", or a positive int window, got {history!r}'
    )


def _sky_xy(orbit_or_radec, t_jd, Ms_kg, dist_pc, trig_solver):
    """Normalize the input to host coordinates of shape ``(K, T, 2)`` arcsec.

    An orbit is propagated with one ``position_arcsec`` call over the
    whole grid; east (Delta alpha cos delta) and north (Delta delta) offsets
    are stacked on the device and transferred to the host once. A bare
    ``(K, T, 2)`` array is borrowed as given (no copy, no
    dtype promotion); an ``(ra, dec)`` pair is stacked into new storage.
    """
    if isinstance(orbit_or_radec, AbstractOrbit):
        missing = [
            name
            for name, value in (("t_jd", t_jd), ("Ms_kg", Ms_kg), ("dist_pc", dist_pc))
            if value is None
        ]
        if missing:
            raise TypeError(
                "preparing an orbit requires " + ", ".join(missing) + " to propagate it"
            )
        ra, dec = orbit_or_radec.position_arcsec(
            trig_solver, t_jd, Ms_kg=Ms_kg, dist_pc=dist_pc
        )
        return np.asarray(jnp.stack([ra, dec], axis=-1))

    if isinstance(orbit_or_radec, (tuple, list)) and len(orbit_or_radec) == 2:
        ra = np.atleast_2d(np.asarray(orbit_or_radec[0]))
        dec = np.atleast_2d(np.asarray(orbit_or_radec[1]))
        if ra.shape != dec.shape or ra.ndim != 2:
            raise ValueError(
                "an (ra, dec) pair needs two (T,) or two (K, T) arrays, got "
                f"{ra.shape} and {dec.shape}"
            )
        tracks = np.stack([ra, dec], axis=-1)
    else:
        tracks = np.asarray(orbit_or_radec)
        if tracks.ndim != 3 or tracks.shape[-1] != 2:
            raise ValueError(
                "expected an AbstractOrbit, a (K, T, 2) array of arcsec offsets, "
                f"or an (ra, dec) pair of (T,) or (K, T) arrays; got shape "
                f"{tracks.shape}"
            )
    if not np.issubdtype(tracks.dtype, np.number):
        raise ValueError(f"tracks must be numeric, got dtype {tracks.dtype}")
    return tracks


def _per_track_sources(source_ids, n_tracks):
    if source_ids is None:
        return (DEFAULT_SOURCE,) * n_tracks
    if isinstance(source_ids, str):
        return (source_ids,) * n_tracks
    sources = tuple(source_ids)
    if len(sources) != n_tracks:
        raise ValueError(f"source_ids has {len(sources)} entries for {n_tracks} tracks")
    return sources


def _per_track_weights(weights, n_tracks):
    if weights is None:
        return (1.0,) * n_tracks
    values = np.atleast_1d(np.asarray(weights, dtype=float))
    if values.ndim != 1 or values.shape[0] != n_tracks:
        raise ValueError(f"weights has {values.shape[0]} entries for {n_tracks} tracks")
    return tuple(float(w) for w in values)


def _observation_points(observations, weight):
    """Build the observations mark from ``(xy, xerr, yerr)``, or None."""
    if observations is None:
        return None
    if not isinstance(observations, (tuple, list)) or len(observations) != 3:
        raise ValueError(
            "observations must be (xy_arcsec, xerr_arcsec, yerr_arcsec) shaped "
            "(P, 2), (P,), (P,)"
        )
    xy, xerr, yerr = observations
    return Points(OBSERVATIONS_ID, xy, xerr=xerr, yerr=yerr, weight=weight)


def _iwa_region(iwa_arcsec):
    if iwa_arcsec is None:
        return None
    return Region(IWA_ID, np.zeros(2), float(iwa_arcsec))


# --- the view ------------------------------------------------------------------


def _limits(tracks, observations, iwa):
    """Fixed axis limits enclosing every mark, the star, and the IWA."""
    rows = tracks.reshape(-1, 2)
    finite = rows[np.all(np.isfinite(rows), axis=1)]
    if finite.shape[0] == 0:
        raise ValueError("tracks have no finite coordinates to draw")
    lows = [finite.min(axis=0), np.zeros(2)]
    highs = [finite.max(axis=0), np.zeros(2)]
    if observations is not None:
        xy = np.asarray(observations.xy, dtype=float)
        err = np.stack(
            [
                np.zeros(len(xy)) if observations.xerr is None else observations.xerr,
                np.zeros(len(xy)) if observations.yerr is None else observations.yerr,
            ],
            axis=1,
        )
        keep = np.all(np.isfinite(xy), axis=1)
        if np.any(keep):
            lows.append((xy[keep] - err[keep]).min(axis=0))
            highs.append((xy[keep] + err[keep]).max(axis=0))
    if iwa is not None:
        lows.append(np.full(2, -iwa.outer_radius))
        highs.append(np.full(2, iwa.outer_radius))
    low = np.min(lows, axis=0)
    high = np.max(highs, axis=0)
    span = float(np.max(high - low))
    pad = _MARGIN * span if span > 0 else 1.0
    return (
        (float(low[0] - pad), float(high[0] + pad)),
        (float(low[1] - pad), float(high[1] + pad)),
    )


def _build_view(
    tracks,
    *,
    source_ids,
    weights,
    observations,
    iwa_arcsec,
    trail=False,
    heads=False,
    clock_text=None,
):
    """Build the one sky ``TrackView`` both public functions share.

    Args:
        tracks: Host coordinates shaped ``(K, T, 2)``, arcsec.
        source_ids: Per-track source identity input (see the public
            functions).
        weights: Per-track weights input.
        observations: ``(xy, xerr, yerr)`` or None.
        iwa_arcsec: IWA radius or None.
        trail: Whether to add a revealed ``trail/<k>`` per track, sharing
            the full path's storage, visible through sample 0.
        heads: Whether to add a current-position ``head/<k>`` per track at
            sample 0.
        clock_text: Text for the acquisition-time label, or None for none.

    Returns:
        A ``TrackView`` with marks in draw order: IWA, full paths,
        trails, star, heads, observations, clock.
    """
    n_tracks = tracks.shape[0]
    sources = _per_track_sources(source_ids, n_tracks)
    track_weights = _per_track_weights(weights, n_tracks)
    # the star and the observations sit at the top weight, so the Points
    # normalization is the tracks' own and each head fades with its path
    top_weight = max(track_weights)
    points = _observation_points(observations, top_weight)
    iwa = _iwa_region(iwa_arcsec)
    x_limits, y_limits = _limits(tracks, points, iwa)

    marks = [] if iwa is None else [iwa]
    marks += [
        Path(
            _element_id(PATH_PREFIX, k),
            tracks[k],
            source_id=sources[k],
            weight=track_weights[k],
        )
        for k in range(n_tracks)
    ]
    if trail:
        marks += [
            Path(
                _element_id(TRAIL_PREFIX, k),
                tracks[k],
                source_id=sources[k],
                weight=track_weights[k],
                visible=(0, 1),
            )
            for k in range(n_tracks)
        ]
    marks.append(Points(STAR_ID, np.zeros((1, 2)), weight=top_weight))
    if heads:
        marks += [
            Points(
                _element_id(HEAD_PREFIX, k),
                tracks[k, :1],
                source_id=sources[k],
                weight=track_weights[k],
            )
            for k in range(n_tracks)
        ]
    # the measurements go above every candidate, so a cluster of heads
    # passing an observed epoch never hides it
    if points is not None:
        marks.append(points)
    if clock_text is not None:
        marks.append(Label(CLOCK_ID, clock_text, _CLOCK_XY))

    axes = AxisSpec(
        *_AXIS_LABELS,
        x_limits=x_limits,
        y_limits=y_limits,
        aspect="equal",
        x_reverse=True,
    )
    return TrackView(VIEW_ID, axes, marks=tuple(marks))


def _clock_format(offsets_d):
    """A clock format that prints every sample's offset exactly, when it can.

    The fewest decimals (at most three) that write every offset exactly,
    so a 1.5-day grid reads 0.0, 1.5, 3.0, ... and a quarter-day grid 0.00,
    0.25, 0.50, ... rather than rounding to repeated or misleading labels.
    Julian-day differences carry float noise near a microday, which the
    comparison tolerates. When no three-decimal number writes every offset
    (a grid of 12 samples over 400 days, say), the fewest decimals that
    place every offset within a quarter of the smallest step are used, so
    adjacent labels stay distinct and ordered without printing noise
    digits.
    """
    offsets = np.asarray(offsets_d, dtype=float)
    choices = range(_MAX_CLOCK_DECIMALS + 1)

    def error(decimals):
        return float(np.max(np.abs(np.round(offsets, decimals) - offsets)))

    exact = [d for d in choices if error(d) <= _CLOCK_EXACT_D]
    if exact:
        decimals = exact[0]
    else:
        steps = np.diff(offsets)
        tolerance = _CLOCK_STEP_FRACTION * float(steps.min()) if steps.size else 0.0
        close = [d for d in choices if error(d) <= tolerance]
        decimals = close[0] if close else _MAX_CLOCK_DECIMALS
    return "t = +{value:." + str(decimals) + "f} {unit}"


# --- public entry points ------------------------------------------------------------


def prepare_sky_track(
    orbit_or_radec,
    *,
    t_jd,
    Ms_kg=None,
    dist_pc=None,
    trig_solver=None,
    source_ids=None,
    weights=None,
    observations=None,
    iwa_arcsec=None,
    history=None,
    clock_fmt=None,
):
    """Prepare sky-plane tracks once, as a replayable ``Sequence``.

    Args:
        orbit_or_radec: An ``AbstractOrbit`` (propagated here once over
            ``t_jd``, requiring ``Ms_kg`` and ``dist_pc``), or bare tracks
            in arcsec: an array shaped ``(K, T, 2)`` of (RA, Dec) offsets,
            borrowed as given, or an ``(ra, dec)`` pair of ``(T,)`` or
            ``(K, T)`` arrays. A tuple or list of two is always read as an
            ``(ra, dec)`` pair. A row with a nonfinite coordinate is a gap.
        t_jd: Sample times in Julian days, shape ``(T,)``, finite and
            strictly increasing. They time the orbit's propagation and,
            for either door, the sequence samples.
        Ms_kg: Stellar mass in kg. Orbit door only.
        dist_pc: Distance to the star in parsecs. Orbit door only.
        trig_solver: Optional Kepler solver forwarded to the orbit; None
            uses orbix's default.
        source_ids: The scientific source of each track: None (every track
            is one source, ``"track"``), one name for every track, or K
            names. A name may repeat, so K posterior draws of one planet
            share a color while each stays its own element.
        weights: Optional per-track weights (length K), typically the
            posterior mass of each draw; renderers map them to opacity.
            Each track's head carries its track's weight, so it fades
            with the track.
        observations: Optional ``(xy_arcsec, xerr_arcsec, yerr_arcsec)``
            of observed positions, shaped ``(P, 2)``, ``(P,)``, ``(P,)``,
            with symmetric one-sigma errors along RA and Dec.
        iwa_arcsec: Optional inner-working-angle radius in arcsec, drawn
            as a shaded disk about the star.
        history: None for no trail, ``"all"`` for a trail growing from the
            first sample, or a positive int for a trail of at most that
            many samples ending at the current one.
        clock_fmt: ``str.format`` template for the acquisition-time label,
            using only ``{value}`` (days since the first sample, a float)
            and ``{unit}``, e.g. ``"t = {value:.1f} {unit}"``. None writes
            ``"t = +<offset> d"`` with the fewest decimals (at most three)
            that print every sample's offset exactly, or, when no such
            precision exists, that keep every label within a quarter of the
            smallest step. Frames and a strip's panels share it.

    Returns:
        An ``eyepiece.prepared.Sequence`` over a ``TrackView`` with element
        IDs ``path/<k>`` (full tracks), ``trail/<k>`` (when ``history`` is
        given), ``head/<k>`` (the held sample), ``observations``, ``iwa``,
        ``star``, and ``clock``. RA increases to the left, with equal x
        and y scale. Times are days since the first sample; the first
        sample's Julian day is the sequence ``epoch``.

    Raises:
        TypeError: If the orbit door is missing a propagation argument.
        ValueError: If the tracks, times, per-track arguments,
            observations, IWA radius, history, or clock format are
            malformed, naming the argument. Time, history, and clock-format
            errors are raised before the orbit is propagated.
    """
    times_jd = _sample_times(t_jd)
    trail, window = _trail_window(history)
    custom_clock = None if clock_fmt is None else Clock(CLOCK_ID, fmt=clock_fmt)
    tracks = _sky_xy(orbit_or_radec, t_jd, Ms_kg, dist_pc, trig_solver)
    if tracks.shape[1] != times_jd.size:
        raise ValueError(
            f"t_jd has {times_jd.size} entries for a {tracks.shape[1]}-sample track"
        )

    offsets_d = times_jd - times_jd[0]
    clock = (
        Clock(CLOCK_ID, fmt=_clock_format(offsets_d))
        if custom_clock is None
        else custom_clock
    )
    template = _build_view(
        tracks,
        source_ids=source_ids,
        weights=weights,
        observations=observations,
        iwa_arcsec=iwa_arcsec,
        trail=trail,
        heads=True,
        clock_text=clock.fmt.format(value=0.0, unit=_TIME_UNIT),
    )
    n_tracks = tracks.shape[0]
    channels = [
        ArrayChannel(_element_id(HEAD_PREFIX, k), "xy", tracks[k][:, None, :])
        for k in range(n_tracks)
    ]
    if trail:
        channels += [
            PathWindow(_element_id(TRAIL_PREFIX, k), history=window)
            for k in range(n_tracks)
        ]
    channels.append(clock)
    return Sequence(
        template,
        offsets_d,
        _TIME_UNIT,
        channels=tuple(channels),
        epoch=float(times_jd[0]),
    )


def prepare_sky_view(
    orbit_or_radec,
    *,
    t_jd=None,
    Ms_kg=None,
    dist_pc=None,
    trig_solver=None,
    source_ids=None,
    weights=None,
    observations=None,
    iwa_arcsec=None,
):
    """Prepare a static sky-plane ``TrackView``: full tracks, no time axis.

    The same view ``prepare_sky_track`` builds, without heads, trails, or
    a clock, so bare tracks that carry no timestamps need none invented.

    Args:
        orbit_or_radec: As for ``prepare_sky_track``.
        t_jd: Times in Julian days, shape ``(T,)``. Orbit door only: bare
            tracks are drawn as given.
        Ms_kg: As for ``prepare_sky_track``.
        dist_pc: As for ``prepare_sky_track``.
        trig_solver: As for ``prepare_sky_track``.
        source_ids: As for ``prepare_sky_track``.
        weights: As for ``prepare_sky_track``.
        observations: As for ``prepare_sky_track``.
        iwa_arcsec: As for ``prepare_sky_track``.

    Returns:
        A ``TrackView`` with ``path/<k>``, ``observations``, ``iwa``, and
        ``star`` marks, RA increasing to the left.

    Raises:
        TypeError: If the orbit door is missing a propagation argument, or
            ``t_jd`` is given with bare tracks.
        ValueError: As for ``prepare_sky_track``.
    """
    if t_jd is not None and not isinstance(orbit_or_radec, AbstractOrbit):
        raise TypeError(
            "t_jd only applies when propagating an orbit; bare tracks are "
            "drawn as given"
        )
    tracks = _sky_xy(orbit_or_radec, t_jd, Ms_kg, dist_pc, trig_solver)
    return _build_view(
        tracks,
        source_ids=source_ids,
        weights=weights,
        observations=observations,
        iwa_arcsec=iwa_arcsec,
    )
