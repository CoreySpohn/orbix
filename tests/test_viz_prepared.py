"""prepare_sky_track / prepare_sky_view: sky tracks prepared once, as records.

Propagation, projection to arcsec, and the device-to-host transfer happen
in preparation, once for the whole time grid. Frames, strips, seeks, and
both renderers then only index the prepared storage, so the head of a track
is always a propagated sample -- never a point interpolated along a chord,
which could invent an inner-working-angle crossing the orbit never makes.
"""

import os

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

pytest.importorskip("eyepiece")

from eyepiece.prepared import (
    AxisSpec,
    Label,
    Points,
    Region,
    Sequence,
    TrackView,
    find_element,
    weight_opacity,
)
from hwoutils.constants import G, Msun2kg, pc2AU, rad2arcsec

from orbix.orbit import KeplerianOrbit
from orbix.viz import prepare_sky_track, prepare_sky_view

DIST_PC = 10.0


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every figure a test creates."""
    yield
    plt.close("all")


def _require_manim():
    """Import Manim, failing (not skipping) when EYEPIECE_REQUIRE_MANIM=1."""
    if os.environ.get("EYEPIECE_REQUIRE_MANIM") == "1":
        import manim

        return manim
    return pytest.importorskip("manim")


def _circle(n, radius=1.0, turns=1.0, phase=0.0):
    """A (1, n, 2) circular track, counterclockwise in (RA, Dec)."""
    theta = phase + np.linspace(0.0, 2.0 * np.pi * turns, n, endpoint=False)
    return np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=-1)[None]


def _circular_orbit(K=1):
    """A face-on circular orbit: its sky track is a circle of known radius."""
    return KeplerianOrbit(
        a_AU=jnp.full((K,), 1.0),
        e=jnp.zeros(K),
        W_rad=jnp.zeros(K),
        i_rad=jnp.zeros(K),
        w_rad=jnp.zeros(K),
        M0_rad=jnp.linspace(0.0, 0.5, K),
        t0_d=jnp.zeros(K),
    )


def _circle_radius_arcsec(a_AU=1.0, dist_pc=DIST_PC):
    return a_AU * rad2arcsec / (dist_pc * pc2AU)


def _eccentric_orbit(K=1):
    """An eccentric, face-on orbit: projected speed peaks at periapsis."""
    return KeplerianOrbit(
        a_AU=jnp.full((K,), 1.3),
        e=jnp.full((K,), 0.5),
        W_rad=jnp.zeros(K),
        i_rad=jnp.zeros(K),
        w_rad=jnp.zeros(K),
        M0_rad=jnp.zeros(K),
        t0_d=jnp.zeros(K),
    )


STELLAR = dict(Ms_kg=jnp.atleast_1d(Msun2kg), dist_pc=jnp.atleast_1d(DIST_PC))


# --- the brief's contract ------------------------------------------------------


def test_sampling_does_not_invent_iwa_crossing():
    """Between two samples the head holds the first, outside the IWA."""
    xy = np.array([[[1.0, 0.0], [-1.0, 0.0]]])
    sequence = prepare_sky_track(
        xy,
        t_jd=np.array([0.0, 10.0]),
        source_ids=("b",),
        iwa_arcsec=0.5,
    )
    head = find_element(sequence.at(5.0).view, "head/0")
    np.testing.assert_allclose(head.xy, [[1.0, 0.0]])
    assert np.linalg.norm(head.xy[0]) > 0.5


# --- raw tracks ------------------------------------------------------------------


def test_raw_track_is_borrowed_and_every_frame_holds_a_sample():
    """A bare (K, T, 2) track is borrowed, and frame i holds sample i."""
    tracks = _circle(8).astype(np.float32)
    t_jd = 2460000.0 + np.arange(8.0) * 5.0
    sequence = prepare_sky_track(tracks, t_jd=t_jd, source_ids="b")

    assert isinstance(sequence, Sequence)
    assert sequence.time_unit == "d"
    assert sequence.epoch == pytest.approx(2460000.0)
    np.testing.assert_allclose(sequence.times, np.arange(8.0) * 5.0)

    template = sequence.template
    assert isinstance(template, TrackView)
    path = find_element(template, "path/0")
    assert np.shares_memory(path.xy, tracks)
    assert path.xy.dtype == np.float32
    assert path.visible == (0, None)
    for i in range(8):
        head = find_element(sequence.frame(i), "head/0")
        assert np.shares_memory(head.xy, tracks)
        np.testing.assert_array_equal(head.xy, tracks[0, i : i + 1])


def test_ra_dec_pair_door_matches_the_stacked_door():
    """An (ra, dec) pair prepares the same paths as the stacked array."""
    tracks = np.concatenate([_circle(6), _circle(6, radius=0.5)])
    t_jd = np.arange(6.0)
    pair = prepare_sky_track(
        (tracks[..., 0], tracks[..., 1]), t_jd=t_jd, source_ids="b"
    )
    stacked = prepare_sky_track(tracks, t_jd=t_jd, source_ids="b")
    for k in range(2):
        np.testing.assert_array_equal(
            find_element(pair.template, f"path/{k}").xy,
            find_element(stacked.template, f"path/{k}").xy,
        )
    single = prepare_sky_track(
        (tracks[0, :, 0], tracks[0, :, 1]), t_jd=t_jd, source_ids="b"
    )
    with pytest.raises(KeyError):
        find_element(single.template, "path/1")


def test_the_clock_reports_days_since_the_first_epoch():
    """The clock reads days since the first sample, as distinct labels."""
    sequence = prepare_sky_track(
        _circle(12), t_jd=2460000.0 + np.linspace(0.0, 400.0, 12)
    )
    assert find_element(sequence.frame(0), "clock").text == "t = +0 d"
    assert find_element(sequence.frame(11), "clock").text == "t = +400 d"
    strip = sequence.strip([11])
    assert find_element(strip, "0/clock").text == "t = +400 d"
    with pytest.raises(KeyError):
        find_element(strip, "0/time")  # the clock is the only time label

    finer = prepare_sky_track(_circle(3), t_jd=np.array([0.0, 0.25, 0.5]))
    assert find_element(finer.frame(1), "clock").text == "t = +0.25 d"


def test_fractional_multi_day_steps_print_every_offset_exactly():
    """A 1.5-day grid keeps its half days instead of rounding them away."""
    t_jd = 2460000.3 + np.array([0.0, 1.5, 3.0, 4.5, 6.0])
    sequence = prepare_sky_track(_circle(5), t_jd=t_jd)
    texts = [find_element(sequence.frame(i), "clock").text for i in range(5)]
    assert texts == [f"t = +{v} d" for v in ("0.0", "1.5", "3.0", "4.5", "6.0")]

    # A step no precision writes exactly keeps the fewest decimals that
    # stay within a quarter step: whole days for a 3-day step, tenths for a
    # 1.5-day one.
    dense = prepare_sky_track(_circle(200), t_jd=np.linspace(0.0, 600.0, 200))
    assert find_element(dense.frame(100), "clock").text == "t = +302 d"
    denser = prepare_sky_track(_circle(400), t_jd=np.linspace(0.0, 600.0, 400))
    assert find_element(denser.frame(200), "clock").text == "t = +300.8 d"


def test_clock_fmt_sets_the_label_and_is_checked_before_propagating():
    """A caller's clock format labels frames and strips; a bad one fails first."""
    sequence = prepare_sky_track(
        _circle(3), t_jd=np.array([0.0, 1.0, 2.5]), clock_fmt="{value:.2f} {unit}"
    )
    assert find_element(sequence.frame(2), "clock").text == "2.50 d"
    assert find_element(sequence.strip([2]), "0/clock").text == "2.50 d"
    # The orbit door is missing Ms_kg and dist_pc, so propagating would raise
    # a TypeError; the format error comes first.
    with pytest.raises(ValueError, match="fmt"):
        prepare_sky_track(
            _circular_orbit(), t_jd=np.array([0.0, 1.0, 2.5]), clock_fmt="{when}"
        )


def test_the_view_has_the_named_marks_in_one_ra_left_frame():
    """One RA-left, equal-aspect TrackView holds every named mark."""
    obs = (
        np.array([[0.2, 0.1], [-0.3, 0.4]]),
        np.array([0.01, 0.02]),
        np.array([0.03, 0.04]),
    )
    sequence = prepare_sky_track(
        np.concatenate([_circle(5), _circle(5, radius=0.8)]),
        t_jd=np.arange(5.0),
        source_ids="b",
        observations=obs,
        iwa_arcsec=0.1,
        history="all",
    )
    view = sequence.frame(2)
    ids = [mark.id for mark in view.marks]
    for expected in (
        "iwa",
        "path/0",
        "path/1",
        "trail/0",
        "trail/1",
        "observations",
        "star",
        "head/0",
        "head/1",
        "clock",
    ):
        assert expected in ids
    # measurements draw above every candidate head, the IWA below the paths
    assert ids.index("observations") > ids.index("head/1")
    assert ids.index("iwa") < ids.index("path/0")
    assert isinstance(find_element(view, "iwa"), Region)
    assert isinstance(find_element(view, "star"), Points)
    assert isinstance(find_element(view, "clock"), Label)
    np.testing.assert_array_equal(find_element(view, "star").xy, [[0.0, 0.0]])

    axes = view.axes
    assert isinstance(axes, AxisSpec)
    assert axes.x_reverse is True
    assert axes.aspect == "equal"
    assert axes.x_label == "RA offset [arcsec]"
    assert axes.y_label == "Dec offset [arcsec]"
    # every track sample and every error-bar end lies inside the fixed limits
    rows = np.concatenate(
        [
            np.concatenate([_circle(5), _circle(5, radius=0.8)]).reshape(-1, 2),
            obs[0] - np.stack([obs[1], obs[2]], axis=1),
            obs[0] + np.stack([obs[1], obs[2]], axis=1),
        ]
    )
    assert np.all(rows[:, 0] > axes.x_limits[0]) and np.all(
        rows[:, 0] < axes.x_limits[1]
    )
    assert np.all(rows[:, 1] > axes.y_limits[0]) and np.all(
        rows[:, 1] < axes.y_limits[1]
    )


# --- observations, IWA, orientation ---------------------------------------------


def test_observation_error_endpoints_are_data_coordinates():
    """Observation error bars end at xy -/+ err along each axis."""
    xy = np.array([[0.5, 0.25], [-0.4, 0.1]])
    xerr = np.array([0.02, 0.05])
    yerr = np.array([0.03, 0.01])
    sequence = prepare_sky_track(
        _circle(4), t_jd=np.arange(4.0), observations=(xy, xerr, yerr)
    )
    points = find_element(sequence.template, "observations")
    assert points.source_id is None
    assert np.shares_memory(points.xy, xy)
    np.testing.assert_array_equal(points.xerr, xerr)
    np.testing.assert_array_equal(points.yerr, yerr)

    from eyepiece import mpl

    result = mpl.render(sequence.frame(0))
    x_segments = np.asarray(result.parts["observations/xerr"].get_segments())
    y_segments = np.asarray(result.parts["observations/yerr"].get_segments())
    np.testing.assert_allclose(x_segments[:, 0, 0], xy[:, 0] - xerr)
    np.testing.assert_allclose(x_segments[:, 1, 0], xy[:, 0] + xerr)
    np.testing.assert_allclose(x_segments[:, :, 1], xy[:, [1, 1]])
    np.testing.assert_allclose(y_segments[:, 0, 1], xy[:, 1] - yerr)
    np.testing.assert_allclose(y_segments[:, 1, 1], xy[:, 1] + yerr)


def test_observations_must_be_the_documented_triple():
    """Observations must be the (xy, xerr, yerr) triple."""
    with pytest.raises(ValueError, match="observations"):
        prepare_sky_track(
            _circle(4), t_jd=np.arange(4.0), observations=(np.zeros((2, 2)),)
        )
    with pytest.raises(ValueError, match="observations"):
        prepare_sky_track(
            _circle(4),
            t_jd=np.arange(4.0),
            observations=(np.zeros((2, 2)), np.zeros(3), np.zeros(2)),
        )


def test_iwa_region_radius_and_shading():
    """The IWA is a shaded reference disk of the given radius."""
    sequence = prepare_sky_track(_circle(4), t_jd=np.arange(4.0), iwa_arcsec=0.3)
    region = find_element(sequence.template, "iwa")
    assert region.outer_radius == pytest.approx(0.3)
    assert region.inner_radius == 0.0
    assert region.role == "reference"
    np.testing.assert_array_equal(region.center, [0.0, 0.0])

    from eyepiece import mpl

    patch = mpl.render(sequence.frame(0)).parts["iwa"]
    assert patch.get_radius() == pytest.approx(0.3)
    assert tuple(patch.center) == (0.0, 0.0)
    assert patch.get_facecolor()[3] > 0.0  # a TrackView region keeps its fill

    with pytest.raises(ValueError, match="iwa"):
        prepare_sky_track(_circle(4), t_jd=np.arange(4.0), iwa_arcsec=-0.1)


def test_positive_ra_offset_draws_left_of_the_star():
    """Positive RA draws left of the star, at equal x and y scale."""
    tracks = np.array([[[1.0, 0.0], [1.0, 0.5]]])
    sequence = prepare_sky_track(tracks, t_jd=np.array([0.0, 1.0]))

    from eyepiece import mpl

    result = mpl.render(sequence.frame(0))
    result.fig.canvas.draw()  # the equal aspect is applied at draw time
    to_pixels = result.ax.transData.transform
    head_px = to_pixels(result.parts["head/0"].get_offsets()[0])
    star_px = to_pixels(result.parts["star"].get_offsets()[0])
    assert head_px[0] < star_px[0]
    assert head_px[1] == pytest.approx(star_px[1])
    # equal physical scale: one arcsec spans the same pixels on both axes
    unit_x = abs(to_pixels((1.0, 0.0))[0] - to_pixels((0.0, 0.0))[0])
    unit_y = abs(to_pixels((0.0, 1.0))[1] - to_pixels((0.0, 0.0))[1])
    assert unit_x == pytest.approx(unit_y, rel=1e-6)


# --- weights and source identity --------------------------------------------------


def test_weights_and_repeated_source_identity():
    """Weights set opacity; a repeated source shares one color."""
    tracks = np.concatenate(
        [_circle(4), _circle(4, radius=0.9), _circle(4, radius=0.5)]
    )
    weights = np.array([1.0, 0.5, 0.0])
    sequence = prepare_sky_track(
        tracks, t_jd=np.arange(4.0), source_ids=("b", "b", "c"), weights=weights
    )
    view = sequence.frame(0)
    assert [find_element(view, f"path/{k}").source_id for k in range(3)] == [
        "b",
        "b",
        "c",
    ]
    assert [find_element(view, f"head/{k}").source_id for k in range(3)] == [
        "b",
        "b",
        "c",
    ]
    assert [find_element(view, f"path/{k}").weight for k in range(3)] == [
        1.0,
        0.5,
        0.0,
    ]
    assert [find_element(view, f"head/{k}").weight for k in range(3)] == [
        1.0,
        0.5,
        0.0,
    ]

    from eyepiece import mpl

    result = mpl.render(view)
    alphas = [result.parts[f"path/{k}"].get_alpha() for k in range(3)]
    np.testing.assert_allclose(alphas, weight_opacity(weights))
    # Each head fades exactly as its track does.
    heads = [result.parts[f"head/{k}"].get_alpha() for k in range(3)]
    np.testing.assert_allclose(heads, alphas)
    colors = [result.parts[f"path/{k}"].get_color() for k in range(3)]
    assert colors[0] == colors[1] != colors[2]
    assert result.cast.names == ("b", "c")


def test_heads_fade_with_their_tracks_when_weights_do_not_peak_at_one():
    """Star and observations never set the Points scale; heads match paths."""
    tracks = np.concatenate([_circle(4), _circle(4, radius=0.5)])
    weights = np.array([0.5, 0.2])
    observations = (np.array([[0.1, 0.2]]), np.array([0.05]), np.array([0.05]))
    sequence = prepare_sky_track(
        tracks, t_jd=np.arange(4.0), weights=weights, observations=observations
    )
    from eyepiece import mpl

    result = mpl.render(sequence.frame(0))
    paths = [result.parts[f"path/{k}"].get_alpha() for k in range(2)]
    heads = [result.parts[f"head/{k}"].get_alpha() for k in range(2)]
    np.testing.assert_allclose(paths, weight_opacity(weights))
    np.testing.assert_allclose(heads, paths)
    # The measurements and the star stay at the top of the mapping.
    top = float(weight_opacity(weights).max())
    assert find_element(sequence.template, "observations").weight == 0.5
    assert find_element(sequence.template, "star").weight == 0.5
    for part in (result.parts["observations"], result.parts["star"]):
        alpha = part.get_alpha()
        assert alpha is not None and alpha == pytest.approx(top)


def test_default_source_is_one_shared_identity():
    """With no source IDs, every track is one source."""
    sequence = prepare_sky_track(
        np.concatenate([_circle(4), _circle(4, radius=0.5)]), t_jd=np.arange(4.0)
    )
    ids = {find_element(sequence.template, f"path/{k}").source_id for k in range(2)}
    assert len(ids) == 1 and None not in ids


def test_per_path_arguments_must_match_the_track_count():
    """Per-track arguments must have one entry per track."""
    tracks = np.concatenate([_circle(4), _circle(4, radius=0.5)])
    with pytest.raises(ValueError, match="source_ids has 3"):
        prepare_sky_track(tracks, t_jd=np.arange(4.0), source_ids=("a", "b", "c"))
    with pytest.raises(ValueError, match="weights has 1"):
        prepare_sky_track(tracks, t_jd=np.arange(4.0), weights=[1.0])
    with pytest.raises(ValueError, match="weight"):
        prepare_sky_track(tracks, t_jd=np.arange(4.0), weights=[1.0, -1.0])


# --- shapes and times ------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [np.zeros((1, 4, 3)), np.zeros((4, 2)), np.zeros((1, 1, 4, 2)), np.zeros((2, 4))],
)
def test_unknown_track_shapes_raise_naming_the_shape(bad):
    """An unsupported track shape raises, naming the accepted shape."""
    with pytest.raises(ValueError, match=r"\(K, T, 2\)"):
        prepare_sky_track(bad, t_jd=np.arange(4.0))
    with pytest.raises(ValueError, match=r"\(K, T, 2\)"):
        prepare_sky_view(bad)


def test_times_must_match_and_increase():
    """Times must match the track and be finite and increasing."""
    with pytest.raises(ValueError, match="t_jd has 3"):
        prepare_sky_track(_circle(4), t_jd=np.arange(3.0))
    with pytest.raises(ValueError, match="increasing"):
        prepare_sky_track(_circle(4), t_jd=np.array([0.0, 2.0, 1.0, 3.0]))
    with pytest.raises(ValueError, match="finite"):
        prepare_sky_track(_circle(4), t_jd=np.array([0.0, 1.0, np.nan, 3.0]))
    with pytest.raises(ValueError, match="history"):
        prepare_sky_track(_circle(4), t_jd=np.arange(4.0), history=0)
    with pytest.raises(ValueError, match="history"):
        prepare_sky_track(_circle(4), t_jd=np.arange(4.0), history="window")


def test_a_bad_time_grid_is_refused_before_propagating(monkeypatch):
    """Time and history errors are raised before propagation."""
    calls = []
    original = KeplerianOrbit.position_arcsec

    def counting(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(KeplerianOrbit, "position_arcsec", counting)
    with pytest.raises(ValueError, match="increasing"):
        prepare_sky_track(_circular_orbit(), t_jd=np.array([0.0, 2.0, 1.0]), **STELLAR)
    with pytest.raises(ValueError, match="history"):
        prepare_sky_track(_circular_orbit(), t_jd=np.arange(3.0), history=-2, **STELLAR)
    assert calls == []


def test_the_orbit_door_names_what_is_missing():
    """The orbit door names each missing propagation argument."""
    with pytest.raises(TypeError, match="Ms_kg, dist_pc"):
        prepare_sky_track(_circular_orbit(), t_jd=np.arange(3.0))
    with pytest.raises(TypeError, match="t_jd"):
        prepare_sky_view(_circular_orbit(), **STELLAR)


def test_nonfinite_rows_are_gaps_not_coordinates():
    """A nonfinite sample hides the head without losing its handle."""
    tracks = _circle(4)
    tracks[0, 2] = np.nan
    sequence = prepare_sky_track(tracks, t_jd=np.arange(4.0))

    from eyepiece import mpl

    result = mpl.render(sequence.frame(1))
    head = result.parts["head/0"]
    result.update(sequence.frame(2))
    assert result.parts["head/0"] is head
    assert np.ma.getmaskarray(head.get_offsets()).all()
    result.update(sequence.frame(3))
    np.testing.assert_allclose(head.get_offsets()[0], tracks[0, 3])

    with pytest.raises(ValueError, match="finite"):
        prepare_sky_track(np.full((1, 3, 2), np.nan), t_jd=np.arange(3.0))


# --- trails and seeking ---------------------------------------------------------


def test_growing_and_sliding_trails_and_order_free_seeking():
    """Trails grow or slide by sample, and seeking is order free."""
    tracks = _circle(10)
    t_jd = np.arange(10.0)
    grown = prepare_sky_track(tracks, t_jd=t_jd, history="all")
    sliding = prepare_sky_track(tracks, t_jd=t_jd, history=3)
    bare = prepare_sky_track(tracks, t_jd=t_jd)

    for i in (0, 7, 2, 9, 0):
        assert find_element(grown.at(t_jd[i]).view, "trail/0").visible == (0, i + 1)
        assert find_element(sliding.at(t_jd[i]).view, "trail/0").visible == (
            max(0, i - 2),
            i + 1,
        )
        assert find_element(grown.at(t_jd[i]).view, "path/0").visible == (0, None)
    assert (
        find_element(grown.frame(2), "trail/0").xy
        is find_element(grown.frame(7), "trail/0").xy
    )
    with pytest.raises(KeyError):
        find_element(bare.template, "trail/0")

    direct = find_element(sliding.frame(2), "trail/0")
    seeked = find_element(sliding.at(t_jd[2]).view, "trail/0")
    assert direct.visible == seeked.visible


# --- propagation happens once ------------------------------------------------------


def test_orbit_preparation_propagates_once(monkeypatch):
    """Prepare east/north once; circular solution from Green (1985), ch. 17."""
    orbit = _circular_orbit(3)
    t_jd = jnp.linspace(0.0, 400.0, 30)
    phase = np.linspace(0.0, 0.5, 3)[:, None] + np.sqrt(G * Msun2kg) * np.asarray(t_jd)
    expected_ra = 0.1 * np.sin(phase)
    expected_dec = 0.1 * np.cos(phase)

    calls = {"position_arcsec": 0, "propagate": 0}
    original_position = KeplerianOrbit.position_arcsec
    original_propagate = KeplerianOrbit.propagate

    def counting_position(self, *args, **kwargs):
        calls["position_arcsec"] += 1
        return original_position(self, *args, **kwargs)

    def counting_propagate(self, *args, **kwargs):
        calls["propagate"] += 1
        return original_propagate(self, *args, **kwargs)

    monkeypatch.setattr(KeplerianOrbit, "position_arcsec", counting_position)
    monkeypatch.setattr(KeplerianOrbit, "propagate", counting_propagate)

    sequence = prepare_sky_track(orbit, t_jd=t_jd, history=5, **STELLAR)
    assert calls == {"position_arcsec": 1, "propagate": 1}
    for k in range(3):
        path = find_element(sequence.template, f"path/{k}")
        assert isinstance(path.xy, np.ndarray)
        assert path.xy.shape == (30, 2)
        np.testing.assert_allclose(path.xy[:, 0], expected_ra[k], atol=1e-6, rtol=0)
        np.testing.assert_allclose(path.xy[:, 1], expected_dec[k], atol=1e-6, rtol=0)

    def refuse(self, *args, **kwargs):
        raise AssertionError("renderers must not propagate")

    monkeypatch.setattr(KeplerianOrbit, "position_arcsec", refuse)
    monkeypatch.setattr(KeplerianOrbit, "propagate", refuse)

    from eyepiece import mpl

    sequence.frame(29)
    sequence.strip([0, 15, 29])
    sequence.at(123.4)
    sequence.schedule(run_time=2.0, fps=10)
    result = mpl.render(sequence.frame(0))
    result.update(sequence.frame(10))
    anim = mpl.animate(sequence, run_time=1.0, fps=10)
    for i in range(anim.n_frames):
        anim.draw(anim.fig, i)


def test_dense_circular_orbit_matches_analytic_checkpoints():
    """Prepared east/north follows Green (1985), ch. 17 circular motion."""
    radius = _circle_radius_arcsec()
    period_d = 365.25
    t_jd = np.linspace(0.0, period_d, 400)
    sequence = prepare_sky_track(
        _circular_orbit(), t_jd=t_jd, iwa_arcsec=0.99 * radius, **STELLAR
    )

    for index in (0, 57, 199, 311, 399):
        phase = np.sqrt(G * Msun2kg) * t_jd[index]
        expected = radius * np.array([np.sin(phase), np.cos(phase)])
        head = find_element(sequence.frame(index), "head/0")
        np.testing.assert_allclose(head.xy[0], expected, rtol=0, atol=1e-6)

    # the radius is held to the default Kepler solver's accuracy, not exactly
    path = find_element(sequence.template, "path/0")
    np.testing.assert_allclose(np.linalg.norm(path.xy, axis=1), radius, rtol=1e-5)

    # between samples the head holds the earlier propagated sample exactly:
    # it stays on the orbit and never cuts the chord toward the IWA
    for t in np.linspace(0.0, period_d, 1111):
        held = int(np.searchsorted(t_jd, t, side="right")) - 1
        head = find_element(sequence.at(t).view, "head/0")
        np.testing.assert_array_equal(head.xy[0], path.xy[held])
        assert np.linalg.norm(head.xy[0]) > 0.99 * radius

    # the Matplotlib path is exactly the propagated samples, not densified
    from eyepiece import mpl

    result = mpl.render(sequence.frame(399))
    np.testing.assert_array_equal(result.parts["path/0"].get_xydata(), path.xy)


def test_no_renderer_interpolates_a_sparse_chord_through_the_iwa():
    """Playback of a sparse circle never cuts a chord into the IWA."""
    radius = 1.0
    tracks = _circle(3, radius=radius)
    iwa = 0.6 * radius  # every chord's midpoint sits at 0.5 * radius
    sequence = prepare_sky_track(tracks, t_jd=np.array([0.0, 1.0, 2.0]), iwa_arcsec=iwa)

    from eyepiece import mpl

    anim = mpl.animate(sequence, run_time=2.0, fps=30)
    head = anim.fig.axes[0].collections[-1]  # head/0 is the last point set drawn
    for i in range(anim.n_frames):
        anim.draw(anim.fig, i)
        xy = np.asarray(head.get_offsets())[0]
        assert np.linalg.norm(xy) == pytest.approx(radius)

    result = mpl.render(sequence.frame(0))
    for t in np.linspace(0.0, 2.0, 61):
        result.update(sequence.at(t).view)
        xy = np.asarray(result.parts["head/0"].get_offsets())[0]
        assert np.linalg.norm(xy) == pytest.approx(radius)
        assert np.linalg.norm(xy) > iwa


def test_equal_time_marks_spread_where_the_planet_moves_fast():
    """Equal-time samples spread near periapsis, bunch near apoapsis."""
    t_jd = np.linspace(0.0, 541.0, 120)
    sequence = prepare_sky_track(_eccentric_orbit(), t_jd=t_jd, **STELLAR)
    xy = find_element(sequence.template, "path/0").xy
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    separation = np.linalg.norm(xy[:-1], axis=1)
    # the widest equal-time step sits near periapsis, the narrowest near apoapsis
    assert separation[np.argmax(step)] < np.percentile(separation, 10)
    assert separation[np.argmin(step)] > np.percentile(separation, 90)


# --- the static view ---------------------------------------------------------------


def test_static_view_invents_no_timestamps():
    """The static view has no heads, trails, or clock."""
    tracks = np.concatenate([_circle(5), _circle(5, radius=0.5)])
    view = prepare_sky_view(tracks, source_ids="b", iwa_arcsec=0.1)
    assert isinstance(view, TrackView)
    ids = [mark.id for mark in view.marks]
    assert ids == ["iwa", "path/0", "path/1", "star"]
    assert all(find_element(view, f"path/{k}").visible == (0, None) for k in range(2))
    assert np.shares_memory(find_element(view, "path/0").xy, tracks)

    with pytest.raises(TypeError, match="t_jd"):
        prepare_sky_view(tracks, t_jd=np.arange(5.0))


def test_static_view_from_an_orbit_matches_the_sequence_paths():
    """The static and sequence views share one builder's paths."""
    orbit = _eccentric_orbit(2)
    t_jd = np.linspace(0.0, 300.0, 25)
    view = prepare_sky_view(orbit, t_jd=t_jd, **STELLAR)
    sequence = prepare_sky_track(orbit, t_jd=t_jd, **STELLAR)
    for k in range(2):
        np.testing.assert_array_equal(
            find_element(view, f"path/{k}").xy,
            find_element(sequence.template, f"path/{k}").xy,
        )
    assert view.axes.x_limits == sequence.template.axes.x_limits


# --- Manim places the same data coordinates -------------------------------------


def _scene_to_data(points, frame, spec):
    """Invert a panel frame's data-to-scene map, for (P, 3) scene points."""
    _, upper_left, lower_left, lower_right = frame.get_vertices()
    x_vec, y_vec = lower_right - lower_left, upper_left - lower_left
    rel = np.asarray(points, float) - lower_left
    u = rel @ x_vec / (x_vec @ x_vec)
    v = rel @ y_vec / (y_vec @ y_vec)
    if spec.x_reverse:
        u = 1.0 - u
    x_low, x_high = spec.x_limits
    y_low, y_high = spec.y_limits
    return np.stack([x_low + u * (x_high - x_low), y_low + v * (y_high - y_low)], 1)


def test_manim_places_heads_observations_and_iwa_in_data_coordinates(tmp_path):
    """Manim places heads, observations, star, and IWA in data space."""
    manim = _require_manim()
    import eyepiece.manim as em

    obs = (
        np.array([[0.5, 0.25], [-0.4, 0.1]]),
        np.array([0.02, 0.05]),
        np.array([0.03, 0.01]),
    )
    tracks = np.concatenate([_circle(6), _circle(6, radius=0.7, phase=0.3)])
    sequence = prepare_sky_track(
        tracks,
        t_jd=np.arange(6.0),
        source_ids=("b", "c"),
        observations=obs,
        iwa_arcsec=0.2,
        history="all",
    )
    with manim.tempconfig({"media_dir": str(tmp_path / "media")}):
        result = em.render(sequence.frame(0))
        frame = result.parts["sky/frame"]
        spec = sequence.template.axes
        for i in (3, 0, 5):
            result.update(sequence.frame(i))
            for k in range(2):
                (marker,) = result.parts[f"head/{k}"]
                xy = _scene_to_data([marker.get_center()], frame, spec)[0]
                np.testing.assert_allclose(xy, tracks[k, i], atol=1e-9)
        centers = [m.get_center() for m in result.parts["observations"]]
        np.testing.assert_allclose(
            _scene_to_data(centers, frame, spec), obs[0], atol=1e-9
        )
        (star,) = result.parts["star"]
        np.testing.assert_allclose(
            _scene_to_data([star.get_center()], frame, spec)[0], [0.0, 0.0], atol=1e-9
        )
        iwa_xy = _scene_to_data(result.parts["iwa"].get_anchors(), frame, spec)
        np.testing.assert_allclose(np.linalg.norm(iwa_xy, axis=1), 0.2, rtol=1e-6)
        # positive RA is drawn left of the star
        (head,) = result.parts["head/0"]
        result.update(sequence.frame(0))
        assert head.get_center()[0] < star.get_center()[0]


def test_manim_error_bar_endpoints_match_data_and_matplotlib(tmp_path):
    """Manim error bars end at xy -/+ err, RA-left, where Matplotlib draws them."""
    manim = _require_manim()
    import eyepiece.manim as em
    from eyepiece import mpl

    xy = np.array([[0.5, 0.25], [-0.4, 0.1], [0.05, -0.3]])
    xerr = np.array([0.02, 0.05, 0.08])
    yerr = np.array([0.03, 0.01, 0.06])
    sequence = prepare_sky_track(
        _circle(4), t_jd=np.arange(4.0), observations=(xy, xerr, yerr)
    )
    view = sequence.frame(1)
    spec = view.axes
    mpl_result = mpl.render(view)
    with manim.tempconfig({"media_dir": str(tmp_path / "media")}):
        result = em.render(view)
        frame = result.parts["sky/frame"]
        for name, axis, err in (("xerr", 0, xerr), ("yerr", 1, yerr)):
            bars = list(result.parts[f"observations/{name}"])
            assert len(bars) == len(xy)
            starts = np.array([bar.points[0] for bar in bars])
            ends = np.array([bar.points[-1] for bar in bars])
            ends_data = np.stack(
                [
                    _scene_to_data(starts, frame, spec),
                    _scene_to_data(ends, frame, spec),
                ],
                axis=1,
            )
            ends_data = np.sort(ends_data, axis=1)  # (P, 2 ends, 2), low end first
            low, high = xy.copy(), xy.copy()
            low[:, axis] -= err
            high[:, axis] += err
            order = np.argsort(np.stack([low[:, axis], high[:, axis]], 1), axis=1)
            assert np.all(order[:, 0] == 0)
            np.testing.assert_allclose(ends_data[:, 0], low, atol=1e-9)
            np.testing.assert_allclose(ends_data[:, 1], high, atol=1e-9)

            mpl_segments = np.sort(
                np.asarray(mpl_result.parts[f"observations/{name}"].get_segments()),
                axis=1,
            )
            np.testing.assert_allclose(ends_data, mpl_segments, atol=1e-9)

        # positive RA is on the left: the +xerr end sits left of the -xerr end
        for bar, x_low, x_high in zip(
            result.parts["observations/xerr"], xy[:, 0] - xerr, xy[:, 0] + xerr
        ):
            scene_x = [bar.points[0][0], bar.points[-1][0]]
            data_x = _scene_to_data(
                np.array([bar.points[0], bar.points[-1]]), frame, spec
            )[:, 0]
            at_high = scene_x[int(np.argmax(data_x))]
            at_low = scene_x[int(np.argmin(data_x))]
            assert at_high < at_low
            np.testing.assert_allclose(sorted(data_x), [x_low, x_high], atol=1e-9)
