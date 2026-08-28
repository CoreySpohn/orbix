"""Contract tests for orbix.viz plot_sky_track and plot_orbit.

Fixtures are deliberately eccentric, inclined, and rotated so no element
collapses to a symmetry a wrong implementation could hide behind.
"""

import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")

MSUN_KG = 1.988409870698051e30


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every figure a test creates."""
    yield
    plt.close("all")


def _orbit(K=1):
    """An eccentric, inclined, rotated orbit; K > 1 spreads the elements."""
    from orbix.orbit import KeplerianOrbit

    spread = jnp.linspace(0.0, 0.2, K)
    return KeplerianOrbit(
        a_AU=1.3 + spread,
        e=0.31 + 0.1 * spread,
        W_rad=2.3 + spread,
        i_rad=1.05 + 0.3 * spread,
        w_rad=-0.7 + spread,
        M0_rad=0.4 + spread,
        t0_d=jnp.full((K,), 2460000.0),
    )


T_JD = jnp.linspace(2460000.0, 2460500.0, 60)
STELLAR = dict(Ms_kg=jnp.atleast_1d(MSUN_KG), dist_pc=jnp.atleast_1d(10.0))


def test_plot_sky_track_single_orbit_contract():
    """One orbit: PlotResult, vocabulary keys, labels, equal aspect, RA flip."""
    import eyepiece as ep

    from orbix.viz import plot_sky_track

    result = plot_sky_track(_orbit(), T_JD, **STELLAR)
    assert isinstance(result, ep.PlotResult)
    assert set(result.artists) <= ep.ARTIST_KEYS
    assert len(result.artists["lines"]) == 1
    assert result.ax.get_aspect() == 1.0
    assert result.ax.xaxis_inverted()
    assert "RA offset" in result.ax.get_xlabel()
    assert "Dec offset" in result.ax.get_ylabel()


def test_plot_sky_track_fan_with_weights_iwa_data():
    """K orbits fan out in one shared color with IWA disk and epochs."""
    from orbix.viz import plot_sky_track

    K = 4
    data = (np.array([0.05, -0.03]), np.array([0.02, 0.06]), np.array([0.01, 0.01]))
    result = plot_sky_track(
        _orbit(K),
        T_JD,
        **STELLAR,
        weights=[1.0, 0.5, 0.2, 0.1],
        iwa=0.06,
        data=data,
    )
    lines = result.artists["lines"]
    assert len(lines) == K
    assert len({line.get_color() for line in lines}) == 1
    assert "ellipse" in result.artists
    assert "collection" in result.artists


def test_plot_sky_track_bare_arrays_door():
    """Bare (ra, dec) tracks draw without labels or propagation arguments."""
    from orbix.viz import plot_sky_track

    theta = np.linspace(0.0, 2.0 * np.pi, 50)
    ra, dec = 0.3 * np.cos(theta), 0.2 * np.sin(theta)

    result = plot_sky_track((ra, dec))
    assert len(result.artists["lines"]) == 1
    assert result.ax.get_xlabel() == ""

    stacked = np.stack([np.stack([ra, dec]), np.stack([0.5 * ra, 0.5 * dec])])
    result = plot_sky_track(stacked)
    assert len(result.artists["lines"]) == 2

    with pytest.raises(TypeError, match="t_jd"):
        plot_sky_track((ra, dec), T_JD)


def test_plot_sky_track_shape_errors_name_shapes():
    """A wrong-shaped array raises ValueError naming got and accepted shapes."""
    from orbix.viz import plot_sky_track

    with pytest.raises(ValueError, match=r"\(2, T\)"):
        plot_sky_track(np.zeros((4, 5)))
    with pytest.raises(ValueError, match=r"\(3, 5\)"):
        plot_sky_track(np.zeros((3, 5)))


def test_plot_sky_track_missing_propagation_args_named():
    """Omitting the stellar context names exactly what is missing."""
    from orbix.viz import plot_sky_track

    with pytest.raises(TypeError, match="Ms_kg, dist_pc"):
        plot_sky_track(_orbit(), T_JD)


def test_plot_sky_track_overplot_keeps_ra_inverted():
    """A second call on the same axes does not flip the RA axis back."""
    from orbix.viz import plot_sky_track

    result = plot_sky_track(_orbit(), T_JD, **STELLAR)
    left_before = result.ax.get_xlim()[0]
    plot_sky_track(_orbit(3), T_JD, **STELLAR, ax=result.ax)
    assert result.ax.xaxis_inverted()
    assert result.ax.get_xlim()[0] == pytest.approx(left_before, rel=0.5)


def test_plot_sky_track_geometry_stays_in_handed_slot():
    """Drawing into axes[0] of a 1x2 grid leaves the slots equal width."""
    from orbix.viz import plot_sky_track

    fig, axes = plt.subplots(1, 2, layout="constrained")
    plot_sky_track(_orbit(), T_JD, **STELLAR, ax=axes[0])
    w0 = axes[0].get_position(original=True).width
    w1 = axes[1].get_position(original=True).width
    assert w0 == pytest.approx(w1)


def test_plot_orbit_single_contract():
    """One orbit: 3D axes, line + scatter artists, AU labels, cubic box."""
    import eyepiece as ep

    from orbix.viz import plot_orbit

    result = plot_orbit(_orbit(), T_JD, Ms_kg=STELLAR["Ms_kg"])
    assert isinstance(result, ep.PlotResult)
    assert result.ax.name == "3d"
    assert "line" in result.artists and "scatter" in result.artists
    assert "AU" in result.ax.get_xlabel()
    xlim, zlim = result.ax.get_xlim(), result.ax.get_zlim()
    assert xlim[0] == pytest.approx(-xlim[1])
    assert zlim == pytest.approx(xlim)


def test_plot_orbit_batch_returns_lists():
    """K orbits return lists under lines/scatter in track order."""
    from orbix.viz import plot_orbit

    K = 3
    result = plot_orbit(_orbit(K), T_JD, Ms_kg=STELLAR["Ms_kg"])
    assert len(result.artists["lines"]) == K
    assert len(result.artists["scatter"]) == K


def test_plot_orbit_bare_arrays_door():
    """Bare (T, 3) positions draw with no labels; (K, T, 3) batches."""
    from orbix.viz import plot_orbit

    theta = np.linspace(0.0, 2.0 * np.pi, 40)
    track = np.stack([np.cos(theta), 0.8 * np.sin(theta), 0.3 * np.sin(theta)], axis=-1)
    result = plot_orbit(track)
    assert result.ax.get_xlabel() == ""

    result = plot_orbit(np.stack([track, 0.5 * track]))
    assert len(result.artists["lines"]) == 2

    with pytest.raises(ValueError, match=r"\(T, 3\)"):
        plot_orbit(np.zeros((7, 4)))


def test_plot_orbit_mark_geometry_is_exact():
    """Periapsis sits at a(1-e) and both nodes sit in the sky plane."""
    from orbix.viz.orbit import _mark_geometry

    orbit = _orbit(3)
    periapsis, ascending, descending = _mark_geometry(orbit)
    np.testing.assert_allclose(
        np.linalg.norm(periapsis, axis=-1),
        np.asarray(orbit.a_AU) * (1.0 - np.asarray(orbit.e)),
        rtol=1e-10,
    )
    np.testing.assert_allclose(ascending[:, 2], 0.0, atol=1e-12)
    np.testing.assert_allclose(descending[:, 2], 0.0, atol=1e-12)
    assert not np.allclose(ascending, descending)


def test_plot_orbit_ascending_node_rises():
    """Z increases through the ascending node along the propagated track."""
    from orbix.viz.orbit import _mark_geometry

    orbit = _orbit()
    _, ascending, _ = _mark_geometry(orbit)

    t_dense = jnp.linspace(2460000.0, 2460900.0, 4000)
    r_AU, _, _ = orbit.propagate(t_jd=t_dense, Ms_kg=STELLAR["Ms_kg"])
    xyz = np.moveaxis(np.asarray(r_AU), 1, 2)[0]
    nearest = np.argmin(np.linalg.norm(xyz - ascending[0], axis=-1))
    assert 0 < nearest < len(xyz) - 1
    assert xyz[nearest + 1, 2] > xyz[nearest - 1, 2]


def test_plot_orbit_marks_smoke_and_errors():
    """Marks draw on the orbit door and raise clearly everywhere else."""
    from orbix.viz import plot_orbit

    result = plot_orbit(
        _orbit(),
        T_JD,
        Ms_kg=STELLAR["Ms_kg"],
        marks={"periapsis", "nodes"},
    )
    assert len(result.artists["scatter"]) == 4
    assert len(result.artists["lines"]) == 2

    with pytest.raises(ValueError, match="elements"):
        plot_orbit(np.zeros((10, 3)), marks={"periapsis"})
    with pytest.raises(ValueError, match="unknown marks"):
        plot_orbit(_orbit(), T_JD, Ms_kg=STELLAR["Ms_kg"], marks={"apoapsis"})


def test_plot_sky_track_per_track_colors():
    """A per-track colors list overrides style and must match K."""
    from orbix.viz import plot_sky_track

    theta = np.linspace(0.0, 2.0 * np.pi, 30)
    tracks = np.stack(
        [
            np.stack([0.3 * np.cos(theta), 0.2 * np.sin(theta)]),
            np.stack([0.2 * np.cos(theta), 0.3 * np.sin(theta)]),
        ]
    )
    result = plot_sky_track(tracks, colors=["#aa3311", "#1133aa"])
    drawn = [line.get_color() for line in result.artists["lines"]]
    assert drawn == ["#aa3311", "#1133aa"]

    with pytest.raises(ValueError, match="colors has 1"):
        plot_sky_track(tracks, colors=["#aa3311"])


def test_plot_orbit_panes_take_the_axes_facecolor():
    """The 3D panes follow the axes facecolor instead of matplotlib gray."""
    from orbix.viz import plot_orbit

    result = plot_orbit(_orbit(), T_JD, Ms_kg=STELLAR["Ms_kg"])
    face = result.ax.get_facecolor()
    for pane_axis in (result.ax.xaxis, result.ax.yaxis, result.ax.zaxis):
        assert tuple(pane_axis.pane.get_facecolor()) == pytest.approx(face)


def test_plot_orbit_star_chart_look_under_markers():
    """depth="markers", no style: dashed gray path, text-color markers."""
    import matplotlib as mpl

    from orbix.viz import plot_orbit

    result = plot_orbit(_orbit(), T_JD, Ms_kg=STELLAR["Ms_kg"], depth="markers")
    line = result.artists["line"]
    assert line.get_linestyle() == "--"
    assert line.get_alpha() == pytest.approx(0.5)
    text_rgba = matplotlib.colors.to_rgba(mpl.rcParams["text.color"])
    scatter_rgba = tuple(result.artists["scatter"].get_facecolor()[0])
    assert scatter_rgba == pytest.approx(text_rgba)

    styled = plot_orbit(
        _orbit(), T_JD, Ms_kg=STELLAR["Ms_kg"], style="#22aabb", depth="markers"
    )
    assert styled.artists["line"].get_linestyle() == "-"
    assert styled.artists["line"].get_color() == "#22aabb"


def test_plot_orbit_unstyled_path_is_visible_without_markers():
    """The star-chart gray only works because markers carry the color.

    Under the default hidden-line cue there are no markers, so an unstyled
    orbit has to put the text color on the path itself; leaving it at the
    dim gray would render the orbit as a barely-there line.
    """
    import matplotlib as mpl

    from orbix.viz import plot_orbit

    result = plot_orbit(_orbit(), T_JD, Ms_kg=STELLAR["Ms_kg"])
    text_rgba = matplotlib.colors.to_rgba(mpl.rcParams["text.color"])
    line_rgba = matplotlib.colors.to_rgba(result.artists["line"].get_color())
    assert line_rgba == pytest.approx(text_rgba)
    # the per-track cue is now the solid near-half overlay, not a scatter
    near = result.artists["scatter"]
    assert near.get_linestyle() in ("-", "solid")
    assert matplotlib.colors.to_rgba(near.get_color()) == pytest.approx(text_rgba)


def test_size_by_radius_geometric_mapping():
    """Endpoints hit the ms range, midpoints interpolate geometrically."""
    from orbix.viz import size_by_radius

    lo, hi = 0.38, 11.2
    ms = size_by_radius([lo, hi, np.sqrt(lo * hi)])
    np.testing.assert_allclose(ms[0], 3.0, rtol=1e-12)
    np.testing.assert_allclose(ms[1], 9.0, rtol=1e-12)
    np.testing.assert_allclose(ms[2], np.sqrt(3.0 * 9.0), rtol=1e-12)

    clipped = size_by_radius([0.01, 100.0])
    np.testing.assert_allclose(clipped, [3.0, 9.0], rtol=1e-12)


def test_plot_orbit_per_track_marker_scale():
    """A marker_scale sequence sizes each track's beads independently."""
    from orbix.viz import plot_orbit

    result = plot_orbit(
        _orbit(2),
        T_JD,
        Ms_kg=STELLAR["Ms_kg"],
        marker_scale=[0.0, 20.0],
        depth="markers",
    )
    first, second = result.artists["scatter"]
    assert np.allclose(first.get_sizes(), 0.0)
    assert np.max(second.get_sizes()) > 0.0

    with pytest.raises(ValueError, match="marker_scale has 3"):
        plot_orbit(_orbit(2), T_JD, Ms_kg=STELLAR["Ms_kg"], marker_scale=[1, 2, 3])


def test_depth_helpers_are_public():
    """Two consumer scripts hand-rolled the head depth cue from privates."""
    import orbix.viz as viz

    assert viz.depth_size(0.0) == pytest.approx(1.0)
    assert viz.depth_size(1.0) == pytest.approx(np.sqrt(1.5))

    positions = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    scales = viz.depth_scale(positions, azim_deg=0.0, elev_deg=0.0)
    assert scales.shape == (2,)
    assert scales[0] > scales[1]
