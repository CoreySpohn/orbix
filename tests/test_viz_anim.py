"""Contract tests for orbix.viz.animate_orbit.

Renders go through ``.jshtml()`` only, keeping ffmpeg off the test path.
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


T_JD = jnp.linspace(2460000.0, 2460400.0, 12)


def test_animate_orbit_returns_lazy_animation_with_frame_count():
    """The result is an eyepiece.Animation with one frame per epoch."""
    import eyepiece as ep

    from orbix.viz import animate_orbit

    anim = animate_orbit(_orbit(), T_JD, Ms_kg=MSUN_KG, dist_pc=10.0)
    assert isinstance(anim, ep.Animation)
    assert anim.n_frames == len(T_JD)
    assert anim.fps == 10


def test_animate_orbit_update_mode_creates_no_artists_per_frame():
    """Walking frames mutates artists; the line count never grows."""
    from orbix.viz import animate_orbit

    anim = animate_orbit(_orbit(), T_JD, Ms_kg=MSUN_KG, dist_pc=10.0)
    ax = anim.fig.axes[0]
    n_before = len(ax.lines)
    for i in range(3):
        anim.draw(anim.fig, i)
    assert len(ax.lines) == n_before


def test_animate_orbit_history_modes():
    """All grows the trail, an int caps it, none draws only the head."""
    from orbix.viz import animate_orbit

    orbit, kwargs = _orbit(), dict(Ms_kg=MSUN_KG, dist_pc=10.0)

    grown = animate_orbit(orbit, T_JD, **kwargs, history="all")
    grown.draw(grown.fig, 7)
    ghost_and_trail = [
        line for line in grown.fig.axes[0].lines if len(line.get_xdata()) == 8
    ]
    assert ghost_and_trail, "history='all' trail should hold frames 0..7"

    windowed = animate_orbit(orbit, T_JD, **kwargs, history=3)
    windowed.draw(windowed.fig, 7)
    assert any(len(line.get_xdata()) == 3 for line in windowed.fig.axes[0].lines), (
        "history=3 trail should hold exactly 3 frames"
    )

    headless = animate_orbit(orbit, T_JD, **kwargs, history="none")
    headless.draw(headless.fig, 7)
    n_lines_none = len(headless.fig.axes[0].lines)
    with_trail = len(grown.fig.axes[0].lines)
    assert n_lines_none == with_trail - 1


def test_animate_orbit_3d_with_marks_and_batch():
    """kind='3d' animates a K-batch on a 3D axes with marks intact."""
    from orbix.viz import animate_orbit

    anim = animate_orbit(
        _orbit(3),
        T_JD,
        Ms_kg=MSUN_KG,
        kind="3d",
        marks={"periapsis"},
        weights=[1.0, 0.4, 0.1],
    )
    ax = anim.fig.axes[0]
    assert ax.name == "3d"
    anim.draw(anim.fig, 5)
    assert anim.n_frames == len(T_JD)


def test_animate_orbit_time_label_advances():
    """The elapsed-time label tracks the epoch being drawn."""
    from orbix.viz import animate_orbit

    anim = animate_orbit(_orbit(), T_JD, Ms_kg=MSUN_KG, dist_pc=10.0)
    ax = anim.fig.axes[0]
    anim.draw(anim.fig, 0)
    label = [t for t in ax.texts if t.get_text().startswith("t = ")][0]
    assert label.get_text() == "t = +0 d"
    anim.draw(anim.fig, len(T_JD) - 1)
    assert label.get_text() == "t = +400 d"


def test_animate_orbit_bare_tracks_door():
    """Bare tracks animate; t_jd labels them when lengths agree."""
    from orbix.viz import animate_orbit

    theta = np.linspace(0.0, 2.0 * np.pi, 15)
    ra, dec = 0.3 * np.cos(theta), 0.2 * np.sin(theta)

    anim = animate_orbit((ra, dec))
    assert anim.n_frames == 15

    anim = animate_orbit((ra, dec), np.linspace(0.0, 140.0, 15))
    anim.draw(anim.fig, 1)

    with pytest.raises(ValueError, match="epoch"):
        animate_orbit((ra, dec), np.linspace(0.0, 1.0, 4))


def test_animate_orbit_jshtml_renders():
    """A short animation renders to embedded HTML without ffmpeg."""
    from orbix.viz import animate_orbit

    t_short = jnp.linspace(2460000.0, 2460100.0, 4)
    anim = animate_orbit(_orbit(), t_short, Ms_kg=MSUN_KG, dist_pc=10.0)
    html = anim.jshtml(dpi=50)
    assert "animation" in html.lower()
    assert html.count("data:image/png") >= 1


def test_animate_orbit_argument_errors():
    """Kind and history outside the vocabulary raise ValueError."""
    from orbix.viz import animate_orbit

    with pytest.raises(ValueError, match="kind"):
        animate_orbit(_orbit(), T_JD, Ms_kg=MSUN_KG, dist_pc=10.0, kind="2d")
    with pytest.raises(ValueError, match="history"):
        animate_orbit(_orbit(), T_JD, Ms_kg=MSUN_KG, dist_pc=10.0, history="window")


def test_animate_orbit_3d_head_carries_depth_cue():
    """The 3D head marker shrinks on the far side; the ghost has no beads."""
    from orbix.viz import animate_orbit

    anim = animate_orbit(_orbit(), T_JD, Ms_kg=MSUN_KG, kind="3d")
    ax = anim.fig.axes[0]

    sizes = []
    for i in range(len(T_JD)):
        anim.draw(anim.fig, i)
        head = ax.lines[-1]
        sizes.append(head.get_markersize())
    # the swing follows the hand-tuned original: a near-side swell of at
    # most sqrt(1.5) in diameter, anchored at the base size on the far
    # side, so marker size stays free to encode physical meaning
    assert max(sizes) > min(sizes) * 1.02
    assert max(sizes) <= min(sizes) * np.sqrt(1.5) + 1e-9

    beads = [c for c in ax.collections if len(c.get_offsets()) > 1]
    for bead in beads:
        assert np.allclose(bead.get_sizes(), 0.0)


def test_animate_orbit_rotate_sweeps_the_camera():
    """A rotate sweep drives view angles across frames, endpoints exact."""
    from orbix.viz import animate_orbit

    anim = animate_orbit(
        _orbit(),
        T_JD,
        Ms_kg=MSUN_KG,
        kind="3d",
        rotate={"azim": (-80.0, -30.0), "elev": (15.0, 35.0)},
    )
    ax = anim.fig.axes[0]
    anim.draw(anim.fig, 0)
    assert ax.azim == pytest.approx(-80.0)
    assert ax.elev == pytest.approx(15.0)
    anim.draw(anim.fig, len(T_JD) - 1)
    assert ax.azim == pytest.approx(-30.0)
    assert ax.elev == pytest.approx(35.0)


def test_animate_orbit_rotate_errors():
    """Rotate is 3d-only and rejects unknown angle names."""
    from orbix.viz import animate_orbit

    with pytest.raises(ValueError, match="3d"):
        animate_orbit(
            _orbit(),
            T_JD,
            Ms_kg=MSUN_KG,
            dist_pc=10.0,
            rotate={"azim": (0.0, 30.0)},
        )
    with pytest.raises(ValueError, match="unknown rotate"):
        animate_orbit(_orbit(), T_JD, Ms_kg=MSUN_KG, kind="3d", rotate={"spin": (0, 1)})


def test_animate_orbit_3d_default_look():
    """3D default: dashed gray trails, text-color heads with edge."""
    import matplotlib as mpl

    from orbix.viz import animate_orbit

    anim = animate_orbit(_orbit(3), T_JD, Ms_kg=MSUN_KG, kind="3d")
    ax = anim.fig.axes[0]
    anim.draw(anim.fig, 4)

    text_rgba = matplotlib.colors.to_rgba(mpl.rcParams["text.color"])
    heads = [
        line
        for line in ax.lines
        if line.get_marker() == "o" and line.get_linestyle() == "None"
    ]
    assert len(heads) == 3
    for head in heads:
        assert matplotlib.colors.to_rgba(head.get_color()) == pytest.approx(text_rgba)

    trails = [line for line in ax.lines if line.get_linestyle() == "--"]
    assert len(trails) >= 4  # 3 ghosts + at least the drawn trails


def test_animate_orbit_default_rotation_is_single_axis():
    """The default sweep turns azimuth alone; rotate=None holds still."""
    from orbix.viz import animate_orbit

    anim = animate_orbit(_orbit(), T_JD, Ms_kg=MSUN_KG, kind="3d")
    ax = anim.fig.axes[0]
    anim.draw(anim.fig, 0)
    azim0, elev0 = ax.azim, ax.elev
    anim.draw(anim.fig, len(T_JD) - 1)
    assert ax.azim == pytest.approx(azim0 + 40.0)
    assert ax.elev == pytest.approx(elev0)

    still = animate_orbit(_orbit(), T_JD, Ms_kg=MSUN_KG, kind="3d", rotate=None)
    ax = still.fig.axes[0]
    azim_before = ax.azim
    still.draw(still.fig, len(T_JD) - 1)
    assert ax.azim == pytest.approx(azim_before)

    with pytest.raises(ValueError, match="rotate must be"):
        animate_orbit(_orbit(), T_JD, Ms_kg=MSUN_KG, kind="3d", rotate="spin")
