"""Import mechanics for orbix.viz: lazy exports, clean base install."""

import subprocess
import sys

import pytest

BLOCK_EYEPIECE = "import sys; sys.modules['eyepiece'] = None; "


def _run(code):
    """Run a code snippet in a fresh interpreter and return the result."""
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )


def test_orbix_imports_without_eyepiece():
    """The base install imports clean with eyepiece blocked."""
    result = _run(BLOCK_EYEPIECE + "import orbix")
    assert result.returncode == 0, result.stderr


def test_viz_attribute_without_eyepiece_names_the_extra():
    """Touching a plot function without eyepiece names the viz extra."""
    result = _run(
        BLOCK_EYEPIECE
        + "import orbix.viz\n"
        + "try:\n"
        + "    orbix.viz.plot_sky_track\n"
        + "except ImportError as err:\n"
        + "    assert 'orbix[viz]' in str(err), str(err)\n"
        + "else:\n"
        + "    raise SystemExit('expected ImportError')\n"
    )
    assert result.returncode == 0, result.stderr


def test_importing_viz_package_imports_no_plotting_stack():
    """Import orbix.viz pulls neither eyepiece nor matplotlib."""
    result = _run(
        "import sys; import orbix.viz; "
        "assert 'eyepiece' not in sys.modules, 'eyepiece imported'; "
        "assert 'matplotlib' not in sys.modules, 'matplotlib imported'"
    )
    assert result.returncode == 0, result.stderr


def test_importing_orbix_does_not_import_viz():
    """The top-level package does not eagerly import the viz package."""
    result = _run("import sys; import orbix; assert 'orbix.viz' not in sys.modules")
    assert result.returncode == 0, result.stderr


def test_dir_lists_lazy_exports():
    """dir(orbix.viz) advertises the lazily provided functions."""
    import orbix.viz

    listed = dir(orbix.viz)
    for name in (
        "plot_sky_track",
        "plot_orbit",
        "animate_orbit",
        "prepare_sky_track",
        "prepare_sky_view",
    ):
        assert name in listed


# A meta-path finder whose find_spec refuses the named modules. This is the
# blocker that works on Python 3.12+: the legacy find_module hook is never
# consulted there, so a find_module-based blocker silently blocks nothing.
_BLOCKER = (
    "import sys\n"
    "class _Block:\n"
    "    def __init__(self, names):\n"
    "        self.names = names\n"
    "    def find_spec(self, name, path=None, target=None):\n"
    "        if any(name == n or name.startswith(n + '.') for n in self.names):\n"
    "            raise ModuleNotFoundError(f'blocked: {name}', name=name)\n"
    "        return None\n"
)


def _run_blocked(names, body):
    """Run ``body`` in a fresh interpreter with ``names`` unimportable."""
    code = _BLOCKER + f"sys.meta_path.insert(0, _Block({names!r}))\n" + body
    return _run(code)


def test_base_import_works_under_a_find_spec_blocker_for_eyepiece():
    """The base install imports, and names the extra, with eyepiece unfindable."""
    body = (
        "import orbix\n"
        "import orbix.viz\n"
        "assert 'eyepiece' not in sys.modules\n"
        "try:\n"
        "    orbix.viz.prepare_sky_track\n"
        "except ImportError as err:\n"
        "    assert 'orbix[viz]' in str(err), str(err)\n"
        "else:\n"
        "    raise SystemExit('expected ImportError')\n"
        "print('survived')\n"
    )
    result = _run_blocked(["eyepiece"], body)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "survived"


def test_preparation_runs_without_either_renderer_interface():
    """Preparing, seeking, and stripping never load a renderer or matplotlib."""
    body = (
        "import numpy as np\n"
        "from orbix.viz import prepare_sky_track, prepare_sky_view\n"
        "theta = np.linspace(0.0, 6.0, 7)\n"
        "xy = np.stack([np.cos(theta), np.sin(theta)], axis=-1)[None]\n"
        "obs = (np.array([[0.5, 0.2]]), np.array([0.01]), np.array([0.02]))\n"
        "seq = prepare_sky_track(xy, t_jd=np.arange(7.0), observations=obs,\n"
        "                        iwa_arcsec=0.3, history=3)\n"
        "seq.frame(6); seq.strip([0, 3]); seq.at(2.5); seq.schedule(1.0, 10)\n"
        "prepare_sky_view(xy, iwa_arcsec=0.3)\n"
        "for name in ('matplotlib', 'manim', 'eyepiece.mpl', 'eyepiece.manim',\n"
        "             'orbix.viz.orbit', 'orbix.viz.anim'):\n"
        "    assert name not in sys.modules, name\n"
        "print('prepared')\n"
    )
    blocked = ["matplotlib", "manim", "eyepiece.mpl", "eyepiece.manim"]
    result = _run_blocked(blocked, body)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "prepared"


def test_unknown_attribute_raises_attribute_error():
    """A name outside the lazy table raises AttributeError, not ImportError."""
    import orbix.viz

    with pytest.raises(AttributeError, match="no attribute"):
        orbix.viz.plot_nonexistent
