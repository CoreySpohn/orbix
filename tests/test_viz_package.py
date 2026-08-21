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
    for name in ("plot_sky_track", "plot_orbit", "animate_orbit"):
        assert name in listed


def test_unknown_attribute_raises_attribute_error():
    """A name outside the lazy table raises AttributeError, not ImportError."""
    import orbix.viz

    with pytest.raises(AttributeError, match="no attribute"):
        orbix.viz.plot_nonexistent
