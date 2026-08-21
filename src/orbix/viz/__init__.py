"""Plotting for orbix types, built on eyepiece primitives.

Requires the ``viz`` extra (``pip install 'orbix[viz]'``), which brings
eyepiece and, through it, matplotlib and hwostyle. The base install stays
free of all three: names are re-exported lazily (PEP 562), so importing
this package imports no plotting stack, and the eyepiece requirement is
checked only when a plot function is first touched.
"""

import importlib

_LAZY = {
    "animate_orbit": "orbix.viz.anim",
    "plot_orbit": "orbix.viz.orbit",
    "plot_sky_track": "orbix.viz.orbit",
    "size_by_radius": "orbix.viz.orbit",
}

__all__ = sorted(_LAZY)


def __getattr__(name):
    """Resolve a lazy re-export, checking the eyepiece requirement first.

    Args:
        name: Attribute being looked up on ``orbix.viz``.

    Returns:
        The requested plot function.

    Raises:
        AttributeError: If ``name`` is not one of the lazy re-exports.
    """
    if name in _LAZY:
        from orbix.viz import _require

        _require.eyepiece()
        module = importlib.import_module(_LAZY[name])
        return getattr(module, name)
    raise AttributeError(f"module 'orbix.viz' has no attribute {name!r}")


def __dir__():
    """List the lazy re-exports alongside the module's real attributes.

    Returns:
        Sorted attribute names, including the lazily provided functions.
    """
    return sorted(set(globals()) | set(__all__))
