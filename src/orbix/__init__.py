"""orbix: differentiable Keplerian orbit propagation in JAX."""

from orbix._version import __version__

__all__ = ["AbstractOrbit", "KeplerianOrbit", "__version__"]


def __getattr__(name):
    if name in __all__[:2]:
        from importlib import import_module

        value = getattr(import_module("orbix.orbit"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
