"""Handles the availability check for the optional EXOSIMS dependency."""

from importlib.util import find_spec

_EXOSIMS_AVAILABLE = find_spec("EXOSIMS") is not None


def is_available() -> bool:
    """Check if the EXOSIMS integration dependencies are installed."""
    return _EXOSIMS_AVAILABLE
