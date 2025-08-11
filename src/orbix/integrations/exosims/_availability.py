"""Handles the availability check for the optional EXOSIMS dependency."""

_EXOSIMS_AVAILABLE = False

try:
    import EXOSIMS

    _EXOSIMS_AVAILABLE = True
except ImportError:
    pass


def is_available() -> bool:
    """Check if the EXOSIMS integration dependencies are installed."""
    return _EXOSIMS_AVAILABLE
