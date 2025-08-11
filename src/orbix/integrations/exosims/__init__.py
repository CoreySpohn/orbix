"""EXOSIMS integrations."""

from ._availability import is_available

_EXOSIMS_IMPORT_ERROR_MSG = (
    "EXOSIMS package is required for this integration functionality.\n"
    "Install orbix with the exosims extra: pip install 'orbix[exosims]'"
)


# Define the generic error-raising function first
def _raise_exosims_import_error(*args, **kwargs):
    """Raises an ImportError indicating EXOSIMS is missing."""
    raise ImportError(_EXOSIMS_IMPORT_ERROR_MSG)


if is_available():
    # If EXOSIMS is available, import the actual implementations
    from .dMag0 import dMag0_grid
else:
    # EXOSIMS is not available, assign the error-raising function
    dMag0_grid = _raise_exosims_import_error

__all__ = ["dMag0_grid", "is_available"]
