"""Import guard for the optional eyepiece dependency."""


def eyepiece():
    """Import eyepiece, or raise with the install hint.

    Returns:
        The imported ``eyepiece`` module.

    Raises:
        ImportError: If eyepiece is not installed; the message names the
            ``orbix[viz]`` extra that provides it.
    """
    try:
        import eyepiece
    except ImportError:
        raise ImportError(
            "orbix.viz requires eyepiece: pip install 'orbix[viz]'"
        ) from None
    return eyepiece
