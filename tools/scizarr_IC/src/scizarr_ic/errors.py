"""Exception types for scizarr-ic."""


class ScizarrError(RuntimeError):
    """Raised for any user-facing failure (bad path, missing branch, unknown snapshot...)."""
