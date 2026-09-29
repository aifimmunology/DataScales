from __future__ import annotations

__all__ = ["AnzError", "ConversionError", "RepoError", "StorageError", "ValidationError"]


class AnzError(RuntimeError):
    """Base class for every error annizarr raises on purpose; the CLI prints these without a traceback."""


class ConversionError(AnzError):
    """A conversion or store-editing operation could not complete."""


class StorageError(AnzError):
    """A store or repository location could not be opened, created, or finalised."""


class ValidationError(AnzError, ValueError):
    """An input or configuration value does not satisfy annizarr's constraints."""


class RepoError(AnzError):
    """An Icechunk repository operation failed: missing branch, unknown snapshot, no writable origin."""
