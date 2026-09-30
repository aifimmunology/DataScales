from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

from annizarr import config, errors, ic, sources, typing
from annizarr._core._runtime import pin_blas
from annizarr._version import __version__
from annizarr.errors import AnzError, ConversionError, RepoError, StorageError, ValidationError

pin_blas()  # OpenBLAS reads the thread env vars at import, so this runs before anything loads numpy

logging.getLogger(__name__).addHandler(logging.NullHandler())  # quiet by default; the CLI adds its own handler

if TYPE_CHECKING:
    from annizarr._core._config import AppConfig, load_config
    from annizarr.ic import Repo
    from annizarr.ops import AppendPlan, OpResult, add_expr, append, convert, plan_append, rechunk, sort

__all__ = [
    "AnzError",
    "AppConfig",
    "AppendPlan",
    "ConversionError",
    "OpResult",
    "Repo",
    "RepoError",
    "StorageError",
    "ValidationError",
    "__version__",
    "add_expr",
    "append",
    "config",
    "convert",
    "errors",
    "ic",
    "load_config",
    "ops",
    "plan_append",
    "rechunk",
    "sort",
    "sources",
    "typing",
]

_LAZY: dict[str, str] = {
    "AppConfig": "annizarr._core._config",
    "load_config": "annizarr._core._config",
    "Repo": "annizarr.ic",
    "AppendPlan": "annizarr.ops",
    "OpResult": "annizarr.ops",
    "add_expr": "annizarr.ops",
    "append": "annizarr.ops",
    "convert": "annizarr.ops",
    "plan_append": "annizarr.ops",
    "rechunk": "annizarr.ops",
    "sort": "annizarr.ops",
}


def __getattr__(name: str) -> object:
    # lazy so `import annizarr` and `annizarr --version` never load anndata, scipy or zarr
    if name == "ops":  # az.ops itself, not a name lazily re-exported from it (see _LAZY below)
        value: object = importlib.import_module("annizarr.ops")
        globals()[name] = value
        return value
    try:
        module = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'annizarr' has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
