from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

from annizarr import config, errors, ic, sources, typing
from annizarr._runtime import pin_blas
from annizarr._version import __version__
from annizarr.errors import AnzError, ConversionError, RepoError, StorageError, ValidationError

pin_blas()  # OpenBLAS reads the thread env vars at import, so this runs before anything loads numpy

logging.getLogger(__name__).addHandler(logging.NullHandler())  # quiet by default; the CLI adds its own handler

if TYPE_CHECKING:
    from annizarr._config import AppConfig, load_config
    from annizarr._ic import Repo
    from annizarr._ops import AppendPlan, OpResult, add_expr, append, convert, plan_append, rechunk, sort

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
    "plan_append",
    "rechunk",
    "sort",
    "sources",
    "typing",
]

_LAZY: dict[str, str] = {
    "AppConfig": "annizarr._config",
    "load_config": "annizarr._config",
    "Repo": "annizarr._ic",
    "AppendPlan": "annizarr._ops",
    "OpResult": "annizarr._ops",
    "add_expr": "annizarr._ops",
    "append": "annizarr._ops",
    "convert": "annizarr._ops",
    "plan_append": "annizarr._ops",
    "rechunk": "annizarr._ops",
    "sort": "annizarr._ops",
}


def __getattr__(name: str) -> object:
    # lazy so `import annizarr` and `annizarr --version` never load anndata, scipy or zarr
    try:
        module = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'annizarr' has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
