from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from ._runtime import pin_blas
from ._version import __version__
from .errors import AnzError, ConversionError, RepoError, StorageError, ValidationError

pin_blas()  # OpenBLAS reads the thread env vars at import, so this runs before anything loads numpy

if TYPE_CHECKING:
    from ._config import AppConfig, load_config
    from ._ic import Repo
    from ._ops import AppendPlan, OpResult, add_expr, append, convert, plan_append, rechunk, sort

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
    "convert",
    "load_config",
    "plan_append",
    "rechunk",
    "sort",
]

_LAZY: dict[str, str] = {
    "AppConfig": "._config",
    "load_config": "._config",
    "Repo": "._ic",
    "AppendPlan": "._ops",
    "OpResult": "._ops",
    "add_expr": "._ops",
    "append": "._ops",
    "convert": "._ops",
    "plan_append": "._ops",
    "rechunk": "._ops",
    "sort": "._ops",
}


def __getattr__(name: str) -> object:
    # lazy so `import annizarr` and `annizarr --version` never load anndata, scipy or zarr
    try:
        module = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'annizarr' has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
