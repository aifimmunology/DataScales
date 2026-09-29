from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from annizarr._storage._uri import bucket_prefix, is_remote, scheme
from annizarr.errors import StorageError
from annizarr.typing import PathLike

if TYPE_CHECKING:
    from types import ModuleType


def require_icechunk() -> ModuleType:
    # every icechunk code path calls this first, so a missing optional dep fails with an
    # install hint instead of a bare ImportError (which can't share a base with AnzError)
    try:
        import icechunk
    except ImportError as exc:
        raise StorageError(
            "Icechunk support is not installed. Install it with: pip install 'annizarr[icechunk]'"
        ) from exc
    return icechunk


def storage_for(path: PathLike) -> Any:
    # the one icechunk.Storage constructor used everywhere; from_env=True resolves
    # credentials from AWS_*/GOOGLE_* variables, a profile, or an instance/container role
    icechunk = require_icechunk()
    if is_remote(path):
        bucket, prefix = bucket_prefix(path)
        if scheme(path) == "s3":
            return icechunk.s3_storage(bucket=bucket, prefix=prefix, from_env=True)
        return icechunk.gcs_storage(bucket=bucket, prefix=prefix, from_env=True)
    return icechunk.local_filesystem_storage(str(path))


def is_icechunk_repo(path: PathLike) -> bool:
    # local icechunk repos carry repo/ + snapshots/ and no zarr.json at the root;
    # remote URIs always return False here — callers combine this with is_remote()
    p = Path(path)
    return p.is_dir() and not (p / "zarr.json").exists() and (p / "snapshots").is_dir() and (p / "repo").exists()
