from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from annizarr._storage._uri import bucket_prefix, is_remote, scheme
from annizarr.errors import StorageError
from annizarr.typing import PathLike

if TYPE_CHECKING:
    from types import ModuleType


def require_icechunk() -> ModuleType:
    try:
        import icechunk
    except ImportError as exc:
        raise StorageError(
            "Icechunk support is not installed. Install it with: pip install 'annizarr[icechunk]'"
        ) from exc
    return icechunk


def storage_for(path: PathLike) -> Any:
    icechunk = require_icechunk()  # from_env=True: AWS_*/GOOGLE_* vars, a profile, or instance role
    if is_remote(path):
        bucket, prefix = bucket_prefix(path)
        if scheme(path) == "s3":
            return icechunk.s3_storage(bucket=bucket, prefix=prefix, from_env=True)
        return icechunk.gcs_storage(bucket=bucket, prefix=prefix, from_env=True)
    return icechunk.local_filesystem_storage(str(path))


def is_icechunk_repo(path: PathLike) -> bool:
    p = Path(path)  # local repos carry repo/ + snapshots/, no zarr.json; remote always False here
    return p.is_dir() and not (p / "zarr.json").exists() and (p / "snapshots").is_dir() and (p / "repo").exists()
