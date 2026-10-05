from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from annizarr._storage._uri import bucket_prefix, is_remote, scheme
from annizarr.errors import RepoError, StorageError
from annizarr.typing import PathLike

if TYPE_CHECKING:
    from types import ModuleType

_logs_quieted = False


def require_icechunk() -> ModuleType:
    try:
        import icechunk
    except ImportError as exc:
        raise StorageError(
            "Icechunk support is not installed. Install it with: pip install 'annizarr[icechunk]'"
        ) from exc
    return icechunk


def storage_for(path: PathLike, *, anonymous: bool = False) -> Any:
    """Build an icechunk ``Storage`` for ``path``.

    Local paths use ``local_filesystem_storage``. ``s3://bucket/prefix`` and
    ``gs://bucket/prefix`` use the object-store backends with credentials from the
    environment (AWS_* variables / profile / instance role; gcloud application-default
    credentials), or unsigned requests when ``anonymous`` (public buckets).
    """
    icechunk = require_icechunk()
    _quiet_icechunk_logs()
    if is_remote(path):
        bucket, prefix = bucket_prefix(path)
        creds = {"anonymous": True} if anonymous else {"from_env": True}
        make = icechunk.s3_storage if scheme(path) == "s3" else icechunk.gcs_storage
        return make(bucket=bucket, prefix=prefix, **creds)
    return icechunk.local_filesystem_storage(str(path))


def open_repository(path: PathLike, *, anonymous: bool = False) -> Any:
    """``icechunk.Repository.open`` with the failure explained (no repo vs. no credentials)."""
    icechunk = require_icechunk()
    try:
        return icechunk.Repository.open(storage_for(path, anonymous=anonymous))
    except Exception as exc:
        raise explain_open_failure(path, exc) from exc


def explain_open_failure(path: PathLike, exc: Exception) -> RepoError:
    """Turn an icechunk open/exists failure into a ``RepoError`` that names the cause.

    Icechunk raises ``RepositoryNotFoundError`` for a missing repo and ``StorageError``
    when the object store cannot be reached, which on a remote path is almost always a
    credentials problem (``from_env`` found nothing, or the identity lacks access).
    """
    icechunk = require_icechunk()
    first_line = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
    if isinstance(exc, icechunk.RepositoryNotFoundError):
        return RepoError(f"No icechunk repository at '{path}'")
    if isinstance(exc, icechunk.StorageError) and is_remote(path):
        how = (
            "`gcloud auth application-default login`"
            if scheme(path) in ("gs", "gcs")
            else "`aws sso login` (or AWS_* variables / an instance role)"
        )
        return RepoError(
            f"Cannot access '{path}': {first_line}. Credentials come from the environment: "
            f"run {how}, or pass anonymous=True for a public bucket."
        )
    return RepoError(f"Cannot open '{path}': {first_line}")


def _quiet_icechunk_logs() -> None:
    # silences icechunk's benign Rust-core WARN spam, once, unless ICECHUNK_LOG is set
    global _logs_quieted
    if _logs_quieted:
        return
    _logs_quieted = True
    if os.environ.get("ICECHUNK_LOG"):
        return
    try:
        import icechunk

        icechunk.set_logs_filter("error")
    except Exception:
        pass


def is_icechunk_repo(path: PathLike) -> bool:
    p = Path(path)  # local repos carry repo/ + snapshots/, no zarr.json; remote always False here
    return p.is_dir() and not (p / "zarr.json").exists() and (p / "snapshots").is_dir() and (p / "repo").exists()
