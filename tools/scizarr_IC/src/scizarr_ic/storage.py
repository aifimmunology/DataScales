"""Map a repo path/URI to an icechunk Storage (local dir, s3://, gs://) and open it.

Also tells a read-only local path (a mounted data asset, a shared read-only volume)
from a writable one, so ``Repo`` can route writes elsewhere or fail early.
"""
from __future__ import annotations

import os
from urllib.parse import urlparse

from .errors import ScizarrError

_REMOTE_SCHEMES = {"s3", "gs", "gcs"}
_logs_quieted = False


def is_remote(path: str) -> bool:
    return urlparse(str(path)).scheme in _REMOTE_SCHEMES


def is_readonly_path(path: str) -> bool:
    """True for a LOCAL path this process cannot write to.

    ``os.access(W_OK)`` reports ``False`` on read-only mounts even for root. A path
    that doesn't exist yet is judged by its nearest existing ancestor — the directory
    it would be created in. Remote URIs are never read-only.
    """
    if is_remote(path):
        return False
    probe = os.path.abspath(str(path))
    while not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            return False
        probe = parent
    return not os.access(probe, os.W_OK)


def canonical_location(path: str) -> str:
    """Stable identity for a repo location: the URI as given, or the real local path."""
    return str(path) if is_remote(path) else os.path.realpath(str(path))


def storage_for(path: str, *, anonymous: bool = False):
    """Build an icechunk Storage for ``path``.

    Local paths use ``local_filesystem_storage``. ``s3://bucket/prefix`` and
    ``gs://bucket/prefix`` use the object-store backends with credentials from the
    environment (AWS_* variables / profile / instance role; gcloud application-default
    credentials), or unsigned requests when ``anonymous`` (public buckets).
    """
    import icechunk

    _quiet_icechunk_logs()
    parsed = urlparse(str(path))
    if parsed.scheme not in _REMOTE_SCHEMES:
        return icechunk.local_filesystem_storage(str(path))
    if not parsed.netloc:
        raise ScizarrError(f"Missing bucket in '{path}'")
    creds = {"anonymous": True} if anonymous else {"from_env": True}
    make = icechunk.s3_storage if parsed.scheme == "s3" else icechunk.gcs_storage
    return make(bucket=parsed.netloc, prefix=parsed.path.lstrip("/") or None, **creds)


def open_repository(path: str, *, anonymous: bool = False):
    """``icechunk.Repository.open`` with the failure explained (no repo vs. no credentials)."""
    import icechunk

    try:
        return icechunk.Repository.open(storage_for(path, anonymous=anonymous))
    except Exception as exc:
        raise explain_open_failure(path, exc) from exc


def explain_open_failure(path: str, exc: Exception) -> ScizarrError:
    """Turn an icechunk open/exists failure into a ``ScizarrError`` that names the cause.

    Icechunk raises ``RepositoryNotFoundError`` for a missing repo and ``StorageError``
    when the object store cannot be reached, which on a remote path is almost always a
    credentials problem (``from_env`` found nothing, or the identity lacks access).
    """
    import icechunk

    first_line = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
    if isinstance(exc, icechunk.RepositoryNotFoundError):
        return ScizarrError(f"No icechunk repository at '{path}'")
    if isinstance(exc, icechunk.StorageError) and is_remote(path):
        scheme = urlparse(path).scheme
        how = (
            "`gcloud auth application-default login`"
            if scheme in ("gs", "gcs")
            else "`aws sso login` (or AWS_* variables / an instance role)"
        )
        return ScizarrError(
            f"Cannot access '{path}': {first_line}. Credentials come from the environment: "
            f"run {how}, or pass anonymous=True for a public bucket."
        )
    return ScizarrError(f"Cannot open '{path}': {first_line}")


def _quiet_icechunk_logs() -> None:
    """Silence icechunk's benign Rust-core WARN spam, once, unless ICECHUNK_LOG is set."""
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
