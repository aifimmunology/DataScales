"""Map a repo path/URI to an icechunk Storage (local dir, s3://, gs://)."""
from __future__ import annotations

from urllib.parse import urlparse

from .errors import ScizarrError

_REMOTE_SCHEMES = {"s3", "gs", "gcs"}


def is_remote(path: str) -> bool:
    return urlparse(str(path)).scheme in _REMOTE_SCHEMES


def storage_for(path: str):
    """Build an icechunk Storage for ``path``.

    Local paths use ``local_filesystem_storage``; ``s3://bucket/prefix`` and
    ``gs://bucket/prefix`` use the object-store backends with credentials
    resolved from the environment (``from_env=True``).
    """
    import icechunk

    parsed = urlparse(str(path))
    if parsed.scheme in _REMOTE_SCHEMES:
        if not parsed.netloc:
            raise ScizarrError(f"Missing bucket in '{path}'")
        prefix = parsed.path.lstrip("/") or None
        if parsed.scheme == "s3":
            return icechunk.s3_storage(bucket=parsed.netloc, prefix=prefix, from_env=True)
        return icechunk.gcs_storage(bucket=parsed.netloc, prefix=prefix, from_env=True)
    return icechunk.local_filesystem_storage(str(path))
