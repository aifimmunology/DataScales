"""Map a repo path/URI to an icechunk Storage (local dir, s3://, gs://).

Also tells a read-only local path (a mounted data asset, a shared read-only volume)
from a writable one, so ``Repo`` can route writes elsewhere or fail early.
"""
from __future__ import annotations

import os
from urllib.parse import urlparse

from .errors import ScizarrError

_REMOTE_SCHEMES = {"s3", "gs", "gcs"}

# Env override for the writable location when the repo path itself can't be written.
ENV_ORIGIN = "SCIZARR_IC_ORIGIN"


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


def storage_for(path: str):
    """Build an icechunk Storage for ``path``.

    Local paths use ``local_filesystem_storage``; ``s3://bucket/prefix`` and
    ``gs://bucket/prefix`` use the object-store backends with credentials resolved
    from the environment (``from_env=True`` — AWS_* variables, profiles, or an
    instance/container role).
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
