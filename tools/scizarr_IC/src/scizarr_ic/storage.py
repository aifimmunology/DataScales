"""Map a repo path/URI to an icechunk Storage (local dir, s3://, gs://).

Also knows how to tell a READ-ONLY local mount (e.g. a Code Ocean data asset under
``/data``) from a writable path, whether such a mount is a *frozen copy* rather than a
live view of its source, and how to read the store's stamped *origin* — the
``s3://``/``gs://`` location writes go to when the local path can't be written.
"""
from __future__ import annotations

import os
from urllib.parse import urlparse

from .errors import ScizarrError

_REMOTE_SCHEMES = {"s3", "gs", "gcs"}

# Repo-metadata key holding the store's writable location. Stamped by ``Repo.init``
# (and by DataScales' ingest tooling); read back through any read-only mount of the
# same repo so consumers never need the bucket/prefix as an input.
ORIGIN_KEY = "origin_url"

# Env overrides. READONLY_PREFIXES forces paths under these prefixes (comma-separated)
# to count as read-only even when ``os.access`` says otherwise (deployments where the
# check lies; hermetic tests running as root). ORIGIN overrides the stamped origin.
ENV_READONLY_PREFIXES = "SCIZARR_IC_READONLY_PREFIXES"
ENV_ORIGIN = "SCIZARR_IC_ORIGIN"

# Filesystem types of read-only mounts that are a one-time COPY of their source, so
# writing to the stamped origin would never show up in the mount. Code Ocean's
# internal data assets are EFS (NFS) copies; its linked external S3 assets are FUSE
# views that do track the bucket.
FROZEN_FSTYPES = ("nfs",)


def is_remote(path: str) -> bool:
    return urlparse(str(path)).scheme in _REMOTE_SCHEMES


def _readonly_prefixes() -> tuple[str, ...]:
    raw = os.environ.get(ENV_READONLY_PREFIXES, "")
    return tuple(os.path.abspath(p) for p in raw.split(",") if p.strip())


def is_readonly_path(path: str) -> bool:
    """True for a LOCAL path this process cannot write to.

    Uses ``os.access(W_OK)`` (which reports ``False`` on read-only mounts even for
    root) plus the ``SCIZARR_IC_READONLY_PREFIXES`` override. A path that doesn't
    exist yet is judged by its nearest existing ancestor — the directory it would be
    created in. Remote URIs are never read-only.
    """
    if is_remote(path):
        return False
    p = os.path.abspath(str(path))
    if any(p == pre or p.startswith(pre + os.sep) for pre in _readonly_prefixes()):
        return True
    probe = p
    while not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            return False
        probe = parent
    return not os.access(probe, os.W_OK)


def mount_fstype(path: str) -> str | None:
    """Filesystem type of the mount holding ``path`` (from ``/proc/mounts``), or None."""
    p = os.path.realpath(str(path))
    try:
        with open("/proc/mounts") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return None
    best: tuple[str, str] | None = None
    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        mp = parts[1].replace("\\040", " ")
        if p == mp or p.startswith(mp.rstrip("/") + "/"):
            if best is None or len(mp) > len(best[0]):
                best = (mp, parts[2])
    return best[1] if best else None


def is_frozen_mount(path: str) -> bool:
    """True for a read-only local mount that is a copy of its source, not a view of it.

    Such a mount's stamped origin is disconnected: writes there never reach the
    mount, so scizarr-ic ignores the stamp and offers ``copy`` instead.
    """
    if not is_readonly_path(path):
        return False
    fs = mount_fstype(path)
    return fs is not None and fs.startswith(FROZEN_FSTYPES)


def origin_of(ic_repo) -> str | None:
    """The store's stamped writable location (``origin_url`` metadata), if any."""
    try:
        value = ic_repo.get_metadata().get(ORIGIN_KEY)
    except Exception:
        return None
    return str(value) if value else None


def canonical_location(path: str) -> str:
    """Stable identity for a repo location: the URI as given, or the real local path."""
    return str(path) if is_remote(path) else os.path.realpath(str(path))


def storage_for(path: str):
    """Build an icechunk Storage for ``path``.

    Local paths use ``local_filesystem_storage``; ``s3://bucket/prefix`` and
    ``gs://bucket/prefix`` use the object-store backends with credentials
    resolved from the environment (``from_env=True`` — AWS_* variables, profiles,
    or a container/instance role such as a Code Ocean AWS secret).
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
