"""Copy routines.

* :func:`copy_group` — stream a zarr hierarchy into another root group. Used by
  ``Repo.init`` to seed a repo from a zarr store: arrays are re-created with the source
  layout (chunks, shards, codecs, fill value, attrs) and data is copied in
  chunk-grid-aligned bands along axis 0 so memory stays bounded.
* :func:`copy_repo` — replicate a whole icechunk repo (every branch and snapshot) to a
  new location byte for byte. Icechunk's object tree is content-addressed, so a plain
  file copy is a faithful clone. Used by ``Repo.copy`` to take a writable copy of a
  read-only data asset.
"""
from __future__ import annotations

import math
import shutil
import subprocess
from typing import Any
from urllib.parse import urlparse

from .errors import ScizarrError
from .head import HEAD_FILE
from .storage import is_remote

BAND_BYTES = 128 * 1024**2


def check_copyable(src: str, dst: str) -> None:
    """Raise unless ``src`` -> ``dst`` is a copy this module can perform (no I/O)."""
    for loc in (src, dst):
        if is_remote(loc) and urlparse(loc).scheme != "s3":
            raise ScizarrError(f"copy supports local paths and s3:// only, got '{loc}'")
    if (is_remote(src) or is_remote(dst)) and shutil.which("aws") is None:
        raise ScizarrError("Copying to/from s3:// needs the AWS CLI ('aws') on PATH")


def copy_repo(src: str, dst: str) -> None:
    """Copy the repo object tree at ``src`` to ``dst`` (local dirs and/or ``s3://``).

    Local→local uses ``shutil``; anything involving ``s3://`` shells out to
    ``aws s3 sync`` (credentials from the environment). The local HEAD file is never
    copied. Callers run :func:`check_copyable` and make sure ``dst`` is a fresh location.
    """
    if not (is_remote(src) or is_remote(dst)):
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(HEAD_FILE), dirs_exist_ok=True)
        return
    cmd = ["aws", "s3", "sync", "--only-show-errors", "--exclude", HEAD_FILE,
           src.rstrip("/"), dst.rstrip("/")]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ScizarrError(f"aws s3 sync failed ({proc.returncode}): {proc.stderr.strip()}")


def copy_group(src: Any, dst: Any, *, band_bytes: int = BAND_BYTES) -> int:
    """Replicate ``src`` (groups, arrays, attrs, layout) into ``dst``; return array count."""
    import zarr

    dst.update_attributes(dict(src.attrs))
    n_arrays = 0
    # sorted → every group path precedes its children
    for path, node in sorted(src.members(max_depth=None), key=lambda kv: kv[0]):
        if isinstance(node, zarr.Group):
            dst.create_group(path, attributes=dict(node.attrs))
        else:
            _copy_array(node, dst, path, band_bytes)
            n_arrays += 1
    return n_arrays


def _copy_array(src: Any, dst_root: Any, path: str, band_bytes: int) -> None:
    kwargs: dict[str, Any] = {
        "shape": src.shape,
        "dtype": src.dtype,
        "chunks": src.chunks,
        "shards": src.shards,
        "fill_value": src.fill_value,
        "attributes": dict(src.attrs),
    }
    if getattr(src.metadata, "zarr_format", 3) == 3:
        kwargs.update(
            filters=src.filters,
            compressors=src.compressors,
            serializer=src.serializer or "auto",
            dimension_names=getattr(src.metadata, "dimension_names", None),
        )
    dst = dst_root.create_array(path, **kwargs)

    if src.size == 0:
        return
    if src.ndim == 0:
        dst[...] = src[...]
        return
    row_bytes = max(1, src.dtype.itemsize) * max(1, math.prod(src.shape[1:]))
    step = src.chunks[0] * max(1, band_bytes // (row_bytes * src.chunks[0]))
    for start in range(0, src.shape[0], step):
        band = slice(start, min(start + step, src.shape[0]))
        dst[band] = src[band]
