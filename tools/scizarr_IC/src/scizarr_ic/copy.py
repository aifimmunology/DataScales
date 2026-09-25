"""Stream-copy a zarr hierarchy into another root group.

Used by ``Repo.init`` to seed an icechunk repo from an existing zarr store. Arrays are
re-created with the source layout (chunks, shards, codecs, fill value, attrs) and data
is copied in chunk-grid-aligned bands along axis 0 so memory stays bounded and no
read-modify-write is triggered on the destination.
"""
from __future__ import annotations

import math
from typing import Any

BAND_BYTES = 128 * 1024**2


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
