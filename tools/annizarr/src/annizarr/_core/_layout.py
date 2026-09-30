from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import zarr

# read as `_layout.BATCH_BYTES` (module attribute, not a `from`-import) so tests can monkeypatch it.
BATCH_BYTES = 256 * 1024 * 1024

__all__ = ["BATCH_BYTES", "DenseLayout", "band_plan", "dense_shards", "write_grid", "x_compressors"]


def write_grid(arr: zarr.Array[Any]) -> tuple[int, ...]:
    # a partial-shard write read-modify-writes the whole shard, so writers must partition here.
    return arr.shards if arr.shards is not None else arr.chunks


def x_compressors() -> tuple[object, ...]:
    from zarr.codecs import BloscCodec

    return (BloscCodec(cname="zstd", clevel=5, shuffle="shuffle"),)  # zarr's default has no shuffle


@dataclass(frozen=True, slots=True)
class DenseLayout:
    chunks: tuple[int, int]
    shards: tuple[int, int] | None
    block: tuple[int, int]


def dense_shards(row_chunk: int, col_chunk: int, n_rows: int, n_cols: int, factor: int) -> DenseLayout:
    if factor <= 1:
        return DenseLayout(chunks=(row_chunk, col_chunk), shards=None, block=(row_chunk, col_chunk))
    import math

    # zarr requires shard shape to be an integer multiple of the chunk shape
    rf = min(factor, math.ceil(n_rows / row_chunk))
    cf = min(factor, math.ceil(n_cols / col_chunk))
    shard_row = row_chunk * rf
    shard_col = col_chunk * cf
    return DenseLayout(chunks=(row_chunk, col_chunk), shards=(shard_row, shard_col), block=(shard_row, shard_col))


def band_plan(n_rows: int, band_rows: int) -> tuple[tuple[int, int], ...]:
    if n_rows <= 0 or band_rows <= 0:
        return ()
    return tuple((r0, min(r0 + band_rows, n_rows)) for r0 in range(0, n_rows, band_rows))
