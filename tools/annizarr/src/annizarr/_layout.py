from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import zarr

# Target bytes per streamed write/copy batch. Shared by every band/segment loop across
# _ops/_append.py, _ops/_rechunk.py, _ops/_expr.py, _writers/*, and _sorting.py — read as
# `_layout.BATCH_BYTES` (module attribute, not a `from`-import) so tests can monkeypatch it.
BATCH_BYTES = 256 * 1024 * 1024

__all__ = ["BATCH_BYTES", "DenseLayout", "band_plan", "dense_shards", "write_grid", "x_compressors"]


def write_grid(arr: zarr.Array[Any]) -> tuple[int, ...]:
    """Return the array's write-partition grid: its shard shape if sharded, else its chunks.

    Every concurrent or process-pool writer must partition its write blocks/segments along
    this grid, taken from the array **as actually created** (not recomputed from config) —
    writing a partial shard makes zarr's sharding codec read-modify-write the whole shard, so
    two tasks writing into the same shard would race. Safe for any array, sharded or not,
    however it came to be sharded (an explicit ``x_shard_factor``, or ``shards="auto"``).

    Parameters
    ----------
    arr
        The zarr array to partition writes against.

    Returns
    -------
    tuple[int, ...]
        ``arr.shards`` when sharded, else ``arr.chunks``.
    """
    return arr.shards if arr.shards is not None else arr.chunks


def x_compressors() -> tuple[object, ...]:
    """Return the codec tuple used for every X (and layer) array.

    Returns
    -------
    tuple[object, ...]
        A single ``BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")``. zarr's own
        default is bare zstd level-0 with no shuffle; the byte-shuffle gives a large
        ratio/throughput win on numeric matrices.
    """
    from zarr.codecs import BloscCodec

    return (BloscCodec(cname="zstd", clevel=5, shuffle="shuffle"),)


@dataclass(frozen=True, slots=True)
class DenseLayout:
    """Resolved chunk/shard/write-block shape for a dense X (or layer) array.

    Parameters
    ----------
    chunks
        The array's inner chunk shape — the read granularity, unaffected by sharding.
    shards
        The ``shards=`` kwarg for ``zarr.Group.require_array``; ``None`` when unsharded.
    block
        The shape callers must write at. Equals ``shards`` when sharding is on (writing a
        partial shard makes zarr's sharding codec read-modify-write the whole shard), else
        equals ``chunks``.
    """

    chunks: tuple[int, int]
    shards: tuple[int, int] | None
    block: tuple[int, int]


def dense_shards(row_chunk: int, col_chunk: int, n_rows: int, n_cols: int, factor: int) -> DenseLayout:
    """Resolve the zarr v3 shard shape and write-block shape for a dense X array.

    With sharding on (``factor`` > 1) the inner chunk stays ``(row_chunk, col_chunk)`` —
    that remains the read granularity — and many inner chunks are packed into one shard
    object, cutting file/object count. zarr requires the shard shape to be an integer
    multiple of the inner chunk shape, so the shard is ``chunk * factor`` per axis, capped
    at the number of chunks the array actually spans (no point in a shard reaching far past
    the data). Peak dense RAM per write block grows by ``~factor**2`` when sharding — the
    documented cost of fewer, larger objects.

    Parameters
    ----------
    row_chunk, col_chunk
        Inner chunk shape.
    n_rows, n_cols
        Full array shape.
    factor
        Shards per axis relative to the inner chunk; ``1`` disables sharding.

    Returns
    -------
    DenseLayout
        ``shards`` is ``None`` when ``factor <= 1``.
    """
    if factor <= 1:
        return DenseLayout(chunks=(row_chunk, col_chunk), shards=None, block=(row_chunk, col_chunk))
    import math

    rf = min(factor, math.ceil(n_rows / row_chunk))
    cf = min(factor, math.ceil(n_cols / col_chunk))
    shard_row = row_chunk * rf
    shard_col = col_chunk * cf
    return DenseLayout(chunks=(row_chunk, col_chunk), shards=(shard_row, shard_col), block=(shard_row, shard_col))


def band_plan(n_rows: int, band_rows: int) -> tuple[tuple[int, int], ...]:
    """Split ``[0, n_rows)`` into contiguous ``(start, end)`` bands of ``band_rows`` each.

    Parameters
    ----------
    n_rows
        Total number of rows (or flat elements) to cover.
    band_rows
        Rows per band; the last band is clipped to ``n_rows``.

    Returns
    -------
    tuple[tuple[int, int], ...]
        Empty when ``n_rows <= 0`` or ``band_rows <= 0``.
    """
    if n_rows <= 0 or band_rows <= 0:
        return ()
    return tuple((r0, min(r0 + band_rows, n_rows)) for r0 in range(0, n_rows, band_rows))
