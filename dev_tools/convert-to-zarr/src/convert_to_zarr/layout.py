from __future__ import annotations


def _x_compressors():
    """Blosc(zstd) + byte-shuffle. zarr's default is bare zstd level-0 with no
    shuffle; the shuffle gives a large ratio/throughput win on numeric matrices."""
    from zarr.codecs import BloscCodec
    return (BloscCodec(cname="zstd", clevel=5, shuffle="shuffle"),)


def _dense_shards(row_chunk, col_chunk, n_rows, n_cols, factor):
    """Resolve the zarr v3 shard shape and the write-block shape for a dense X array.

    With sharding on (``factor`` > 1) the inner chunk stays (row_chunk, col_chunk) — that
    remains the read granularity — and many inner chunks are packed into one shard object,
    cutting file/object count. zarr requires the shard shape to be an integer multiple of the
    inner chunk shape, so the shard is ``chunk * factor`` per axis, capped at the number of
    chunks the array actually spans (no point in a shard reaching far past the data).

    Returns ``(shards, block_row, block_col)`` where ``shards`` is the shards= kwarg (None when
    no sharding) and (block_row, block_col) is the granularity callers must write at. Writing a
    *partial* shard makes zarr's sharding codec read-modify-write the whole shard (silent perf
    killer #2), so the block shape equals the shard shape when sharding is on, and the inner
    chunk shape otherwise. Peak dense RAM per write block therefore grows by ~factor**2 when
    sharding — the documented cost of fewer, larger objects.
    """
    if factor <= 1:
        return None, row_chunk, col_chunk
    import math
    rf = min(factor, math.ceil(n_rows / row_chunk))
    cf = min(factor, math.ceil(n_cols / col_chunk))
    shard_row = row_chunk * rf
    shard_col = col_chunk * cf
    return (shard_row, shard_col), shard_row, shard_col
