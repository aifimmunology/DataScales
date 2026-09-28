from __future__ import annotations

import sys
from typing import Any

import numpy as np
import zarr

from .._config import AppConfig
from .._layout import dense_shards, x_compressors
from .._runtime import run_parallel
from ._encoding import set_array_attrs
from ._workers import _densify_band_segment


def _build_tiled_dense_dask(matrix: Any, row_chunk: int, col_chunk: int) -> Any:
    """Build a 2D-tiled dask array from a (backed or in-memory) CSR matrix.

    Each block is exactly (row_chunk × col_chunk), matching the zarr chunk grid so
    da.store() writes whole chunks (no read-modify-write) and never materialises more
    than one chunk-sized dense block per worker — peak RAM is bounded by the chunk
    size, independent of total column count.

    A single delayed row-slice (sparse CSR) is shared across that band's column tiles,
    so each row band is sliced from source storage only once. da.from_delayed is used
    (not da.from_array) because backed _CSRDataset has no ndim / array protocol.
    """
    import dask
    import dask.array as da
    import numpy as np

    n_rows, n_cols = matrix.shape
    dtype = matrix.dtype

    def _row_band(r0: int, r1: int):
        # Row-slicing returns scipy CSR for both backed and in-memory inputs.
        return matrix[r0:r1]

    row_blocks = []
    for r0 in range(0, n_rows, row_chunk):
        r1 = min(r0 + row_chunk, n_rows)
        band = dask.delayed(_row_band)(r0, r1)  # sparse; computed once, reused per tile
        col_blocks = []
        for c0 in range(0, n_cols, col_chunk):
            c1 = min(c0 + col_chunk, n_cols)
            col_blocks.append(da.from_delayed(
                dask.delayed(lambda b, a=c0, z=c1: np.asarray(b[:, a:z].toarray()))(band),
                shape=(r1 - r0, c1 - c0),
                dtype=dtype,
            ))
        row_blocks.append(da.concatenate(col_blocks, axis=1))
    return da.concatenate(row_blocks, axis=0)


def _write_sparse_as_dense_dask(
    group: zarr.Group, matrix: Any, key: str, cfg: AppConfig
) -> None:
    """Write a CSR matrix (in-memory or backed _CSRDataset) as a dense zarr array.

    Blocks match the zarr chunk grid (row_chunk × col_chunk) so the full dense
    matrix is never materialised. In-memory CSR streams through a 2D-tiled dask
    array (cfg.chunks.cpus threads). Backed _CSRDataset is written by row band in
    parallel processes (each opens its own h5py handle; bands are chunk-aligned).
    """
    import dask.array as da
    from dask.diagnostics import ProgressBar

    n_rows, n_cols = matrix.shape
    dtype = matrix.dtype

    row_chunk = min(cfg.chunks.x_row_chunk, n_rows)
    col_chunk = min(cfg.chunks.x_col_chunk, n_cols)
    shards, block_row, block_col = dense_shards(
        row_chunk, col_chunk, n_rows, n_cols, cfg.chunks.x_shard_factor
    )

    zarr_arr = group.require_array(
        key,
        shape=(n_rows, n_cols),
        dtype=dtype,
        chunks=(row_chunk, col_chunk),
        shards=shards,
        compressors=x_compressors(),
        overwrite=True,
    )
    set_array_attrs(zarr_arr)

    backed = not hasattr(matrix, "indices")  # backed _CSRDataset lacks .indices
    if backed:
        # Bands are block_row tall and densified in block_col-wide tiles so each write
        # covers whole shards (or whole chunks when unsharded) — no read-modify-write,
        # and disjoint bands never share a shard so the parallel workers need no lock.
        src = matrix.group
        out_root = zarr_arr.store_path.store.root
        jobs = [
            (out_root, zarr_arr.store_path.path, src.file.filename, src.name,
             r0, min(r0 + block_row, n_rows), block_col, n_cols)
            for r0 in range(0, n_rows, block_row)
        ]
        run_parallel(_densify_band_segment, jobs, cfg.chunks.cpus, mode="processes")
        return

    dask_dense = _build_tiled_dense_dask(matrix, block_row, block_col)
    with ProgressBar(out=sys.stderr, dt=1.0, minimum=0):
        # lock=False: this single-file path tiles from row 0 with block == the shard shape
        # (dense_shards), so every da.store task writes one whole, disjoint shard — no shared
        # chunk, so no write lock is needed. dask's default lock=True serializes the
        # compress+write and pins the threaded path to ~1 core (measured ~6.5x slower on the
        # unsharded dense path). NOTE: do NOT copy lock=False to the concat/_append_* paths —
        # those write at a misaligned row/nnz offset and read-modify-write the seam chunk, so
        # they must keep the lock or concurrent writes corrupt data.
        da.store(dask_dense, zarr_arr, scheduler="threads",
                 num_workers=cfg.chunks.cpus, lock=False)


def _write_dense_streaming(
    group: zarr.Group, matrix: Any, key: str, cfg: AppConfig
) -> None:
    """Stream an already-dense matrix to a dense zarr array via da.store.

    anndata's write_elem assigns the whole array at once (full materialisation);
    da.from_array + da.store writes chunk-by-chunk to the zarr grid instead.
    """
    import dask.array as da
    from dask.diagnostics import ProgressBar

    n_rows, n_cols = matrix.shape
    row_chunk = min(cfg.chunks.x_row_chunk, n_rows)
    col_chunk = min(cfg.chunks.x_col_chunk, n_cols)
    shards, block_row, block_col = dense_shards(
        row_chunk, col_chunk, n_rows, n_cols, cfg.chunks.x_shard_factor
    )

    zarr_arr = group.require_array(
        key, shape=(n_rows, n_cols), dtype=matrix.dtype,
        chunks=(row_chunk, col_chunk), shards=shards,
        compressors=x_compressors(), overwrite=True,
    )
    set_array_attrs(zarr_arr)

    # Block to the shard grid (or chunk grid when unsharded) so each da.store write covers
    # whole shards and never triggers a read-modify-write of a partial shard.
    backed = not isinstance(matrix, np.ndarray)  # h5py-backed dense isn't thread-safe
    arr = da.from_array(matrix, chunks=(block_row, block_col))
    with ProgressBar(out=sys.stderr, dt=1.0, minimum=0):
        da.store(
            arr, zarr_arr,
            scheduler="synchronous" if backed else "threads",
            num_workers=1 if backed else cfg.chunks.cpus,
            lock=False,  # single-file, block==shard tiled from 0 -> disjoint whole-shard writes (see _write_sparse_as_dense_dask)
        )
