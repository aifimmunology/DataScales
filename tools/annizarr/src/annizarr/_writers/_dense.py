from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import numpy as np

from annizarr._layout import band_plan, dense_shards, x_compressors
from annizarr._runtime import run_parallel
from annizarr._sources._matrix import is_backed
from annizarr._writers._encoding import set_array_attrs
from annizarr._writers._workers import _densify_band_segment

if TYPE_CHECKING:
    import zarr

    from annizarr._config import AppConfig


def _build_tiled_dense_dask(matrix: Any, row_chunk: int, col_chunk: int) -> Any:
    # blocks are exactly (row_chunk x col_chunk), matching the zarr chunk grid, so
    # da.store() writes whole chunks (no read-modify-write); a single delayed row-slice
    # (sparse CSR) is shared across that band's column tiles, so each row band is sliced
    # from source storage only once. da.from_delayed (not da.from_array) is used because
    # backed _CSRDataset has no ndim / array protocol.
    # dask ships py.typed but delayed/from_delayed/concatenate are lazily re-exported and
    # untyped at the call boundary; dask is removed in Phase 3.
    import dask
    import dask.array as da

    n_rows, n_cols = matrix.shape
    dtype = matrix.dtype

    def _row_band(r0: int, r1: int) -> Any:
        # Row-slicing returns scipy CSR for both backed and in-memory inputs.
        return matrix[r0:r1]

    row_blocks = []
    for r0, r1 in band_plan(n_rows, row_chunk):
        band = dask.delayed(_row_band)(r0, r1)  # type: ignore[attr-defined]  # sparse; computed once, reused per tile
        col_blocks = []
        for c0, c1 in band_plan(n_cols, col_chunk):
            col_blocks.append(
                da.from_delayed(  # type: ignore[no-untyped-call]
                    dask.delayed(lambda b, a=c0, z=c1: np.asarray(b[:, a:z].toarray()))(band),  # type: ignore[attr-defined]
                    shape=(r1 - r0, c1 - c0),
                    dtype=dtype,
                )
            )
        row_blocks.append(da.concatenate(col_blocks, axis=1))  # type: ignore[no-untyped-call]
    return da.concatenate(row_blocks, axis=0)  # type: ignore[no-untyped-call]


def _write_sparse_as_dense_dask(group: zarr.Group, matrix: Any, key: str, cfg: AppConfig) -> None:
    # blocks match the zarr chunk grid (row_chunk x col_chunk) so the full dense matrix is
    # never materialised. In-memory CSR streams through a 2D-tiled dask array
    # (cfg.chunks.cpus threads); backed _CSRDataset is written by row band in parallel
    # processes (each opens its own h5py handle; bands are chunk-aligned).
    import dask.array as da
    from dask.diagnostics import ProgressBar  # type: ignore[attr-defined]  # dask lazy-export; removed in Phase 3

    n_rows, n_cols = matrix.shape
    dtype = matrix.dtype

    row_chunk = min(cfg.chunks.x_row_chunk, n_rows)
    col_chunk = min(cfg.chunks.x_col_chunk, n_cols)
    layout = dense_shards(row_chunk, col_chunk, n_rows, n_cols, cfg.chunks.x_shard_factor)

    zarr_arr = group.require_array(
        key,
        shape=(n_rows, n_cols),
        dtype=dtype,
        chunks=layout.chunks,
        shards=layout.shards,
        compressors=x_compressors(),
        overwrite=True,
    )
    set_array_attrs(zarr_arr)

    block_row, block_col = layout.block
    if is_backed(matrix):
        from zarr.storage import LocalStore

        # Bands are block_row tall and densified in block_col-wide tiles so each write
        # covers whole shards (or whole chunks when unsharded) — no read-modify-write,
        # and disjoint bands never share a shard so the parallel workers need no lock.
        src = matrix.group
        store = zarr_arr.store_path.store
        # the process-pool workers re-open the store by filesystem path (see _workers.py)
        assert isinstance(store, LocalStore), "backed dense write requires a local zarr store"
        out_root = store.root
        jobs = [
            (out_root, zarr_arr.store_path.path, src.file.filename, src.name, r0, r1, block_col, n_cols)
            for r0, r1 in band_plan(n_rows, block_row)
        ]
        run_parallel(_densify_band_segment, jobs, cfg.chunks.cpus, mode="processes")
        return

    dask_dense = _build_tiled_dense_dask(matrix, block_row, block_col)
    with ProgressBar(out=sys.stderr, dt=1.0, minimum=0):  # type: ignore[no-untyped-call]
        # lock=False: this single-file path tiles from row 0 with block == the shard shape
        # (dense_shards), so every da.store task writes one whole, disjoint shard — no shared
        # chunk, so no write lock is needed. dask's default lock=True serializes the
        # compress+write and pins the threaded path to ~1 core (measured ~6.5x slower on the
        # unsharded dense path). NOTE: do NOT copy lock=False to the concat/_append_* paths —
        # those write at a misaligned row/nnz offset and read-modify-write the seam chunk, so
        # they must keep the lock or concurrent writes corrupt data.
        # zarr.Array satisfies dask's array-store target at runtime; the dask stub's Buffer
        # protocol match on __array__ doesn't recognise it (untyped internals, Phase 3 removes dask).
        da.store(dask_dense, zarr_arr, scheduler="threads", num_workers=cfg.chunks.cpus, lock=False)  # type: ignore[arg-type]


def _write_dense_streaming(group: zarr.Group, matrix: Any, key: str, cfg: AppConfig) -> None:
    # anndata's write_elem assigns the whole array at once (full materialisation);
    # da.from_array + da.store writes chunk-by-chunk to the zarr grid instead
    import dask.array as da
    from dask.diagnostics import ProgressBar  # type: ignore[attr-defined]  # dask lazy-export; removed in Phase 3

    n_rows, n_cols = matrix.shape
    row_chunk = min(cfg.chunks.x_row_chunk, n_rows)
    col_chunk = min(cfg.chunks.x_col_chunk, n_cols)
    layout = dense_shards(row_chunk, col_chunk, n_rows, n_cols, cfg.chunks.x_shard_factor)

    zarr_arr = group.require_array(
        key,
        shape=(n_rows, n_cols),
        dtype=matrix.dtype,
        chunks=layout.chunks,
        shards=layout.shards,
        compressors=x_compressors(),
        overwrite=True,
    )
    set_array_attrs(zarr_arr)

    # Block to the shard grid (or chunk grid when unsharded) so each da.store write covers
    # whole shards and never triggers a read-modify-write of a partial shard.
    backed = is_backed(matrix)  # h5py-backed dense isn't thread-safe
    arr = da.from_array(matrix, chunks=layout.block)  # type: ignore[no-untyped-call]
    # lock=False: single-file, tiled from row 0 with block == the shard shape, so every
    # da.store task writes one whole, disjoint shard (see _write_sparse_as_dense_dask).
    with ProgressBar(out=sys.stderr, dt=1.0, minimum=0):  # type: ignore[no-untyped-call]
        da.store(
            arr,
            zarr_arr,  # type: ignore[arg-type]  # zarr.Array is a valid da.store target; dask stub mismatch
            scheduler="synchronous" if backed else "threads",
            num_workers=1 if backed else cfg.chunks.cpus,
            lock=False,
        )
