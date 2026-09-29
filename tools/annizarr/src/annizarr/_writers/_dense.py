from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from annizarr._layout import band_plan, dense_shards, x_compressors
from annizarr._runtime import progress, run_parallel
from annizarr._sources._matrix import is_backed
from annizarr._writers._concat import _is_thread_unsafe
from annizarr._writers._encoding import set_array_attrs
from annizarr._writers._workers import _densify_band_segment

if TYPE_CHECKING:
    from collections.abc import Callable

    import zarr

    from annizarr._config import AppConfig


def _write_dense_block(
    zarr_arr: zarr.Array[Any], matrix: Any, r0: int, r1: int, c0: int, c1: int, tick: Callable[[], None]
) -> None:
    # r0:r1 x c0:c1 is one write-block (the shard grid when sharded, else the chunk grid);
    # blocks are disjoint across tasks, so no two threads ever touch the same chunk and no
    # read-modify-write happens
    zarr_arr[r0:r1, c0:c1] = np.asarray(matrix[r0:r1, c0:c1])
    tick()


def _densify_row_band(
    zarr_arr: zarr.Array[Any], matrix: Any, r0: int, r1: int, block_col: int, n_cols: int, tick: Callable[[], None]
) -> None:
    # slices the CSR row band once and reuses it for every column tile, so an in-memory
    # sparse source is sliced n_rows/block_row times total, not once per (row, col) tile;
    # each column tile write is block_col-wide, matching the write grid (no read-modify-write)
    band = matrix[r0:r1]
    for c0, c1 in band_plan(n_cols, block_col):
        zarr_arr[r0:r1, c0:c1] = np.asarray(band[:, c0:c1].toarray())
    tick()


def _write_sparse_as_dense(group: zarr.Group, matrix: Any, key: str, cfg: AppConfig) -> None:
    # densifies without ever materialising the full matrix: row bands are block_row tall,
    # densified in block_col-wide tiles matching the zarr write grid. In-memory CSR, and a
    # zarr-backed CSR (e.g. write_matrix's backed-CSC-to-dense temp store), are thread-pooled
    # over row bands (zarr is thread-safe); an h5py-backed _CSRDataset is densified by row
    # band in parallel processes instead (each opens its own h5py handle — h5py is not
    # thread-safe, but independent read-only file handles across processes are).
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
    if _is_thread_unsafe(matrix):
        from zarr.storage import LocalStore

        # Bands are block_row tall and densified in block_col-wide tiles so each write
        # covers whole shards (or whole chunks when unsharded) — no read-modify-write;
        # disjoint bands never share a shard, so the process-pool workers need no
        # cross-worker synchronisation.
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

    row_bands = band_plan(n_rows, block_row)
    tick = progress(len(row_bands), f"Writing {key} (sparse as dense)")
    thread_jobs = [(zarr_arr, matrix, r0, r1, block_col, n_cols, tick) for r0, r1 in row_bands]
    run_parallel(_densify_row_band, thread_jobs, cfg.chunks.cpus, mode="threads")


def _write_dense_streaming(group: zarr.Group, matrix: Any, key: str, cfg: AppConfig) -> None:
    # anndata's write_elem assigns the whole array at once (full materialisation); this
    # streams block-by-block onto the zarr write grid instead
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

    # Blocks match the write grid (shard shape when sharded, else chunk shape), so every
    # write covers a whole, disjoint block — no read-modify-write, no synchronisation needed.
    block_row, block_col = layout.block
    backed = is_backed(matrix)  # h5py-backed dense isn't thread-safe: force a serial pass
    blocks = [(r0, r1, c0, c1) for r0, r1 in band_plan(n_rows, block_row) for c0, c1 in band_plan(n_cols, block_col)]
    tick = progress(len(blocks), f"Writing {key}")
    jobs = [(zarr_arr, matrix, r0, r1, c0, c1, tick) for r0, r1, c0, c1 in blocks]
    run_parallel(_write_dense_block, jobs, 1 if backed else cfg.chunks.cpus, mode="threads")
