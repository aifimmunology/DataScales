from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from annizarr._core._layout import band_plan, dense_shards, write_grid, x_compressors
from annizarr._core._runtime import progress, run_parallel
from annizarr._sources._matrix import is_backed
from annizarr._writers._concat import _is_thread_unsafe
from annizarr._writers._encoding import set_array_attrs
from annizarr._writers._workers import _densify_band_segment

if TYPE_CHECKING:
    from collections.abc import Callable

    import zarr

    from annizarr._core._config import AppConfig


def _write_dense_block(
    zarr_arr: zarr.Array[Any], matrix: Any, r0: int, r1: int, c0: int, c1: int, tick: Callable[[], None]
) -> None:
    # disjoint write-grid blocks: no two threads touch the same chunk (no read-modify-write)
    zarr_arr[r0:r1, c0:c1] = np.asarray(matrix[r0:r1, c0:c1])
    tick()


def _densify_row_band(
    zarr_arr: zarr.Array[Any], matrix: Any, r0: int, r1: int, block_col: int, n_cols: int, tick: Callable[[], None]
) -> None:
    band = matrix[r0:r1]
    for c0, c1 in band_plan(n_cols, block_col):
        zarr_arr[r0:r1, c0:c1] = np.asarray(band[:, c0:c1].toarray())
    tick()


def _write_sparse_as_dense(group: zarr.Group, matrix: Any, key: str, cfg: AppConfig) -> None:
    # h5py-backed input uses processes (h5py is not thread-safe); zarr-backed uses threads.
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

    block_row, block_col = write_grid(zarr_arr)
    if _is_thread_unsafe(matrix):
        from zarr.storage import LocalStore

        # disjoint row bands never share a shard/chunk, so no synchronisation is needed.
        src = matrix.group
        store = zarr_arr.store_path.store
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

    block_row, block_col = write_grid(zarr_arr)
    backed = is_backed(matrix)  # h5py-backed dense isn't thread-safe: force a serial pass
    blocks = [(r0, r1, c0, c1) for r0, r1 in band_plan(n_rows, block_row) for c0, c1 in band_plan(n_cols, block_col)]
    tick = progress(len(blocks), f"Writing {key}")
    jobs = [(zarr_arr, matrix, r0, r1, c0, c1, tick) for r0, r1, c0, c1 in blocks]
    run_parallel(_write_dense_block, jobs, 1 if backed else cfg.chunks.cpus, mode="threads")
