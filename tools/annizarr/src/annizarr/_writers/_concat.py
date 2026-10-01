from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp
import zarr

from annizarr._core._layout import band_plan, dense_shards, write_grid, x_compressors
from annizarr._core._runtime import progress, run_parallel
from annizarr._sources._matrix import get_indptr, is_backed, matrix_format
from annizarr._writers._encoding import make_sparse_group, set_array_attrs, sparse_shards, suppress_autoshard_warning
from annizarr._writers._sparse import flat_segments
from annizarr.errors import ConversionError

if TYPE_CHECKING:
    from collections.abc import Callable

    from annizarr._core._config import AppConfig


def _is_thread_unsafe(matrix: Any) -> bool:
    # h5py-backed inputs aren't thread-safe; zarr-backed ones are.
    return is_backed(matrix) and not isinstance(getattr(matrix, "group", None), zarr.Group)


def _flat_arrays(matrix: Any) -> tuple[Any, Any]:
    if sp.issparse(matrix):
        return matrix.data, matrix.indices
    if hasattr(matrix, "group"):
        g = matrix.group
        return g["data"], g["indices"]
    raise ConversionError(f"Cannot locate flat data/indices on sparse input of type {type(matrix).__name__}")


def _write_dense_concat_band(
    zarr_arr: zarr.Array[Any],
    matrices: tuple[Any, ...],
    fmts: tuple[str, ...],
    offsets: tuple[int, ...],
    R0: int,
    R1: int,
    block_col: int,
    n_cols: int,
    tick: Callable[[], None],
) -> None:
    # column tiles match block_col (the write grid), so every write is a whole, disjoint
    # block even when an input seam falls inside it — no read-modify-write.
    overlaps = []
    for matrix, fmt, off_lo, off_hi in zip(matrices, fmts, offsets[:-1], offsets[1:], strict=True):
        lo, hi = max(R0, off_lo), min(R1, off_hi)
        if lo >= hi:
            continue
        local_lo, local_hi = lo - off_lo, hi - off_lo
        band = matrix if fmt == "dense" else matrix[local_lo:local_hi]
        overlaps.append((lo - R0, hi - R0, band, fmt, local_lo, local_hi))

    dtype = zarr_arr.dtype
    for c0, c1 in band_plan(n_cols, block_col):
        tile = np.empty((R1 - R0, c1 - c0), dtype=dtype)
        for out_lo, out_hi, band, fmt, local_lo, local_hi in overlaps:
            if fmt == "dense":
                piece = np.asarray(band[local_lo:local_hi, c0:c1], dtype=dtype)
            else:
                piece = np.asarray(band[:, c0:c1].toarray(), dtype=dtype)
            tile[out_lo:out_hi, :] = piece
        zarr_arr[R0:R1, c0:c1] = tile
    tick()


def _write_concatenated_dense(
    group: zarr.Group,
    key: str,
    matrices: list[Any],
    n_obs_each: list[int],
    n_vars: int,
    data_dtype: Any,
    cfg: AppConfig,
) -> None:
    n_obs_total = sum(n_obs_each)

    row_chunk = min(cfg.chunks.x_row_chunk, n_obs_total)
    col_chunk = min(cfg.chunks.x_col_chunk, n_vars)
    layout = dense_shards(row_chunk, col_chunk, n_obs_total, n_vars, cfg.chunks.x_shard_factor)

    zarr_arr = group.require_array(
        key,
        shape=(n_obs_total, n_vars),
        dtype=data_dtype,
        chunks=layout.chunks,
        shards=layout.shards,
        compressors=x_compressors(),
        overwrite=True,
    )
    set_array_attrs(zarr_arr)

    block_row, block_col = write_grid(zarr_arr)
    offsets = tuple(int(v) for v in np.cumsum([0, *n_obs_each]))
    fmts = tuple(matrix_format(m) for m in matrices)
    matrices_t = tuple(matrices)
    any_unsafe = any(_is_thread_unsafe(m) for m in matrices)

    row_bands = band_plan(n_obs_total, block_row)
    tick = progress(len(row_bands), f"Writing {key} (concat)")
    jobs = [(zarr_arr, matrices_t, fmts, offsets, R0, R1, block_col, n_vars, tick) for R0, R1 in row_bands]
    run_parallel(_write_dense_concat_band, jobs, 1 if any_unsafe else cfg.chunks.cpus, mode="threads")


def _write_csr_concat_segment(
    data_arr: zarr.Array[Any],
    indices_arr: zarr.Array[Any],
    flat_pairs: tuple[tuple[Any, Any], ...],
    nnz_offsets: tuple[int, ...],
    s0: int,
    s1: int,
    data_dtype: Any,
    indices_dtype: Any,
    tick: Callable[[], None],
) -> None:
    # a segment spanning an input seam gathers the overlapping pieces before one write, so
    # every output chunk is written exactly once regardless of where the inputs seam.
    data_buf = np.empty(s1 - s0, dtype=data_dtype)
    indices_buf = np.empty(s1 - s0, dtype=indices_dtype)
    for (data_src, indices_src), off_lo, off_hi in zip(flat_pairs, nnz_offsets[:-1], nnz_offsets[1:], strict=True):
        lo, hi = max(s0, off_lo), min(s1, off_hi)
        if lo >= hi:
            continue
        local_lo, local_hi = lo - off_lo, hi - off_lo
        data_buf[lo - s0 : hi - s0] = np.asarray(data_src[local_lo:local_hi], dtype=data_dtype)
        indices_buf[lo - s0 : hi - s0] = np.asarray(indices_src[local_lo:local_hi], dtype=indices_dtype)
    data_arr[s0:s1] = data_buf
    indices_arr[s0:s1] = indices_buf
    tick()


def _write_concatenated_csr(
    group: zarr.Group,
    key: str,
    matrices: list[Any],
    n_obs_each: list[int],
    n_vars: int,
    data_dtype: Any,
    cfg: AppConfig,
) -> None:
    indptrs = [get_indptr(m) for m in matrices]
    nnz_each = [int(ip[-1]) for ip in indptrs]
    nnz_total = sum(nnz_each)
    n_obs_total = sum(n_obs_each)

    indices_dtype = np.int32
    indptr_dtype = np.int64 if nnz_total > np.iinfo(np.int32).max else np.int32

    sp_group = make_sparse_group(group, key, csr=True, shape=(n_obs_total, n_vars))

    flat_chunk = min(cfg.chunks.sparse_flat_chunk, max(1, nnz_total))
    shards = sparse_shards(cfg.chunks.auto_shard)
    with suppress_autoshard_warning(cfg.chunks.auto_shard):
        data_arr = sp_group.require_array(
            "data",
            shape=(nnz_total,),
            dtype=data_dtype,
            chunks=(flat_chunk,),
            shards=shards,
            compressors=x_compressors(),
            overwrite=True,
        )
        indices_arr = sp_group.require_array(
            "indices",
            shape=(nnz_total,),
            dtype=indices_dtype,
            chunks=(flat_chunk,),
            shards=shards,
            compressors=x_compressors(),
            overwrite=True,
        )
    indptr_arr = sp_group.require_array(
        "indptr",
        shape=(n_obs_total + 1,),
        dtype=indptr_dtype,
        chunks=(n_obs_total + 1,),
        overwrite=True,
    )
    for a in (data_arr, indices_arr, indptr_arr):
        set_array_attrs(a)

    full_indptr = np.empty(n_obs_total + 1, dtype=indptr_dtype)
    full_indptr[0] = 0
    row_offset = 0
    nnz_offset = 0
    for ip, n_obs_i, nnz_i in zip(indptrs, n_obs_each, nnz_each, strict=True):
        full_indptr[row_offset + 1 : row_offset + 1 + n_obs_i] = ip[1:].astype(indptr_dtype, copy=False) + nnz_offset
        row_offset += n_obs_i
        nnz_offset += nnz_i
    indptr_arr[:] = full_indptr

    if nnz_total == 0:
        return

    nnz_offsets = tuple(int(v) for v in np.cumsum([0, *nnz_each]))
    flat_pairs = tuple(_flat_arrays(m) for m in matrices)
    any_unsafe = any(_is_thread_unsafe(m) for m in matrices)

    bytes_per_nnz = np.dtype(data_dtype).itemsize + np.dtype(indices_dtype).itemsize
    segments = flat_segments(nnz_total, write_grid(data_arr)[0], bytes_per_nnz)
    tick = progress(len(segments), f"Writing {key} (concat)")
    jobs = [
        (data_arr, indices_arr, flat_pairs, nnz_offsets, s0, s1, data_dtype, indices_dtype, tick) for s0, s1 in segments
    ]
    run_parallel(_write_csr_concat_segment, jobs, 1 if any_unsafe else cfg.chunks.cpus, mode="threads")
