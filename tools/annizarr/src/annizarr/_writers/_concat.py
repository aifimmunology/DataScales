from __future__ import annotations

import sys
from typing import Any

import numpy as np
import scipy.sparse as sp
import zarr

from .._config import AppConfig
from .._layout import dense_shards, x_compressors
from .._sources._matrix import get_indptr
from ._dense import _build_tiled_dense_dask
from ._encoding import make_sparse_group, set_array_attrs


def _append_dense_region(
    zarr_arr: Any,
    matrix: Any,
    row_offset: int,
    cfg: AppConfig,
) -> None:
    """Write a single matrix into zarr_arr[row_offset:row_offset+n_rows, :]."""
    import dask.array as da
    from dask.diagnostics import ProgressBar

    n_rows, n_cols = matrix.shape
    # Block to the array's shard grid (or chunk grid when unsharded) so writes cover whole
    # shards. NB: each input file starts at an arbitrary row_offset, so the shard straddling
    # a file seam is read-modify-written — unavoidable with per-file concat region writes.
    block_row, block_col = zarr_arr.shards or zarr_arr.chunks
    block_row = min(block_row, n_rows)
    block_col = min(block_col, n_cols)
    region = (slice(row_offset, row_offset + n_rows), slice(None))

    is_backed_sparse = not sp.issparse(matrix) and hasattr(matrix, "format")
    if not sp.issparse(matrix) and not is_backed_sparse:
        # Already dense (ndarray-like); 2D-tile to the zarr chunk grid via region store.
        arr = da.from_array(np.asarray(matrix), chunks=(block_row, block_col))
        with ProgressBar(out=sys.stderr, dt=1.0, minimum=0):
            da.store(
                arr, zarr_arr,
                regions=region,
                scheduler="threads",
                num_workers=cfg.chunks.cpus,
            )
        return

    backed = is_backed_sparse
    dask_dense = _build_tiled_dense_dask(matrix, block_row, block_col)
    scheduler = "synchronous" if backed else "threads"
    with ProgressBar(out=sys.stderr, dt=1.0, minimum=0):
        da.store(
            dask_dense, zarr_arr,
            regions=region,
            scheduler=scheduler,
            num_workers=1 if backed else cfg.chunks.cpus,
        )


def _csr_dask_parts(
    matrix: Any,
    indptr_full: Any,
    n_rows: int,
    nnz_total: int,
    data_dtype: Any,
    indices_dtype: Any,
) -> tuple[list[Any], list[Any]]:
    """Build (data_parts, indices_parts) dask arrays for ONE CSR matrix, batched by ~256 MB.

    Returns lazy parts rather than writing them, so callers can concatenate parts across many
    matrices and issue a SINGLE ``da.store`` — see :func:`_write_concatenated_csr` for why that
    matters.
    """
    import dask
    import dask.array as da

    _TARGET_BATCH_BYTES = 256 * 1024 * 1024
    avg_nnz = max(1, nnz_total // max(1, n_rows))
    bpm = avg_nnz * (np.dtype(data_dtype).itemsize + np.dtype(indices_dtype).itemsize)
    batch_size = max(1_000, min(200_000, _TARGET_BATCH_BYTES // max(1, bpm)))

    data_parts, indices_parts = [], []
    for b0 in range(0, n_rows, batch_size):
        b1 = min(b0 + batch_size, n_rows)
        bnnz = int(indptr_full[b1] - indptr_full[b0])
        if bnnz == 0:
            continue
        batch = dask.delayed(lambda m, a, b: m[a:b])(matrix, b0, b1)
        data_parts.append(da.from_delayed(
            dask.delayed(lambda b: np.asarray(b.data, dtype=data_dtype))(batch),
            shape=(bnnz,), dtype=data_dtype,
        ))
        indices_parts.append(da.from_delayed(
            dask.delayed(lambda b: np.asarray(b.indices, dtype=indices_dtype))(batch),
            shape=(bnnz,), dtype=indices_dtype,
        ))
    return data_parts, indices_parts


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
    data_arr = sp_group.require_array(
        "data", shape=(nnz_total,), dtype=data_dtype,
        chunks=(flat_chunk,), compressors=x_compressors(), overwrite=True,
    )
    indices_arr = sp_group.require_array(
        "indices", shape=(nnz_total,), dtype=indices_dtype,
        chunks=(flat_chunk,), compressors=x_compressors(), overwrite=True,
    )
    indptr_arr = sp_group.require_array(
        "indptr", shape=(n_obs_total + 1,), dtype=indptr_dtype,
        chunks=(n_obs_total + 1,), overwrite=True,
    )
    for a in (data_arr, indices_arr, indptr_arr):
        set_array_attrs(a)

    # Build full indptr in memory (small: ~8B per row) then write once.
    full_indptr = np.empty(n_obs_total + 1, dtype=indptr_dtype)
    full_indptr[0] = 0
    row_offset = 0
    nnz_offset = 0
    for ip, n_obs_i, nnz_i in zip(indptrs, n_obs_each, nnz_each):
        full_indptr[row_offset + 1 : row_offset + 1 + n_obs_i] = (
            ip[1:].astype(indptr_dtype, copy=False) + nnz_offset
        )
        row_offset += n_obs_i
        nnz_offset += nnz_i
    indptr_arr[:] = full_indptr

    # ── ONE da.store across ALL matrices, rechunked to the output chunk grid ──
    # This used to be one `da.store` per matrix, writing into an arbitrary nnz region. Two costs
    # made that pathological once the matrix count got large (e.g. `--sort-by` on high-cardinality
    # keys, which buckets one matrix per distinct key tuple):
    #   1. Every `da.store` is wrapped in `ProgressBar(dt=1.0)`, whose teardown joins a timer
    #      thread sleeping `dt` — ~1 s of fixed cost per matrix regardless of payload. Measured
    #      slope was 1.02 s/matrix, so 10^5 groups meant tens of hours of pure overhead.
    #   2. A per-matrix nnz region is never chunk-aligned, so zarr read-modify-wrote a whole
    #      `flat_chunk` (data + indices) for every matrix — amplification ~= flat_chunk/mean nnz.
    # Concatenating the lazy parts and rechunking to `flat_chunk` fixes both: one ProgressBar for
    # the whole write, and each output chunk written exactly once.
    import dask.array as da
    from dask.diagnostics import ProgressBar

    all_data, all_indices = [], []
    backed_any = False
    for matrix, ip, n_obs_i, nnz_i in zip(matrices, indptrs, n_obs_each, nnz_each):
        if nnz_i == 0:
            continue
        if not sp.issparse(matrix) and not isinstance(
            getattr(matrix, "group", None), zarr.Group
        ):
            backed_any = True  # h5py-backed: not thread-safe (zarr-backed temps are)
        d_parts, i_parts = _csr_dask_parts(
            matrix, ip, n_obs_i, nnz_i, data_dtype, indices_dtype
        )
        all_data.extend(d_parts)
        all_indices.extend(i_parts)

    if not all_data:
        return

    data_dask = da.concatenate(all_data).rechunk((flat_chunk,))
    indices_dask = da.concatenate(all_indices).rechunk((flat_chunk,))
    scheduler = "synchronous" if backed_any else "threads"
    with ProgressBar(out=sys.stderr, dt=1.0, minimum=0):
        da.store(
            [data_dask, indices_dask],
            [data_arr, indices_arr],
            scheduler=scheduler,
            num_workers=1 if backed_any else cfg.chunks.cpus,
        )


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

    # _append_dense_region 2D-tiles each file to the array's write-block grid (shard shape
    # when sharding, else chunk shape), so peak RAM is bounded by one block — no need to
    # shrink row_chunk for wide matrices.
    row_chunk = min(cfg.chunks.x_row_chunk, n_obs_total)
    col_chunk = min(cfg.chunks.x_col_chunk, n_vars)
    shards, _, _ = dense_shards(
        row_chunk, col_chunk, n_obs_total, n_vars, cfg.chunks.x_shard_factor
    )

    zarr_arr = group.require_array(
        key, shape=(n_obs_total, n_vars), dtype=data_dtype,
        chunks=(row_chunk, col_chunk), shards=shards,
        compressors=x_compressors(), overwrite=True,
    )
    set_array_attrs(zarr_arr)

    row_offset = 0
    for matrix, n_rows in zip(matrices, n_obs_each):
        _append_dense_region(zarr_arr, matrix, row_offset, cfg)
        row_offset += n_rows
