from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp
import zarr

from annizarr._core import _layout
from annizarr._core._layout import write_grid, x_compressors
from annizarr._core._runtime import progress, run_parallel
from annizarr._core._zarr import get_array, shape_attr
from annizarr._sources._matrix import get_indptr, is_backed
from annizarr._writers._encoding import make_sparse_group, set_array_attrs, sparse_shards, suppress_autoshard_warning
from annizarr._writers._workers import _copy_sparse_segment
from annizarr.errors import ConversionError

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

    from numpy.typing import NDArray

    from annizarr._core._config import AppConfig


def flat_segments(nnz_total: int, step: int, bytes_per_nnz: int) -> list[tuple[int, int]]:
    # step is the output's write grid (not the raw chunk size), so segments never share one
    if nnz_total <= 0:
        return []
    seg = max(1, _layout.BATCH_BYTES // max(1, step * bytes_per_nnz)) * step
    return [(s, min(s + seg, nnz_total)) for s in range(0, nnz_total, seg)]


def _write_flat_segment(
    data_arr: zarr.Array[Any],
    indices_arr: zarr.Array[Any],
    data: Any,
    indices: Any,
    s0: int,
    s1: int,
    indices_dtype: Any,
    tick: Callable[[], None],
) -> None:
    data_arr[s0:s1] = data[s0:s1]
    indices_arr[s0:s1] = np.asarray(indices[s0:s1], dtype=indices_dtype)
    tick()


def _write_sparse_streaming(group: zarr.Group, matrix: Any, key: str, cfg: AppConfig, csr: bool) -> None:
    n_rows, n_cols = matrix.shape
    n_major = n_rows if csr else n_cols

    # backed _CSRDataset/_CSCDataset doesn't expose .indptr directly — read the h5py group.
    if sp.issparse(matrix):
        indptr_full = np.asarray(matrix.indptr)
    elif hasattr(matrix, "indptr"):
        indptr_full = np.asarray(matrix.indptr[:])
    elif hasattr(matrix, "group"):
        indptr_full = np.asarray(matrix.group["indptr"][:])
    else:
        raise ConversionError(f"Cannot locate indptr on sparse input of type {type(matrix).__name__}")
    nnz_total = int(indptr_full[-1])

    backed = is_backed(matrix)
    indices_dtype = np.int32  # match scipy default; values fit unless > 2^31 cols/rows

    sp_group = make_sparse_group(group, key, csr=csr, shape=(n_rows, n_cols))

    flat_chunk = min(cfg.chunks.sparse_flat_chunk, max(1, nnz_total))
    shards = sparse_shards(cfg.chunks.auto_shard)
    with suppress_autoshard_warning(cfg.chunks.auto_shard):
        data_arr = sp_group.require_array(
            "data",
            shape=(nnz_total,),
            dtype=matrix.dtype,
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
        shape=(n_major + 1,),
        dtype=indptr_full.dtype,
        chunks=(n_major + 1,),
        overwrite=True,
    )
    for a in (data_arr, indices_arr, indptr_arr):
        set_array_attrs(a)

    indptr_arr[:] = indptr_full

    bytes_per_nnz = np.dtype(matrix.dtype).itemsize + np.dtype(indices_dtype).itemsize
    step = write_grid(data_arr)[0]

    if backed:
        from zarr.storage import LocalStore

        # h5py is not thread-safe, so backed segments copy in worker processes instead.
        src = matrix.group
        store = data_arr.store_path.store
        assert isinstance(store, LocalStore), "backed sparse write requires a local zarr store"
        out_root = store.root
        jobs = [
            (
                out_root,
                data_arr.store_path.path,
                indices_arr.store_path.path,
                src.file.filename,
                src.name,
                s0,
                s1,
                indices_dtype,
            )
            for s0, s1 in flat_segments(nnz_total, step, bytes_per_nnz)
        ]
        run_parallel(_copy_sparse_segment, jobs, cfg.chunks.cpus, mode="processes")
        return

    segments = flat_segments(nnz_total, step, bytes_per_nnz)
    tick = progress(len(segments), f"Writing {key}")
    thread_jobs = [
        (data_arr, indices_arr, matrix.data, matrix.indices, s0, s1, indices_dtype, tick) for s0, s1 in segments
    ]
    run_parallel(_write_flat_segment, thread_jobs, cfg.chunks.cpus, mode="threads")


def _sparse_source(matrix: Any) -> tuple[Any, Any, NDArray[np.int64], tuple[int, int], Any]:
    if isinstance(matrix, zarr.Group):
        data = get_array(matrix, "data")
        indices = get_array(matrix, "indices")
        indptr = np.asarray(get_array(matrix, "indptr")[:], dtype=np.int64)
        return data, indices, indptr, shape_attr(matrix), data.dtype
    indptr = get_indptr(matrix).astype(np.int64, copy=False)
    return matrix.group["data"], matrix.group["indices"], indptr, matrix.shape, matrix.dtype


def _local_tmp_dir(group: zarr.Group) -> str | None:
    from zarr.storage import LocalStore

    store = group.store_path.store
    return str(store.root) if isinstance(store, LocalStore) else None


def write_transposed_sparse(
    group: zarr.Group,
    key: str,
    matrix: Any,
    cfg: AppConfig,
    *,
    row_scale: NDArray[np.float64] | None = None,
    target: Literal["csc", "csr"],
) -> zarr.Group:
    # two BATCH_BYTES-bounded passes so the whole matrix is never materialised.
    data_src, idx_src, indptr_src, (n_rows, n_cols), src_dtype = _sparse_source(matrix)
    n_source_major = n_rows if target == "csc" else n_cols
    n_target_major = n_cols if target == "csc" else n_rows
    nnz = int(indptr_src[-1])
    source_major_nnz = np.diff(indptr_src)

    value_dtype = np.dtype(np.float32) if row_scale is not None else np.dtype(src_dtype)
    indices_dtype = np.int32  # source-major positions in the output; matches scipy's default
    indptr_dtype = np.int64 if nnz > np.iinfo(np.int32).max else np.int32

    target_nnz = np.zeros(n_target_major, dtype=np.int64)
    flat_step = max(cfg.chunks.sparse_flat_chunk, _layout.BATCH_BYTES // 8)
    for s0 in range(0, nnz, flat_step):
        s1 = min(s0 + flat_step, nnz)
        target_nnz += np.bincount(np.asarray(idx_src[s0:s1]), minlength=n_target_major)
    target_indptr = np.concatenate([[0], np.cumsum(target_nnz)]).astype(np.int64)

    bytes_per_entry = value_dtype.itemsize + 16
    max_band_nnz = max(1, _layout.BATCH_BYTES // bytes_per_entry)
    edges = [0]
    while edges[-1] < n_target_major:
        want = target_indptr[edges[-1]] + max_band_nnz
        nxt = int(np.searchsorted(target_indptr, want, side="right")) - 1
        edges.append(min(max(nxt, edges[-1] + 1), n_target_major))
    n_bands = len(edges) - 1
    band_nnz = [int(target_indptr[edges[i + 1]] - target_indptr[edges[i]]) for i in range(n_bands)]

    bytes_per_source_unit = max(1, nnz // max(1, n_source_major)) * 12
    source_step = max(1_000, min(200_000, _layout.BATCH_BYTES // bytes_per_source_unit))

    tmp_root = Path(tempfile.mkdtemp(prefix="annizarr_transpose_", dir=_local_tmp_dir(group)))
    try:
        buckets: list[dict[str, NDArray[Any]]] = []
        for i, m in enumerate(band_nnz):
            m = max(1, m)
            buckets.append(
                {
                    "src": np.memmap(tmp_root / f"s{i}", dtype=np.int32, mode="w+", shape=(m,)),
                    "tgt": np.memmap(tmp_root / f"t{i}", dtype=np.int32, mode="w+", shape=(m,)),
                    "val": np.memmap(tmp_root / f"v{i}", dtype=value_dtype, mode="w+", shape=(m,)),
                }
            )
        edges_arr = np.asarray(edges[1:], dtype=np.int64)
        cursors = [0] * n_bands

        bands = range(0, n_source_major, source_step)
        tick = progress(len(bands), f"Transposing {key} to {target}")
        for b0 in bands:
            b1 = min(b0 + source_step, n_source_major)
            s0, s1 = int(indptr_src[b0]), int(indptr_src[b1])
            data_band = np.asarray(data_src[s0:s1])
            tgt_band = np.asarray(idx_src[s0:s1], dtype=np.int32)
            src_pos = np.repeat(np.arange(b0, b1, dtype=np.int32), source_major_nnz[b0:b1])
            if row_scale is not None:
                factor = np.repeat(row_scale[b0:b1], source_major_nnz[b0:b1])
                vals = np.log1p(data_band.astype(np.float64) * factor).astype(np.float32)
            else:
                vals = data_band
            band_ids = np.searchsorted(edges_arr, tgt_band, side="right")
            order = np.argsort(band_ids, kind="stable")
            bounds = np.searchsorted(band_ids[order], np.arange(n_bands + 1))
            for bi in range(n_bands):
                lo, hi = int(bounds[bi]), int(bounds[bi + 1])
                if lo == hi:
                    continue
                sel = order[lo:hi]
                c = cursors[bi]
                buckets[bi]["src"][c : c + hi - lo] = src_pos[sel]
                buckets[bi]["tgt"][c : c + hi - lo] = tgt_band[sel]
                buckets[bi]["val"][c : c + hi - lo] = vals[sel]
                cursors[bi] = c + hi - lo
            tick()

        sp_group = make_sparse_group(group, key, csr=(target == "csr"), shape=(n_rows, n_cols))
        flat_chunk = min(cfg.chunks.sparse_flat_chunk, max(1, nnz))
        shards = sparse_shards(cfg.chunks.auto_shard)
        with suppress_autoshard_warning(cfg.chunks.auto_shard):
            data_arr = sp_group.require_array(
                "data",
                shape=(nnz,),
                dtype=value_dtype,
                chunks=(flat_chunk,),
                shards=shards,
                compressors=x_compressors(),
                overwrite=True,
            )
            indices_arr = sp_group.require_array(
                "indices",
                shape=(nnz,),
                dtype=indices_dtype,
                chunks=(flat_chunk,),
                shards=shards,
                compressors=x_compressors(),
                overwrite=True,
            )
        indptr_arr = sp_group.require_array(
            "indptr", shape=(n_target_major + 1,), dtype=indptr_dtype, chunks=(n_target_major + 1,), overwrite=True
        )
        for a in (data_arr, indices_arr, indptr_arr):
            set_array_attrs(a)
        indptr_arr[:] = target_indptr.astype(indptr_dtype)

        for bi in range(n_bands):
            m = band_nnz[bi]
            if m == 0:
                continue
            order = np.argsort(np.asarray(buckets[bi]["tgt"][:m]), kind="stable")
            o0, o1 = int(target_indptr[edges[bi]]), int(target_indptr[edges[bi + 1]])
            # not write-grid aligned, but serial (no concurrent writers): a bounded perf cost.
            data_arr[o0:o1] = np.asarray(buckets[bi]["val"][:m])[order]
            indices_arr[o0:o1] = np.asarray(buckets[bi]["src"][:m])[order].astype(indices_dtype)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    return sp_group
