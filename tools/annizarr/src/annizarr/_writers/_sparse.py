from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import zarr

from annizarr._core import _layout
from annizarr._core._layout import write_grid, x_compressors
from annizarr._core._runtime import ArrayTarget, pipeline_parallel, progress, run_parallel, writer_parallel_mode
from annizarr._writers._encoding import make_sparse_group, set_array_attrs, sparse_shards, suppress_autoshard_warning

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

    from numpy.typing import NDArray

    from annizarr._core._config import AppConfig
    from annizarr._sources._readers import Reader

type _CsrPayload = tuple[int, int, NDArray[Any], NDArray[Any]]


def _segment_size(step: int, bytes_per_nnz: int, budget: int, grain: int | None) -> int:
    # always a multiple of the output write grid; also of the source chunk when that fits the budget
    base = step
    if grain is not None and grain > 0:
        import math

        unit = math.lcm(step, grain)
        if unit * bytes_per_nnz <= budget:
            base = unit
    return max(1, budget // max(1, base * bytes_per_nnz)) * base


def flat_segments(nnz_total: int, step: int, bytes_per_nnz: int, *, grain: int | None = None) -> list[tuple[int, int]]:
    if nnz_total <= 0:
        return []
    seg = _segment_size(step, bytes_per_nnz, _layout.BATCH_BYTES, grain)
    return [(s, min(s + seg, nnz_total)) for s in range(0, nnz_total, seg)]


def _no_tick() -> None:
    pass


def _csr_job(
    data_target: ArrayTarget,
    indices_target: ArrayTarget,
    reader: Reader,
    s0: int,
    s1: int,
    indices_dtype: Any,
    tick: Callable[[], None],
) -> None:
    try:
        data_arr = data_target.resolve()
        indices_arr = indices_target.resolve()
        data, indices = reader.flat(s0, s1)
        data_arr[s0:s1] = data
        indices_arr[s0:s1] = np.asarray(indices, dtype=indices_dtype)
        tick()
    finally:
        if data_target.in_worker:
            reader.close()


def _csr_pipeline_produce(reader: Reader, s0: int, s1: int) -> _CsrPayload:
    try:
        data, indices = reader.flat(s0, s1)
        return s0, s1, data, indices
    finally:
        reader.close()


def _local_tmp_dir(group: zarr.Group) -> str | None:
    from zarr.storage import LocalStore

    store = group.store_path.store
    return str(store.root) if isinstance(store, LocalStore) else None


def write_csr(group: zarr.Group, key: str, reader: Reader, cfg: AppConfig) -> None:
    _write_compressed(group, key, reader, cfg, csr=True, shape=reader.shape)


def write_csc(group: zarr.Group, key: str, reader: Reader, cfg: AppConfig) -> None:
    if reader.lazy:
        write_transposed_sparse(group, key, reader, cfg, target="csc")
        return
    from annizarr._sources._readers import CSRMemoryReader

    # in memory: scipy's tocsc, then the parallel flat copy over the column-major view (.T is zero-copy)
    n_rows = reader.shape[0]
    columns = CSRMemoryReader(reader.csr_rows(0, n_rows).tocsc().T)
    columns.cpus = reader.cpus
    _write_compressed(group, key, columns, cfg, csr=False, shape=reader.shape)


def _write_compressed(
    group: zarr.Group, key: str, reader: Reader, cfg: AppConfig, *, csr: bool, shape: tuple[int, int]
) -> None:
    # reader is major-axis ordered: rows of a CSR output, columns of a CSC output
    n_major = reader.shape[0]
    indptr_full = reader.indptr
    nnz_total = int(indptr_full[-1])
    indices_dtype = np.int32  # matches scipy's default; values fit unless > 2^31 along the minor axis

    sp_group = make_sparse_group(group, key, csr=csr, shape=shape)
    flat_chunk = min(cfg.chunks.sparse_flat_chunk, max(1, nnz_total))
    shards = sparse_shards(cfg.chunks.auto_shard)
    with suppress_autoshard_warning(cfg.chunks.auto_shard):
        data_arr = sp_group.require_array(
            "data",
            shape=(nnz_total,),
            dtype=reader.dtype,
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
    indptr_dtype = np.int64 if nnz_total > np.iinfo(np.int32).max else np.int32
    indptr_arr = sp_group.require_array(
        "indptr", shape=(n_major + 1,), dtype=indptr_dtype, chunks=(n_major + 1,), overwrite=True
    )
    for a in (data_arr, indices_arr, indptr_arr):
        set_array_attrs(a)
    indptr_arr[:] = indptr_full.astype(indptr_dtype)

    if nnz_total == 0:
        return

    bytes_per_nnz = np.dtype(reader.dtype).itemsize + np.dtype(indices_dtype).itemsize
    step = write_grid(data_arr)[0]
    segments = flat_segments(nnz_total, step, bytes_per_nnz, grain=reader.grain)
    mode = writer_parallel_mode(reader, data_arr.store_path.store)

    if mode == "processes":
        data_target = ArrayTarget.detached(data_arr)
        indices_target = ArrayTarget.detached(indices_arr)
        jobs = [(data_target, indices_target, reader, s0, s1, indices_dtype, _no_tick) for s0, s1 in segments]
        run_parallel(_csr_job, jobs, cfg.chunks.cpus, mode="processes")
        return

    if mode == "pipeline":
        budget = min(_layout.BATCH_BYTES, _layout.PIPELINE_BATCH_BYTES)
        pipeline_seg = _segment_size(step, bytes_per_nnz, budget, reader.grain)
        pipeline_jobs = [(reader, s0, min(s0 + pipeline_seg, nnz_total)) for s0 in range(0, nnz_total, pipeline_seg)]

        def consume(payload: _CsrPayload) -> None:
            ps0, ps1, data, indices = payload
            data_arr[ps0:ps1] = data
            indices_arr[ps0:ps1] = np.asarray(indices, dtype=indices_dtype)

        pipeline_parallel(_csr_pipeline_produce, consume, pipeline_jobs, cfg.chunks.cpus)
        return

    tick = progress(len(segments), f"Writing {key}")
    data_target = ArrayTarget.attached(data_arr)
    indices_target = ArrayTarget.attached(indices_arr)
    cpus = cfg.chunks.cpus if mode == "threads" else 1
    jobs = [(data_target, indices_target, reader, s0, s1, indices_dtype, tick) for s0, s1 in segments]
    run_parallel(_csr_job, jobs, cpus, mode="threads")


def write_transposed_sparse(
    group: zarr.Group,
    key: str,
    reader: Reader,
    cfg: AppConfig,
    *,
    row_scale: NDArray[np.float64] | None = None,
    target: Literal["csc", "csr"],
) -> zarr.Group:
    # two BATCH_BYTES-bounded passes so the whole matrix is never materialised. `reader`
    # always represents data in source-major ("row-like") terms: reader.shape[0] is the
    # source major-axis count, reader.indptr is over it, reader.flat(...) gives (data,
    # target-axis-index) in source-major flat order. The true output shape is reader.shape
    # when target=="csc" (source is a normal CSR-shaped reader); when target=="csr" the
    # source was CSC and is fed in pre-swapped (as_reader's _SwappedCSRView), so the true
    # shape is reader.shape reversed.
    n_source_major, n_target_major = reader.shape
    true_shape = reader.shape if target == "csc" else (n_target_major, n_source_major)
    indptr_src = reader.indptr
    nnz = reader.nnz
    source_major_nnz = np.diff(indptr_src)

    value_dtype = np.dtype(np.float32) if row_scale is not None else np.dtype(reader.dtype)
    indices_dtype = np.int32  # source-major positions in the output; matches scipy's default
    indptr_dtype = np.int64 if nnz > np.iinfo(np.int32).max else np.int32

    target_nnz = np.zeros(n_target_major, dtype=np.int64)
    flat_step = max(cfg.chunks.sparse_flat_chunk, _layout.BATCH_BYTES // 8)
    for s0 in range(0, nnz, flat_step):
        s1 = min(s0 + flat_step, nnz)
        _, idx = reader.flat(s0, s1)
        target_nnz += np.bincount(idx, minlength=n_target_major)
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
            data_band, tgt_band = reader.flat(s0, s1)
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

        sp_group = make_sparse_group(group, key, csr=(target == "csr"), shape=true_shape)
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
