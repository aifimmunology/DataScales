from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp
import zarr

from annizarr import _layout
from annizarr._layout import x_compressors
from annizarr._runtime import progress, run_parallel
from annizarr._sources._matrix import get_indptr, is_backed
from annizarr._writers._encoding import make_sparse_group, set_array_attrs
from annizarr._writers._workers import _copy_sparse_segment
from annizarr._zarr import get_array, shape_attr
from annizarr.errors import ConversionError

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

    from numpy.typing import NDArray

    from annizarr._config import AppConfig


def flat_segments(nnz_total: int, flat_chunk: int, bytes_per_nnz: int) -> list[tuple[int, int]]:
    """Split ``[0, nnz_total)`` into ``flat_chunk``-aligned segments (the last one ragged).

    Segment size is ``k * flat_chunk`` with ``k`` chosen so each segment holds about
    ``_layout.BATCH_BYTES``. Every segment covers whole output chunks, so parallel writers
    never share a chunk and no read-modify-write happens.
    """
    if nnz_total <= 0:
        return []
    seg = max(1, _layout.BATCH_BYTES // max(1, flat_chunk * bytes_per_nnz)) * flat_chunk
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
    # data/indices are already flat, output-ordered arrays (write_matrix converts the
    # matrix to the target format before calling in), so s0:s1 is a straight copy; the
    # range is flat_chunk-aligned, so this write never shares a chunk with another task
    data_arr[s0:s1] = data[s0:s1]
    indices_arr[s0:s1] = np.asarray(indices[s0:s1], dtype=indices_dtype)
    tick()


def _write_sparse_streaming(group: zarr.Group, matrix: Any, key: str, cfg: AppConfig, csr: bool) -> None:
    # reads indptr upfront (small, ~8B per row/col) to know exact output offsets. In-memory
    # scipy sparse already matches the output format by the time this is called (write_matrix
    # converts beforehand), so its .data/.indices ARE the output's flat arrays — a thread pool
    # writes them in flat_chunk-aligned segments. Backed _CSRDataset/_CSCDataset is the same
    # format as the output too, so its data/indices map 1:1 to the output — workers flat-copy
    # chunk-aligned nnz segments in parallel processes (h5py is not thread-safe, but
    # independent process handles are).
    n_rows, n_cols = matrix.shape
    # CSR iterates over rows; CSC iterates over columns.
    n_major = n_rows if csr else n_cols

    # indptr is small (~8B per row); load fully to compute exact offsets.
    # Backed _CSRDataset/_CSCDataset doesn't expose .indptr directly — read from
    # the underlying h5py group instead.
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
    data_arr = sp_group.require_array(
        "data",
        shape=(nnz_total,),
        dtype=matrix.dtype,
        chunks=(flat_chunk,),
        compressors=x_compressors(),
        overwrite=True,
    )
    indices_arr = sp_group.require_array(
        "indices",
        shape=(nnz_total,),
        dtype=indices_dtype,
        chunks=(flat_chunk,),
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

    # indptr is small — write it directly.
    indptr_arr[:] = indptr_full

    bytes_per_nnz = np.dtype(matrix.dtype).itemsize + np.dtype(indices_dtype).itemsize

    if backed:
        from zarr.storage import LocalStore

        # Flat-copy chunk-aligned nnz segments in parallel processes. Segments are
        # multiples of flat_chunk, so each zarr chunk is owned by exactly one worker.
        src = matrix.group
        store = data_arr.store_path.store
        # the process-pool workers re-open the store by filesystem path (see _workers.py)
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
            for s0, s1 in flat_segments(nnz_total, flat_chunk, bytes_per_nnz)
        ]
        run_parallel(_copy_sparse_segment, jobs, cfg.chunks.cpus, mode="processes")
        return

    segments = flat_segments(nnz_total, flat_chunk, bytes_per_nnz)
    tick = progress(len(segments), f"Writing {key}")
    thread_jobs = [
        (data_arr, indices_arr, matrix.data, matrix.indices, s0, s1, indices_dtype, tick) for s0, s1 in segments
    ]
    run_parallel(_write_flat_segment, thread_jobs, cfg.chunks.cpus, mode="threads")


def _sparse_source(matrix: Any) -> tuple[Any, Any, NDArray[np.int64], tuple[int, int], Any]:
    # matrix is either a zarr.Group (an existing store's on-disk sparse X/layer, e.g.
    # add_expr's source) or a backed anndata _CSRDataset/_CSCDataset (h5py-backed convert
    # input) — the latter exposes no public .data/.indices, only .group (the underlying
    # h5py group) and get_indptr's .indptr. Returns (data, indices, indptr, shape, dtype).
    if isinstance(matrix, zarr.Group):
        data = get_array(matrix, "data")
        indices = get_array(matrix, "indices")
        indptr = np.asarray(get_array(matrix, "indptr")[:], dtype=np.int64)
        return data, indices, indptr, shape_attr(matrix), data.dtype
    indptr = get_indptr(matrix).astype(np.int64, copy=False)
    return matrix.group["data"], matrix.group["indices"], indptr, matrix.shape, matrix.dtype


def _local_tmp_dir(group: zarr.Group) -> str | None:
    # bucket temp files sit next to a local store (same filesystem, cheap memmap);
    # fall back to the system tmp dir for a non-local (e.g. icechunk) store.
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
    """Stream a row-major (CSR) or column-major (CSC) sparse source into the other axis.

    Two passes, both bounded to ``_layout.BATCH_BYTES`` per band, so the source and the
    output are each read/written a band at a time — the whole matrix is never
    materialised. Pass 1 histograms ``indices`` into the target major axis to get its
    ``indptr``. Pass 2 re-reads the source in source-major bands, buckets each entry's
    (source position, target position, value) into disk-backed temp files sized per
    target band; each band is then sorted by target position (stable, so ties keep the
    source-ascending append order — canonical CSR/CSC) and written to its slice of the
    output.

    Used for a backed CSR X/layer targeted at ``csc`` output, a backed CSC X/layer
    targeted at ``csr`` output (:func:`annizarr._writers._adata.write_matrix`), and
    ``add_expr``'s CSR X → lognormalized ``csc`` layer (with ``row_scale``).

    Parameters
    ----------
    group
        Parent group the new sparse group is created under.
    key
        Child group name.
    matrix
        Row-major source when ``target="csc"``, column-major source when
        ``target="csr"``: a backed anndata sparse dataset, or a zarr ``Group`` holding
        ``data``/``indices``/``indptr`` (e.g. an existing store's ``X``).
    cfg
        Resolved configuration; ``cfg.chunks.sparse_flat_chunk`` sizes the output's flat
        chunks (and the pass-1 read batch).
    row_scale
        Optional per-source-major-unit multiplicative factor (length = the source's
        major axis — rows, since this is only meaningful for a CSR source). When given,
        each value is written as ``log1p(value * row_scale[unit])`` — what
        ``add_expr``'s lognorm layer needs — instead of copied as-is.
    target
        Output layout: ``"csc"`` for a CSR source, ``"csr"`` for a CSC source.

    Returns
    -------
    zarr.Group
        The newly written sparse group (encoding attrs already set); callers may add
        further attrs (e.g. a target-sum) before the store is finalized.
    """
    data_src, idx_src, indptr_src, (n_rows, n_cols), src_dtype = _sparse_source(matrix)
    n_source_major = n_rows if target == "csc" else n_cols
    n_target_major = n_cols if target == "csc" else n_rows
    nnz = int(indptr_src[-1])
    source_major_nnz = np.diff(indptr_src)

    value_dtype = np.dtype(np.float32) if row_scale is not None else np.dtype(src_dtype)
    indices_dtype = np.int32  # source-major positions in the output; matches scipy's default
    indptr_dtype = np.int64 if nnz > np.iinfo(np.int32).max else np.int32

    # Pass 1: histogram source `indices` (already target-major positions) into the
    # target axis's indptr. Reads only `indices`, in BATCH_BYTES-ish flat batches.
    target_nnz = np.zeros(n_target_major, dtype=np.int64)
    flat_step = max(cfg.chunks.sparse_flat_chunk, _layout.BATCH_BYTES // 8)
    for s0 in range(0, nnz, flat_step):
        s1 = min(s0 + flat_step, nnz)
        target_nnz += np.bincount(np.asarray(idx_src[s0:s1]), minlength=n_target_major)
    target_indptr = np.concatenate([[0], np.cumsum(target_nnz)]).astype(np.int64)

    # Target bands sized so one band's bucket triple (source pos + target pos + value,
    # plus the argsort index at write time) stays within BATCH_BYTES.
    bytes_per_entry = value_dtype.itemsize + 16
    max_band_nnz = max(1, _layout.BATCH_BYTES // bytes_per_entry)
    edges = [0]
    while edges[-1] < n_target_major:
        want = target_indptr[edges[-1]] + max_band_nnz
        nxt = int(np.searchsorted(target_indptr, want, side="right")) - 1
        edges.append(min(max(nxt, edges[-1] + 1), n_target_major))
    n_bands = len(edges) - 1
    band_nnz = [int(target_indptr[edges[i + 1]] - target_indptr[edges[i]]) for i in range(n_bands)]

    # Source bands sized off the average source-major density, same BATCH_BYTES budget.
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
        data_arr = sp_group.require_array(
            "data", shape=(nnz,), dtype=value_dtype, chunks=(flat_chunk,), compressors=x_compressors(), overwrite=True
        )
        indices_arr = sp_group.require_array(
            "indices",
            shape=(nnz,),
            dtype=indices_dtype,
            chunks=(flat_chunk,),
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
            data_arr[o0:o1] = np.asarray(buckets[bi]["val"][:m])[order]
            indices_arr[o0:o1] = np.asarray(buckets[bi]["src"][:m])[order].astype(indices_dtype)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    return sp_group
