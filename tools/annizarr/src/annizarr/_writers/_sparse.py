from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp

from annizarr import _layout
from annizarr._layout import x_compressors
from annizarr._runtime import progress, run_parallel
from annizarr._sources._matrix import is_backed
from annizarr._writers._encoding import make_sparse_group, set_array_attrs
from annizarr._writers._workers import _copy_sparse_segment
from annizarr.errors import ConversionError

if TYPE_CHECKING:
    from collections.abc import Callable

    import zarr

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
