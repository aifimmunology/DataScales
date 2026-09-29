from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp

from annizarr import _layout
from annizarr._layout import x_compressors
from annizarr._runtime import run_parallel
from annizarr._sources._matrix import is_backed
from annizarr._writers._encoding import make_sparse_group, set_array_attrs
from annizarr._writers._workers import _copy_sparse_segment
from annizarr.errors import ConversionError

if TYPE_CHECKING:
    import zarr

    from annizarr._config import AppConfig


def _write_sparse_streaming(group: zarr.Group, matrix: Any, key: str, cfg: AppConfig, csr: bool) -> None:
    # reads indptr upfront (small, ~8B per row/col) to know exact output offsets.
    # In-memory scipy sparse is written in row/col batches via dask threads. Backed
    # _CSRDataset/_CSCDataset is the same format as the output, so its data/indices map
    # 1:1 to the output — workers flat-copy chunk-aligned nnz segments in parallel
    # processes (h5py is not thread-safe, but independent process handles are).
    # dask ships py.typed but delayed/from_delayed/concatenate/ProgressBar are lazily
    # re-exported and untyped at the call boundary; dask is removed in Phase 3.
    import dask
    import dask.array as da
    from dask.diagnostics import ProgressBar  # type: ignore[attr-defined]

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

    if backed:
        from zarr.storage import LocalStore

        # Flat-copy chunk-aligned nnz segments in parallel processes. Segments are
        # multiples of flat_chunk, so each zarr chunk is owned by exactly one worker.
        src = matrix.group
        store = data_arr.store_path.store
        # the process-pool workers re-open the store by filesystem path (see _workers.py)
        assert isinstance(store, LocalStore), "backed sparse write requires a local zarr store"
        out_root = store.root
        bytes_per_nnz = np.dtype(matrix.dtype).itemsize + np.dtype(indices_dtype).itemsize
        seg = max(1, _layout.BATCH_BYTES // (flat_chunk * bytes_per_nnz)) * flat_chunk
        jobs = [
            (
                out_root,
                data_arr.store_path.path,
                indices_arr.store_path.path,
                src.file.filename,
                src.name,
                s,
                min(s + seg, nnz_total),
                indices_dtype,
            )
            for s in range(0, nnz_total, seg)
        ]
        run_parallel(_copy_sparse_segment, jobs, cfg.chunks.cpus, mode="processes")
        return

    # In-memory: build dask arrays for data + indices via delayed row/col batches.
    # Auto-tune batch size: target ~256 MB per batch in RAM. Clamped to [1k, 200k] majors.
    avg_nnz_per_major = max(1, nnz_total // max(1, n_major))
    bytes_per_major = avg_nnz_per_major * (np.dtype(matrix.dtype).itemsize + np.dtype(indices_dtype).itemsize)
    batch_size = max(1_000, min(200_000, _layout.BATCH_BYTES // max(1, bytes_per_major)))
    batch_starts = list(range(0, n_major, batch_size))

    def _load_batch(m: Any, b0: int, b1: int) -> Any:
        # Slicing returns scipy sparse for both backed and in-memory inputs.
        return m[b0:b1] if csr else m[:, b0:b1]

    data_parts = []
    indices_parts = []
    nonempty = 0
    for b0 in batch_starts:
        b1 = min(b0 + batch_size, n_major)
        batch_nnz = int(indptr_full[b1] - indptr_full[b0])
        if batch_nnz == 0:
            continue
        nonempty += 1
        # One delayed batch shared by both data and indices to avoid loading twice.
        batch = dask.delayed(_load_batch)(matrix, b0, b1)  # type: ignore[attr-defined]
        data_parts.append(
            da.from_delayed(  # type: ignore[no-untyped-call]
                dask.delayed(lambda b: np.asarray(b.data))(batch),  # type: ignore[attr-defined]
                shape=(batch_nnz,),
                dtype=matrix.dtype,
            )
        )
        indices_parts.append(
            da.from_delayed(  # type: ignore[no-untyped-call]
                dask.delayed(lambda b: np.asarray(b.indices, dtype=indices_dtype))(batch),  # type: ignore[attr-defined]
                shape=(batch_nnz,),
                dtype=indices_dtype,
            )
        )

    if nonempty == 0:
        # all-zero matrix: indptr already written, data/indices are empty
        return

    data_dask = da.concatenate(data_parts)  # type: ignore[no-untyped-call]
    indices_dask = da.concatenate(indices_parts)  # type: ignore[no-untyped-call]

    with ProgressBar(out=sys.stderr, dt=1.0, minimum=0):  # type: ignore[no-untyped-call]
        da.store(
            [data_dask, indices_dask],
            [data_arr, indices_arr],  # type: ignore[list-item]  # zarr.Array is a valid da.store target
            scheduler="threads",
            num_workers=cfg.chunks.cpus,
        )
