from __future__ import annotations

from typing import Any

from annizarr._core._zarr import get_array

# module-level so the process pool can pickle them: h5py is not thread-safe, but independent
# read-only file handles across processes are. Each worker writes a write-grid-aligned region
# (see the callers in _sparse.py/_dense.py), so no two workers touch the same shard/chunk.


def _copy_sparse_segment(
    out_root: Any,
    data_path: str,
    indices_path: str,
    src_file: str,
    src_group: str,
    s0: int,
    s1: int,
    indices_dtype: Any,
) -> None:
    import h5py
    import numpy as np
    import zarr
    from zarr.storage import LocalStore

    zarr.config.set({"async.concurrency": 8, "threading.max_workers": 2})  # small: pools multiply across the pool
    with h5py.File(src_file, "r") as f:
        g = f[src_group]
        data = g["data"][s0:s1]
        indices = np.asarray(g["indices"][s0:s1], dtype=indices_dtype)
    root = zarr.open_group(store=LocalStore(str(out_root)), mode="r+")
    get_array(root, data_path)[s0:s1] = data
    get_array(root, indices_path)[s0:s1] = indices


def _densify_band_segment(
    out_root: Any, data_path: str, src_file: str, src_group: str, r0: int, r1: int, col_chunk: int, n_cols: int
) -> None:
    import h5py
    import zarr
    from anndata.io import sparse_dataset
    from zarr.storage import LocalStore

    zarr.config.set({"async.concurrency": 8, "threading.max_workers": 2})
    with h5py.File(src_file, "r") as f:
        band = sparse_dataset(f[src_group])[r0:r1]
    root = zarr.open_group(store=LocalStore(str(out_root)), mode="r+")
    arr = get_array(root, data_path)
    for c0 in range(0, n_cols, col_chunk):
        c1 = min(c0 + col_chunk, n_cols)
        arr[r0:r1, c0:c1] = band[:, c0:c1].toarray()
