from __future__ import annotations

# These band workers run in separate processes (h5py is not thread-safe, but independent
# read-only file handles across processes are). They are module-level so the process
# pool can pickle them. Each worker writes a chunk-aligned region, so no two workers
# ever touch the same zarr chunk and no lock is needed.


def _copy_sparse_segment(out_root, data_path, indices_path, src_file, src_group,
                         s0, s1, indices_dtype):
    """Copy a chunk-aligned nnz segment [s0:s1) from a backed sparse h5ad to zarr.

    CSR→CSR / CSC→CSC keeps row/col order, so source and output flat positions
    map 1:1 — this is a straight flat copy, no scipy needed.
    """
    import h5py
    import numpy as np
    import zarr
    from zarr.storage import LocalStore

    # pools multiply across the process pool — keep each worker's zarr pools small
    zarr.config.set({"async.concurrency": 8, "threading.max_workers": 2})
    with h5py.File(src_file, "r") as f:
        g = f[src_group]
        data = g["data"][s0:s1]
        indices = np.asarray(g["indices"][s0:s1], dtype=indices_dtype)
    root = zarr.open_group(store=LocalStore(str(out_root)), mode="r+")
    root[data_path][s0:s1] = data
    root[indices_path][s0:s1] = indices


def _densify_band_segment(out_root, data_path, src_file, src_group, r0, r1,
                          col_chunk, n_cols):
    """Read CSR row band [r0:r1) from a backed sparse h5ad and write it dense,
    one column tile at a time so dense RAM is bounded by one chunk."""
    import h5py
    import zarr
    from anndata.io import sparse_dataset
    from zarr.storage import LocalStore

    zarr.config.set({"async.concurrency": 8, "threading.max_workers": 2})
    with h5py.File(src_file, "r") as f:
        band = sparse_dataset(f[src_group])[r0:r1]
    arr = zarr.open_group(store=LocalStore(str(out_root)), mode="r+")[data_path]
    for c0 in range(0, n_cols, col_chunk):
        c1 = min(c0 + col_chunk, n_cols)
        arr[r0:r1, c0:c1] = band[:, c0:c1].toarray()
