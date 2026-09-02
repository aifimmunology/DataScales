from __future__ import annotations

import sys
import time
from contextlib import contextmanager

# The band workers run in separate processes (h5py is not thread-safe, but independent
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
    import zarr
    import h5py
    from anndata.io import sparse_dataset
    from zarr.storage import LocalStore

    zarr.config.set({"async.concurrency": 8, "threading.max_workers": 2})
    with h5py.File(src_file, "r") as f:
        band = sparse_dataset(f[src_group])[r0:r1]
    arr = zarr.open_group(store=LocalStore(str(out_root)), mode="r+")[data_path]
    for c0 in range(0, n_cols, col_chunk):
        c1 = min(c0 + col_chunk, n_cols)
        arr[r0:r1, c0:c1] = band[:, c0:c1].toarray()


def _run_parallel(worker, jobs, cpus):
    """Run worker(*job) for each job — in a process pool when cpus>1, else inline.
    With a single job the pool would be pure spawn overhead, so run inline."""
    if cpus <= 1 or len(jobs) <= 1:
        for job in jobs:
            worker(*job)
        return
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=cpus) as ex:
        for fut in [ex.submit(worker, *job) for job in jobs]:
            fut.result()


_configured = False


def configure_runtime(cpus: int) -> None:
    """Pin BLAS threads and size zarr's dispatch/decode pools, once per process."""
    global _configured
    if _configured:
        return
    _configured = True
    import os
    import zarr

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    workers = max(cpus, os.cpu_count() or 1)
    zarr.config.set({"async.concurrency": 64, "threading.max_workers": workers})


def _run_parallel_threads(worker, jobs, cpus):
    """Threaded variant for zarr→zarr copies (blosc codecs release the GIL)."""
    if cpus <= 1 or len(jobs) <= 1:
        for job in jobs:
            worker(*job)
        return
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=cpus) as ex:
        for fut in [ex.submit(worker, *job) for job in jobs]:
            fut.result()


@contextmanager
def _stage(label: str):
    """Print a labelled progress line with elapsed time. Flushes immediately."""
    print(f"→ {label} ...", flush=True, file=sys.stderr)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        print(f"  done ({time.perf_counter() - t0:.1f}s)", flush=True, file=sys.stderr)
