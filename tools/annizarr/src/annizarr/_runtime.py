from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from typing import Literal

_BLAS_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def pin_blas() -> None:
    for var in _BLAS_VARS:
        os.environ.setdefault(var, "1")


_configured = False


def configure_runtime(cpus: int) -> None:
    """Pin BLAS threads and size zarr's dispatch/decode pools, once per process."""
    global _configured
    if _configured:
        return
    _configured = True
    import zarr

    pin_blas()
    workers = max(cpus, os.cpu_count() or 1)
    zarr.config.set({"async.concurrency": 64, "threading.max_workers": workers})


def run_parallel(worker, jobs, cpus, *, mode: Literal["threads", "processes"] = "threads") -> None:
    """Run worker(*job) for each job — in a thread or process pool when cpus>1, else inline.

    ``mode="processes"`` is for backed inputs (h5py is not thread-safe, but independent
    read-only file handles across processes are); ``mode="threads"`` (default) is for
    zarr-to-zarr copies, where blosc codecs release the GIL. With a single job the pool
    would be pure spawn overhead, so run inline.
    """
    if cpus <= 1 or len(jobs) <= 1:
        for job in jobs:
            worker(*job)
        return
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

    executor_cls = ProcessPoolExecutor if mode == "processes" else ThreadPoolExecutor
    with executor_cls(max_workers=cpus) as ex:
        for fut in [ex.submit(worker, *job) for job in jobs]:
            fut.result()


@contextmanager
def stage(label: str):
    """Print a labelled progress line with elapsed time. Flushes immediately."""
    print(f"→ {label} ...", flush=True, file=sys.stderr)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        print(f"  done ({time.perf_counter() - t0:.1f}s)", flush=True, file=sys.stderr)
