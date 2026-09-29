from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

logger = logging.getLogger(__name__)

_BLAS_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def pin_blas() -> None:
    """Pin every BLAS/OpenMP thread env var to 1 unless the caller already set it.

    Guards against thread oversubscription (dask/process-pool workers x BLAS threads):
    must run before numpy/scipy/blosc are imported, since OpenBLAS reads these once at
    import time.
    """
    for var in _BLAS_VARS:
        os.environ.setdefault(var, "1")


_configured = False


def configure_runtime(cpus: int) -> None:
    """Pin BLAS threads and size zarr's dispatch/decode pools, once per process.

    Parameters
    ----------
    cpus
        Requested worker count; zarr's thread pool is sized to at least this and the
        physical core count, whichever is larger.
    """
    global _configured
    if _configured:
        return
    _configured = True
    import zarr

    pin_blas()
    workers = max(cpus, os.cpu_count() or 1)
    zarr.config.set({"async.concurrency": 64, "threading.max_workers": workers})


def run_parallel(
    worker: Callable[..., None],
    jobs: Sequence[tuple[object, ...]],
    cpus: int,
    *,
    mode: Literal["threads", "processes"] = "threads",
) -> None:
    """Run ``worker(*job)`` for each job in ``jobs`` — in a pool when ``cpus > 1``, else inline.

    Parameters
    ----------
    worker
        Callable invoked as ``worker(*job)`` for each entry in ``jobs``.
    jobs
        Positional-argument tuples, one per unit of work.
    cpus
        Pool size; ``<= 1`` (or a single job) runs inline, skipping pool spawn overhead.
    mode
        ``"processes"`` for backed inputs (h5py is not thread-safe, but independent
        read-only file handles across processes are); ``"threads"`` (default) for
        zarr-to-zarr copies, where blosc codecs release the GIL.
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
def stage(label: str) -> Iterator[None]:
    """Log a labelled progress stage at INFO, with elapsed time on exit."""
    logger.info(f"{label} ...")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        logger.info(f"  done ({time.perf_counter() - t0:.1f}s)")
