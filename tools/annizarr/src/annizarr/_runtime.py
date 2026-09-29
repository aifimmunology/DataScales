from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

logger = logging.getLogger(__name__)

_BLAS_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def pin_blas() -> None:
    """Pin every BLAS/OpenMP thread env var to 1 unless the caller already set it.

    Guards against thread oversubscription (thread-pool/process-pool workers x BLAS threads):
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
    """Run ``worker(*job)`` for each job in ``jobs`` in a pool (``cpus > 1``) or inline,
    cancelling not-yet-started jobs and re-raising as soon as one worker fails.

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
        futures = [ex.submit(worker, *job) for job in jobs]
        try:
            for fut in futures:
                fut.result()
        except BaseException:
            # a worker already failed: drop every job that hasn't started instead of
            # draining the whole queue first.
            ex.shutdown(wait=False, cancel_futures=True)
            raise


def progress(total: int, label: str) -> Callable[[], None]:
    """Return a thread-safe ``tick()`` that logs ``label`` progress for a writer's tasks.

    Parameters
    ----------
    total
        Number of ticks expected (tasks/blocks/segments); paces the percentage-based log.
    label
        Text prefixed to each progress line.

    Returns
    -------
    Callable[[], None]
        ``tick()``; logs at INFO at most once every ~5s or every 10% of ``total``,
        whichever comes first, plus always on the final call. Safe to call concurrently.
    """
    lock = threading.Lock()
    step = max(1, total // 10)
    count = 0
    last_time = time.perf_counter()
    last_count = 0

    def tick() -> None:
        nonlocal count, last_time, last_count
        with lock:
            count += 1
            now = time.perf_counter()
            done = count >= total
            if done or count - last_count >= step or now - last_time >= 5.0:
                logger.info(f"{label}: {count}/{total}" + (" done" if done else ""))
                last_time = now
                last_count = count

    return tick


@contextmanager
def stage(label: str) -> Iterator[None]:
    """Log a labelled progress stage at INFO, with elapsed time on exit."""
    logger.info(f"{label} ...")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        logger.info(f"  done ({time.perf_counter() - t0:.1f}s)")
