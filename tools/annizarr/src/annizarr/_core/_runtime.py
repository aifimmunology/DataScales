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
    # must run before numpy/scipy/blosc import: OpenBLAS reads these env vars once, at import.
    for var in _BLAS_VARS:
        os.environ.setdefault(var, "1")


_configured = False


def configure_runtime(cpus: int) -> None:
    global _configured
    if _configured:
        return
    _configured = True
    import zarr

    pin_blas()
    # both zarr knobs: async.concurrency only dispatches fetches, max_workers runs decompression.
    workers = max(cpus, os.cpu_count() or 1)
    zarr.config.set({"async.concurrency": 64, "threading.max_workers": workers})


def run_parallel(
    worker: Callable[..., None],
    jobs: Sequence[tuple[object, ...]],
    cpus: int,
    *,
    mode: Literal["threads", "processes"] = "threads",
) -> None:
    # mode="processes" for backed inputs (h5py is not thread-safe); "threads" otherwise.
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
            ex.shutdown(wait=False, cancel_futures=True)
            raise


def progress(total: int, label: str) -> Callable[[], None]:
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
    logger.info(f"{label} ...")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        logger.info(f"  done ({time.perf_counter() - t0:.1f}s)")
