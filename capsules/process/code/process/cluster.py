"""The Dask-CUDA cluster the pipeline runs on, and a sampler for peak host and GPU memory."""
from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger("process")


@contextmanager
def gpu_cluster() -> Iterator:
    """A LocalCUDACluster over every visible GPU, with managed memory spilling to host RAM."""
    import cupy as cp
    import dask
    import rmm
    from dask.distributed import Client
    from dask_cuda import LocalCUDACluster
    from rmm.allocators.cupy import rmm_cupy_allocator

    dask.config.set({"distributed.scheduler.worker-ttl": None})
    cluster = LocalCUDACluster(rmm_managed_memory=True, enable_cudf_spill=True,
                               rmm_allocator_external_lib_list="cupy")
    try:
        with Client(cluster) as client:
            rmm.reinitialize(managed_memory=True)
            cp.cuda.set_allocator(rmm_cupy_allocator)
            yield client
    finally:
        try:
            cluster.close()
        except Exception as exc:
            logger.warning("Cluster shutdown: %s", exc)


def n_workers(client) -> int:
    return len(client.scheduler_info()["workers"])


def sync() -> None:
    import cupy as cp

    cp.cuda.Stream.null.synchronize()


class MemoryMonitor:
    """Polls host RSS (this process and its workers) and GPU memory in a background thread.

    GPU memory is device-wide, so another job on the same GPU counts toward the peak.
    """

    def __init__(self, interval: float = 0.25):
        import psutil
        import pynvml

        pynvml.nvmlInit()
        self._nvml = pynvml
        self._gpus = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        self._proc = psutil.Process()
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.peak_host = self.peak_gpu = 0

    def _host(self) -> int:
        total = self._proc.memory_info().rss
        for child in self._proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except Exception:
                continue
        return total

    def _gpu(self) -> int:
        return sum(self._nvml.nvmlDeviceGetMemoryInfo(h).used for h in self._gpus)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_host = max(self.peak_host, self._host())
            self.peak_gpu = max(self.peak_gpu, self._gpu())
            self._stop.wait(self._interval)

    def __enter__(self) -> MemoryMonitor:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        try:
            self._nvml.nvmlShutdown()
        except Exception:
            pass

    def summary(self) -> dict:
        return {"peak_host_gb": round(self.peak_host / 1024**3, 2), "peak_gpu_gb": round(self.peak_gpu / 1024**3, 2)}
