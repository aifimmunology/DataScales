"""Run-time support: the Dask-CUDA cluster (one worker per visible GPU, RMM memory mode) and per-step wall time with peak host / GPU memory."""
from __future__ import annotations

import gc
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from pathlib import Path

from config import Config


def host_bytes() -> int:
    """RSS of this process and its children (the Dask workers)."""
    import psutil

    proc, total = psutil.Process(), 0
    for p in [proc, *proc.children(recursive=True)]:
        try:
            total += p.memory_info().rss
        except psutil.Error:
            continue
    return total


@cache
def _nvml(ids: tuple[int, ...]) -> tuple:
    try:
        import pynvml

        pynvml.nvmlInit()
        return pynvml, [pynvml.nvmlDeviceGetHandleByIndex(i) for i in ids or range(pynvml.nvmlDeviceGetCount())]
    except Exception:
        return None, []


def gpu_bytes(ids: tuple[int, ...] = ()) -> int:
    """Used memory on GPUs `ids` (physical numbering, as NVML counts; all if empty), by every process."""
    nvml, handles = _nvml(ids)
    return sum(nvml.nvmlDeviceGetMemoryInfo(h).used for h in handles)


def _sync() -> None:
    try:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass


@dataclass
class StepRow:
    label: str
    wall: float
    host: float
    gpu: float


class Report:
    def __init__(self, interval: float = 0.1, gpus: tuple[int, ...] = ()):
        self.interval, self.gpus = interval, gpus
        self.rows: list[StepRow] = []

    def _memory(self) -> list[int]:
        return [host_bytes(), gpu_bytes(self.gpus)]

    @contextmanager
    def step(self, label: str):
        print(f"  {label:<16} ... ", end="", flush=True)
        gc.collect()
        _sync()
        peak = self._memory()
        stop = threading.Event()

        def sample():
            while not stop.wait(self.interval):
                peak[:] = map(max, peak, self._memory())

        thread = threading.Thread(target=sample, daemon=True)
        start = time.perf_counter()
        thread.start()
        try:
            yield
            _sync()
        finally:
            stop.set()
            thread.join()
        wall = time.perf_counter() - start
        host, gpu = (p / 2**20 for p in map(max, peak, self._memory()))
        self.rows.append(StepRow(label, wall, host, gpu))
        print(f"{wall:7.1f}s   host {host:7.0f} MB   gpu {gpu:7.0f} MB", flush=True)

    def table(self) -> str:
        lines = [f"  {'Step':<16}  {'Wall':>8}  {'Peak host':>12}  {'Peak GPU':>12}"]
        for r in self.rows:
            lines.append(f"  {r.label:<16}  {r.wall:7.1f}s  {r.host:9.0f} MB  {r.gpu:9.0f} MB")
        total = sum(r.wall for r in self.rows)
        host, gpu = max((r.host for r in self.rows), default=0), max((r.gpu for r in self.rows), default=0)
        lines.append(f"  {'TOTAL':<16}  {total:7.1f}s  {host:9.0f} MB  {gpu:9.0f} MB")
        return "\n".join(lines)

    def write(self, path: Path, header: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as fh:
            fh.write(f"{header}\n{'=' * 80}\n{self.table()}\n\n")


def header(cfg, n_gpus: int, written: str = "") -> str:
    changed = ", ".join(f"{k}={v}" for k, v in cfg.changed().items() if k != "input") or "(defaults)"
    return (f"process  {datetime.now().isoformat(timespec='seconds')}  input: {cfg.input}  GPUs: {n_gpus}\n"
            f"settings: {changed}" + (f"\nwritten: {written}" if written else ""))


def set_zarr_config(concurrency: int, max_workers: int) -> None:
    import zarr

    zarr.config.set({"async.concurrency": concurrency, "threading.max_workers": max_workers})


class GpuCluster:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cluster = self.client = None

    def start(self) -> None:
        import cupy as cp
        import dask
        import rmm
        from dask.distributed import Client
        from dask_cuda import LocalCUDACluster
        from rmm.allocators.cupy import rmm_cupy_allocator

        cfg = self.cfg
        dask.config.set({"distributed.scheduler.worker-ttl": None})
        kw = dict(protocol="tcp", threads_per_worker=cfg.threads_per_worker, enable_cudf_spill=cfg.enable_cudf_spill,
                  rmm_allocator_external_lib_list="cupy")
        if cfg.rmm_mode == "managed":
            kw["rmm_managed_memory"] = True
        elif cfg.rmm_mode == "pool":
            kw["rmm_pool_size"] = cfg.rmm_pool_size
        self.cluster = LocalCUDACluster(**kw)
        self.client = Client(self.cluster)
        self.client.run(set_zarr_config, cfg.zarr_concurrency, cfg.zarr_max_workers)
        rmm.reinitialize(managed_memory=cfg.rmm_mode == "managed")
        cp.cuda.set_allocator(rmm_cupy_allocator)

    @property
    def n_workers(self) -> int:
        return len(self.client.scheduler_info()["workers"]) if self.client else 1

    def close(self) -> None:
        """Frees every worker's GPU memory; later dask work runs in this process on its first GPU."""
        for part in (self.client, self.cluster):
            try:
                if part is not None:
                    part.close()
            except Exception as exc:
                print(f"cluster shutdown: {exc}")
        self.client = self.cluster = None
