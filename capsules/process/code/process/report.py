"""Per-step wall time and peak host / GPU memory, printed as each step ends and as a table at the end."""
from __future__ import annotations

import gc
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from functools import cache
from pathlib import Path


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


class Report:
    def __init__(self, interval: float = 0.1, gpus: tuple[int, ...] = ()):
        self.interval, self.gpus = interval, gpus
        self.rows: list[tuple[str, float, float, float]] = []

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
        self.rows.append((label, wall, host, gpu))
        print(f"{wall:7.1f}s   host {host:7.0f} MB   gpu {gpu:7.0f} MB", flush=True)

    def table(self) -> str:
        lines = [f"  {'Step':<16}  {'Wall':>8}  {'Peak host':>12}  {'Peak GPU':>12}"]
        for label, wall, host, gpu in self.rows:
            lines.append(f"  {label:<16}  {wall:7.1f}s  {host:9.0f} MB  {gpu:9.0f} MB")
        total = sum(r[1] for r in self.rows)
        host, gpu = (max((r[i] for r in self.rows), default=0) for i in (2, 3))
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
