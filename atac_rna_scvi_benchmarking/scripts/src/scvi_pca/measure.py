"""Stage timer: GPU-synchronised wall time plus peak VRAM (NVML, device-wide) and host RSS (process tree)."""
import contextlib
import gc
import sys
import time

import pandas as pd

from . import paths

paths.add()
import rapids_benchmark as rb  # noqa: E402

INTERVAL_S = 0.2


def init_nvml(gpu=0):
    rb.init_nvml([gpu])


def sync():
    """Drain queued GPU work on cupy and (only if already imported) torch."""
    if "cupy" in sys.modules:
        sys.modules["cupy"].cuda.runtime.deviceSynchronize()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


@contextlib.contextmanager
def time_first_wait():
    """Yields a dict whose 's' becomes the time until the first dask.distributed.wait returns —
    the service's one full pass over X (read + transfer) on the lazy path."""
    import dask.distributed as dd
    orig, t0, out = dd.wait, time.perf_counter(), {"s": 0.0}

    def wrapped(*a, **k):
        r = orig(*a, **k)
        if not out["s"]:
            sync()
            out["s"] = time.perf_counter() - t0
        return r
    dd.wait = wrapped
    try:
        yield out
    finally:
        dd.wait = orig


class _TracingSampler(rb._Sampler):
    """The benchmark's _Sampler, additionally keeping every (t, rss, vram) sample."""

    def __init__(self, t_origin, interval=INTERVAL_S):
        super().__init__(interval)
        self.t_origin, self.samples = t_origin, []

    def _run(self):
        while not self._stop.is_set():
            host, gpu = rb._host_rss_tree(), rb._gpu_vram_used()
            self.peak_host, self.peak_gpu = max(self.peak_host, host), max(self.peak_gpu, gpu)
            self.samples.append((time.perf_counter() - self.t_origin, host, gpu))
            self._stop.wait(self.interval)


class Recorder:
    """Collects per-stage results and raw samples for one run."""

    def __init__(self):
        self.t_origin = time.perf_counter()
        self.stages, self.samples = {}, []

    def stage(self, name):
        return Stage(self, name)

    def run_peaks(self):
        s = self.stages.values()
        return {"peak_vram_bytes": max((x["peak_vram_bytes"] for x in s), default=0),
                "peak_rss_bytes": max((x["peak_rss_bytes"] for x in s), default=0)}

    def samples_frame(self):
        return pd.DataFrame(self.samples, columns=["stage", "t_s", "rss_bytes", "vram_bytes"])


class Stage:
    """Context manager: `with rec.stage("t_pre"): ...`; results land in rec.stages[name]."""

    def __init__(self, rec, name):
        self.rec, self.name = rec, name

    def __enter__(self):
        gc.collect()
        sync()
        self.sampler = _TracingSampler(self.rec.t_origin)
        self.vram_start = self.sampler.peak_gpu
        self.sampler.__enter__()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        sync()
        wall = time.perf_counter() - self.t0
        self.sampler.__exit__(*exc)
        s = self.sampler
        self.rec.stages[self.name] = {"wall_s": wall, "peak_vram_bytes": s.peak_gpu,
                                      "peak_rss_bytes": s.peak_host, "vram_start_bytes": self.vram_start}
        self.rec.samples += [(self.name, *x) for x in s.samples]
        return False
