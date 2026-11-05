"""The Dask-CUDA cluster: one worker per visible GPU, over tcp, with the chosen RMM memory mode."""
from __future__ import annotations

from config import Config


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
