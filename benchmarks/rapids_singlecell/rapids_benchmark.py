#!/usr/bin/env python
"""Configurable per-step benchmark for the rapids-singlecell GPU pipeline.

Measures wall-clock + peak host RSS (process tree) + peak GPU VRAM (device-wide,
NVML) for each stage of an out-of-core, multi-GPU single-cell pipeline reading a
Zarr store lazily via Dask-CUDA.

Steps measured:
  1. load_zarr      lazy zarr -> AnnData (X = dask array of sparse blocks)
  2. h2d_transfer   anndata_to_GPU (block structure preserved; one block/GPU)
  3. preprocessing  calculate_qc_metrics + normalize_total + log1p
  4. hvg            highly_variable_genes + subset + rechunk/persist
  5. scaling        (optional float64 cast) + scale
  6. pca            pca + materialize X_pca to host
  7. harmony        harmony_integrate -> X_pca_harmony  (only if --batch-key set)
  8. neighbors      neighbors
  9. umap           umap
 10. leiden         leiden
 11. write_results  (default on) append the UMAP embedding (obsm/X_umap) + leiden
                    labels (obs/leiden) onto the zarr store — anndata-readable, and
                    crucially no X rematerialization; skip with --no-write-results

Everything that was hardcoded is now a knob: GPUs, zarr read concurrency/threads,
dask-cuda threads/protocol/RMM, chunk size, data path, and the pipeline params.
Defaults live in the `Config` dataclass; every field has a matching `--flag`, so
you can script a sweep of near-identical runs without editing source:

    # 4-GPU baseline, capacity preset (tcp + managed memory)
    pixi run python rapids_benchmark.py \
        --data-path /home/workspace/temp/2M_50M.zarr --gpus 0,1,2,3 \
        --label 2M_4gpu

    # same store, single GPU, smaller chunk, more zarr read threads per worker
    pixi run python rapids_benchmark.py \
        --data-path /home/workspace/temp/2M_50M.zarr --gpus 0 \
        --chunk-rows 12000 --zarr-max-workers 8 --label 2M_1gpu

    # speed preset (ucx + rmm pool) with a hard protocol override
    pixi run python rapids_benchmark.py --gpus 0,1,2,3 --preset speed \
        --protocol tcp --rmm-pool-size 0.7

Key correctness notes (see CLAUDE.md):
  * ZARR CONFIG REACHES THE WORKERS. `zarr.config` is a *runtime* (donfig) setting,
    not an env var, so setting it in the client process does NOT propagate to the
    dask-cuda worker processes — and the lazy chunk reads happen ON the workers.
    We apply it on every worker via `client.run(_set_zarr_config, ...)` (and on the
    client too). Env-var thread pins (OMP/BLAS below) DO inherit, because workers
    are spawned as child processes.
  * THREADS MULTIPLY. host decode budget ≈ n_gpus × threads_per_worker ×
    zarr.threading.max_workers. The out-of-core doc recommends threads_per_worker=1
    for GPU work (more threads spike VRAM); we log the effective product.
  * GPU SELECTION IS SINGLE-SOURCED. `--gpus` is a list of PHYSICAL device ids.
    NVML enumerates physical devices (ignores CUDA_VISIBLE_DEVICES), so it uses the
    physical ids directly; the cluster and client-RMM use the same list. Keeping the
    client unrestricted (it sees all GPUs) is what keeps the three consistent.

Environment: GPU/CUDA-only (rapids-singlecell, dask-cuda, cupy, rmm). Not runnable
on a CPU box; run it in the GPU node's pixi env. `--help` works anywhere (heavy
deps are imported lazily inside functions).
"""
from __future__ import annotations

import argparse
import gc
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from datetime import datetime

# Pin host BLAS/OpenMP threads so N worker processes don't each spawn N BLAS
# threads (CLAUDE.md silent perf killer #1). These are ENV VARS, read at library
# init in each process — set here at module top so the dask-cuda worker children
# inherit them. Deliberate + recorded in provenance. Host BLAS is not the GPU
# pipeline's bottleneck, so pinning to 1 costs ~nothing and prevents N×N contention.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")


# ── configuration ─────────────────────────────────────────────────────────────
@dataclass
class Config:
    """Every knob for one run. Defaults = the tuned baseline; override any field on
    the CLI (field `foo_bar` -> `--foo-bar`)."""

    # -- GPUs / dask-cuda cluster --
    gpus: str = "0,1,2,3"              # physical device ids (drives cluster+NVML+client RMM)
    preset: str = "capacity"           # "capacity" (tcp+managed) | "speed" (ucx+rmm pool)
    protocol: str | None = None        # hard override of preset transport ("tcp"|"ucx")
    rmm_mode: str | None = None        # hard override: "managed" | "pool" | "none"
    rmm_pool_size: float = 0.8         # per-worker pool as fraction of device mem (pool mode only)
    threads_per_worker: int = 1        # dask-cuda worker threadpool (GPU-safe default=1)
    enable_cudf_spill: bool = True     # spill cuDF (obs/var) from VRAM to host under pressure

    # -- zarr read path (applied on EVERY worker + client) --
    zarr_concurrency: int = 4          # async.concurrency (dispatch semaphore)
    zarr_max_workers: int = 4          # threading.max_workers (decode threadpool)

    # -- data / layout --
    data_path: str = "/home/workspace/temp/2M_50M_pbmc.zarr"
    chunk_rows: int = 24_000           # row block for read (multiple of the store's row chunk)

    # -- result write-back (UMAP embedding + leiden labels onto the zarr, no h5ad) --
    write_results: bool = True         # append obsm/X_umap + obs/leiden to the store as a final step
    results_store: str = ""            # target store; "" -> write back into data_path (a layer on the input)
    umap_key: str = "X_umap"           # obsm key for the embedding (anndata default)
    leiden_key: str = "leiden"         # obs column for the labels (anndata default)

    # -- pipeline params --
    n_top_genes: int = 2000
    n_comps: int = 30                  # PCA comps (also n_pcs for neighbors)
    n_neighbors: int = 20
    neighbors_algorithm: str = "mg_ivfflat"  # brute|ivfflat|ivfpq|mg_ivfflat|mg_ivfpq
    leiden_resolution: float = 1.1
    leiden_iterations: int = 100
    umap_min_dist: float = 0.45
    pca_float64: bool = True           # cast X to float64 before scale for stable std
    batch_key: str = "pool_id"         # obs column for harmony; "" disables the harmony step
    random_seed: int = 5671

    # -- sampling / output --
    sample_interval_s: float = 0.02    # memory poll interval (GPU changes fast)
    results_txt: str = "results/Run_results.txt"  # human-readable append log ("" to skip)
    label: str = ""                    # free tag shown in the run header/log

    @property
    def gpu_ids(self) -> list[int]:
        return [int(x) for x in str(self.gpus).split(",") if x != ""]

    def resolve_cluster(self) -> tuple[str, str]:
        """(protocol, rmm_mode) after applying the preset then any hard override."""
        presets = {"capacity": ("tcp", "managed"), "speed": ("ucx", "pool")}
        protocol, rmm_mode = presets.get(self.preset, presets["capacity"])
        if self.protocol:
            protocol = self.protocol
        if self.rmm_mode:
            rmm_mode = self.rmm_mode
        return protocol, rmm_mode


def parse_config(argv=None) -> Config:
    """Build a Config from CLI. The dataclass defaults are the single source of
    truth; a flag only overrides when it is actually passed (default sentinel None)."""
    base = Config()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    bool_fields = {"pca_float64", "enable_cudf_spill", "write_results"}
    for f in fields(Config):
        flag = "--" + f.name.replace("_", "-")
        default = getattr(base, f.name)
        # escape % (e.g. rmm_pool_size '80%') so argparse doesn't treat it as a format spec
        helptxt = f"(default: {default!r})".replace("%", "%%")
        if f.name in bool_fields:
            p.add_argument(flag, dest=f.name, action=argparse.BooleanOptionalAction,
                           default=None, help=helptxt)
        elif f.type == "int":
            p.add_argument(flag, dest=f.name, type=int, default=None, help=helptxt)
        elif f.type == "float":
            p.add_argument(flag, dest=f.name, type=float, default=None, help=helptxt)
        else:  # str / str | None
            p.add_argument(flag, dest=f.name, type=str, default=None, help=helptxt)
    args = p.parse_args(argv)
    overrides = {k: v for k, v in vars(args).items() if v is not None}
    return replace(base, **overrides)


# ── module state populated at runtime (kept simple for the sampler thread) ─────
_NVML_OK = False
_GPU_HANDLES: list = []
_PROC = None
_SAMPLE_INTERVAL = 0.02
# (label, wall_s, peak_host_mb, peak_gpu_mb)
RESULTS: list[tuple[str, float, float, float]] = []


# ── memory probes ─────────────────────────────────────────────────────────────
def _host_rss_tree() -> int:
    """Sum RSS of this process + all descendants (dask-cuda workers)."""
    import psutil
    global _PROC
    if _PROC is None:
        _PROC = psutil.Process()
    total = _PROC.memory_info().rss
    for child in _PROC.children(recursive=True):
        try:
            total += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def _gpu_vram_used() -> int:
    """Total used VRAM across the selected GPUs (device-wide, all PIDs). Per-PID
    NVML attribution is fragile across dask-cuda workers + RMM pools; on a dedicated
    benchmark box the device-wide number is what you actually care about."""
    import pynvml
    if not _NVML_OK:
        return 0
    total = 0
    for h in _GPU_HANDLES:
        try:
            total += pynvml.nvmlDeviceGetMemoryInfo(h).used
        except pynvml.NVMLError:
            continue
    return total


def init_nvml(physical_ids: list[int]) -> None:
    """Init NVML and grab handles by PHYSICAL index (NVML ignores CUDA_VISIBLE_DEVICES)."""
    global _NVML_OK, _GPU_HANDLES
    try:
        import pynvml
        pynvml.nvmlInit()
        _GPU_HANDLES = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_ids]
        _NVML_OK = True
    except Exception as e:  # NVML absent/blocked -> VRAM reports 0, benchmark still runs
        print(f"WARNING: pynvml init failed ({e}); GPU memory will report 0.")
        _NVML_OK = False


class _Sampler:
    """Polls host RSS (process tree) and GPU VRAM in a background thread."""

    def __init__(self, interval: float | None = None):
        self.interval = interval if interval is not None else _SAMPLE_INTERVAL
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.peak_host = _host_rss_tree()
        self.peak_gpu = _gpu_vram_used()

    def _run(self):
        while not self._stop.is_set():
            self.peak_host = max(self.peak_host, _host_rss_tree())
            self.peak_gpu = max(self.peak_gpu, _gpu_vram_used())
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak_host = max(self.peak_host, _host_rss_tree())
        self.peak_gpu = max(self.peak_gpu, _gpu_vram_used())


def _cuda_sync():
    """Block until all queued GPU work on the default stream finishes (kernels are
    async — sync so the clock reflects real GPU work, not just launch)."""
    import cupy as cp
    cp.cuda.Stream.null.synchronize()


def _drain_cupy_pools():
    """Return cached VRAM blocks to the driver so the next step starts from a clean
    high-water mark. RMM-managed allocations may still be pooled by RMM itself; this
    only drains cupy's own pools."""
    import cupy as cp
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


@contextmanager
def step(label: str):
    print(f"  {label:<16} ... ", end="", flush=True)
    gc.collect()
    _drain_cupy_pools()
    _cuda_sync()  # finish any prior queued work first

    sampler = _Sampler()
    t0 = time.perf_counter()
    with sampler:
        yield
        _cuda_sync()  # ensure this step's GPU work is done before stopping the clock
    wall = time.perf_counter() - t0

    host_mb = sampler.peak_host / (1024 * 1024)
    gpu_mb = sampler.peak_gpu / (1024 * 1024)
    RESULTS.append((label, wall, host_mb, gpu_mb))
    print(f"{wall:7.2f}s   host {host_mb:7.0f} MB   gpu {gpu_mb:7.0f} MB")


# ── cluster / device setup ────────────────────────────────────────────────────
def make_cluster(cfg: Config, protocol: str, rmm_mode: str):
    """Build the LocalCUDACluster. `rmm_allocator_external_lib_list='cupy'` matches
    the dask-cuda version on the GPU node (the out-of-core doc's singular
    `rmm_allocator_external_lib` is a different release)."""
    from dask_cuda import LocalCUDACluster
    kw = dict(
        CUDA_VISIBLE_DEVICES=cfg.gpus,
        protocol=protocol,
        threads_per_worker=cfg.threads_per_worker,
        rmm_allocator_external_lib_list="cupy",
        enable_cudf_spill=cfg.enable_cudf_spill,
    )
    if rmm_mode == "managed":
        kw["rmm_managed_memory"] = True
    elif rmm_mode == "pool":
        kw["rmm_pool_size"] = cfg.rmm_pool_size
    # rmm_mode == "none": leave RMM allocator kwargs off entirely
    return LocalCUDACluster(**kw)


def init_client_rmm(physical_ids: list[int], rmm_mode: str) -> None:
    """Configure RMM for the CLIENT process (which gathers X_pca via .compute()).
    Kept consistent with the workers' RMM mode."""
    import cupy as cp
    import rmm
    from rmm.allocators.cupy import rmm_cupy_allocator
    rmm.reinitialize(
        managed_memory=(rmm_mode == "managed"),
        pool_allocator=(rmm_mode == "pool"),
        devices=physical_ids,
    )
    cp.cuda.set_allocator(rmm_cupy_allocator)
    print("client RMM resource:", type(rmm.mr.get_current_device_resource()).__name__)


def _set_zarr_config(concurrency: int, max_workers: int) -> None:
    """Module-level so it is picklable for `client.run`. Runs on each worker (and
    the client) to size that process's zarr dispatch + decode pools."""
    import zarr
    zarr.config.set({
        "async.concurrency": concurrency,
        "threading.max_workers": max_workers,
    })


# ── run header (date + the knobs the user actually set) ─────────────────────────
def _changed_cfg(cfg: Config) -> dict:
    """Config fields the user actually overrode (value != dataclass default). The store,
    GPUs, and label are surfaced on their own line, so drop them from this diff."""
    base = Config()
    shown = {"data_path", "gpus", "label"}
    return {f.name: getattr(cfg, f.name) for f in fields(cfg)
            if f.name not in shown and getattr(cfg, f.name) != getattr(base, f.name)}


def provenance(cfg: Config) -> dict:
    """Just what identifies/parameterises this run: timestamp + the user's inputs (store,
    GPUs, and any cfg field left off its default). No auto-collected env facts."""
    return {
        "label": cfg.label,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_path": cfg.data_path,
        "gpus": cfg.gpu_ids,
        "changed": _changed_cfg(cfg),
    }


def _format_header(prov: dict) -> str:
    """The run header shared by stdout and the Run_results.txt append log."""
    changed = prov["changed"]
    changed_str = ", ".join(f"{k}={v}" for k, v in changed.items()) if changed else "(defaults)"
    return (
        f"Rapids-Singlecell Benchmark  —  {prov['timestamp']}  —  label: {prov['label'] or '-'}\n"
        f"store: {prov['data_path']}  —  GPUs: {prov['gpus']}\n"
        f"changed cfg: {changed_str}"
    )


# ── result write-back (UMAP + leiden -> zarr, anndata-readable) ─────────────────
def _write_results(adata, cfg: Config) -> None:
    """Writing the UMAP and Leiden clustering results.

    Target defaults to `data_path` (add a layer onto the input store); `--results-store`
    points it elsewhere. `write_elem` overwrites existing keys, so re-runs are idempotent."""
    import numpy as np
    import zarr
    from anndata.io import write_elem

    store = cfg.results_store or cfg.data_path

    umap = adata.obsm[cfg.umap_key]
    if hasattr(umap, "get"):            # cupy -> host (rsc already gathers; belt-and-braces)
        umap = umap.get()
    umap = np.asarray(umap)
    leiden = adata.obs[cfg.leiden_key].values
         
    #Check if consolidated. Important when reclosing the store
    was_consolidated = (
        zarr.open_group(store, mode="r").metadata.consolidated_metadata is not None
    )
    root = zarr.open_group(store, mode="r+", use_consolidated=False)

    obsm = root["obsm"] if "obsm" in root else root.create_group("obsm")
    obsm.attrs["encoding-type"] = "dict"
    obsm.attrs["encoding-version"] = "0.1.0"
    write_elem(obsm, cfg.umap_key, umap)

    obs = root["obs"]
    write_elem(obs, cfg.leiden_key, leiden)
    order = list(obs.attrs.get("column-order", []))
    if cfg.leiden_key not in order:
        obs.attrs["column-order"] = order + [cfg.leiden_key]

    if was_consolidated:
        zarr.consolidate_metadata(store)

    print(f"  results -> {store}: obsm/{cfg.umap_key} + obs/{cfg.leiden_key}"
          + ("  (metadata re-consolidated)" if was_consolidated else ""))


# ── the pipeline ──────────────────────────────────────────────────────────────
def run_pipeline(cfg: Config) -> None:
    import numpy as np
    import anndata as ad
    import zarr
    import cupy as cp
    import rapids_singlecell as rsc
    from anndata.experimental import read_elem_lazy as read_dask

    n_gpus = max(1, len(cfg.gpu_ids))

    # Warm the CUDA context so its 1-3s init isn't charged to the first step.
    print("  (warming CUDA context...)")
    cp.zeros(1, dtype=cp.float32)
    _cuda_sync()

    with step("load_zarr"):
        f = zarr.open(cfg.data_path)
        X = f["X"]
        shape = tuple(X.attrs["shape"]) if "shape" in getattr(X, "attrs", {}) else tuple(X.shape)
        X_dask = read_dask(X, (cfg.chunk_rows, shape[1]))
        if np.issubdtype(X_dask.dtype, np.integer):
            X_dask = X_dask.astype(np.float32)
        adata = ad.AnnData(
            X=X_dask,                       # (chunk_rows, all genes) blocks; align to zarr row chunk
            obs=ad.io.read_elem(f["obs"]),
            var=ad.io.read_elem(f["var"]),
        )

    with step("h2d_transfer"):
        rsc.get.anndata_to_GPU(adata)       # lazy: map_blocks, still block-structured

    with step("preprocessing"):
        rsc.pp.calculate_qc_metrics(adata)
        rsc.pp.normalize_total(adata)
        rsc.pp.log1p(adata)

    with step("hvg"):
        rsc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=cfg.n_top_genes)
        hvg_mask = np.asarray(adata.var["highly_variable"])
        adata = adata[:, hvg_mask].copy()
        n_rows, n_cols = adata.shape
        rows_per_worker = (n_rows + n_gpus - 1) // n_gpus   # one even band per GPU
        adata.X = adata.X.rechunk((rows_per_worker, n_cols)).persist()
        adata.X.compute_chunk_sizes()

    with step("scaling"):
        if cfg.pca_float64 and hasattr(adata.X, "astype"):
            adata.X = adata.X.astype("float64")             # stable std; matters here
        rsc.pp.scale(adata, max_value=10, zero_center=False)

    with step("pca"):
        rsc.pp.pca(adata, n_comps=cfg.n_comps, svd_solver="covariance_eigh",
                   random_state=cfg.random_seed)
        adata.obsm["X_pca"] = adata.obsm["X_pca"].persist()
        adata.obsm["X_pca"].compute_chunk_sizes()
        adata.obsm["X_pca"] = adata.obsm["X_pca"].compute()

    if cfg.batch_key:
        with step("harmony"):
            adata.obs[cfg.batch_key] = adata.obs[cfg.batch_key].astype("category")
            rsc.pp.harmony_integrate(adata, key=cfg.batch_key, basis="X_pca",
                                     adjusted_basis="X_pca_harmony")

    with step("neighbors"):
        rsc.pp.neighbors(adata, n_neighbors=cfg.n_neighbors, n_pcs=cfg.n_comps,
                         algorithm=cfg.neighbors_algorithm, random_state=cfg.random_seed)

    with step("umap"):
        rsc.tl.umap(adata, min_dist=cfg.umap_min_dist, init_pos="spectral",
                    n_components=2, random_state=cfg.random_seed)

    with step("leiden"):
        rsc.tl.leiden(adata, resolution=cfg.leiden_resolution,
                      n_iterations=cfg.leiden_iterations, random_state=cfg.random_seed)

    if cfg.write_results:
        with step("write_results"):
            _write_results(adata, cfg)


# ── report ────────────────────────────────────────────────────────────────────
def report(cfg: Config, prov: dict) -> None:
    total_wall = sum(r[1] for r in RESULTS)
    overall_host = max((r[2] for r in RESULTS), default=0.0)
    overall_gpu = max((r[3] for r in RESULTS), default=0.0)

    lines = [
        f"  {'Step':<16}  {'Wall':>8}  {'Peak Host':>12}  {'Peak GPU':>12}",
        f"  {'----':<16}  {'----':>8}  {'---------':>12}  {'--------':>12}",
    ]
    for label, wall, host_mb, gpu_mb in RESULTS:
        lines.append(f"  {label:<16}  {wall:7.2f}s  {host_mb:9.0f} MB  {gpu_mb:9.0f} MB")
    lines.append(f"  {'-' * 16}  {'-' * 8}  {'-' * 12}  {'-' * 12}")
    lines.append(f"  {'TOTAL':<16}  {total_wall:7.2f}s  {overall_host:9.0f} MB  {overall_gpu:9.0f} MB")
    summary = "\n".join(lines)
    print("\n" + summary)

    if cfg.results_txt:
        os.makedirs(os.path.dirname(cfg.results_txt) or ".", exist_ok=True)
        header = _format_header(prov)
        with open(cfg.results_txt, "a") as fh:
            fh.write(header + "\n" + "=" * 80 + "\n" + summary + "\n\n")
        print(f"Summary appended to: {cfg.results_txt}")


# ── entrypoint ────────────────────────────────────────────────────────────────
def main(cfg: Config) -> None:
    global _SAMPLE_INTERVAL
    _SAMPLE_INTERVAL = cfg.sample_interval_s

    import dask
    from dask.distributed import Client
    # Don't let the scheduler kill idle workers during long non-dask steps.
    dask.config.set({"distributed.scheduler.worker-ttl": None})

    physical_ids = cfg.gpu_ids
    protocol, rmm_mode = cfg.resolve_cluster()
    init_nvml(physical_ids)

    cluster = make_cluster(cfg, protocol, rmm_mode)
    client = Client(cluster)
    try:
        # Push the zarr read config to EVERY worker (donfig is not inherited across
        # processes), then apply it on the client for its own eager obs/var read.
        client.run(_set_zarr_config, cfg.zarr_concurrency, cfg.zarr_max_workers)
        _set_zarr_config(cfg.zarr_concurrency, cfg.zarr_max_workers)

        init_client_rmm(physical_ids, rmm_mode)

        prov = provenance(cfg)
        print()
        print(_format_header(prov))
        print("=" * 80)

        run_pipeline(cfg)
        report(cfg, prov)
    finally:
        # dask-cuda workers keep heartbeating; stop it before shutdown to avoid noise.
        def _stop_heartbeat(dask_worker):
            try:
                dask_worker.periodic_callbacks["heartbeat"].stop()
            except Exception:
                pass
        for fn in (lambda: client.run(_stop_heartbeat), client.shutdown):
            try:
                fn()
            except Exception:
                pass
        if _NVML_OK:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass


if __name__ == "__main__":
    # https://rapids-singlecell.readthedocs.io/en/latest/out_of_core.html
    main(parse_config())
