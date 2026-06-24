"""Per-step wall-clock + peak host RSS + peak GPU VRAM benchmark for the
rapids-singlecell pipeline.

Steps measured:
  1. load_zarr      (lazy dense zarr -> per-chunk host CSR -> AnnData)
  2. h2d_transfer   (anndata_to_GPU; CSR blocks -> cupyx CSR, non-zeros only)
  3. preprocessing  (calculate_qc_metrics, normalize_total, log1p)
  4. hvg            (highly_variable_genes)
  5. scaling        (scale)
  6. pca            (pca + materialize X_pca to host)
  7. harmony        (harmony_integrate -> X_pca_harmony; optional, BATCH_KEY)
  8. neighbors      (neighbors)
  9. umap           (umap)
 10. leiden         (leiden)

Notes vs. the scanpy CPU benchmark:
  * CUDA kernels are async — we synchronize before/after each step so wall
    time reflects actual GPU work, not just kernel launch.
  * GPU VRAM is invisible to RSS, so we sample it via NVML alongside host RSS.
  * cupy memory pools cache freed VRAM — we drain them between steps so each
    step's peak GPU reflects its own working set, not the pool high-water.
  * First CUDA call initializes the context (~1-3s); we do an explicit warmup
    so it isn't charged to the first real step.
  * dask-cuda spawns one worker process per GPU. Their host RSS is captured
    by the process-tree sampler. Their VRAM is captured device-wide via NVML.

Usage:
    pixi run python rapids_benchmark.py
"""
from __future__ import annotations

import os
import time
import gc
import threading
from contextlib import contextmanager
from datetime import datetime

import psutil

# ── CUDA cluster setup ────────────────────────────────────────────────────────
# Must happen before rapids_singlecell imports so RMM is configured first.
#OTHER CLUSTER SETU UP IN NOTEBOOK
import dask
import dask.array as da
# needed so dask doesnt time out on steps its not used
dask.config.set({"distributed.scheduler.worker-ttl": None})

from dask_cuda import LocalCUDACluster
from dask.distributed import Client, wait
import cupy as cp

import rapids_singlecell as rsc
import anndata as ad
import zarr
from anndata.experimental import read_elem_lazy as read_dask

zarr.config.set({
    'async.concurrency': 16, 
    'threading.max_workers': 16, 
})

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_OK = True
except Exception as e:
    print(f"WARNING: pynvml init failed ({e}); GPU memory will report 0.")
    _NVML_OK = False

# Hardcoded to GPUs 0–3. NVML enumerates ALL physical GPUs, so we map
# explicitly to the same indices passed to LocalCUDACluster.
_GPU_INDICES = [0, 1, 2, 3]
_GPU_HANDLES = (
    [pynvml.nvmlDeviceGetHandleByIndex(i) for i in _GPU_INDICES]
    if _NVML_OK else []
)

######## CONFIG ################
#for now hardocded config ===================================================================
BENCHMARK_FILE = "Run_results.txt"
DATA_PATH = "/home/workspace/temp/2M_Dense_5k.zarr"
H5AD_PATH = ""#"/home/workspace/private/rapids_h5ads/rapids_sc64_16_2M50_24k_S2k_20-30_mgff_45sU_11-100-.h5ad"
PCA_FLOAT64 = True
RANDOM_SEED = 5671
CHUNK_ROWS = 5000
BATCH_KEY = "pool_id"
SAMPLE_INTERVAL_S = 0.02  #gpu changes faster, query more.

_PROC = psutil.Process()


def _write_h5ad(adata, path):
    """Materialize the dask X to a single host CSR (chunk-by-chunk, never
    gathering the full matrix onto one GPU), move the rest of adata to host,
    and write a standard h5ad in one shot.

    Peak host RAM ≈ one full HVG CSR. Peak VRAM ≈ one chunk above whatever
    is already resident.
    """
    import scipy.sparse as sp
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    X = adata.X
    if hasattr(X, "numblocks"):
        parts = []
        for bi in range(X.numblocks[0]):
            ck = X.blocks[bi, 0].compute()
            if hasattr(ck, "get"):              # cupyx CSR → scipy CSR
                ck = ck.get()
            parts.append(ck)
        adata.X = sp.vstack(parts, format="csr")
        del parts
    rsc.get.anndata_to_CPU(adata)
    adata.write_h5ad(path, compression="gzip")
    print(f"  h5ad written: {path}")


#######################################
######## objects that will probe memory usage for processes
##############################
def _host_rss_tree() -> int:
    """Sum RSS of this process + all descendants (dask-cuda workers)."""
    total = _PROC.memory_info().rss
    for child in _PROC.children(recursive=True):
        try:
            total += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def _gpu_vram_used() -> int:
    """Total used VRAM across visible GPUs (device-wide, all PIDs).

    Per-PID NVML attribution is fragile across dask-cuda workers + RMM pools;
    on a dedicated benchmark box the device-wide number is what you actually
    care about anyway.
    """
    if not _NVML_OK:
        return 0
    total = 0
    for h in _GPU_HANDLES:
        try:
            total += pynvml.nvmlDeviceGetMemoryInfo(h).used
        except pynvml.NVMLError:
            continue
    return total


################################
#Build sampler to run the above probes for each sc step
###########################
class _Sampler:
    """Polls host RSS (process tree) and GPU VRAM in a background thread."""

    def __init__(self, interval: float = SAMPLE_INTERVAL_S):
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.peak_host = _host_rss_tree()
        self.peak_gpu = _gpu_vram_used()

    def _run(self):
        while not self._stop.is_set():
            h = _host_rss_tree()
            g = _gpu_vram_used()
            if h > self.peak_host:
                self.peak_host = h
            if g > self.peak_gpu:
                self.peak_gpu = g
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        h = _host_rss_tree()
        g = _gpu_vram_used()
        if h > self.peak_host:
            self.peak_host = h
        if g > self.peak_gpu:
            self.peak_gpu = g


###########################
###interlude the sampler and a step function, call step function for each sc step#
########################
# (label, wall_s, peak_host_mb, peak_gpu_mb)
RESULTS: list[tuple[str, float, float, float]] = []

def _cuda_sync():
    """Block until all queued GPU work on the default stream finishes."""
    cp.cuda.Stream.null.synchronize()

def _drain_cupy_pools():
    """Return cached VRAM blocks to the driver so the next step starts from a
    clean high-water mark. RMM-managed allocations may still be pooled by RMM
    itself; this only drains cupy's own pools."""
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
    _cuda_sync()# finish any prior queued work first

    sampler = _Sampler()
    t0 = time.perf_counter()
    with sampler:
        yield
        _cuda_sync() # ensure step's GPU work is done before stopping clock
    wall = time.perf_counter() - t0

    host_mb = sampler.peak_host / (1024 * 1024)
    gpu_mb = sampler.peak_gpu / (1024 * 1024)
    RESULTS.append((label, wall, host_mb, gpu_mb))
    print(f"{wall:7.2f}s   host {host_mb:7.0f} MB   gpu {gpu_mb:7.0f} MB")




####################
#####main
##############
def main():
    
    import rmm
    from rmm.allocators.cupy import rmm_cupy_allocator
    rmm.reinitialize(
        managed_memory=True,
        pool_allocator=False,    
        devices=[0, 1, 2, 3]                 
    )
    cp.cuda.set_allocator(rmm_cupy_allocator)
    
    print("client RMM resource:", type(rmm.mr.get_current_device_resource()).__name__)

    
    """Call the step context manager for each process that we want to bench mark individually.
    Preprocessing will all be tied together"""
    header = (
        f"Rapids-Singlecell Benchmark  —  {datetime.now():%Y-%m-%d %H:%M}  —  "
        f"input: {DATA_PATH}  —  GPUs: {_GPU_INDICES}"
    )
    divider = "=" * 80

    print()
    print(header)
    print(divider)
    print()
    print(f"  {'Step':<16}  {'Wall':>8}  {'Peak Host':>12}  {'Peak GPU':>12}")
    print(f"  {'----':<16}  {'----':>8}  {'---------':>12}  {'--------':>12}")

    # Warm CUDA context so its 1-3s init isn't charged to the first step.
    print("  (warming CUDA context...)")
    cp.zeros(1, dtype=cp.float32)
    _cuda_sync()


    #################################
    ##For now just load the full zarr lazily, #TODO load certain cell type or other filter of larger dataset
    ##################################
    with step("load_zarr"):
        import scipy.sparse as sp

        f = zarr.open(DATA_PATH, mode="r")

        # Read the dense X as a lazy Dask array (full-width row-band chunks),
        # then sparsify each block to a scipy CSR on the host. Staying lazy keeps
        # peak host RAM ≈ one dense chunk per worker: each block is read +
        # decompressed, converted to CSR, and consumed downstream — the full
        # host matrix is never gathered. Because each block's _meta is a scipy
        # csr_matrix, the h2d_transfer step (anndata_to_GPU -> X_to_GPU) maps
        # every block to a cupyx CSR, so only the non-zeros (data/indices/indptr)
        # cross the PCIe bus instead of the dense block (zeros included).
        X_dask = da.from_zarr(f["X"], chunks=(CHUNK_ROWS, f["X"].shape[1]))
        X_dask = X_dask.map_blocks(
            sp.csr_matrix,
            meta=sp.csr_matrix((0, 0), dtype=X_dask.dtype),
        )

        adata = ad.AnnData(
            X=X_dask,
            obs=ad.io.read_elem(f["obs"]),
            var=ad.io.read_elem(f["var"]),
        )

        print(adata)
        print(f"X type: {type(adata.X)}")                       # dask.array.Array
        print(f"X block type: {type(adata.X._meta).__name__}")  # csr_matrix
        print(f"X dtype: {adata.X.dtype}")                      # float32
        print(f"X chunks: {adata.X.chunks}")

    #Loads dask array to gpu, just converts it to be ready to be loaded, still a lazy load
    with step("h2d_transfer"):
        rsc.get.anndata_to_GPU(adata)

    with step("preprocessing"):
        rsc.pp.calculate_qc_metrics(adata)
        rsc.pp.normalize_total(adata)
        rsc.pp.log1p(adata)

    with step("hvg"):
        rsc.pp.highly_variable_genes(
            adata, flavor="seurat", n_top_genes=2000,
        )
        # adata_full = adata                                      #To save full adata instead of just hvg, see write_h5ad step for other part
        
        import numpy as _np_hvg
        hvg_mask = _np_hvg.asarray(adata.var["highly_variable"])
        adata = adata[:, hvg_mask].copy()

        n_rows = adata.shape[0]
        n_cols = adata.shape[1]
        rows_per_worker = (n_rows+4-1)//4
        adata.X = adata.X.rechunk((rows_per_worker, n_cols)).persist()
        adata.X.compute_chunk_sizes()

    with step("scaling"):
        if PCA_FLOAT64 and hasattr(adata.X, "astype"):
            adata.X = adata.X.astype("float64")     # stable std; this is where it matters
        rsc.pp.scale(adata, max_value=10, zero_center=False)

    with step("pca"):
        rsc.pp.pca(adata, n_comps=30, svd_solver='covariance_eigh', random_state=RANDOM_SEED) # Covariance_eigh with dask https://rapids-singlecell.readthedocs.io/en/latest/api/generated/rapids_singlecell.pp.pca.html#rapids_singlecell.pp.pca
        
        adata.obsm["X_pca"]=adata.obsm["X_pca"].persist()
        adata.obsm["X_pca"].compute_chunk_sizes()
        adata.obsm["X_pca"]=adata.obsm["X_pca"].compute()


    rep = "X_pca"
    if BATCH_KEY:
        with step("harmony"):
            adata.obs[BATCH_KEY] = adata.obs[BATCH_KEY].astype("category")
            rsc.pp.harmony_integrate(
                adata, key=BATCH_KEY,
                basis="X_pca", adjusted_basis="X_pca_harmony",
                #flavor="harmony1"
            )
        rep = "X_pca_harmony"
    

    with step("neighbors"):
        rsc.pp.neighbors(
            ##'brute' alg best but long, 'mg_ivfflat', 'mg_ivfpq' for multi-gpu. ivfflat more accurate, ivgpq max speed
            adata, n_neighbors=20, n_pcs=30, algorithm="mg_ivfflat", random_state=RANDOM_SEED 
        )

    with step("umap"):
        rsc.tl.umap(adata, min_dist=0.45, init_pos='spectral', n_components=2, random_state=RANDOM_SEED)

    with step("leiden"):
        rsc.tl.leiden(adata, resolution=1.1, n_iterations=100, random_state=RANDOM_SEED)#, use_dask=True)

    if H5AD_PATH:
        with step("write_h5ad"):
            #when writing full adata, add embeddings to object first
            # adata_full.obs  = adata.obs    # includes leiden
            # adata_full.obsm = adata.obsm   # X_pca, X_umap
            # adata_full.obsp = adata.obsp   # neighbors graph
            # adata_full.uns  = adata.uns    # neighbors/umap/leiden/t-test
            # _write_h5ad(adata_full, H5AD_PATH)
            _write_h5ad(adata, H5AD_PATH)

    # Summary
    total_wall = sum(r[1] for r in RESULTS)
    overall_host = max(r[2] for r in RESULTS)
    overall_gpu = max(r[3] for r in RESULTS)

    lines = []
    lines.append(f"H5AD path: {H5AD_PATH}")
    lines.append(divider)
    lines.append("")
    lines.append(f"  {'Step':<16}  {'Wall':>8}  {'Peak Host':>12}  {'Peak GPU':>12}")
    lines.append(f"  {'----':<16}  {'----':>8}  {'---------':>12}  {'--------':>12}")
    for label, wall, host_mb, gpu_mb in RESULTS:
        lines.append(
            f"  {label:<16}  {wall:7.2f}s  {host_mb:9.0f} MB  {gpu_mb:9.0f} MB"
        )
    lines.append(f"  {'-' * 16}  {'-' * 8}  {'-' * 12}  {'-' * 12}")
    lines.append(
        f"  {'TOTAL':<16}  {total_wall:7.2f}s  {overall_host:9.0f} MB  {overall_gpu:9.0f} MB"
    )
    lines.append("")

    summary = "\n".join(lines)
    print(summary)

    if BENCHMARK_FILE:
        with open(BENCHMARK_FILE, "a") as f:
            f.write(header + "\n")
            f.write(divider + "\n")
            f.write(summary + "\n")
        print(f"Results appended to: {BENCHMARK_FILE}")


if __name__ == "__main__":
    # https://rapids-singlecell.readthedocs.io/en/latest/out_of_core.html
    cluster = LocalCUDACluster(
        CUDA_VISIBLE_DEVICES="0,1,2,3", 
        protocol="tcp",  
        threads_per_worker=4,   #change based on data size, 16 worked for 5M cell set but was close to max gpu memory. 
        rmm_managed_memory=True,    
        rmm_allocator_external_lib_list="cupy",
    )
    client = Client(cluster)
    try:
        main()
    finally:
        def _stop_heartbeat(dask_worker):
            try:
                dask_worker.periodic_callbacks["heartbeat"].stop()
            except Exception:
                pass
        try:
            client.run(_stop_heartbeat)
        except Exception:
            pass
        try:
            client.shutdown()
        except Exception:
            pass
        if _NVML_OK:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
