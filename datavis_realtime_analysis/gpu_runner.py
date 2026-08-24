"""Warm GPU runner for datavis "generate new UMAP" jobs.

Started once by the app backend over ssh (auto-restarts on stale heartbeat).
Imports the RAPIDS stack and caches obs/var from RUNNER_DATA at warm-up, then
loops on /tmp/datavis_jobs/<id>.json. Each job runs the eager single-GPU
pipeline and uploads the view store; stage lines plus a terminal
done/failed/fallback line stream to /tmp/datavis_jobs/<id>.log.

Selections above RUNNER_EAGER_MAX are handed back ("fallback") for the
dask-cuda cold path. Restart the runner after the source store changes —
obs is cached at warm-up.
"""

import json
import os
import subprocess
import time
import traceback
from pathlib import Path

DATA = os.environ.get("RUNNER_DATA", "/mnt/subset3M_megazarr_v1.0.zarr")
EAGER_MAX = int(os.environ.get("RUNNER_EAGER_MAX", "150000"))
JOB_DIR = Path("/tmp/datavis_jobs")
HEARTBEAT = Path("/tmp/datavis_runner.alive")
RANDOM_SEED = 4242


def _pipeline(ad, np, rsc, X, obs_sel, var):
    raw = np.issubdtype(X.dtype, np.integer)
    adata = ad.AnnData(X=X.astype(np.float32) if raw else X, obs=obs_sel, var=var.copy())
    rsc.get.anndata_to_GPU(adata)
    if raw:
        rsc.pp.normalize_total(adata)
        rsc.pp.log1p(adata)
    rsc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)
    adata = adata[:, adata.var["highly_variable"].to_numpy()].copy()
    adata.X = adata.X.astype("float64")
    rsc.pp.scale(adata, zero_center=False, max_value=10)
    rsc.pp.pca(adata, n_comps=50, random_state=RANDOM_SEED)
    rsc.pp.neighbors(adata, n_neighbors=20, n_pcs=30, use_rep="X_pca",
                     algorithm="brute", random_state=RANDOM_SEED)
    rsc.tl.umap(adata, min_dist=0.45, init_pos="spectral", n_components=2,
                random_state=RANDOM_SEED)
    rsc.tl.leiden(adata, resolution=1.1, n_iterations=100, random_state=RANDOM_SEED)
    umap = adata.obsm["X_umap"]
    if hasattr(umap, "get"):
        umap = umap.get()
    return ad.AnnData(obs=adata.obs.copy(),
                      obsm={"X_umap": np.asarray(umap, dtype=np.float32)})


def _upload(view, dest: str, job_id: str) -> None:
    try:
        view.write_zarr(dest)  # direct gs:// via gcsfs — needs the instance SA on the bucket
    except Exception:
        local = f"/tmp/datavis_view_{job_id}"
        view.write_zarr(local)
        subprocess.run(["gcloud", "-q", "storage", "rsync", "-r", local, dest],
                       check=True, capture_output=True)
        subprocess.run(["rm", "-rf", local])


def main():
    JOB_DIR.mkdir(exist_ok=True)
    print("warming: importing RAPIDS", flush=True)
    import numpy as np
    import zarr
    import anndata as ad
    import rapids_singlecell as rsc

    import rmm
    import cupy as cp
    from rmm.allocators.cupy import rmm_cupy_allocator

    rmm.reinitialize(managed_memory=True, pool_allocator=False)
    cp.cuda.set_allocator(rmm_cupy_allocator)
    zarr.config.set({"async.concurrency": 32, "threading.max_workers": 8})
    ad.settings.zarr_write_format = 3

    print("warming: caching obs from", DATA, flush=True)
    f = zarr.open(DATA, mode="r")
    obs = ad.io.read_elem(f["obs"])
    var = ad.io.read_elem(f["var"])
    xds = ad.io.sparse_dataset(f["X"])
    print("ready", flush=True)

    while True:
        HEARTBEAT.touch()
        pending = sorted(JOB_DIR.glob("*.json"))
        if not pending:
            time.sleep(0.5)
            continue
        jf = pending[0]
        job_id = jf.stem
        log = open(JOB_DIR / f"{job_id}.log", "a")

        def stage(msg):
            log.write(f"stage: {msg}\n")
            log.flush()

        try:
            sel = json.loads(jf.read_text())
            if len(sel["barcodes"]) > EAGER_MAX:
                jf.rename(jf.with_suffix(".json.fallback"))
                stage("selection too large for warm runner")
                log.write("fallback\n")
                continue
            jf.unlink()
            stage("loading selected cells")
            rows = obs.index.get_indexer(sel["barcodes"])
            assert (rows >= 0).all(), "some barcodes are not in this store"
            rows = np.unique(rows)
            stage("pipeline (hvg -> pca -> umap -> leiden)")
            view = _pipeline(ad, np, rsc, xds[rows], obs.iloc[rows].copy(), var)
            stage("uploading view")
            _upload(view, sel["dest"], job_id)
            log.write("done\n")
        except Exception as e:
            log.write(f"stage: {e}\n{traceback.format_exc()}\nfailed\n")
        finally:
            log.flush()
            log.close()


if __name__ == "__main__":
    main()
