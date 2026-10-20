"""Open stores, read obs/X/latent rows (service eager/lazy rule), pull missing stores."""
import subprocess
import time
from pathlib import Path

import numpy as np

from . import config, paths

paths.add()


def store_path(ds):
    return Path(config.dataset(ds)["store"])


def ensure_local(ds):
    """Return the store path, pulling it from GCS (logged to setup/) if missing; raise if unavailable."""
    cfg, store = config.dataset(ds), store_path(ds)
    if store.exists():
        return store
    if not cfg.get("gcs"):
        raise FileNotFoundError(f"{store} missing and {ds} has no gcs source")
    log = config.out_root() / "logs" / f"pull_{ds}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with open(log, "a") as f:
        subprocess.run(["gcloud", "storage", "cp", "-r", cfg["gcs"], str(store.parent)],
                       stdout=f, stderr=subprocess.STDOUT, check=True)
        f.write(f"pull_seconds={time.perf_counter() - t0:.1f}\n")
    return store


def open_root(ds):
    import zarr
    return zarr.open(str(ensure_local(ds)), mode="r")


def n_obs(root):
    return int(root["X"].attrs["shape"][0])


def read_obs_cols(root, cols, ds=None, index=False):
    """Selected obs columns as a DataFrame; with ds, label columns are renamed to plot_labels."""
    import anndata as ad
    import pandas as pd

    grp = root["obs"]
    if index:
        obs = pd.DataFrame(index=pd.Index(ad.io.read_elem(grp[grp.attrs["_index"]]), name="barcode"))
    else:
        obs = pd.DataFrame()
    for c in cols:
        obs[c] = ad.io.read_elem(grp[c])
    if ds is not None:
        cfg = config.dataset(ds)
        obs = obs.rename(columns=dict(zip(cfg["labels"], cfg["plot_labels"])))
    return obs


def label_obs(ds, root):
    """Label columns under their plot names (AIFI_L1/2/3 for rna*, L1/L2/L2.5 for mmatac)."""
    cfg = config.dataset(ds)
    return read_obs_cols(root, cfg["labels"], ds=ds)


def batch_obs(ds, root):
    """Index + batch column (categorical) for building AnnData of selected rows."""
    cfg = config.dataset(ds)
    obs = read_obs_cols(root, [cfg["batch_col"]], index=True)
    obs[cfg["batch_col"]] = obs[cfg["batch_col"]].astype("category")
    return obs


def is_eager(n_rows, rep=None):
    """The service's rule: latent routes and selections <= RERUN_EAGER_MAX cells run eagerly."""
    import rerun_umap_on_selection as svc
    return rep is not None or n_rows <= svc.EAGER_MAX


def _rows_arg(rows, n):
    return None if rows is None or len(rows) == n else rows


def load_X(root, obs, rows, eager):
    """AnnData of the selected rows (None = all): eager host CSR, or lazy dask (cluster path)."""
    import rerun_umap_on_selection as svc
    return svc.load_cells(root, obs, _rows_arg(rows, len(obs)), None, eager)


def load_stored_latent(root, obs, rows, key):
    """AnnData holding only obsm[key] for the selected rows (service `load_cells` latent path)."""
    import rerun_umap_on_selection as svc
    return svc.load_cells(root, obs, _rows_arg(rows, len(obs)), key, True)


def load_own_latent(ds, rows, p=None, rep=1):
    """Rows of our setup-run latent (models/<ds>/full_r<rep>/latent.npy)."""
    path = config.model_latent_path(ds, p, rep)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run the full-model setup run first")
    lat = np.load(path, mmap_mode="r")
    return np.ascontiguousarray(lat if _rows_arg(rows, len(lat)) is None else lat[rows])
