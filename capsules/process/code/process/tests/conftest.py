import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

PACKAGE = Path(__file__).resolve().parents[1]
RUN = PACKAGE.parent / "run"
sys.path.insert(0, str(PACKAGE))
warnings.filterwarnings("ignore", category=FutureWarning)

N_CELLS, N_GENES, N_TYPES = 3000, 400, 4
SMALL = ["--n_top_genes", "200", "--n_comps", "20", "--n_pcs", "10", "--n_neighbors", "15", "--chunk_rows", "1000"]


def pipeline_mode() -> str:
    """'gpu', 'stub' (the CPU stand-ins from /scratch/process_rework/stubs are on PYTHONPATH) or ''."""
    try:
        import rapids_singlecell as rsc
    except ImportError:
        return ""
    if getattr(rsc, "__version__", "") == "cpu-stub":
        return "stub"
    try:
        import cupy

        return "gpu" if cupy.cuda.runtime.getDeviceCount() > 0 else ""
    except Exception:
        return ""


MODE = pipeline_mode()
needs_pipeline = pytest.mark.skipif(not MODE, reason="needs a GPU, or the CPU stubs on PYTHONPATH")
needs_gpu = pytest.mark.skipif(MODE != "gpu", reason="needs a CUDA GPU")


def synthetic(seed: int = 0) -> ad.AnnData:
    """Counts with N_TYPES cell types, three samples on two chips, and a per-chip shift in gene scale."""
    rng = np.random.default_rng(seed)
    cell_type = rng.integers(0, N_TYPES, N_CELLS)
    sample = rng.integers(0, 3, N_CELLS)
    chip = sample % 2
    profile = rng.gamma(0.3, 1.0, (N_TYPES, N_GENES))
    for t in range(N_TYPES):
        profile[t, t * 20:(t + 1) * 20] += 8
    chip_scale = np.where(chip[:, None] == 1, rng.uniform(0.5, 1.5, N_GENES), 1.0)
    counts = sp.csr_matrix(rng.poisson(profile[cell_type] * chip_scale * 2).astype(np.int32))
    obs = pd.DataFrame({
        "sample_id": pd.Categorical([f"SP{s}" for s in sample]),
        "chip_id": pd.Categorical([f"C{c}" for c in chip]),
        "cell_type": pd.Categorical([f"T{t}" for t in cell_type]),
    }, index=[f"cell{i}" for i in range(N_CELLS)])
    return ad.AnnData(X=counts, obs=obs, var=pd.DataFrame(index=[f"GENE{i}" for i in range(N_GENES)]))


def ingest_like(adata: ad.AnnData, sample_id: bool = True) -> ad.AnnData:
    """As anndata_ingest writes it: uint32 CSR counts, categorical obs, "" for missing strings, no layers."""
    missing = np.random.default_rng(2).random(adata.n_obs) < 0.05
    obs = pd.DataFrame({"batch": pd.Categorical(np.where(missing, "", adata.obs["chip_id"].astype(str)))},
                       index=adata.obs_names)
    if sample_id:
        obs.insert(0, "sample_id", pd.Categorical(np.where(missing, "", adata.obs["sample_id"].astype(str))))
    return ad.AnnData(X=sp.csr_matrix(adata.X, dtype=np.uint32), obs=obs, var=adata.var.copy())


def write_zarr(adata: ad.AnnData, path: Path) -> Path:
    import annizarr

    annizarr.convert(adata, output=str(path))
    return path


def write_spatialdata(adata: ad.AnnData, path: Path) -> Path:
    import spatialdata as sd
    from spatialdata.models import Image2DModel, ShapesModel, TableModel

    rng = np.random.default_rng(0)
    table = adata.copy()
    table.obs["region"] = pd.Categorical(["cells"] * table.n_obs)
    table.obs["cell_id"] = np.arange(table.n_obs)
    table.obsm["spatial"] = rng.uniform(0, 100, (table.n_obs, 2))
    sd.SpatialData(
        images={"image": Image2DModel.parse(rng.integers(0, 255, (1, 64, 64), dtype=np.uint8), dims=("c", "y", "x"))},
        shapes={"cells": ShapesModel.parse(table.obsm["spatial"], geometry=0, radius=np.ones(table.n_obs))},
        tables={"table": TableModel.parse(table, region="cells", region_key="region", instance_key="cell_id")},
    ).write(path)
    return path


def read_only(path: Path) -> Path:
    for folder, dirs, files in os.walk(path):
        for f in files:
            os.chmod(Path(folder) / f, 0o444)
        os.chmod(folder, 0o555)
    return path


def writable(path: Path) -> None:
    for folder, _, _ in os.walk(path):
        os.chmod(folder, 0o755)


@pytest.fixture(scope="session")
def adata():
    return synthetic()


@pytest.fixture(scope="session")
def stores(adata, tmp_path_factory):
    """Read-only inputs, made once: an AnnData zarr, an ingest-style zarr, a SpatialData zarr."""
    root = tmp_path_factory.mktemp("stores")
    paths = {
        "pbmc": write_zarr(adata, root / "pbmc.zarr"),
        "ingest_no_sample": write_zarr(ingest_like(adata, sample_id=False), root / "ingest_no_sample.zarr"),
        "tissue": write_spatialdata(adata, root / "tissue.zarr"),
    }
    yield paths
    for p in paths.values():
        writable(p)


@pytest.fixture
def icechunk(stores, tmp_path):
    """A writable icechunk origin and a read-only copy of it, like a data asset mounted from S3."""
    from annizarr import Repo

    origin = tmp_path / "origin_icechunk"
    Repo.init(stores["pbmc"], origin)
    mount = tmp_path / "mount_icechunk"
    shutil.copytree(origin, mount)
    read_only(mount)
    yield {"origin": origin, "mount": mount}
    writable(mount)


@pytest.fixture(scope="session", autouse=True)
def lock_inputs(stores):
    for p in stores.values():
        read_only(p)


def run_pipeline(*args, cwd=None, env=None, check=True) -> subprocess.CompletedProcess:
    full_env = {**os.environ, "MPLBACKEND": "Agg", **(env or {})}
    full_env.pop("PROCESS_CWD", None)
    done = subprocess.run([str(RUN), *map(str, args)], cwd=cwd, env=full_env, capture_output=True, text=True)
    if check and done.returncode:
        raise AssertionError(f"run failed ({done.returncode}):\n{done.stdout[-3000:]}\n{done.stderr[-5000:]}")
    return done
