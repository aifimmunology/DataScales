import sys
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", category=FutureWarning)

N_CELLS, N_GENES, N_TYPES = 3000, 400, 4


def has_gpu() -> bool:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(not has_gpu(), reason="process needs a CUDA GPU")


def synthetic_adata(seed: int = 0) -> ad.AnnData:
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
        "sample_id": pd.Categorical([f"SP{s:04d}" for s in sample]),
        "chip_id": pd.Categorical([f"C{c}" for c in chip]),
        "cell_type": pd.Categorical([f"T{t}" for t in cell_type]),
    }, index=[f"cell{i}" for i in range(N_CELLS)])
    var = pd.DataFrame(index=[f"MT-{i}" if i < 5 else f"GENE{i}" for i in range(N_GENES)])
    return ad.AnnData(X=counts, obs=obs, var=var)


def write_zarr(adata: ad.AnnData, path: Path) -> str:
    import annizarr

    annizarr.convert(adata, output=str(path))
    return str(path)


def write_icechunk(adata: ad.AnnData, path: Path, branch: str = "qc") -> str:
    import annizarr
    from annizarr import AppConfig, Repo
    from annizarr.config import IOConfig

    annizarr.convert(adata, output=str(path), cfg=AppConfig(io=IOConfig(backend="icechunk")), message="seed")
    Repo(str(path)).checkout(branch, create=True)
    return str(path)


def write_spatialdata(adata: ad.AnnData, path: Path) -> str:
    """A SpatialData zarr: a small image, and `adata` as the table annotating cell circles."""
    import spatialdata as sd
    from spatialdata.models import Image2DModel, ShapesModel, TableModel

    rng = np.random.default_rng(0)
    table = adata.copy()
    table.obs["region"] = pd.Categorical(["cells"] * table.n_obs)
    table.obs["cell_id"] = np.arange(table.n_obs)
    table.obsm["spatial"] = rng.uniform(0, 100, (table.n_obs, 2))
    circles = ShapesModel.parse(table.obsm["spatial"], geometry=0, radius=np.ones(table.n_obs))
    sdata = sd.SpatialData(
        images={"image": Image2DModel.parse(rng.integers(0, 255, (1, 64, 64), dtype=np.uint8), dims=("c", "y", "x"))},
        shapes={"cells": circles},
        tables={"table": TableModel.parse(table, region="cells", region_key="region", instance_key="cell_id")},
    )
    sdata.write(path)
    return str(path)


def write_spatial_icechunk(adata: ad.AnnData, path: Path, branch: str = "qc") -> str:
    """An icechunk repo laid out like a SpatialData store: `tables/table` and an image."""
    import zarr
    from anndata.io import write_elem
    from annizarr import Repo

    repo = Repo.create(str(path))
    root = zarr.open_group(repo.open_zarr("w").store, mode="a")
    root.attrs["spatialdata_attrs"] = {"version": "0.2"}
    write_elem(root.require_group("tables"), "table", adata)
    image = root.require_group("images").require_group("image").create_array("s0", shape=(1, 64, 64), dtype="u1")
    image[:] = np.random.default_rng(0).integers(0, 255, (1, 64, 64), dtype=np.uint8)
    repo.commit("seed")
    Repo(str(path)).checkout(branch, create=True)
    return str(path)


@pytest.fixture(scope="session")
def adata():
    return synthetic_adata()


@pytest.fixture
def zarr_store(adata, tmp_path):
    return write_zarr(adata, tmp_path / "curated.zarr")


@pytest.fixture
def icechunk_store(adata, tmp_path):
    return write_icechunk(adata, tmp_path / "repo")


@pytest.fixture
def env(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    return {
        "RESULTS_DIR": str(results),
        "ROWS_PER_CHUNK": "1000",
        "N_TOP_GENES": "200",
        "N_COMPS": "20",
        "N_PCS": "10",
        "N_NEIGHBORS": "15",
        "LEIDEN_RESOLUTION": "0.3",
    }


@pytest.fixture
def spatial_store(adata, tmp_path):
    return write_spatialdata(adata, tmp_path / "sample.zarr")


@pytest.fixture
def spatial_icechunk(adata, tmp_path):
    return write_spatial_icechunk(adata, tmp_path / "spatial_repo")
