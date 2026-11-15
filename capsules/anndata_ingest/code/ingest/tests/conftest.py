import sys
import warnings
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", category=FutureWarning)

import main  # noqa: E402
from config import load_settings  # noqa: E402

GENES = [f"MT-{i}" for i in range(5)] + [f"GENE{i}" for i in range(495)]
HISE_STR = {
    "adt_qc_flag": "Good", "batch_id": "B092", "chip_id": "C1", "hto_barcode": "ACGT", "hto_category": "singlet",
    "pool_id": "P1", "well_id": "W1",
}
SAMPLES = {"SP00001": 160, "SP00002": 180, "SP00003": 200}
CONTROLS = ["NegControlProbe_0001", "BLANK_0001"]


def ingest(env, store, samples, *flags, **extra):
    """A run with these flags; `extra` adds environment variables."""
    return main.run(load_settings(["--store", str(store), "--samples", str(samples), *flags], {**env, **extra}))


def create(env, store, samples, *flags, **extra):
    return ingest(env, store, samples, "--create", "true", *flags, **extra)


def counts_matrix(n_cells: int, seed: int) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    counts = sp.random(n_cells, len(GENES), density=0.15, format="csr", random_state=seed)
    counts.data = (rng.integers(1, 40, size=counts.nnz) + 1).astype(np.uint16)
    counts = counts.tolil()
    counts[:, :3] = rng.integers(1, 10, size=(n_cells, 3))
    return counts.tocsr()


def write_hise_h5(path: Path, sample: str, n_cells: int, seed: int, genes: list[str] | None = None) -> Path:
    genes = genes or GENES
    rng = np.random.default_rng(seed)
    counts = counts_matrix(n_cells, seed)
    csc = sp.csc_matrix(counts.T.tocsr())
    barcodes = np.array([f"BC{i:05d}-1" for i in range(n_cells)], dtype="S")
    with h5py.File(path, "w") as f:
        m = f.create_group("matrix")
        m["data"] = csc.data.astype(np.uint16)
        m["indices"] = csc.indices.astype(np.uint16)
        m["indptr"] = csc.indptr.astype(np.uint32)
        m["shape"] = np.array(csc.shape, dtype=np.uint32)
        m["barcodes"] = barcodes
        feat = m.create_group("features")
        feat["name"] = np.array(genes, dtype="S")
        feat["id"] = np.array([f"ENSG{i}" for i in range(len(genes))], dtype="S")
        feat["feature_type"] = np.array(["Gene Expression"] * len(genes), dtype="S")
        obs = m.create_group("observations")
        for col, value in HISE_STR.items():
            obs[col] = np.array([value] * n_cells, dtype="S")
        obs["cell_name"] = np.array([f"{sample}_{i}" for i in range(n_cells)], dtype="S")
        obs["cell_uuid"] = np.array([f"uuid-{sample}-{i}" for i in range(n_cells)], dtype="S")
        obs["original_barcodes"] = barcodes
        obs["pbmc_sample_id"] = np.array([sample] * n_cells, dtype="S")
        obs["n_genes"] = np.diff(counts.indptr).astype(np.uint16)
        obs["n_umis"] = np.asarray(counts.sum(axis=1)).ravel().astype(np.uint32)
        obs["n_mito_umis"] = np.asarray(counts[:, :5].sum(axis=1)).ravel().astype(np.uint16)
        obs["n_reads"] = (np.asarray(counts.sum(axis=1)).ravel() * 3).astype(np.uint32)
        obs["adt_umis"] = rng.integers(100, 500, size=n_cells).astype(np.uint16)
    return path


def write_spatialdata(path, n_cells, seed, genes=None, controls=True):
    """A minimal SpatialData .zarr: root attrs plus a tables/table AnnData, Xenium-style var."""
    genes = genes or GENES
    x = counts_matrix(n_cells, seed)[:, :len(genes)]
    names, types = list(genes), ["Gene Expression"] * len(genes)
    if controls:
        x = ad.AnnData(x).X
        x = sp.hstack([x, sp.csr_matrix(np.ones((n_cells, len(CONTROLS))))]).tocsr()
        names += CONTROLS
        types += ["Negative Control Probe", "Unassigned Codeword"]
    table = ad.AnnData(x.astype(np.float32), obs=pd.DataFrame(
        {"cell_id": [f"cell{i}-1" for i in range(n_cells)]}, index=[f"cell{i}-1" for i in range(n_cells)]),
        var=pd.DataFrame({"feature_types": types}, index=names))
    table.obsm["spatial"] = np.zeros((n_cells, 2))
    root = zarr.open_group(path, mode="w")
    root.attrs["spatialdata_attrs"] = {"version": "0.2"}
    ad.io.write_elem(root.create_group("tables"), "table", table)
    return path


@pytest.fixture
def samples_dir(tmp_path):
    d = tmp_path / "samples"
    d.mkdir()
    for i, (sid, n) in enumerate(SAMPLES.items()):
        write_hise_h5(d / f"B092-P1_{sid}-01.h5", sid, n, seed=i)
    return d


@pytest.fixture
def base_env(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    return {
        "RESULTS_DIR": str(results),
        "STAGING_DIR": str(tmp_path / "staging"),
        "CHUNK_CELLS": "100",
        "WRITE_CPUS": "2",
        "RUN_SCRUBLET": "false",
        "RUN_CELLTYPIST": "false",
    }


@pytest.fixture(params=["icechunk", "zarr"])
def store_path(request, tmp_path):
    return str(tmp_path / ("store" if request.param == "icechunk" else "store.zarr"))


@pytest.fixture(scope="session")
def models_dir(tmp_path_factory):
    """Two tiny CellTypist models, `tiny_L1` (2 labels) and `tiny_L2` (3 labels), trained on synthetic counts."""
    import celltypist

    X = sp.vstack([counts_matrix(60, seed) for seed in range(2)]).astype(np.float32)
    X = sp.csr_matrix(np.log1p(X.multiply(1e4 / X.sum(axis=1)).toarray()))
    train = ad.AnnData(X, var={"_": np.zeros(len(GENES))})
    train.var_names = GENES
    d = tmp_path_factory.mktemp("models")
    for name, labels in {"tiny_L1": ["T cell", "B cell"], "tiny_L2": ["CD4 T", "CD8 T", "B naive"]}.items():
        y = np.resize(labels, train.n_obs)
        celltypist.train(train, labels=y, use_SGD=True, max_iter=50, check_expression=False).write(str(d / f"{name}.pkl"))
    return d
