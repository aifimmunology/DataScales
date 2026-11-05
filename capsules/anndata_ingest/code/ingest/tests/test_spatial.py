"""SpatialData tables as samples, non-gene features, the early gene-panel check, and per-sample sources."""
import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import zarr
from annizarr import Repo

import contract
import hise
import main
import store as stores
from config import load_settings
from conftest import GENES, counts_matrix, write_hise_h5

CONTROLS = ["NegControlProbe_0001", "BLANK_0001"]


def write_spatialdata(path, n_cells, seed, genes=None, controls=True):
    """A minimal SpatialData .zarr: root attrs plus a tables/table AnnData, Xenium-style var."""
    genes = genes or GENES
    x = counts_matrix(n_cells, seed)[:, :len(genes)]
    names, types = list(genes), ["Gene Expression"] * len(genes)
    if controls:
        x = ad.AnnData(x).X
        import scipy.sparse as sp
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


def run(env, store, samples, **extra):
    return main.run(load_settings(["--store", str(store), "--samples", str(samples), "--create", "true"],
                                  {**env, **extra}))


def test_a_spatialdata_table_is_read_without_its_control_features(tmp_path):
    path = write_spatialdata(tmp_path / "Breast_A.zarr", 40, seed=1)
    adata = hise.read_sample(path)
    assert hise.zarr_kind(path) == "spatialdata" and list(adata.var_names) == GENES == list(hise.gene_names(path))
    assert hise.sample_id(path) == "Breast_A"


def test_generic_10x_file_names_take_the_bundle_folder_as_sample_id():
    assert hise.sample_id("/data/Xenium_Lung_2fov/extract/cell_feature_matrix.h5") == "Xenium_Lung_2fov"
    assert hise.sample_id("/data/run1/outs/filtered_feature_bc_matrix.h5") == "run1"
    assert hise.sample_id("/data/multi/outs/per_sample_outs/S1/count/sample_filtered_feature_bc_matrix.h5") == "S1"
    assert hise.sample_id("/data/B092-P1_SP02425-01.h5") == "SP02425"


def test_spatialdata_folders_are_listed_and_ingested_with_their_sources(base_env, tmp_path):
    folder = tmp_path / "sdata"
    folder.mkdir()
    a = write_spatialdata(folder / "Breast_A.zarr", 40, seed=1)
    b = write_spatialdata(folder / "Breast_B.zarr", 50, seed=2)
    store = tmp_path / "store"
    report = run(base_env, store, folder)
    assert report["appended_samples"] == ["Breast_A", "Breast_B"] and report["n_cells"] == 90
    assert report["sources"] == {"Breast_A": str(a), "Breast_B": str(b)}
    head = Repo(str(store)).log()[0]
    assert head.metadata["sources"] == report["sources"] and f"- Breast_B: {b}" in head.message
    assert run(base_env, store, folder)["appended_samples"] == []


def test_a_zarr_that_is_not_anndata_or_spatialdata_is_refused(base_env, tmp_path):
    zarr.open_group(tmp_path / "plain.zarr", mode="w")
    with pytest.raises(main.IngestError, match="must be an AnnData, or a SpatialData"):
        run(base_env, tmp_path / "store", tmp_path / "plain.zarr")


def test_mudata_is_refused_by_name(base_env, tmp_path):
    root = zarr.open_group(tmp_path / "sample_mudata", mode="w")
    root.attrs["encoding-type"] = "MuData"
    root.create_group("mod")
    with pytest.raises(main.IngestError, match="is MuData"):
        run(base_env, tmp_path / "store", tmp_path / "sample_mudata")


def test_a_different_gene_panel_warns_and_is_fitted_to_the_store_genes(base_env, tmp_path, caplog):
    store = tmp_path / "store"
    run(base_env, store, write_hise_h5(tmp_path / "B092-P1_SP00001-01.h5", "SP00001", 40, seed=1))
    path = write_spatialdata(tmp_path / "B.zarr", 30, seed=2, genes=GENES[3:] + ["NEW1", "NEW2"])
    report = run(base_env, store, path)
    assert "B: 497 shared, 3 only in the store (e.g. MT-0, MT-1, MT-2), " \
           "2 only in the sample (e.g. NEW1, NEW2)" in caplog.text
    assert report["appended_samples"] == ["B"] and report["n_cells"] == 70
    note = "fitted to the store's genes: 3 zero-filled, 2 dropped"
    head = Repo(str(store)).log()[0]
    assert report["fitted"] == {"B": note} == head.metadata["fitted"] and f"- B: {path} ({note})" in head.message
    stored = ad.read_zarr(Repo(str(store)).open_zarr("r").store)
    assert list(stored.var_names) == GENES
    cells = stored[stored.obs["sample_id"] == "B"]
    assert cells.X[:, :3].nnz == 0
    assert (cells.X[:, 3:].toarray() == hise.read_sample(path)[:, GENES[3:]].X.toarray()).all()


def test_a_new_store_fits_later_samples_to_the_first_samples_genes(base_env, tmp_path):
    store = tmp_path / "store.zarr"
    a = write_spatialdata(tmp_path / "A.zarr", 40, seed=1)
    b = write_spatialdata(tmp_path / "B.zarr", 30, seed=2, genes=GENES[::-1])
    report = run(base_env, store, f"{a},{b}")
    assert report["n_cells"] == 70 and report["fitted"] == {"B": "genes reordered to the store's"}
    stored = ad.read_zarr(store)
    assert list(stored.var_names) == GENES
    cells = stored[stored.obs["sample_id"] == "B"]
    assert (cells.X.toarray() == hise.read_sample(b)[:, GENES].X.toarray()).all()


def test_fitting_handles_duplicate_gene_names():
    sample = ad.AnnData(sp.csr_matrix(np.array([[1, 2, 3]], dtype=np.uint32)),
                        var=pd.DataFrame(index=["A", "B", "A"]))
    fitted = contract.conform_var(sample, ["B", "A", "C"])
    assert fitted.X.toarray().tolist() == [[2, 1, 0]] and fitted.X.dtype == np.uint32


def test_storage_warning_names_refreshing_locations():
    assert "durable S3" in main.storage_warning("/scratch/x") and "data asset is read-only" in main.storage_warning("/results/x")
    assert main.storage_warning("s3://bucket/x") is None and main.storage_warning("/data/asset") is None
