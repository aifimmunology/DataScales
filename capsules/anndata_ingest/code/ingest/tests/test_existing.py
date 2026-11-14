"""Existing AnnData objects ingested as they are, alone or mixed with raw samples."""
import shutil

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from annizarr import Repo

import contract
import main
from config import load_settings
from conftest import GENES, counts_matrix, write_hise_h5


def existing(n_cells, seed, samples, genes=None, extra=None, dtype=np.float32):
    genes = genes or GENES
    x = counts_matrix(n_cells, seed)[:, :len(genes)].astype(dtype)
    obs = pd.DataFrame({"cell_type": "T", **(extra or {})}, index=[f"s{seed}_c{i}" for i in range(n_cells)])
    if samples:
        obs["sample_id"] = np.resize(samples, n_cells)
    adata = ad.AnnData(x, obs=obs, var=pd.DataFrame(index=genes))
    adata.obsm["X_umap"] = np.zeros((n_cells, 2))
    adata.layers["counts"] = x.copy()
    return adata


def run(env, store, samples, input_type="existing", **extra):
    return main.run(load_settings(["--store", str(store), "--samples", str(samples), "--create", "true",
                                   "--input-type", input_type], {**env, **extra}))


def read(store):
    return ad.read_zarr(Repo(str(store)).open_zarr("r")) if Repo.exists(str(store)) else ad.read_zarr(store)


def test_a_new_store_keeps_an_existing_anndata_whole(base_env, tmp_path):
    path = tmp_path / "atlas.h5ad"
    existing(60, 1, ["A", "B"], extra={"score": 0.5}).write_h5ad(path)
    report = run(base_env, tmp_path / "store", path)
    assert report["appended_samples"] == ["A", "B"] and report["n_cells"] == 60
    stored = read(tmp_path / "store")
    assert {"cell_type", "score", "sample_id"} <= set(stored.obs) and "X_umap" in stored.obsm
    assert "counts" in stored.layers and stored.X.dtype == np.float32
    assert run(base_env, tmp_path / "store", path)["appended_samples"] == []


def test_later_objects_append_their_cells_and_extra_columns_are_dropped(base_env, tmp_path, caplog):
    first, second = tmp_path / "first.h5ad", tmp_path / "second.zarr"
    existing(60, 1, ["A"]).write_h5ad(first)
    existing(40, 2, ["B"], extra={"novel": 1}, dtype=np.uint32).write_zarr(second)
    run(base_env, tmp_path / "store", first)
    report = run(base_env, tmp_path / "store", second)
    assert report["appended_samples"] == ["B"] and report["n_cells"] == 100
    assert report["dropped"] == ["obsm/X_umap", "layers/counts"]
    assert "obs/novel" in caplog.text and "obsm/X_umap" in caplog.text
    stored = read(tmp_path / "store")
    assert "novel" not in stored.obs and stored.X.dtype == np.float32
    assert (stored[stored.obs["sample_id"] == "B"].X.toarray() == counts_matrix(40, 2).toarray()).all()


def test_an_object_partly_in_the_store_is_refused(base_env, tmp_path):
    existing(60, 1, ["A"]).write_h5ad(tmp_path / "a.h5ad")
    existing(40, 2, ["A", "C"]).write_h5ad(tmp_path / "ac.h5ad")
    run(base_env, tmp_path / "store", tmp_path / "a.h5ad")
    with pytest.raises(main.IngestError, match="would be duplicated"):
        run(base_env, tmp_path / "store", tmp_path / "ac.h5ad")


def test_an_object_without_sample_id_is_named_after_its_file(base_env, tmp_path):
    path = tmp_path / "donor7.h5ad"
    existing(30, 3, None).write_h5ad(path)
    report = run(base_env, tmp_path / "store.zarr", path)
    assert report["appended_samples"] == ["donor7"] and set(read(tmp_path / "store.zarr").obs["sample_id"]) == {"donor7"}


def test_sample_ids_can_come_from_another_column(base_env, tmp_path):
    adata = existing(40, 4, None, extra={"donor": np.resize(["d1", "d2"], 40)})
    adata.write_h5ad(tmp_path / "rna.h5ad")
    report = run(base_env, tmp_path / "store", tmp_path / "rna.h5ad", SAMPLE_COLUMN="donor")
    assert report["appended_samples"] == ["d1", "d2"]
    assert set(read(tmp_path / "store").obs["sample_id"]) == {"d1", "d2"}


def test_an_anndata_zarr_folder_without_the_suffix_is_one_object(base_env, tmp_path):
    existing(30, 3, ["Z"]).write_zarr(tmp_path / "asset")
    assert run(base_env, tmp_path / "store", tmp_path / "asset")["appended_samples"] == ["Z"]


def test_raw_samples_join_a_store_seeded_from_an_existing_anndata(base_env, tmp_path, caplog):
    existing(60, 1, ["A"]).write_h5ad(tmp_path / "a.h5ad")
    run(base_env, tmp_path / "store", tmp_path / "a.h5ad")
    raw = write_hise_h5(tmp_path / "B092-P1_SP00001-01.h5", "SP00001", 50, seed=5)
    report = run(base_env, tmp_path / "store", raw, input_type="raw", CELLTYPIST_MODELS="None")
    assert report["appended_samples"] == ["SP00001"] and report["n_cells"] == 110
    assert "raw-sample columns aren't kept" in caplog.text
    assert read(tmp_path / "store").X.dtype == np.float32


def test_normalized_data_cannot_join_a_counts_store(base_env, tmp_path, samples_dir):
    raw = write_hise_h5(tmp_path / "B092-P1_SP00001-01.h5", "SP00001", 50, seed=5)
    run(base_env, tmp_path / "store", raw, input_type="raw", CELLTYPIST_MODELS="None")
    norm = existing(30, 2, ["N"])
    norm.X = norm.X * 0.5
    norm.write_h5ad(tmp_path / "norm.h5ad")
    with pytest.raises(contract.ContractError, match="non-count values"):
        run(base_env, tmp_path / "store", tmp_path / "norm.h5ad")


def test_existing_mode_points_raw_h5_files_to_raw_mode(base_env, tmp_path):
    raw = write_hise_h5(tmp_path / "B092-P1_SP00001-01.h5", "SP00001", 50, seed=5)
    with pytest.raises(main.IngestError, match="Raw samples"):
        run(base_env, tmp_path / "store", raw)


def test_several_objects_seed_a_new_store_in_one_run(base_env, tmp_path):
    folder = tmp_path / "objects"
    (folder / "a").mkdir(parents=True)
    (folder / "b").mkdir()
    existing(30, 1, ["A"]).write_h5ad(folder / "a" / "rna.h5ad")
    existing(30, 2, ["B"]).write_h5ad(folder / "b" / "rna.h5ad")
    report = run(base_env, tmp_path / "store", f"{folder / 'a' / 'rna.h5ad'},{folder / 'b' / 'rna.h5ad'}")
    assert report["appended_samples"] == ["A", "B"] and report["n_cells"] == 60
    assert report["dropped"] == ["obsm/X_umap", "layers/counts"]
    assert set(report["sources"]) == {"rna", "rna1"}


def test_shared_barcodes_from_different_samples_stay_distinct(base_env, tmp_path):
    a, b = existing(30, 1, ["A"]), existing(30, 1, ["B"])
    a.write_h5ad(tmp_path / "a.h5ad")
    b.write_h5ad(tmp_path / "b.h5ad")
    run(base_env, tmp_path / "store", tmp_path / "a.h5ad")
    report = run(base_env, tmp_path / "store", tmp_path / "b.h5ad")
    stored = read(tmp_path / "store")
    assert report["n_cells"] == 60 and stored.obs_names.is_unique and stored.obs_names[0] == "A:s1_c0"


def test_a_scanpy_qc_seed_still_takes_raw_samples(base_env, tmp_path, caplog):
    seed = existing(40, 1, ["A"])
    for c in ("n_genes_by_counts", "total_counts", "total_counts_mt", "pct_counts_mt"):
        seed.obs[c] = 1.0
    seed.write_h5ad(tmp_path / "a.h5ad")
    run(base_env, tmp_path / "store", tmp_path / "a.h5ad")
    raw = write_hise_h5(tmp_path / "B092-P1_SP00001-01.h5", "SP00001", 50, seed=5)
    report = run(base_env, tmp_path / "store", raw, input_type="raw", CELLTYPIST_MODELS="None")
    assert report["n_cells"] == 90 and "raw-sample columns aren't kept" in caplog.text


def test_obs_values_that_dont_fit_the_store_types_are_left_empty(base_env, tmp_path, caplog):
    existing(30, 1, ["A"], extra={"score": 0.5}).write_h5ad(tmp_path / "a.h5ad")
    existing(30, 2, ["B"], extra={"score": "high"}).write_h5ad(tmp_path / "b.h5ad")
    run(base_env, tmp_path / "store", tmp_path / "a.h5ad")
    report = run(base_env, tmp_path / "store", tmp_path / "b.h5ad")
    assert report["n_cells"] == 60 and "obs/score (values don't fit the store's type)" in caplog.text


def test_normalized_values_cannot_join_a_float_counts_store(base_env, tmp_path):
    existing(30, 1, ["A"]).write_h5ad(tmp_path / "a.h5ad")
    norm = existing(30, 2, ["B"])
    norm.X = norm.X * 0.5
    norm.write_h5ad(tmp_path / "b.h5ad")
    run(base_env, tmp_path / "store", tmp_path / "a.h5ad")
    with pytest.raises(contract.ContractError, match="non-count values"):
        run(base_env, tmp_path / "store", tmp_path / "b.h5ad")


def test_missing_sample_ids_are_refused(base_env, tmp_path):
    adata = existing(30, 1, ["A"])
    adata.obs.loc[adata.obs_names[:3], "sample_id"] = np.nan
    adata.write_h5ad(tmp_path / "a.h5ad")
    with pytest.raises(main.IngestError, match="some cells have no sample_id"):
        run(base_env, tmp_path / "store", tmp_path / "a.h5ad")


def test_counts_too_large_for_the_store_dtype_are_refused():
    import scipy.sparse as sp
    with pytest.raises(contract.ContractError, match="don't fit"):
        contract.fit_x(sp.csr_matrix(np.array([[5e9]])), np.dtype(np.uint32), True, "big")


def test_a_missing_sample_column_falls_back_to_sample_id(base_env, tmp_path, caplog):
    existing(30, 1, ["A"]).write_h5ad(tmp_path / "a.h5ad")
    report = run(base_env, tmp_path / "store", tmp_path / "a.h5ad", SAMPLE_COLUMN="donor")
    assert report["appended_samples"] == ["A"] and "no 'donor' column" in caplog.text


def test_a_different_panel_is_fitted_block_by_block(base_env, tmp_path, monkeypatch):
    import objects
    monkeypatch.setattr(objects, "ROWS_PER_BLOCK", 7)
    existing(30, 1, ["A"]).write_h5ad(tmp_path / "a.h5ad")
    genes = GENES[10:] + ["NEW"]
    existing(25, 2, ["B"], genes=genes, dtype=np.uint32).write_h5ad(tmp_path / "b.h5ad")
    run(base_env, tmp_path / "store", tmp_path / "a.h5ad")
    report = run(base_env, tmp_path / "store", tmp_path / "b.h5ad")
    assert report["fitted"] == {"b": "fitted to the store's genes: 10 zero-filled, 1 dropped"}
    stored = read(tmp_path / "store")
    cells = stored[stored.obs["sample_id"] == "B"]
    assert list(stored.var_names) == GENES and cells.X[:, :10].nnz == 0
    assert (cells.X[:, 10:].toarray() == counts_matrix(25, 2)[:, :len(GENES) - 10].toarray()).all()


def test_raw_is_never_kept_so_later_objects_still_append(base_env, tmp_path, caplog):
    for name, seed, sample in (("a", 1, "A"), ("b", 2, "B"), ("c", 3, "C")):
        adata = existing(30, seed, [sample])
        adata.raw = adata
        adata.write_h5ad(tmp_path / f"{name}.h5ad")
    run(base_env, tmp_path / "store", tmp_path / "a.h5ad")
    assert "not added, since the store has no place for them: raw" in caplog.text
    report = run(base_env, tmp_path / "store", f"{tmp_path / 'b.h5ad'},{tmp_path / 'c.h5ad'}")
    assert report["appended_samples"] == ["B", "C"] and report["n_cells"] == 90
    assert read(tmp_path / "store").raw is None


def test_a_zarr_v2_anndata_is_copied_into_v3_staging(base_env, tmp_path, monkeypatch):
    monkeypatch.setattr(ad.settings, "zarr_write_format", 2)
    existing(30, 1, ["A"]).write_zarr(tmp_path / "a.zarr")
    existing(25, 2, ["B"], genes=GENES[10:]).write_zarr(tmp_path / "b.zarr")
    monkeypatch.setattr(ad.settings, "zarr_write_format", 3)
    run(base_env, tmp_path / "store", tmp_path / "a.zarr")
    assert "X_umap" in read(tmp_path / "store").obsm
    report = run(base_env, tmp_path / "store", tmp_path / "b.zarr")
    assert report["n_cells"] == 55 and report["fitted"] == {"b": "fitted to the store's genes: 10 zero-filled, 0 dropped"}
    cells = read(tmp_path / "store")
    cells = cells[cells.obs["sample_id"] == "B"]
    assert (cells.X[:, 10:].toarray() == counts_matrix(25, 2)[:, :len(GENES) - 10].toarray()).all()


def test_a_stale_mount_does_not_block_an_append_after_derived_elements_were_dropped(base_env, tmp_path):
    origin, mount = tmp_path / "origin", tmp_path / "mount"
    for name, seed, sample in (("a", 1, "A"), ("b", 2, "B"), ("c", 3, "C")):
        existing(30, seed, [sample]).write_h5ad(tmp_path / f"{name}.h5ad")
    run(base_env, origin, tmp_path / "a.h5ad")
    shutil.copytree(origin, mount)
    first = run(base_env, mount, tmp_path / "b.h5ad", ORIGIN=str(origin))
    assert first["dropped"] == ["obsm/X_umap", "layers/counts"]
    second = run(base_env, mount, f"{tmp_path / 'b.h5ad'},{tmp_path / 'c.h5ad'}", ORIGIN=str(origin))
    assert second["appended_samples"] == ["C"] and second["dropped"] == [] and second["n_cells"] == 90


def test_objects_without_cells_are_skipped_before_staging(base_env, tmp_path, caplog):
    empty = existing(0, 1, None)
    empty.write_h5ad(tmp_path / "empty.h5ad")
    existing(30, 2, ["A"], genes=GENES[::-1]).write_h5ad(tmp_path / "a.h5ad")
    existing(30, 3, ["B"]).write_h5ad(tmp_path / "b.h5ad")
    report = run(base_env, tmp_path / "store", f"{tmp_path / 'empty.h5ad'},{tmp_path / 'a.h5ad'}")
    assert report["appended_samples"] == ["A"] and "empty.h5ad has no cells; skipped" in caplog.text
    report = run(base_env, tmp_path / "store", f"{tmp_path / 'empty.h5ad'},{tmp_path / 'b.h5ad'}")
    assert report["appended_samples"] == ["B"] and list(read(tmp_path / "store").var_names) == GENES[::-1]
