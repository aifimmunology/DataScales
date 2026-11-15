"""Inputs and stores that must stop a run with one clear message (no traceback, nothing written), and
store layouts the run must still read."""
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from annizarr import Repo

import config
import main
import store as stores
from config import load_settings
from conftest import GENES, counts_matrix, write_hise_h5

APP_PANEL = Path(__file__).resolve().parents[3] / ".codeocean" / "app-panel.json"


def ingest(env, store, samples, *flags):
    return main.run(load_settings(["--store", str(store), "--samples", str(samples), "--create", "true", *flags], env))


def test_the_app_panel_matches_the_positional_order():
    if not APP_PANEL.is_file():
        pytest.skip("no .codeocean/app-panel.json next to code/")
    params = json.loads(APP_PANEL.read_text())["parameters"]
    assert len(params) == len(config.PANEL)
    by_name = dict(zip(config.PANEL, params))
    assert set(by_name["input_type"]["extra_data"]) <= set(config.INPUT_TYPES)
    assert set(by_name["celltypist_models"]["extra_data"]) <= set(config.CELLTYPIST_PRESETS)
    assert by_name["store"]["default_value"] == config.Settings().store


def test_a_sample_sharing_no_genes_is_refused(samples_dir, base_env, store_path, tmp_path):
    ingest(base_env, store_path, sorted(samples_dir.glob("*.h5"))[0])
    other = write_hise_h5(tmp_path / "B092-P1_SP00009-01.h5", "SP00009", 50, seed=9,
                          genes=[f"ENSG{i:011d}" for i in range(len(GENES))])
    with pytest.raises(main.contract.ContractError, match="shares no gene names"):
        ingest(base_env, store_path, other)


def test_an_existing_object_sharing_no_genes_is_refused(samples_dir, base_env, tmp_path):
    store = tmp_path / "store"
    ingest(base_env, store, sorted(samples_dir.glob("*.h5"))[0])
    obj = tmp_path / "symbols.h5ad"
    ad.AnnData(counts_matrix(30, 1).astype(np.float32), var=pd.DataFrame(index=[f"X{i}" for i in range(len(GENES))]),
               obs=pd.DataFrame({"sample_id": "Z"}, index=[f"c{i}" for i in range(30)])).write_h5ad(obj)
    with pytest.raises(main.contract.ContractError, match="shares no gene names"):
        ingest(base_env, store, obj, "--input-type", "existing")


def test_a_truncated_input_stops_before_the_store_is_created(samples_dir, base_env, tmp_path):
    good = sorted(samples_dir.glob("*.h5"))[0]
    bad = tmp_path / "in" / "B092-P1_SP00008-01.h5"
    bad.parent.mkdir()
    bad.write_bytes(good.read_bytes()[:2000])
    with pytest.raises(main.IngestError, match="can't be read"):
        ingest(base_env, tmp_path / "store", bad.parent)
    assert not (tmp_path / "store").exists()


def test_a_garbage_model_stops_before_the_store_is_created(samples_dir, base_env, tmp_path):
    pkl = tmp_path / "junk.pkl"
    pkl.write_bytes(b"not a model")
    env = {**base_env, "RUN_CELLTYPIST": "true", "CELLTYPIST_PKL": str(pkl)}
    with pytest.raises(main.IngestError, match="can't be loaded"):
        ingest(env, tmp_path / "store", samples_dir)
    assert not (tmp_path / "store").exists()


def test_an_unknown_dropdown_value_lists_the_options(samples_dir, base_env, tmp_path):
    env = {**base_env, "RUN_CELLTYPIST": "true", "CELLTYPIST_MODELS": "AIFI L9 celltypes"}
    with pytest.raises(main.IngestError, match="Celltypist Model options are"):
        ingest(env, tmp_path / "store", samples_dir)


def test_a_plain_zarr_on_s3_is_refused():
    with pytest.raises(stores.StoreError, match="must be local"):
        stores.open_store("s3://bucket/atlas.zarr")


def test_an_icechunk_repo_named_zarr_opens_as_icechunk(samples_dir, base_env, tmp_path):
    store = tmp_path / "atlas.zarr"
    Repo.create(str(store))
    assert isinstance(stores.open_store(str(store)), stores.IcechunkStore)
    assert ingest(base_env, store, sorted(samples_dir.glob("*.h5"))[0])["kind"] == "icechunk"


def test_a_folder_of_other_files_is_not_made_an_icechunk_store(samples_dir, base_env, tmp_path):
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "readme.txt").write_text("x")
    with pytest.raises(stores.StoreError, match="not an icechunk repo"):
        ingest(base_env, folder, samples_dir)


def test_a_store_without_sample_id_is_a_clean_error(base_env, tmp_path, samples_dir):
    path = tmp_path / "foreign.zarr"
    ad.AnnData(counts_matrix(10, 0), var=pd.DataFrame(index=GENES)).write_zarr(path)
    with pytest.raises(stores.StoreError, match="no sample_id column"):
        ingest(base_env, path, samples_dir)


def test_a_dense_x_store_counts_its_cells(base_env, tmp_path):
    obj = tmp_path / "dense.zarr"
    ad.AnnData(counts_matrix(40, 2).toarray().astype(np.float32), var=pd.DataFrame(index=GENES),
               obs=pd.DataFrame({"sample_id": "D"}, index=[f"c{i}" for i in range(40)])).write_zarr(obj)
    report = ingest(base_env, tmp_path / "store", obj, "--input-type", "existing")
    assert report["n_cells"] == 40


def test_a_seed_with_other_features_keeps_its_layers_on_the_genes(base_env, tmp_path):
    n, extra = 30, ["ADT1", "ADT2"]
    x = np.hstack([counts_matrix(n, 3).toarray(), np.ones((n, 2))]).astype(np.float32)
    var = pd.DataFrame({"feature_types": ["Gene Expression"] * len(GENES) + ["Antibody Capture"] * 2},
                       index=GENES + extra)
    obj = ad.AnnData(x, var=var, obs=pd.DataFrame({"sample_id": "M"}, index=[f"c{i}" for i in range(n)]))
    obj.layers["counts"] = x.copy()
    obj.varm["loadings"] = np.zeros((len(var), 2))
    obj.write_h5ad(tmp_path / "multi.h5ad")
    report = ingest(base_env, tmp_path / "store", tmp_path / "multi.h5ad", "--input-type", "existing")
    stored = ad.read_zarr(Repo(str(tmp_path / "store")).open_zarr("r"))
    assert report["n_cells"] == n and stored.shape == (n, len(GENES))
    assert stored.layers["counts"].shape == (n, len(GENES)) and "loadings" not in stored.varm


def test_staging_cleanup_leaves_the_rest_of_the_staging_folder(samples_dir, base_env, tmp_path):
    keep = tmp_path / "staging" / "keep.txt"
    keep.parent.mkdir()
    keep.write_text("x")
    ingest(base_env, tmp_path / "store", sorted(samples_dir.glob("*.h5"))[0])
    assert keep.exists() and [p.name for p in keep.parent.iterdir()] == ["keep.txt"]


def test_the_second_commit_of_a_new_store_names_only_what_it_appends(samples_dir, base_env, tmp_path):
    report = ingest(base_env, tmp_path / "store", samples_dir)
    log = Repo(str(tmp_path / "store")).log()
    assert "SP00001" not in log[0].message and "SP00002" in log[0].message
    assert report["appended_samples"] == ["SP00001", "SP00002", "SP00003"]
    assert set(report["versions"]) >= {"annizarr", "icechunk", "anndata"}


def test_a_resequenced_sample_is_skipped_with_a_warning(samples_dir, base_env, store_path, tmp_path, caplog):
    first = sorted(samples_dir.glob("*.h5"))[0]
    ingest(base_env, store_path, first)
    again = tmp_path / "B093-P2_SP00001-02.h5"
    write_hise_h5(again, "SP00001", 50, seed=7)
    caplog.clear()
    assert ingest(base_env, store_path, again)["appended_samples"] == []
    assert "re-sequenced batch" in caplog.text and first.name in caplog.text and again.name in caplog.text
    caplog.clear()
    assert ingest(base_env, store_path, first)["appended_samples"] == []
    assert "re-sequenced batch" not in caplog.text
    opened = stores.open_store(store_path)
    opened.open(False)
    assert opened.recorded_sources()["SP00001"] == str(first)


def test_every_input_problem_is_listed_at_once(samples_dir, base_env, tmp_path):
    bad = tmp_path / "B092-P1_SP00008-01.h5"
    bad.write_bytes(b"not an h5 file")
    with pytest.raises(main.IngestError) as err:
        ingest(base_env, tmp_path / "store", f"{samples_dir},{bad},{tmp_path / 'gone.h5'}")
    assert "precheck found problems" in str(err.value)
    assert "not found" in str(err.value) and "can't be read" in str(err.value)
