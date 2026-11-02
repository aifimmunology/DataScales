import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import zarr
from annizarr import Repo
from sklearn.metrics import homogeneity_score

import main
import store as stores
from config import load_settings
from conftest import requires_gpu

pytestmark = requires_gpu

RESULTS = {"X_pca", "X_umap"}


def run(env, *argv):
    return main.run(load_settings(list(argv), env=env))


def check_results(back: ad.AnnData, obsm: set[str]):
    assert set(back.obsm) == obsm
    assert back.obsm["X_umap"].shape == (back.n_obs, 2)
    assert back.obs["leiden"].dtype == "category" and back.obs.columns[-1] == "leiden"
    assert back.var["highly_variable"].sum() == 200
    assert "lognorm" in back.layers
    uns = back.uns["process"]
    assert {"params", "n_gpus", "versions", "pca_variance_ratio", "source"} <= set(uns)


def lognorm(path_or_store) -> np.ndarray:
    return ad.read_zarr(path_or_store).layers["lognorm"].toarray()


def test_icechunk_writes_one_commit_on_the_branch(icechunk_store, adata, env):
    repo = Repo(icechunk_store, branch="qc")
    before, main_tip = len(repo.log()), repo.log(branch="main")[0].id
    report = run(env, "--input", icechunk_store)

    repo = Repo(icechunk_store, branch="qc")
    log = repo.log()
    assert len(log) == before + 1
    meta = log[0].metadata
    assert meta["pipeline"] == "process" and meta["curation_branch"] == "qc"
    assert meta["n_cells"] == adata.n_obs and meta["n_clusters"] == report["n_clusters"]
    assert repo.log(branch="main")[0].id == main_tip

    back = ad.read_zarr(repo.open_zarr("r").store)
    check_results(back, RESULTS)
    assert back.uns["process"]["source"]["snapshot"] == log[1].id
    np.testing.assert_array_equal(back.X.toarray(), adata.X.toarray())
    assert homogeneity_score(adata.obs["cell_type"], back.obs["leiden"]) > 0.95

    results = Path(env["RESULTS_DIR"])
    run_json = json.loads((results / "run.json").read_text())
    assert run_json["written"]["snapshot"] == log[0].id and run_json["peak_gpu_gb"] > 0
    summary = pd.read_csv(results / "cluster_summary.csv", index_col=0)
    assert summary["n_cells"].sum() == adata.n_obs and {"SP0000", "SP0001", "SP0002"} <= set(summary.columns)


def test_harmony_then_rerun_without_replaces_the_derived_elements(icechunk_store, adata, env):
    run(env, "--input", icechunk_store, "--integration", "harmony", "--batch-keys", "chip_id")
    back = ad.read_zarr(Repo(icechunk_store, branch="qc").open_zarr("r").store)
    check_results(back, RESULTS | {"X_pca_harmony"})
    assert back.uns["process"]["params"]["integration"] == "harmony"
    assert homogeneity_score(adata.obs["cell_type"], back.obs["leiden"]) > 0.95
    assert not np.allclose(back.obsm["X_pca_harmony"], back.obsm["X_pca"])

    run(env, "--input", icechunk_store)
    repo = Repo(icechunk_store, branch="qc")
    assert [s.metadata.get("integration") for s in list(repo.log())[:2]] == ["none", "harmony"]
    check_results(ad.read_zarr(repo.open_zarr("r").store), RESULTS)


def tree_digest(path) -> dict:
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(path).rglob("*")) if p.is_file()}


def test_writable_plain_zarr_gets_results_in_place(zarr_store, adata, env):
    zarr.consolidate_metadata(zarr_store)
    report = run(env, "--input", zarr_store)
    assert report["written"] == {"output": zarr_store, "mode": "in-place"}
    back = ad.read_zarr(zarr_store)
    check_results(back, RESULTS)
    assert "leiden" in zarr.open_group(zarr_store, mode="r", use_consolidated=True)["obs"]
    assert homogeneity_score(adata.obs["cell_type"], back.obs["leiden"]) > 0.95
    assert len(back.uns["process"]["source"]["fingerprint"]) == 64


def test_read_only_plain_zarr_full_and_results_only(zarr_store, adata, env, tmp_path, monkeypatch):
    monkeypatch.setattr(stores, "writable", lambda path: False)
    untouched = tree_digest(zarr_store)

    full = run({**env, "OUTPUT": str(tmp_path / "full.zarr")}, "--input", zarr_store)
    assert full["written"]["mode"] == "full"
    back = ad.read_zarr(tmp_path / "full.zarr")
    check_results(back, RESULTS)
    np.testing.assert_array_equal(back.X.toarray(), adata.X.toarray())
    pd.testing.assert_frame_equal(back.obs.drop(columns="leiden"), adata.obs)

    slim = run({**env, "OUTPUT": str(tmp_path / "slim.zarr")}, "--input", zarr_store, "--output-mode", "results-only")
    assert slim["written"]["mode"] == "results-only"
    back = ad.read_zarr(tmp_path / "slim.zarr")
    assert back.X is None
    check_results(back, RESULTS)
    np.testing.assert_allclose(lognorm(tmp_path / "slim.zarr"), lognorm(tmp_path / "full.zarr"), rtol=1e-5, atol=1e-6)

    assert homogeneity_score(adata.obs["cell_type"], back.obs["leiden"]) > 0.95
    assert tree_digest(zarr_store) == untouched


def test_missing_batch_key_fails_with_one_line_and_run_json(zarr_store, env, caplog, monkeypatch):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    code = main.main(["--input", zarr_store, "--integration", "harmony", "--batch-keys", "batch_id"])
    assert code == 1
    assert "no column(s) ['batch_id']" in caplog.text
    failed = json.loads((Path(env["RESULTS_DIR"]) / "run.json").read_text())
    assert failed["status"] == "failed" and "batch_id" in failed["error"]


def test_spatialdata_zarr_gets_results_in_its_table_only(spatial_store, adata, env, tmp_path, monkeypatch):
    import spatialdata as sd

    elements = {k: tree_digest(Path(spatial_store) / k) for k in ("images", "shapes")}
    run(env, "--input", spatial_store)
    sdata = sd.read_zarr(spatial_store)
    check_results(sdata.tables["table"], RESULTS | {"spatial"})
    assert {k: tree_digest(Path(spatial_store) / k) for k in elements} == elements

    monkeypatch.setattr(stores, "writable", lambda path: False)
    copy = run({**env, "OUTPUT": str(tmp_path / "copy.zarr")}, "--input", spatial_store)
    assert copy["written"]["mode"] == "full"
    back = sd.read_zarr(tmp_path / "copy.zarr")
    assert set(back.images) == {"image"} and set(back.shapes) == {"cells"}
    check_results(back.tables["table"], RESULTS | {"spatial"})


def test_spatial_icechunk_commit_touches_only_the_table(spatial_icechunk, adata, env):
    import icechunk

    before = Repo(spatial_icechunk, branch="qc").log()[0].id
    report = run(env, "--input", spatial_icechunk)
    after = report["written"]["snapshot"]
    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(spatial_icechunk))
    diff = repo.diff(from_snapshot_id=before, to_snapshot_id=after)
    touched = {*diff.new_arrays, *diff.new_groups, *diff.updated_arrays, *diff.updated_groups,
               *diff.deleted_arrays, *diff.updated_chunks}
    assert touched and all(p.startswith("/tables/table") for p in touched)
    table = ad.io.read_elem(Repo(spatial_icechunk, branch="qc").open_zarr("r")["tables/table"])
    check_results(table, RESULTS)
    assert homogeneity_score(adata.obs["cell_type"], table.obs["leiden"]) > 0.95
