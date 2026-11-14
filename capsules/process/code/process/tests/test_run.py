"""End to end through code/run: on the GPU, or on CPU with the stand-ins in /scratch/process_rework/stubs."""
import os

import anndata as ad
import numpy as np
import scanpy as sc
import zarr
from annizarr import Repo

from conftest import N_CELLS, SMALL, needs_pipeline, run_pipeline

NO_PLOTS = {"PLOTS": "false"}


def check(group, source, obsm, leiden=("leiden",), method="pca", subset=None):
    out = ad.io.read_elem(group)
    assert set(obsm) | {"X_umap"} == set(out.obsm) & {"X_pca", "X_pca_harmony", "X_scVI", "X_umap"}
    p = out.uns["process"]
    assert p["method"] == method and set(p["leiden"]) == set(leiden)
    assert out.var["highly_variable"].sum() == 200
    kept = np.ones(out.n_obs, bool) if subset is None else out.obs[subset[0]].astype(str).isin(subset[1]).to_numpy()
    for key in leiden:
        assert out.obs[key].notna().to_numpy().tolist() == kept.tolist(), key
    for key in [*obsm, "X_umap"]:
        assert not np.isnan(out.obsm[key][kept]).any() and np.isnan(out.obsm[key][~kept]).all(), key
    assert p["n_cells"] == kept.sum()
    assert group["layers/lognorm"].attrs["target_sum"] == 1e4
    ref = sc.AnnData(X=source.X.astype(np.float32))
    sc.pp.normalize_total(ref, target_sum=1e4)
    sc.pp.log1p(ref)
    assert (out.X != source.X).nnz == 0, "X changed"
    assert np.allclose(out.layers["lognorm"].toarray(), ref.X.toarray(), rtol=1e-5, atol=1e-5), "lognorm differs"
    return out


def test_no_input_prints_usage_and_succeeds(tmp_path):
    for args in ([], ["--input", ""], ["--input", "", "--method", "PCA", "--leiden_resolutions", "1.1"]):
        done = run_pipeline(*args, env={"RESULTS_DIR": str(tmp_path)})
        assert done.returncode == 0 and "mainly a RAPIDS GPU environment" in done.stdout
        assert "App Panel" in done.stdout
    assert not list(tmp_path.iterdir())


def test_bad_setting_exits_2():
    done = run_pipeline("--input", "x.zarr", "--method", "umap", check=False)
    assert done.returncode == 2 and "not a valid pca or scvi" in done.stderr


def test_input_errors_stop_before_the_cluster(stores, tmp_path):
    cases = [
        ([tmp_path / "missing.zarr"], "error: can't open"),
        ([stores["pbmc"], "--batch_key", "chip"], "error: obs has no column chip; it has sample_id, chip_id, cell_type, pbmc_clean_AIFI_L1_2024_label"),
        ([stores["ingest_no_sample"], "--batch_key", "batch"], 'cells have no batch (missing or "")'),
        ([stores["pbmc"], "--subset_column", "sample_id", "--subset_values", "SP9"], "no cells have sample_id in"),
    ]
    for args, message in cases:
        done = run_pipeline("--input", *args, "--results_dir", tmp_path, check=False)
        assert done.returncode == 1 and message in done.stderr, (args, done.stderr[-2000:])
        assert "start_cluster" not in done.stdout and "Traceback" not in done.stderr
    assert not list(tmp_path.iterdir())


@needs_pipeline
def test_pca_relative_paths(stores, tmp_path):
    run_pipeline("--input", os.path.relpath(stores["pbmc"], tmp_path), "--results_dir", "res", *SMALL, cwd=tmp_path)
    res = tmp_path / "res"
    check(zarr.open_group(res / "pbmc.processed.zarr", mode="r"), ad.read_zarr(stores["pbmc"]), ["X_pca"])
    for name in ("umap.png", "cluster_summary.csv", "process_report.txt"):
        assert (res / name).exists(), name
    assert "TOTAL" in (res / "process_report.txt").read_text()


@needs_pipeline
def test_pca_harmony(stores, tmp_path):
    run_pipeline("--input", stores["pbmc"], "--batch_key", "chip_id", "--results_dir", tmp_path, *SMALL, env=NO_PLOTS)
    out = check(zarr.open_group(tmp_path / "pbmc.processed.zarr", mode="r"), ad.read_zarr(stores["pbmc"]),
                ["X_pca", "X_pca_harmony"])
    assert out.uns["process"]["basis"] == "X_pca_harmony" and list(out.uns["process"]["batch_keys"]) == ["chip_id"]


@needs_pipeline
def test_scvi(stores, tmp_path):
    env = {**NO_PLOTS, "SCVI_N_LATENT": "8", "SCVI_MAX_EPOCHS": "5"}
    run_pipeline("--input", stores["pbmc"], "--method", "scVI (train)", "--batch_key", "sample_id",
                 "--results_dir", tmp_path, *SMALL, env=env)
    out = check(zarr.open_group(tmp_path / "pbmc.processed.zarr", mode="r"), ad.read_zarr(stores["pbmc"]),
                ["X_scVI"], method="scvi")
    assert out.obsm["X_scVI"].shape == (N_CELLS, 8) and out.uns["process"]["basis"] == "X_scVI"
    assert out.uns["process"]["scvi_model"] == str(tmp_path / "scvi_model")
    assert (tmp_path / "scvi_model" / "model.pt").exists()


@needs_pipeline
def test_multi_resolution_subset(stores, tmp_path):
    run_pipeline("--input", stores["pbmc"], "--leiden_resolutions", "0.5,1.0,2.0", "--subset_column", "sample_id",
                 "--subset_values", "SP0, SP2", "--results_dir", tmp_path, *SMALL)
    leiden = ("leiden_0.5", "leiden_1.0", "leiden_2.0")
    out = check(zarr.open_group(tmp_path / "pbmc.processed.zarr", mode="r"), ad.read_zarr(stores["pbmc"]), ["X_pca"],
                leiden=leiden, subset=("sample_id", ["SP0", "SP2"]))
    assert "leiden" not in out.obs and list(out.uns["process"]["subset"]["values"]) == ["SP0", "SP2"]
    for key in leiden:
        assert (tmp_path / f"cluster_summary_{key}.csv").exists()


@needs_pipeline
def test_ingest_store_without_layers_or_sample_id_twice(stores, tmp_path):
    for _ in range(2):
        run_pipeline("--input", stores["ingest_no_sample"], "--results_copy", tmp_path / "out", "--results_dir", tmp_path,
                     *SMALL)
    for name in ("out.zarr", "out-1.zarr"):
        check(zarr.open_group(tmp_path / name, mode="r"), ad.read_zarr(stores["ingest_no_sample"]), ["X_pca"])
    assert not (tmp_path / "out").exists()
    assert (tmp_path / "umap.png").exists()


@needs_pipeline
def test_spatialdata(stores, tmp_path):
    import spatialdata as sd

    run_pipeline("--input", stores["tissue"], "--results_dir", tmp_path, *SMALL, env=NO_PLOTS)
    out = tmp_path / "tissue.processed.zarr"
    assert set(sd.read_zarr(out).images) == {"image"}
    source = ad.io.read_elem(zarr.open_group(stores["tissue"], mode="r")["tables/table"])
    check(zarr.open_group(out, mode="r")["tables/table"], source, ["X_pca"])


@needs_pipeline
def test_icechunk_external_twice(stores, icechunk, tmp_path):
    origin = str(icechunk["origin"])
    n = len(Repo(origin).log())
    for _ in range(2):
        run_pipeline("--input", icechunk["mount"], "--external", origin, "--results_dir", tmp_path, *SMALL, env=NO_PLOTS)
    log = Repo(origin).log()
    assert len(log) == n + 2 and log[0].metadata["source_snapshot"] == log[1].id
    check(Repo(origin).open_zarr("r"), ad.read_zarr(stores["pbmc"]), ["X_pca"])
    assert not list(tmp_path.glob("*.zarr"))


@needs_pipeline
def test_icechunk_copy_in_results(stores, icechunk, tmp_path):
    mount = icechunk["mount"]
    n = len(Repo(str(mount)).log())
    run_pipeline("--input", mount, "--results_dir", tmp_path, *SMALL, env=NO_PLOTS)
    copy = tmp_path / "mount_icechunk.processed"
    log = Repo(str(copy)).log()
    assert len(log) == n + 1 and log[0].metadata["source_snapshot"] == log[1].id
    check(Repo(str(copy)).open_zarr("r"), ad.read_zarr(stores["pbmc"]), ["X_pca"])
    assert len(Repo(str(mount)).log()) == n


def test_output_errors_stop_before_the_cluster(stores, icechunk, tmp_path):
    cases = [
        (["--input", icechunk["mount"], "--external", tmp_path / "typo"], "nothing there: check"),
        (["--input", stores["pbmc"], "--external", stores["tissue"]], "isn't an AnnData store like the input"),
        (["--input", stores["tissue"], "--results_format", "icechunk"], "SpatialData zarr can't be stored as icechunk"),
    ]
    for args, message in cases:
        done = run_pipeline(*args, "--results_dir", tmp_path / "res", check=False)
        assert done.returncode == 1 and message in done.stderr, (args, done.stderr[-2000:])
        assert "start_cluster" not in done.stdout and "Traceback" not in done.stderr
    done = run_pipeline("--input", stores["pbmc"], "--external", "s3://b/p", "--results_copy", "x", check=False)
    assert done.returncode == 2 and "leave them empty" in done.stderr
