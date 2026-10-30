import anndata as ad
import numpy as np
import pandas as pd
import pytest
import zarr
from anndata.io import write_elem

import store as stores
from annizarr import Repo
from config import Settings
from steps import Embedding, read_obs


def embedding(n_obs, n_vars, harmony=False):
    rng = np.random.default_rng(0)
    obsm = {"X_pca": rng.random((n_obs, 5), dtype=np.float32), "X_umap": rng.random((n_obs, 2), dtype=np.float32)}
    if harmony:
        obsm["X_pca_harmony"] = obsm["X_pca"] + 1
    return Embedding(highly_variable=np.arange(n_vars) % 2 == 0, variance_ratio=np.linspace(0.5, 0.1, 5),
                     obsm=obsm, leiden=pd.Categorical((np.arange(n_obs) % 3).astype(str), categories=["0", "1", "2"]))


def test_plain_zarr_is_written_in_place_when_writable(zarr_store):
    target = stores.open_target(Settings(input=zarr_store))
    assert (target.kind, target.mode, target.dest) == ("zarr", "in-place", zarr_store)
    assert "_write_probe" not in zarr.open_group(zarr_store, mode="r")


def test_read_only_plain_zarr_goes_to_a_new_output(zarr_store, tmp_path, monkeypatch):
    monkeypatch.setattr(stores, "writable", lambda path: False)
    target = stores.open_target(Settings(input=zarr_store, results_dir=str(tmp_path)))
    assert (target.mode, target.dest) == ("full", str(tmp_path / "curated.processed.zarr"))
    (tmp_path / "curated.processed.zarr").mkdir()
    with pytest.raises(stores.StoreError, match="already exists"):
        stores.open_target(Settings(input=zarr_store, results_dir=str(tmp_path)))


def test_origin_on_plain_zarr_and_missing_input_are_refused(zarr_store, tmp_path):
    with pytest.raises(stores.StoreError, match="icechunk repos only"):
        stores.open_target(Settings(input=zarr_store, origin="s3://b/p"))
    with pytest.raises(stores.StoreError, match="No icechunk repo or zarr"):
        stores.open_target(Settings(input=str(tmp_path / "nothing")))


def test_read_only_icechunk_without_origin_is_refused(icechunk_store, monkeypatch):
    import annizarr.ic._repo

    monkeypatch.setattr(annizarr.ic._repo, "is_readonly_path", lambda path: True)
    with pytest.raises(stores.StoreError, match="read-only"):
        stores.open_target(Settings(input=icechunk_store))
    target = stores.open_target(Settings(input=icechunk_store, origin=icechunk_store))
    assert target.location == icechunk_store


def test_icechunk_source_is_the_branch_tip(icechunk_store):
    target = stores.open_target(Settings(input=icechunk_store))
    assert target.kind == "icechunk"
    assert target.source()["snapshot"] == target.repo.log()[0].id
    assert "X" in target.read_root()


def test_missing_branch_is_refused(icechunk_store):
    from annizarr.errors import AnzError

    with pytest.raises(AnzError, match="No branch"):
        stores.open_target(Settings(input=icechunk_store, branch="strict"))


def test_fingerprint_is_stable_and_tracks_obs(zarr_store):
    root = zarr.open_group(zarr_store, mode="r")
    first = stores.fingerprint(root)
    assert stores.fingerprint(zarr.open_group(zarr_store, mode="r")) == first
    obs = zarr.open_group(zarr_store, mode="r+")["obs"]
    names = [f"cell{i}" for i in range(root["X"].attrs["shape"][0])]
    write_elem(obs, obs.attrs["_index"], np.array(["renamed", *names[1:]]))
    assert stores.fingerprint(zarr.open_group(zarr_store, mode="r")) != first


def test_write_embedding_replaces_earlier_results(zarr_store, adata):
    root = zarr.open_group(zarr_store, mode="r+")
    stores.write_embedding(root, embedding(adata.n_obs, adata.n_vars, harmony=True), {"run": 1})
    stores.write_embedding(root, embedding(adata.n_obs, adata.n_vars), {"run": 2})
    back = ad.read_zarr(zarr_store)
    assert set(back.obsm) == {"X_pca", "X_umap"}
    assert back.uns["process"]["run"] == 2
    assert list(back.obs.columns).count("leiden") == 1 and back.obs.columns[-1] == "leiden"
    assert back.var["highly_variable"].dtype == bool
    assert back.obs["leiden"].dtype == "category"


def test_copy_without_counts_keeps_everything_else(adata, tmp_path):
    src = adata.copy()
    src.obsm["spatial"] = np.ones((src.n_obs, 2))
    src.layers["counts"] = src.X.copy()
    src.uns["note"] = "kept"
    src.write_zarr(tmp_path / "src.zarr")
    out = str(tmp_path / "out.zarr")
    stores.copy_without_counts(zarr.open_group(tmp_path / "src.zarr", mode="r"), out)
    stores.write_embedding(zarr.open_group(out, mode="r+"), embedding(adata.n_obs, adata.n_vars), {})
    back = ad.read_zarr(out)
    assert back.X is None and not back.layers
    assert "spatial" in back.obsm and back.uns["note"] == "kept"
    pd.testing.assert_frame_equal(back.obs.drop(columns="leiden"), adata.obs)
    assert list(back.var_names) == list(adata.var_names)


def test_read_obs_reads_only_requested_columns(zarr_store):
    obs = read_obs(zarr.open_group(zarr_store, mode="r"), ["chip_id"])
    assert list(obs.columns) == ["chip_id"] and obs["chip_id"].dtype == "category"
    with pytest.raises(Exception, match="no column"):
        read_obs(zarr.open_group(zarr_store, mode="r"), ["batch_id"])


def commit_marker(path, branch="qc"):
    repo = Repo(path, branch=branch)
    repo.open_zarr("w").attrs["marker"] = 1
    return repo.commit("elsewhere")


def test_stale_mount_reads_and_writes_at_the_origin_tip(icechunk_store, tmp_path, adata, monkeypatch):
    import shutil

    import annizarr.ic._repo

    mount = str(tmp_path / "mount")
    shutil.copytree(icechunk_store, mount)
    tip = commit_marker(icechunk_store)
    monkeypatch.setattr(annizarr.ic._repo, "is_readonly_path", lambda path: path == mount)
    target = stores.open_target(Settings(input=mount, origin=icechunk_store))
    assert target.snapshot == tip
    assert target.read_root().attrs["marker"] == 1


def test_write_refuses_when_the_branch_moved(icechunk_store, adata):
    target = stores.open_target(Settings(input=icechunk_store))
    commit_marker(icechunk_store)
    with pytest.raises(stores.StoreError, match="moved"):
        target.write(embedding(adata.n_obs, adata.n_vars), {}, None, Settings(input=icechunk_store))
    assert "obsm" not in Repo(icechunk_store, branch="qc").open_zarr("r") or \
        "X_umap" not in Repo(icechunk_store, branch="qc").open_zarr("r")["obsm"]


def test_failed_copy_leaves_no_output(zarr_store, adata, tmp_path, monkeypatch):
    monkeypatch.setattr(stores, "writable", lambda path: False)
    out = tmp_path / "out.zarr"
    target = stores.open_target(Settings(input=zarr_store, output=str(out)))

    def boom(*args):
        raise OSError("disk full")

    monkeypatch.setattr(stores, "write_lognorm", lambda *a: None)
    monkeypatch.setattr(stores, "write_embedding", boom)
    with pytest.raises(OSError, match="disk full"):
        target.write(embedding(adata.n_obs, adata.n_vars), {}, None, Settings(input=zarr_store))
    assert not out.exists() and list(tmp_path.glob("out.zarr.partial-*")) == []


def test_remote_plain_zarr_is_refused(monkeypatch):
    monkeypatch.setattr(stores.Repo, "exists", staticmethod(lambda path: False))
    with pytest.raises(stores.StoreError, match="remote plain zarr"):
        stores.open_target(Settings(input="s3://bucket/curated.zarr"))


def test_missing_batch_values_are_refused():
    from steps import PipelineError, check_batch_keys

    obs = pd.DataFrame({"chip_id": pd.Categorical(["C0", None, "C1"])})
    with pytest.raises(PipelineError, match="'chip_id': 1"):
        check_batch_keys(obs, ["chip_id"])
    check_batch_keys(obs, [])


def test_output_given_copies_even_when_writable(zarr_store, tmp_path):
    target = stores.open_target(Settings(input=zarr_store, output=str(tmp_path / "o.zarr")))
    assert (target.in_place, target.mode) == (False, "full")
    target = stores.open_target(Settings(input=zarr_store, output_mode="results-only", results_dir=str(tmp_path)))
    assert (target.in_place, target.mode) == (False, "results-only")


def test_failed_in_place_write_drops_consolidated_metadata(zarr_store, adata, monkeypatch):
    root = zarr.open_group(zarr_store, mode="r+")
    stores.write_embedding(root, embedding(adata.n_obs, adata.n_vars), {"run": 0})
    zarr.consolidate_metadata(zarr_store)
    target = stores.open_target(Settings(input=zarr_store))
    monkeypatch.setattr(stores, "write_lognorm", lambda *a: None)

    def boom(*args):
        raise OSError("killed")

    monkeypatch.setattr(stores, "write_embedding", boom)
    with pytest.raises(OSError, match="killed"):
        target.write(embedding(adata.n_obs, adata.n_vars), {}, None, Settings(input=zarr_store))
    assert zarr.open_group(zarr_store, mode="r").metadata.consolidated_metadata is None
    assert "process" not in ad.read_zarr(zarr_store).uns


def test_input_changed_during_the_run_is_refused(zarr_store, adata):
    target = stores.open_target(Settings(input=zarr_store))
    obs = zarr.open_group(zarr_store, mode="r+")["obs"]
    write_elem(obs, obs.attrs["_index"], np.array([f"x{i}" for i in range(adata.n_obs)]))
    with pytest.raises(stores.StoreError, match="changed during the run"):
        target.write(embedding(adata.n_obs, adata.n_vars), {}, None, Settings(input=zarr_store))


def test_output_that_appeared_is_left_alone(zarr_store, adata, tmp_path):
    out = tmp_path / "o.zarr"
    target = stores.open_target(Settings(input=zarr_store, output=str(out)))
    out.mkdir()
    (out / "theirs").write_text("x")
    with pytest.raises(stores.StoreError, match="appeared"):
        target.write(embedding(adata.n_obs, adata.n_vars), {}, None, Settings(input=zarr_store))
    assert (out / "theirs").exists()


def test_failed_icechunk_write_leaves_the_branch(icechunk_store, adata, monkeypatch):
    target = stores.open_target(Settings(input=icechunk_store))
    tip = target.snapshot
    monkeypatch.setattr(stores, "write_lognorm", lambda *a: None)
    monkeypatch.setattr(stores, "write_embedding", lambda *a: (_ for _ in ()).throw(OSError("killed")))
    with pytest.raises(OSError, match="killed"):
        target.write(embedding(adata.n_obs, adata.n_vars), {}, None, Settings(input=icechunk_store))
    assert not target.repo.dirty
    assert Repo(icechunk_store, branch="qc").log()[0].id == tip


def test_spatialdata_tables_are_found(spatial_store, spatial_icechunk, tmp_path):
    for path in (spatial_store, spatial_icechunk):
        target = stores.open_target(Settings(input=path))
        assert target.table == "tables/table" and "X" in target.read_root()
    with pytest.raises(stores.StoreError, match="no X and no tables/other"):
        stores.open_target(Settings(input=spatial_store, table="other"))


def test_unconsolidate_reaches_a_consolidated_table(adata, tmp_path):
    path = tmp_path / "s.zarr"
    zarr.open_group(path, mode="w").create_group("tables")
    adata.write_zarr(path / "tables" / "table")
    zarr.consolidate_metadata(path)
    stores.unconsolidate(str(path), "tables/table")
    for group in ("", "tables", "tables/table"):
        assert zarr.open_group(path / group, mode="r").metadata.consolidated_metadata is None
    write_elem(zarr.open_group(path, mode="r+", use_consolidated=False)["tables/table/layers"], "new", adata.X)
    zarr.consolidate_metadata(path)
    assert "new" in zarr.open_group(path, mode="r")["tables/table/layers"]
