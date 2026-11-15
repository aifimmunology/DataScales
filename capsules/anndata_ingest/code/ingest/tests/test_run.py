"""End-to-end runs on small synthetic samples, against a local icechunk repo and a plain zarr: raw samples,
existing AnnData objects, SpatialData tables, and the inputs and stores that must stop a run with one
clear message (no traceback, nothing written)."""
import json
import shutil

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import scrublet
import zarr
from anndata.io import read_elem
from annizarr import Repo

import inputs
import main
import prepare
import store as stores
from config import ConfigError, load_settings
from conftest import GENES, SAMPLES, counts_matrix, create, ingest, write_hise_h5, write_spatialdata


def root_of(store):
    return zarr.open_group(store, mode="r") if store.endswith(".zarr") else Repo(store).open_zarr("r")


def log_of(store):
    return None if store.endswith(".zarr") else [s.id for s in Repo(store).log()]


def files_of(samples_dir):
    return sorted(samples_dir.glob("*.h5"))


def read(store):
    return ad.read_zarr(Repo(str(store)).open_zarr("r")) if Repo.exists(str(store)) else ad.read_zarr(store)


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


def ingest_objects(env, store, samples, input_type="existing", **extra):
    return create(env, store, samples, "--input-type", input_type, **extra)


def test_bootstrap_then_append(samples_dir, base_env, store_path, tmp_path):
    files = files_of(samples_dir)
    first = ingest(base_env, store_path, files[0], "--create", "true")
    assert first["appended_samples"] == ["SP00001"] and first["n_cells"] == 160
    if first["kind"] == "icechunk":
        assert len(log_of(store_path)) == 2 and first["snapshot"] == first["bootstrap_snapshot"]

    second = ingest(base_env, store_path, samples_dir)
    assert second["appended_samples"] == ["SP00002", "SP00003"] and second["bootstrap_snapshot"] is None
    root = root_of(store_path)
    assert stores.n_obs(root) == sum(SAMPLES.values()) == second["n_cells"]
    assert not any(k in root and len(root[k]) for k in ("layers", "obsm", "obsp"))
    obs = read_elem(root["obs"])
    assert list(obs.columns) == list(prepare.requested_schema(False, []))
    assert obs.index.is_unique and obs["sample_id"].value_counts().to_dict() == SAMPLES
    if second["kind"] == "icechunk":
        repo = Repo(store_path)
        assert len(repo.log()) == 3 and repo.log()[0].metadata["appended_samples"] == ["SP00002", "SP00003"]
        assert second["snapshot"] == repo.log()[0].id
    else:
        assert second["snapshot"] is None

    summary = pd.read_csv(tmp_path / "results" / "ingest_summary.csv", index_col=0)
    assert summary["n_cells"].to_dict() == {"SP00002": 180, "SP00003": 200}
    assert json.loads((tmp_path / "results" / "ingest_run.json").read_text())["n_samples"] == 3


def test_multi_sample_bootstrap_is_a_seed_plus_one_commit(samples_dir, base_env, tmp_path):
    store = str(tmp_path / "store")
    report = ingest(base_env, store, samples_dir, "--create", "true")
    log = Repo(store).log()
    assert len(log) == 3 and log[1].id == report["bootstrap_snapshot"] and log[0].id == report["snapshot"]
    assert log[0].metadata["n_cells"] == 540 and log[0].metadata["n_samples"] == 3
    seed = log[1].metadata
    assert seed["pipeline"] == "ingest" and seed["appended_samples"] == ["SP00001"] and seed["n_cells"] == 160
    assert set(seed["sources"]) == {"SP00001"} and "SP00001" in log[1].message


def test_stored_counts_and_obs_match_the_source(samples_dir, base_env, store_path):
    files = files_of(samples_dir)
    ingest(base_env, store_path, f"{files[0]},{files[1]}", "--create", "true")
    root = root_of(store_path)
    X = read_elem(root["X"])
    src = inputs.read_sample(files[1])
    assert X.dtype == np.uint32
    np.testing.assert_array_equal(X[160:].toarray(), src.X.toarray())
    obs = read_elem(root["obs"])
    assert obs.index[160] == "SP00002:BC00000-1"
    assert obs["n_umis"].iloc[160:].astype(int).tolist() == src.obs["n_umis"].astype(int).tolist()
    assert obs["batch_id"].iloc[0] == "B092" and obs["n_genes_by_counts"].iloc[0] == X[0].nnz


def test_rerun_with_nothing_new_changes_nothing(samples_dir, base_env, store_path):
    ingest(base_env, store_path, samples_dir, "--create", "true")
    before = log_of(store_path)
    report = ingest(base_env, store_path, samples_dir)
    assert report["appended_samples"] == [] and report["snapshot"] is None and report["n_cells"] == 540
    assert log_of(store_path) == before


def test_mixed_inputs_in_one_run(base_env, store_path, tmp_path):
    d = tmp_path / "mixed"
    d.mkdir()
    write_hise_h5(d / "B092-P1_SP00011-01.h5", "SP00011", 40, seed=11)
    vlen = write_hise_h5(d / "B092-P1_SP00012-01.h5", "SP00012", 30, seed=12)
    with h5py.File(vlen, "r+") as f:
        for key in ("barcodes", "observations/batch_id"):
            values = [v.decode() for v in f["matrix"][key][:]]
            del f["matrix"][key]
            f["matrix"][key] = np.array(values, dtype=h5py.string_dtype())
    tenx = write_hise_h5(d / "SP00013.h5", "SP00013", 50, seed=13)
    with h5py.File(tenx, "r+") as f:
        del f["matrix/observations"]
    h5ad_src = inputs.read_sample(write_hise_h5(tmp_path / "SP00014.h5", "SP00014", 20, seed=14))
    ad.AnnData(h5ad_src.X, var=h5ad_src.var).write_h5ad(d / "SP00014.h5ad")

    report = ingest(base_env, store_path, d, "--create", "true")
    assert report["appended_samples"] == ["SP00011", "SP00012", "SP00013", "SP00014"] and report["n_cells"] == 140
    obs = read_elem(root_of(store_path)["obs"])
    assert obs.loc[obs["sample_id"] == "SP00012", "batch_id"].unique().tolist() == ["B092"]
    assert set(obs.loc[obs["sample_id"].isin(["SP00013", "SP00014"]), "batch_id"]) == {""}
    assert obs.loc[obs["sample_id"] == "SP00013", "n_umis"].isna().all()


def test_scrublet_and_celltypist(samples_dir, base_env, store_path, models_dir, caplog):
    caplog.set_level("INFO")
    files = files_of(samples_dir)
    env = {**base_env, "RUN_SCRUBLET": "true", "RUN_CELLTYPIST": "true", "MODELS_DIR": str(models_dir),
           "CELLTYPIST_MODELS": "tiny_L1,tiny_L2"}
    ingest(env, store_path, files[0], "--create", "true")
    ingest(env, store_path, files[1])
    obs = read_elem(root_of(store_path)["obs"])
    assert list(obs.columns)[-6:] == [
        "scrublet_score", "predicted_doublets", "tiny_L1_label", "tiny_L1_prob", "tiny_L2_label", "tiny_L2_prob"]
    assert obs["scrublet_score"].between(0, 1).all()
    assert set(obs["tiny_L1_label"]) <= {"T cell", "B cell"} and set(obs["tiny_L2_label"]) <= {"CD4 T", "CD8 T", "B naive"}
    assert obs["tiny_L1_prob"].between(0, 1).all() and obs["tiny_L2_prob"].between(0, 1).all()
    assert "CellTypist tiny_L1: " in caplog.text and "Prediction done" not in caplog.text


def test_scrublet_failure_leaves_na_and_the_run_completes(samples_dir, base_env, store_path, monkeypatch):
    def fail(self, **kw):
        raise ValueError("too few cells")

    monkeypatch.setattr(scrublet.Scrublet, "scrub_doublets", fail)
    report = ingest({**base_env, "RUN_SCRUBLET": "true"}, store_path, files_of(samples_dir)[0], "--create", "true")
    obs = read_elem(root_of(store_path)["obs"])
    assert report["n_cells"] == 160 and obs["scrublet_score"].isna().all() and obs["predicted_doublets"].isna().all()


def test_schema_mismatch_is_refused_before_any_write(samples_dir, base_env, store_path):
    files = files_of(samples_dir)
    ingest(base_env, store_path, files[0], "--create", "true")
    before = log_of(store_path)
    with pytest.raises(prepare.ContractError, match="scrublet_score"):
        ingest({**base_env, "RUN_SCRUBLET": "true"}, store_path, files[1])
    assert log_of(store_path) == before and stores.n_obs(root_of(store_path)) == 160


def test_failure_while_preparing_leaves_the_store_untouched(samples_dir, base_env, store_path, monkeypatch):
    files = files_of(samples_dir)
    ingest(base_env, store_path, files[0], "--create", "true")
    before = log_of(store_path)
    real = prepare.qc_metrics
    calls = []

    def fail_on_second(adata):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("sample is corrupt")
        real(adata)

    monkeypatch.setattr(prepare, "qc_metrics", fail_on_second)
    with pytest.raises(RuntimeError, match="corrupt"):
        ingest(base_env, store_path, samples_dir)
    assert log_of(store_path) == before and stores.n_obs(root_of(store_path)) == 160


def test_failure_while_appending_commits_nothing_on_icechunk(samples_dir, base_env, tmp_path, monkeypatch):
    store = str(tmp_path / "store")
    files = files_of(samples_dir)
    ingest(base_env, store, files[0], "--create", "true")
    before = log_of(store)
    real = stores.IcechunkStore.append
    calls = []

    def fail_on_second(self, cells, cfg, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("network dropped")
        real(self, cells, cfg, **kwargs)

    monkeypatch.setattr(stores.IcechunkStore, "append", fail_on_second)
    with pytest.raises(RuntimeError, match="network"):
        ingest(base_env, store, samples_dir)
    assert log_of(store) == before and stores.n_obs(root_of(store)) == 160


def test_failed_bootstrap_append_keeps_the_seed_and_reruns_cleanly(samples_dir, base_env, tmp_path, monkeypatch):
    store = str(tmp_path / "store")
    files = files_of(samples_dir)
    def fail(self, cells, cfg, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(stores.IcechunkStore, "append", fail)
    with pytest.raises(RuntimeError):
        ingest(base_env, store, f"{files[0]},{files[1]}", "--create", "true")
    assert len(log_of(store)) == 2 and stores.n_obs(root_of(store)) == 160
    monkeypatch.undo()
    report = ingest(base_env, store, f"{files[0]},{files[1]}")
    assert report["appended_samples"] == ["SP00002"] and report["n_cells"] == 340


def test_store_state_guards(samples_dir, base_env, store_path):
    with pytest.raises(inputs.IngestError, match="No store"):
        ingest(base_env, store_path, samples_dir)
    ingest(base_env, store_path, samples_dir, "--create", "true")
    again = ingest(base_env, store_path, samples_dir, "--create", "true")
    assert again["appended_samples"] == [] and again["n_cells"] == sum(SAMPLES.values())


def test_a_new_store_is_created_only_inside_the_create_roots(samples_dir, base_env, tmp_path):
    env = {**base_env, "CREATE_ROOTS": str(tmp_path / "results")}
    with pytest.raises(inputs.IngestError, match="created only inside"):
        ingest(env, tmp_path / "elsewhere", samples_dir)
    with pytest.raises(inputs.IngestError, match="created only inside"):
        ingest(env, tmp_path / "results", samples_dir)
    report = ingest(env, tmp_path / "results" / "store", samples_dir)
    assert report["n_samples"] == len(SAMPLES) and report["kind"] == "icechunk"
    assert ingest(env, tmp_path / "results" / "store", samples_dir)["appended_samples"] == []


def add_results(root):
    root.require_group("obsm").create_array("X_umap", shape=(160, 2), dtype="f4")
    ad.io.write_elem(root.require_group("layers"), "lognorm", sp.random(160, len(GENES), format="csr", dtype=np.float32))
    ad.io.write_elem(root.require_group("uns"), "process", {"integration": "none"})
    old = ad.AnnData(sp.csr_matrix((160, len(GENES)), dtype=np.float32), var=pd.DataFrame(index=GENES))
    old.raw = old
    ad.io.write_elem(root, "raw", old.raw)
    if isinstance(root.store, zarr.storage.LocalStore):
        zarr.consolidate_metadata(root.store)


def test_appending_to_a_processed_store_drops_its_results(samples_dir, base_env, store_path, caplog):
    files = files_of(samples_dir)
    ingest(base_env, store_path, files[0], "--create", "true")
    if store_path.endswith(".zarr"):
        add_results(zarr.open_group(store_path, mode="r+"))
    else:
        repo = Repo(store_path)
        add_results(repo.open_zarr("w"))
        repo.commit("add results")
    report = ingest(base_env, store_path, files[1])
    dropped = ["obsm/X_umap", "layers/lognorm", "raw", "uns/process"]
    assert report["dropped"] == dropped and "appending new cells drops them" in caplog.text
    stored = ad.read_zarr(root_of(store_path))
    assert stored.n_obs == report["n_cells"] > 160 and not stored.obsm and not stored.layers
    assert "process" not in stored.uns and stored.raw is None
    if not store_path.endswith(".zarr"):
        assert Repo(store_path).log()[0].metadata["dropped"] == dropped


def test_a_file_as_the_store_is_refused(samples_dir, base_env, tmp_path):
    with pytest.raises(stores.StoreError, match="is a file"):
        ingest(base_env, tmp_path / "store.h5ad", samples_dir, "--create", "true")


def test_a_plain_zarr_without_the_suffix_is_appended_in_place(samples_dir, base_env, tmp_path):
    files = files_of(samples_dir)
    ingest(base_env, tmp_path / "store.zarr", files[0], "--create", "true")
    (tmp_path / "store.zarr").rename(tmp_path / "renamed")
    report = ingest(base_env, tmp_path / "renamed", files[1])
    assert report["kind"] == "zarr" and report["n_cells"] == 340


def test_the_none_preset_skips_celltypist(samples_dir, base_env, store_path):
    env = {**base_env, "RUN_CELLTYPIST": "true"}
    report = ingest(env, store_path, samples_dir, "--create", "true", "--celltypist-models", "None")
    obs_columns = stores.obs_columns(root_of(store_path))
    assert report["n_samples"] == len(SAMPLES) and not [c for c in obs_columns if c.endswith("_label")]


def test_panel_values_in_any_shape_reach_the_run(samples_dir, base_env, tmp_path):
    files = files_of(samples_dir)
    store = str(tmp_path / "store")
    samples = f'["{files[0]}",\n {samples_dir}/ ; {files[1]}]'
    env = {**base_env, "RUN_CELLTYPIST": "true", "CREATE": "true"}
    report = main.run(load_settings([f" {store}/ ", "", samples, "None"], env))
    assert report["appended_samples"] == ["SP00001", "SP00002", "SP00003"]
    report = main.run(load_settings([store, str(samples_dir / "*.h5"), "None"], env))
    assert report["appended_samples"] == [] and report["store"] == store


def test_input_guards(base_env, store_path, tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ConfigError, match="holds no .h5"):
        ingest(base_env, store_path, tmp_path / "empty", "--create", "true")
    with pytest.raises(ConfigError, match="not a .h5/.h5ad file"):
        ingest(base_env, store_path, tmp_path / "notes.txt", "--create", "true")
    with pytest.raises(inputs.IngestError, match="not found"):
        ingest(base_env, store_path, tmp_path / "missing.h5", "--create", "true")


def test_var_mismatch_is_fitted_to_the_store_genes(samples_dir, base_env, store_path, tmp_path):
    ingest(base_env, store_path, files_of(samples_dir)[0], "--create", "true")
    other = write_hise_h5(tmp_path / "B092-P1_SP00009-01.h5", "SP00009", 50, seed=9, genes=GENES[:-1] + ["OTHER"])
    report = ingest(base_env, store_path, other)
    assert report["fitted"] == {"SP00009": "fitted to the store's genes: 1 zero-filled, 1 dropped"}
    assert stores.n_obs(root_of(store_path)) == 210 and list(stores.var_names(root_of(store_path))) == GENES


def test_origin_reads_the_mount_and_writes_the_origin(samples_dir, base_env, tmp_path):
    files = files_of(samples_dir)
    origin = str(tmp_path / "origin")
    ingest(base_env, origin, files[0], "--create", "true")
    mount = str(tmp_path / "mount")
    shutil.copytree(origin, mount)
    report = ingest(base_env, mount, files[1], "--origin", origin)
    assert report["store"] == origin and stores.n_obs(root_of(origin)) == 340 == report["n_cells"]
    assert stores.n_obs(root_of(mount)) == 160
    again = ingest(base_env, mount, f"{files[1]},{files[2]}", "--origin", origin)
    assert again["appended_samples"] == ["SP00003"] and again["n_cells"] == sum(SAMPLES.values())
    obs = read_elem(root_of(origin)["obs"])
    assert obs["sample_id"].value_counts().to_dict() == SAMPLES and stores.n_obs(root_of(mount)) == 160


def test_a_mistyped_origin_does_not_start_a_new_store(samples_dir, base_env, tmp_path):
    files = files_of(samples_dir)
    mount = str(tmp_path / "mount")
    ingest(base_env, mount, files[0], "--create", "true")
    with pytest.raises(inputs.IngestError, match="its origin .* doesn't"):
        ingest(base_env, mount, files[1], "--origin", str(tmp_path / "tpyo"))
    assert not (tmp_path / "tpyo").exists()


def test_read_only_icechunk_without_origin_is_refused(samples_dir, base_env, tmp_path, monkeypatch):
    store = str(tmp_path / "store")
    ingest(base_env, store, files_of(samples_dir)[0], "--create", "true")
    monkeypatch.setattr("annizarr.ic._repo.is_readonly_path", lambda p: True)
    with pytest.raises(stores.StoreError, match="read-only"):
        ingest(base_env, store, files_of(samples_dir)[1])


def test_unwritable_zarr_is_refused(samples_dir, base_env, tmp_path, monkeypatch):
    store = str(tmp_path / "store.zarr")
    ingest(base_env, store, files_of(samples_dir)[0], "--create", "true")
    real = zarr.open_group

    def read_only(path, mode="r", **kw):
        if mode != "r":
            raise PermissionError("read-only file system")
        return real(path, mode=mode, **kw)

    monkeypatch.setattr(stores.zarr, "open_group", read_only)
    with pytest.raises(stores.StoreError, match="not writable"):
        ingest(base_env, store, files_of(samples_dir)[1])


def test_origin_with_a_plain_zarr_is_refused(samples_dir, base_env, tmp_path):
    with pytest.raises(stores.StoreError, match="icechunk stores only"):
        ingest(base_env, tmp_path / "store.zarr", samples_dir, "--create", "true", "--origin", "s3://b/p")


def test_main_reports_user_errors_without_a_traceback(base_env, tmp_path, monkeypatch, caplog):
    for key, value in base_env.items():
        monkeypatch.setenv(key, value)
    code = main.main(["--store", str(tmp_path / "store"), "--samples", str(tmp_path)])
    assert code == 1 and "holds no .h5" in caplog.text and "Traceback" not in caplog.text


def test_main_reads_positional_panel_values_from_the_command_line(samples_dir, base_env, tmp_path, monkeypatch):
    for key, value in base_env.items():
        monkeypatch.setenv(key, value)
    store = str(tmp_path / "store")
    monkeypatch.setattr("sys.argv", ["main.py", store, "", str(samples_dir), "None"])
    monkeypatch.setenv("CREATE", "true")
    assert main.main() == 0 and stores.n_obs(root_of(store)) == sum(SAMPLES.values())


def test_create_refuses_a_folder_that_holds_other_files(samples_dir, base_env, tmp_path):
    folder = tmp_path / "precious.zarr"
    folder.mkdir()
    (folder / "notes.txt").write_text("keep me")
    with pytest.raises(stores.StoreError, match="not a zarr store"):
        ingest(base_env, folder, samples_dir, "--create", "true")
    assert (folder / "notes.txt").read_text() == "keep me" and len(list(folder.iterdir())) == 1


def test_create_can_be_rerun_after_a_create_run_failed(samples_dir, base_env, store_path, tmp_path):
    bad = tmp_path / "B092-P1_SP00009-01.h5"
    bad.write_bytes(b"not an h5 file")
    files = files_of(samples_dir)
    with pytest.raises(inputs.IngestError, match="can't be read"):
        ingest(base_env, store_path, f"{files[0]},{bad}", "--create", "true")
    report = ingest(base_env, store_path, files[0], "--create", "true")
    assert report["appended_samples"] == ["SP00001"] and report["n_cells"] == 160


def test_samples_without_cells_are_skipped(samples_dir, base_env, store_path, tmp_path):
    empty = write_hise_h5(tmp_path / "B092-P1_SP00008-01.h5", "SP00008", 0, seed=8)
    report = ingest(base_env, store_path, f"{empty},{files_of(samples_dir)[0]}", "--create", "true")
    assert report["appended_samples"] == ["SP00001"] and report["n_samples"] == 1
    with pytest.raises(inputs.IngestError, match="none of the samples has cells"):
        ingest(base_env, store_path.replace("store", "other"), empty, "--create", "true")


def test_an_empty_first_sample_does_not_set_a_new_stores_genes(samples_dir, base_env, store_path, tmp_path):
    empty = write_hise_h5(tmp_path / "B092-P1_SP00008-01.h5", "SP00008", 0, seed=8, genes=GENES[::-1])
    report = ingest(base_env, store_path, f"{empty},{files_of(samples_dir)[0]},{files_of(samples_dir)[1]}",
                    "--create", "true")
    assert report["fitted"] == {} and list(stores.var_names(root_of(store_path))) == GENES


def test_a_sample_listed_twice_is_ingested_once(samples_dir, base_env, store_path, tmp_path):
    first = files_of(samples_dir)[0]
    copy = tmp_path / "SP00001.h5"
    shutil.copy(first, copy)
    report = ingest(base_env, store_path, f"{first},{copy}", "--create", "true")
    assert report["appended_samples"] == ["SP00001"] and report["n_cells"] == 160


def test_a_different_model_set_is_refused(samples_dir, base_env, store_path, models_dir):
    files = files_of(samples_dir)
    env = {**base_env, "RUN_CELLTYPIST": "true", "MODELS_DIR": str(models_dir)}
    ingest({**env, "CELLTYPIST_MODELS": "tiny_L1"}, store_path, files[0], "--create", "true")
    with pytest.raises(prepare.ContractError, match="tiny_L2_label"):
        ingest(env, store_path, files[1], "--celltypist-models", "tiny_L1,tiny_L2")
    report = ingest(env, store_path, files[1], "--celltypist-models", str(models_dir / "tiny_L1.pkl"))
    assert report["appended_samples"] == ["SP00002"]


def test_a_missing_model_stops_the_run_before_the_store_is_touched(samples_dir, base_env, store_path, models_dir):
    env = {**base_env, "RUN_CELLTYPIST": "true", "MODELS_DIR": str(models_dir)}
    with pytest.raises(inputs.IngestError, match="not_a_model.pkl"):
        ingest(env, store_path, samples_dir, "--create", "true", "--celltypist-models", "tiny_L1,not_a_model")
    assert not stores.open_store(store_path).exists()


def test_a_sample_sharing_no_genes_is_refused(samples_dir, base_env, store_path, tmp_path):
    create(base_env, store_path, sorted(samples_dir.glob("*.h5"))[0])
    other = write_hise_h5(tmp_path / "B092-P1_SP00009-01.h5", "SP00009", 50, seed=9,
                          genes=[f"ENSG{i:011d}" for i in range(len(GENES))])
    with pytest.raises(prepare.ContractError, match="shares no gene names"):
        create(base_env, store_path, other)


def test_an_existing_object_sharing_no_genes_is_refused(samples_dir, base_env, tmp_path):
    store = tmp_path / "store"
    create(base_env, store, sorted(samples_dir.glob("*.h5"))[0])
    obj = tmp_path / "symbols.h5ad"
    ad.AnnData(counts_matrix(30, 1).astype(np.float32), var=pd.DataFrame(index=[f"X{i}" for i in range(len(GENES))]),
               obs=pd.DataFrame({"sample_id": "Z"}, index=[f"c{i}" for i in range(30)])).write_h5ad(obj)
    with pytest.raises(prepare.ContractError, match="shares no gene names"):
        create(base_env, store, obj, "--input-type", "existing")


def test_a_truncated_input_stops_before_the_store_is_created(samples_dir, base_env, tmp_path):
    good = sorted(samples_dir.glob("*.h5"))[0]
    bad = tmp_path / "in" / "B092-P1_SP00008-01.h5"
    bad.parent.mkdir()
    bad.write_bytes(good.read_bytes()[:2000])
    with pytest.raises(inputs.IngestError, match="can't be read"):
        create(base_env, tmp_path / "store", bad.parent)
    assert not (tmp_path / "store").exists()


def test_a_garbage_model_stops_before_the_store_is_created(samples_dir, base_env, tmp_path):
    pkl = tmp_path / "junk.pkl"
    pkl.write_bytes(b"not a model")
    env = {**base_env, "RUN_CELLTYPIST": "true", "CELLTYPIST_PKL": str(pkl)}
    with pytest.raises(inputs.IngestError, match="can't be loaded"):
        create(env, tmp_path / "store", samples_dir)
    assert not (tmp_path / "store").exists()


def test_an_unknown_dropdown_value_lists_the_options(samples_dir, base_env, tmp_path):
    env = {**base_env, "RUN_CELLTYPIST": "true", "CELLTYPIST_MODELS": "AIFI L9 celltypes"}
    with pytest.raises(inputs.IngestError, match="Celltypist Model options are"):
        create(env, tmp_path / "store", samples_dir)


def test_a_plain_zarr_on_s3_is_refused():
    with pytest.raises(stores.StoreError, match="must be local"):
        stores.open_store("s3://bucket/atlas.zarr")


def test_an_icechunk_repo_named_zarr_opens_as_icechunk(samples_dir, base_env, tmp_path):
    store = tmp_path / "atlas.zarr"
    Repo.create(str(store))
    assert isinstance(stores.open_store(str(store)), stores.IcechunkStore)
    assert create(base_env, store, sorted(samples_dir.glob("*.h5"))[0])["kind"] == "icechunk"


def test_a_folder_of_other_files_is_not_made_an_icechunk_store(samples_dir, base_env, tmp_path):
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "readme.txt").write_text("x")
    with pytest.raises(stores.StoreError, match="not an icechunk repo"):
        create(base_env, folder, samples_dir)


def test_a_store_without_sample_id_is_a_clean_error(base_env, tmp_path, samples_dir):
    path = tmp_path / "foreign.zarr"
    ad.AnnData(counts_matrix(10, 0), var=pd.DataFrame(index=GENES)).write_zarr(path)
    with pytest.raises(stores.StoreError, match="no sample_id column"):
        create(base_env, path, samples_dir)


def test_a_dense_x_store_counts_its_cells(base_env, tmp_path):
    obj = tmp_path / "dense.zarr"
    ad.AnnData(counts_matrix(40, 2).toarray().astype(np.float32), var=pd.DataFrame(index=GENES),
               obs=pd.DataFrame({"sample_id": "D"}, index=[f"c{i}" for i in range(40)])).write_zarr(obj)
    report = create(base_env, tmp_path / "store", obj, "--input-type", "existing")
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
    report = create(base_env, tmp_path / "store", tmp_path / "multi.h5ad", "--input-type", "existing")
    stored = ad.read_zarr(Repo(str(tmp_path / "store")).open_zarr("r"))
    assert report["n_cells"] == n and stored.shape == (n, len(GENES))
    assert stored.layers["counts"].shape == (n, len(GENES)) and "loadings" not in stored.varm


def test_staging_cleanup_leaves_the_rest_of_the_staging_folder(samples_dir, base_env, tmp_path):
    keep = tmp_path / "staging" / "keep.txt"
    keep.parent.mkdir()
    keep.write_text("x")
    create(base_env, tmp_path / "store", sorted(samples_dir.glob("*.h5"))[0])
    assert keep.exists() and [p.name for p in keep.parent.iterdir()] == ["keep.txt"]


def test_the_second_commit_of_a_new_store_names_only_what_it_appends(samples_dir, base_env, tmp_path):
    report = create(base_env, tmp_path / "store", samples_dir)
    log = Repo(str(tmp_path / "store")).log()
    assert "SP00001" not in log[0].message and "SP00002" in log[0].message
    assert report["appended_samples"] == ["SP00001", "SP00002", "SP00003"]
    assert set(report["versions"]) >= {"annizarr", "icechunk", "anndata"}


def test_a_resequenced_sample_is_skipped_with_a_warning(samples_dir, base_env, store_path, tmp_path, caplog):
    first = sorted(samples_dir.glob("*.h5"))[0]
    create(base_env, store_path, first)
    again = tmp_path / "B093-P2_SP00001-02.h5"
    write_hise_h5(again, "SP00001", 50, seed=7)
    caplog.clear()
    assert create(base_env, store_path, again)["appended_samples"] == []
    assert "re-sequenced batch" in caplog.text and first.name in caplog.text and again.name in caplog.text
    caplog.clear()
    assert create(base_env, store_path, first)["appended_samples"] == []
    assert "re-sequenced batch" not in caplog.text
    opened = stores.open_store(store_path)
    opened.open(False)
    assert opened.recorded_sources()["SP00001"] == str(first)


def test_every_input_problem_is_listed_at_once(samples_dir, base_env, tmp_path):
    bad = tmp_path / "B092-P1_SP00008-01.h5"
    bad.write_bytes(b"not an h5 file")
    with pytest.raises(inputs.IngestError) as err:
        create(base_env, tmp_path / "store", f"{samples_dir},{bad},{tmp_path / 'gone.h5'}")
    assert "precheck found problems" in str(err.value)
    assert "not found" in str(err.value) and "can't be read" in str(err.value)


def test_a_new_store_keeps_an_existing_anndata_whole(base_env, tmp_path):
    path = tmp_path / "atlas.h5ad"
    existing(60, 1, ["A", "B"], extra={"score": 0.5}).write_h5ad(path)
    report = ingest_objects(base_env, tmp_path / "store", path)
    assert report["appended_samples"] == ["A", "B"] and report["n_cells"] == 60
    stored = read(tmp_path / "store")
    assert {"cell_type", "score", "sample_id"} <= set(stored.obs) and "X_umap" in stored.obsm
    assert "counts" in stored.layers and stored.X.dtype == np.float32
    assert ingest_objects(base_env, tmp_path / "store", path)["appended_samples"] == []


def test_later_objects_append_their_cells_and_extra_columns_are_dropped(base_env, tmp_path, caplog):
    first, second = tmp_path / "first.h5ad", tmp_path / "second.zarr"
    existing(60, 1, ["A"]).write_h5ad(first)
    existing(40, 2, ["B"], extra={"novel": 1}, dtype=np.uint32).write_zarr(second)
    ingest_objects(base_env, tmp_path / "store", first)
    report = ingest_objects(base_env, tmp_path / "store", second)
    assert report["appended_samples"] == ["B"] and report["n_cells"] == 100
    assert report["dropped"] == ["obsm/X_umap", "layers/counts"]
    assert "obs/novel" in caplog.text and "obsm/X_umap" in caplog.text
    stored = read(tmp_path / "store")
    assert "novel" not in stored.obs and stored.X.dtype == np.float32
    assert (stored[stored.obs["sample_id"] == "B"].X.toarray() == counts_matrix(40, 2).toarray()).all()


def test_an_object_partly_in_the_store_is_refused(base_env, tmp_path):
    existing(60, 1, ["A"]).write_h5ad(tmp_path / "a.h5ad")
    existing(40, 2, ["A", "C"]).write_h5ad(tmp_path / "ac.h5ad")
    ingest_objects(base_env, tmp_path / "store", tmp_path / "a.h5ad")
    with pytest.raises(inputs.IngestError, match="would be duplicated"):
        ingest_objects(base_env, tmp_path / "store", tmp_path / "ac.h5ad")


def test_an_object_without_sample_id_is_named_after_its_file(base_env, tmp_path):
    path = tmp_path / "donor7.h5ad"
    existing(30, 3, None).write_h5ad(path)
    report = ingest_objects(base_env, tmp_path / "store.zarr", path)
    assert report["appended_samples"] == ["donor7"] and set(read(tmp_path / "store.zarr").obs["sample_id"]) == {"donor7"}


def test_sample_ids_can_come_from_another_column(base_env, tmp_path):
    adata = existing(40, 4, None, extra={"donor": np.resize(["d1", "d2"], 40)})
    adata.write_h5ad(tmp_path / "rna.h5ad")
    report = ingest_objects(base_env, tmp_path / "store", tmp_path / "rna.h5ad", SAMPLE_COLUMN="donor")
    assert report["appended_samples"] == ["d1", "d2"]
    assert set(read(tmp_path / "store").obs["sample_id"]) == {"d1", "d2"}


def test_an_anndata_zarr_folder_without_the_suffix_is_one_object(base_env, tmp_path):
    existing(30, 3, ["Z"]).write_zarr(tmp_path / "asset")
    assert ingest_objects(base_env, tmp_path / "store", tmp_path / "asset")["appended_samples"] == ["Z"]


def test_raw_samples_join_a_store_seeded_from_an_existing_anndata(base_env, tmp_path, caplog):
    existing(60, 1, ["A"]).write_h5ad(tmp_path / "a.h5ad")
    ingest_objects(base_env, tmp_path / "store", tmp_path / "a.h5ad")
    raw = write_hise_h5(tmp_path / "B092-P1_SP00001-01.h5", "SP00001", 50, seed=5)
    report = ingest_objects(base_env, tmp_path / "store", raw, input_type="raw", CELLTYPIST_MODELS="None")
    assert report["appended_samples"] == ["SP00001"] and report["n_cells"] == 110
    assert "raw-sample columns aren't kept" in caplog.text
    assert read(tmp_path / "store").X.dtype == np.float32


def test_normalized_data_cannot_join_a_counts_store(base_env, tmp_path, samples_dir):
    raw = write_hise_h5(tmp_path / "B092-P1_SP00001-01.h5", "SP00001", 50, seed=5)
    ingest_objects(base_env, tmp_path / "store", raw, input_type="raw", CELLTYPIST_MODELS="None")
    norm = existing(30, 2, ["N"])
    norm.X = norm.X * 0.5
    norm.write_h5ad(tmp_path / "norm.h5ad")
    with pytest.raises(prepare.ContractError, match="non-count values"):
        ingest_objects(base_env, tmp_path / "store", tmp_path / "norm.h5ad")


def test_existing_mode_points_raw_h5_files_to_raw_mode(base_env, tmp_path):
    raw = write_hise_h5(tmp_path / "B092-P1_SP00001-01.h5", "SP00001", 50, seed=5)
    with pytest.raises(inputs.IngestError, match="Raw samples"):
        ingest_objects(base_env, tmp_path / "store", raw)


def test_several_objects_seed_a_new_store_in_one_run(base_env, tmp_path):
    folder = tmp_path / "objects"
    (folder / "a").mkdir(parents=True)
    (folder / "b").mkdir()
    existing(30, 1, ["A"]).write_h5ad(folder / "a" / "rna.h5ad")
    existing(30, 2, ["B"]).write_h5ad(folder / "b" / "rna.h5ad")
    report = ingest_objects(base_env, tmp_path / "store", f"{folder / 'a' / 'rna.h5ad'},{folder / 'b' / 'rna.h5ad'}")
    assert report["appended_samples"] == ["A", "B"] and report["n_cells"] == 60
    assert report["dropped"] == ["obsm/X_umap", "layers/counts"]
    assert set(report["sources"]) == {"rna", "rna1"}


def test_shared_barcodes_from_different_samples_stay_distinct(base_env, tmp_path):
    a, b = existing(30, 1, ["A"]), existing(30, 1, ["B"])
    a.write_h5ad(tmp_path / "a.h5ad")
    b.write_h5ad(tmp_path / "b.h5ad")
    ingest_objects(base_env, tmp_path / "store", tmp_path / "a.h5ad")
    report = ingest_objects(base_env, tmp_path / "store", tmp_path / "b.h5ad")
    stored = read(tmp_path / "store")
    assert report["n_cells"] == 60 and stored.obs_names.is_unique and stored.obs_names[0] == "A:s1_c0"


def test_a_scanpy_qc_seed_still_takes_raw_samples(base_env, tmp_path, caplog):
    seed = existing(40, 1, ["A"])
    for c in ("n_genes_by_counts", "total_counts", "total_counts_mt", "pct_counts_mt"):
        seed.obs[c] = 1.0
    seed.write_h5ad(tmp_path / "a.h5ad")
    ingest_objects(base_env, tmp_path / "store", tmp_path / "a.h5ad")
    raw = write_hise_h5(tmp_path / "B092-P1_SP00001-01.h5", "SP00001", 50, seed=5)
    report = ingest_objects(base_env, tmp_path / "store", raw, input_type="raw", CELLTYPIST_MODELS="None")
    assert report["n_cells"] == 90 and "raw-sample columns aren't kept" in caplog.text


def test_obs_values_that_dont_fit_the_store_types_are_left_empty(base_env, tmp_path, caplog):
    existing(30, 1, ["A"], extra={"score": 0.5}).write_h5ad(tmp_path / "a.h5ad")
    existing(30, 2, ["B"], extra={"score": "high"}).write_h5ad(tmp_path / "b.h5ad")
    ingest_objects(base_env, tmp_path / "store", tmp_path / "a.h5ad")
    report = ingest_objects(base_env, tmp_path / "store", tmp_path / "b.h5ad")
    assert report["n_cells"] == 60 and "obs/score (values don't fit the store's type)" in caplog.text


def test_normalized_values_cannot_join_a_float_counts_store(base_env, tmp_path):
    existing(30, 1, ["A"]).write_h5ad(tmp_path / "a.h5ad")
    norm = existing(30, 2, ["B"])
    norm.X = norm.X * 0.5
    norm.write_h5ad(tmp_path / "b.h5ad")
    ingest_objects(base_env, tmp_path / "store", tmp_path / "a.h5ad")
    with pytest.raises(prepare.ContractError, match="non-count values"):
        ingest_objects(base_env, tmp_path / "store", tmp_path / "b.h5ad")


def test_missing_sample_ids_are_refused(base_env, tmp_path):
    adata = existing(30, 1, ["A"])
    adata.obs.loc[adata.obs_names[:3], "sample_id"] = np.nan
    adata.write_h5ad(tmp_path / "a.h5ad")
    with pytest.raises(inputs.IngestError, match="some cells have no sample_id"):
        ingest_objects(base_env, tmp_path / "store", tmp_path / "a.h5ad")


def test_a_missing_sample_column_falls_back_to_sample_id(base_env, tmp_path, caplog):
    existing(30, 1, ["A"]).write_h5ad(tmp_path / "a.h5ad")
    report = ingest_objects(base_env, tmp_path / "store", tmp_path / "a.h5ad", SAMPLE_COLUMN="donor")
    assert report["appended_samples"] == ["A"] and "no 'donor' column" in caplog.text


def test_a_different_panel_is_fitted_block_by_block(base_env, tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "ROWS_PER_BLOCK", 7)
    existing(30, 1, ["A"]).write_h5ad(tmp_path / "a.h5ad")
    genes = GENES[10:] + ["NEW"]
    existing(25, 2, ["B"], genes=genes, dtype=np.uint32).write_h5ad(tmp_path / "b.h5ad")
    ingest_objects(base_env, tmp_path / "store", tmp_path / "a.h5ad")
    report = ingest_objects(base_env, tmp_path / "store", tmp_path / "b.h5ad")
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
    ingest_objects(base_env, tmp_path / "store", tmp_path / "a.h5ad")
    assert "not added, since the store has no place for them: raw" in caplog.text
    report = ingest_objects(base_env, tmp_path / "store", f"{tmp_path / 'b.h5ad'},{tmp_path / 'c.h5ad'}")
    assert report["appended_samples"] == ["B", "C"] and report["n_cells"] == 90
    assert read(tmp_path / "store").raw is None


def test_a_zarr_v2_anndata_is_copied_into_v3_staging(base_env, tmp_path, monkeypatch):
    monkeypatch.setattr(ad.settings, "zarr_write_format", 2)
    existing(30, 1, ["A"]).write_zarr(tmp_path / "a.zarr")
    existing(25, 2, ["B"], genes=GENES[10:]).write_zarr(tmp_path / "b.zarr")
    monkeypatch.setattr(ad.settings, "zarr_write_format", 3)
    ingest_objects(base_env, tmp_path / "store", tmp_path / "a.zarr")
    assert "X_umap" in read(tmp_path / "store").obsm
    report = ingest_objects(base_env, tmp_path / "store", tmp_path / "b.zarr")
    assert report["n_cells"] == 55 and report["fitted"] == {"b": "fitted to the store's genes: 10 zero-filled, 0 dropped"}
    cells = read(tmp_path / "store")
    cells = cells[cells.obs["sample_id"] == "B"]
    assert (cells.X[:, 10:].toarray() == counts_matrix(25, 2)[:, :len(GENES) - 10].toarray()).all()


def test_a_stale_mount_does_not_block_an_append_after_derived_elements_were_dropped(base_env, tmp_path):
    origin, mount = tmp_path / "origin", tmp_path / "mount"
    for name, seed, sample in (("a", 1, "A"), ("b", 2, "B"), ("c", 3, "C")):
        existing(30, seed, [sample]).write_h5ad(tmp_path / f"{name}.h5ad")
    ingest_objects(base_env, origin, tmp_path / "a.h5ad")
    shutil.copytree(origin, mount)
    first = ingest_objects(base_env, mount, tmp_path / "b.h5ad", ORIGIN=str(origin))
    assert first["dropped"] == ["obsm/X_umap", "layers/counts"]
    second = ingest_objects(base_env, mount, f"{tmp_path / 'b.h5ad'},{tmp_path / 'c.h5ad'}", ORIGIN=str(origin))
    assert second["appended_samples"] == ["C"] and second["dropped"] == [] and second["n_cells"] == 90


def test_objects_without_cells_are_skipped_before_staging(base_env, tmp_path, caplog):
    empty = existing(0, 1, None)
    empty.write_h5ad(tmp_path / "empty.h5ad")
    existing(30, 2, ["A"], genes=GENES[::-1]).write_h5ad(tmp_path / "a.h5ad")
    existing(30, 3, ["B"]).write_h5ad(tmp_path / "b.h5ad")
    report = ingest_objects(base_env, tmp_path / "store", f"{tmp_path / 'empty.h5ad'},{tmp_path / 'a.h5ad'}")
    assert report["appended_samples"] == ["A"] and "empty.h5ad has no cells; skipped" in caplog.text
    report = ingest_objects(base_env, tmp_path / "store", f"{tmp_path / 'empty.h5ad'},{tmp_path / 'b.h5ad'}")
    assert report["appended_samples"] == ["B"] and list(read(tmp_path / "store").var_names) == GENES[::-1]


def test_spatialdata_folders_are_listed_and_ingested_with_their_sources(base_env, tmp_path):
    folder = tmp_path / "sdata"
    folder.mkdir()
    a = write_spatialdata(folder / "Breast_A.zarr", 40, seed=1)
    b = write_spatialdata(folder / "Breast_B.zarr", 50, seed=2)
    store = tmp_path / "store"
    report = create(base_env, store, folder)
    assert report["appended_samples"] == ["Breast_A", "Breast_B"] and report["n_cells"] == 90
    assert report["sources"] == {"Breast_A": str(a), "Breast_B": str(b)}
    head = Repo(str(store)).log()[0]
    # the seed commit holds Breast_A; the run's commit records only what it appended
    assert head.metadata["sources"] == {"Breast_B": str(b)} and f"- Breast_B: {b}" in head.message
    assert create(base_env, store, folder)["appended_samples"] == []


def test_a_zarr_that_is_not_anndata_or_spatialdata_is_refused(base_env, tmp_path):
    zarr.open_group(tmp_path / "plain.zarr", mode="w")
    with pytest.raises(inputs.IngestError, match="must be an AnnData, or a SpatialData"):
        create(base_env, tmp_path / "store", tmp_path / "plain.zarr")


def test_mudata_is_refused_by_name(base_env, tmp_path):
    root = zarr.open_group(tmp_path / "sample_mudata", mode="w")
    root.attrs["encoding-type"] = "MuData"
    root.create_group("mod")
    with pytest.raises(inputs.IngestError, match="is MuData"):
        create(base_env, tmp_path / "store", tmp_path / "sample_mudata")


def test_a_different_gene_panel_warns_and_is_fitted_to_the_store_genes(base_env, tmp_path, caplog):
    store = tmp_path / "store"
    create(base_env, store, write_hise_h5(tmp_path / "B092-P1_SP00001-01.h5", "SP00001", 40, seed=1))
    path = write_spatialdata(tmp_path / "B.zarr", 30, seed=2, genes=GENES[3:] + ["NEW1", "NEW2"])
    report = create(base_env, store, path)
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
    assert (cells.X[:, 3:].toarray() == inputs.read_sample(path)[:, GENES[3:]].X.toarray()).all()


def test_a_new_store_fits_later_samples_to_the_first_samples_genes(base_env, tmp_path):
    store = tmp_path / "store.zarr"
    a = write_spatialdata(tmp_path / "A.zarr", 40, seed=1)
    b = write_spatialdata(tmp_path / "B.zarr", 30, seed=2, genes=GENES[::-1])
    report = create(base_env, store, f"{a},{b}")
    assert report["n_cells"] == 70 and report["fitted"] == {"B": "genes reordered to the store's"}
    stored = ad.read_zarr(store)
    assert list(stored.var_names) == GENES
    cells = stored[stored.obs["sample_id"] == "B"]
    assert (cells.X.toarray() == inputs.read_sample(b)[:, GENES].X.toarray()).all()


def test_storage_warning_names_refreshing_locations():
    assert "durable S3" in main.storage_warning("/scratch/x") and "data asset is read-only" in main.storage_warning("/results/x")
    assert main.storage_warning("s3://bucket/x") is None and main.storage_warning("/data/asset") is None


# User cases and edge cases: short runs of what people actually hand the capsule.

def test_repeated_barcodes_in_one_sample_get_unique_cell_names(base_env, tmp_path, caplog):
    path = tmp_path / "lanes.h5ad"
    lanes = ad.AnnData(counts_matrix(40, 1).astype(np.float32), var=pd.DataFrame(index=GENES))
    lanes.obs_names = ["AAAC-1"] * 40
    lanes.write_h5ad(path)
    create(base_env, tmp_path / "store", path)
    assert read(tmp_path / "store").obs_names.is_unique and "repeated barcodes" in caplog.text


def test_repeated_barcodes_in_an_existing_object_get_unique_cell_names(base_env, tmp_path):
    path = tmp_path / "atlas.h5ad"
    atlas = ad.AnnData(counts_matrix(30, 2).astype(np.float32), var=pd.DataFrame(index=GENES),
                       obs=pd.DataFrame({"sample_id": "A"}, index=["c"] * 30))
    atlas.write_h5ad(path)
    create(base_env, tmp_path / "store", path, "--input-type", "existing")
    assert read(tmp_path / "store").obs_names.is_unique


@pytest.mark.parametrize("n_cells", [1, 3])
def test_a_tiny_sample_goes_in_with_na_doublets_and_no_traceback(base_env, tmp_path, caplog, n_cells):
    path = write_hise_h5(tmp_path / "B092-P1_SP09001-01.h5", "SP09001", n_cells, 2)
    report = create(base_env, tmp_path / "store", path, RUN_SCRUBLET="true")
    assert report["n_cells"] == n_cells and "Traceback" not in caplog.text


def test_cells_with_zero_counts_are_kept_and_labelled(base_env, tmp_path, models_dir):
    x = counts_matrix(40, 3).tolil()
    x[:5] = 0
    path = tmp_path / "zeros.h5ad"
    ad.AnnData(x.tocsr().astype(np.float32), var=pd.DataFrame(index=GENES)).write_h5ad(path)
    create(base_env, tmp_path / "store", path, RUN_SCRUBLET="true", RUN_CELLTYPIST="true",
           MODELS_DIR=str(models_dir), CELLTYPIST_MODELS="tiny_L1")
    obs = read(tmp_path / "store").obs
    assert len(obs) == 40 and obs["tiny_L1_label"].notna().all()


def test_negative_counts_are_refused(base_env, tmp_path):
    neg = ad.AnnData(counts_matrix(20, 6).astype(np.float32), var=pd.DataFrame(index=GENES))
    neg.X.data[0] = -1
    neg.write_h5ad(tmp_path / "neg.h5ad")
    with pytest.raises(prepare.ContractError, match="raw counts"):
        create(base_env, tmp_path / "store", tmp_path / "neg.h5ad")


def test_a_trailing_slash_on_the_store_path_is_the_same_store(samples_dir, base_env, tmp_path):
    first = files_of(samples_dir)[0]
    create(base_env, f"{tmp_path}/store.zarr/", first)
    assert ingest(base_env, tmp_path / "store.zarr", first)["appended_samples"] == []


def test_the_same_file_listed_by_folder_and_by_path_is_ingested_once(samples_dir, base_env, tmp_path):
    report = create(base_env, tmp_path / "store", f"{samples_dir},{files_of(samples_dir)[0]}")
    assert report["appended_samples"] == ["SP00001", "SP00002", "SP00003"] and report["n_cells"] == sum(SAMPLES.values())


def test_a_run_interrupted_before_the_commit_can_simply_be_rerun(samples_dir, base_env, tmp_path, monkeypatch):
    store, files = tmp_path / "store", files_of(samples_dir)
    create(base_env, store, files[0])
    monkeypatch.setattr(main, "commit", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        ingest(base_env, store, samples_dir)
    monkeypatch.undo()
    assert ingest(base_env, store, samples_dir)["appended_samples"] == ["SP00002", "SP00003"]


def test_a_sample_without_the_models_genes_is_refused_before_the_store_is_created(base_env, tmp_path, models_dir):
    path = write_hise_h5(tmp_path / "B092-P1_SP09009-01.h5", "SP09009", 30, 9, genes=[f"ENSG{i:011d}" for i in range(500)])
    with pytest.raises(inputs.IngestError, match="shares no genes with the CellTypist model tiny_L1"):
        create(base_env, tmp_path / "store", path, RUN_CELLTYPIST="true", MODELS_DIR=str(models_dir),
               CELLTYPIST_MODELS="tiny_L1")
    assert not (tmp_path / "store").exists()


def test_a_model_sharing_few_genes_only_warns(base_env, tmp_path, models_dir, caplog):
    path = write_hise_h5(tmp_path / "B092-P1_SP09010-01.h5", "SP09010", 30, 10, genes=GENES[:100] + [f"X{i}" for i in range(400)])
    create(base_env, tmp_path / "store", path, RUN_CELLTYPIST="true", MODELS_DIR=str(models_dir), CELLTYPIST_MODELS="tiny_L1")
    assert "labels rest on few genes" in caplog.text
