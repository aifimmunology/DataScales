"""End-to-end runs on small synthetic samples, against a local icechunk repo and a plain zarr."""
import json
import shutil

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scrublet
import zarr
from anndata.io import read_elem
from annizarr import Repo

import annotate
import contract
import hise
import main
import store as stores
from config import load_settings
from conftest import GENES, SAMPLES, write_hise_h5


def ingest(env, store, samples, *flags):
    return main.run(load_settings(["--store", str(store), "--samples", str(samples), *flags], env))


def root_of(store):
    return zarr.open_group(store, mode="r") if store.endswith(".zarr") else Repo(store).open_zarr("r")


def log_of(store):
    return None if store.endswith(".zarr") else [s.id for s in Repo(store).log()]


def files_of(samples_dir):
    return sorted(samples_dir.glob("*.h5"))


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
    assert list(obs.columns) == list(contract.requested_schema(False, []))
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


def test_stored_counts_and_obs_match_the_source(samples_dir, base_env, store_path):
    files = files_of(samples_dir)
    ingest(base_env, store_path, f"{files[0]},{files[1]}", "--create", "true")
    root = root_of(store_path)
    X = read_elem(root["X"])
    src = hise.read_sample(files[1])
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
    h5ad_src = hise.read_sample(write_hise_h5(tmp_path / "SP00014.h5", "SP00014", 20, seed=14))
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
    with pytest.raises(contract.ContractError, match="scrublet_score"):
        ingest({**base_env, "RUN_SCRUBLET": "true"}, store_path, files[1])
    assert log_of(store_path) == before and stores.n_obs(root_of(store_path)) == 160


def test_failure_while_preparing_leaves_the_store_untouched(samples_dir, base_env, store_path, monkeypatch):
    files = files_of(samples_dir)
    ingest(base_env, store_path, files[0], "--create", "true")
    before = log_of(store_path)
    real = annotate.qc_metrics
    calls = []

    def fail_on_second(adata):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("sample is corrupt")
        real(adata)

    monkeypatch.setattr(annotate, "qc_metrics", fail_on_second)
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

    def fail_on_second(self, cells, cfg):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("network dropped")
        real(self, cells, cfg)

    monkeypatch.setattr(stores.IcechunkStore, "append", fail_on_second)
    with pytest.raises(RuntimeError, match="network"):
        ingest(base_env, store, samples_dir)
    assert log_of(store) == before and stores.n_obs(root_of(store)) == 160


def test_failed_bootstrap_append_keeps_the_seed_and_reruns_cleanly(samples_dir, base_env, tmp_path, monkeypatch):
    store = str(tmp_path / "store")
    files = files_of(samples_dir)
    def fail(self, cells, cfg):
        raise RuntimeError("boom")

    monkeypatch.setattr(stores.IcechunkStore, "append", fail)
    with pytest.raises(RuntimeError):
        ingest(base_env, store, f"{files[0]},{files[1]}", "--create", "true")
    assert len(log_of(store)) == 2 and stores.n_obs(root_of(store)) == 160
    monkeypatch.undo()
    report = ingest(base_env, store, f"{files[0]},{files[1]}")
    assert report["appended_samples"] == ["SP00002"] and report["n_cells"] == 340


def test_store_state_guards(samples_dir, base_env, store_path):
    with pytest.raises(main.IngestError, match="No store"):
        ingest(base_env, store_path, samples_dir)
    ingest(base_env, store_path, samples_dir, "--create", "true")
    with pytest.raises(main.IngestError, match="already holds a store"):
        ingest(base_env, store_path, samples_dir, "--create", "true")


def test_input_guards(base_env, store_path, tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(main.IngestError, match="No .h5"):
        ingest(base_env, store_path, tmp_path / "empty", "--create", "true")
    with pytest.raises(main.IngestError, match="not found"):
        ingest(base_env, store_path, tmp_path / "missing.h5", "--create", "true")


def test_var_mismatch_is_refused(samples_dir, base_env, store_path, tmp_path):
    ingest(base_env, store_path, files_of(samples_dir)[0], "--create", "true")
    other = write_hise_h5(tmp_path / "B092-P1_SP00009-01.h5", "SP00009", 50, seed=9, genes=GENES[:-1] + ["OTHER"])
    with pytest.raises(contract.ContractError, match="var mismatch"):
        ingest(base_env, store_path, other)
    assert stores.n_obs(root_of(store_path)) == 160


def test_origin_reads_the_mount_and_writes_the_origin(samples_dir, base_env, tmp_path):
    files = files_of(samples_dir)
    origin = str(tmp_path / "origin")
    ingest(base_env, origin, files[0], "--create", "true")
    mount = str(tmp_path / "mount")
    shutil.copytree(origin, mount)
    report = ingest(base_env, mount, files[1], "--origin", origin)
    assert report["store"] == origin and stores.n_obs(root_of(origin)) == 340
    assert stores.n_obs(root_of(mount)) == 160


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
    assert code == 1 and "No .h5" in caplog.text and "Traceback" not in caplog.text


def test_create_refuses_a_folder_that_holds_other_files(samples_dir, base_env, tmp_path):
    folder = tmp_path / "precious.zarr"
    folder.mkdir()
    (folder / "notes.txt").write_text("keep me")
    with pytest.raises(stores.StoreError, match="not a zarr store"):
        ingest(base_env, folder, samples_dir, "--create", "true")
    assert (folder / "notes.txt").read_text() == "keep me" and len(list(folder.iterdir())) == 1


def test_create_can_be_rerun_after_a_create_run_failed(samples_dir, base_env, store_path, tmp_path):
    bad = write_hise_h5(tmp_path / "B092-P1_SP00009-01.h5", "SP00009", 30, seed=9, genes=GENES[::-1])
    files = files_of(samples_dir)
    with pytest.raises(contract.ContractError):
        ingest(base_env, store_path, f"{files[0]},{bad}", "--create", "true")
    report = ingest(base_env, store_path, files[0], "--create", "true")
    assert report["appended_samples"] == ["SP00001"] and report["n_cells"] == 160


def test_samples_without_cells_are_skipped(samples_dir, base_env, store_path, tmp_path):
    empty = write_hise_h5(tmp_path / "B092-P1_SP00008-01.h5", "SP00008", 0, seed=8)
    report = ingest(base_env, store_path, f"{empty},{files_of(samples_dir)[0]}", "--create", "true")
    assert report["appended_samples"] == ["SP00001"] and report["n_samples"] == 1
    with pytest.raises(main.IngestError, match="none of the samples has cells"):
        ingest(base_env, store_path.replace("store", "other"), empty, "--create", "true")


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
    with pytest.raises(contract.ContractError, match="tiny_L2_label"):
        ingest(env, store_path, files[1], "--celltypist-models", "tiny_L1,tiny_L2")
    report = ingest(env, store_path, files[1], "--celltypist-models", str(models_dir / "tiny_L1.pkl"))
    assert report["appended_samples"] == ["SP00002"]


def test_a_missing_model_stops_the_run_before_the_store_is_touched(samples_dir, base_env, store_path, models_dir):
    env = {**base_env, "RUN_CELLTYPIST": "true", "MODELS_DIR": str(models_dir)}
    with pytest.raises(main.IngestError, match="not_a_model.pkl"):
        ingest(env, store_path, samples_dir, "--create", "true", "--celltypist-models", "tiny_L1,not_a_model")
    assert not stores.open_store(store_path).exists()
