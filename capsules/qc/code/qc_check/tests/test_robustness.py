import anndata as ad
import numpy as np
import pandas as pd
import pytest
from annizarr import Repo

import samples
import write
from conftest import LABEL, build_store, cohort
from helpers import accept_all, cell_qc, decisions, exit_code, run, write_json


@pytest.fixture(autouse=True)
def small_cohort(monkeypatch):
    monkeypatch.setattr(samples, "MIN_CELLS", 50)


def awkward_store(path) -> str:
    """Integer sample ids, one sample with no failing cells, and one with an all-NA group value."""
    adata = ad.concat(cohort(5), merge="same")
    ids = adata.obs["sample_id"].astype(str).str[2:].astype(int).to_numpy()
    adata.obs["sample_id"] = pd.Categorical(ids)
    adata.obs.loc[ids == 1, "predicted_doublets"] = 0
    adata.obs["pool_id"] = adata.obs["pool_id"].cat.add_categories(["P2"])
    adata.obs.loc[ids >= 4, "pool_id"] = "P2"
    adata.obs.loc[ids == 5, "pool_id"] = np.nan
    adata.write_zarr(path)
    return str(path)


def test_integer_ids_clean_samples_and_na_groups(tmp_path, results, monkeypatch):
    path = awkward_store(tmp_path / "awkward.zarr")
    monkeypatch.setattr(samples, "MIN_CELLS", 500)
    run(results, input=path)
    d = decisions(results)
    assert d["pending"] == ["1", "2", "3", "4", "5"] and d["samples"] == {}
    table = pd.read_csv(results / "sample_qc.csv", index_col=0)
    assert table.loc[1, "fail_pct"] == 0
    assert (results / "plots" / "groups" / "pool_id_cells_per_sample.png").is_file()


def test_na_metrics_are_missing_metrics(tmp_path, results, caplog):
    adata = ad.concat(cohort(5), merge="same")
    adata.obs.loc[adata.obs_names[:3], "pct_counts_mt"] = np.nan
    path = str(tmp_path / "na.zarr")
    adata.write_zarr(path)
    assert exit_code(results / "refused", input=path) == 1
    assert "pct_counts_mt absent or NA for some cells" in caplog.text
    run(results / "ok", input=path, compute_metrics="true")
    assert cell_qc(results / "ok")["pct_counts_mt"].notna().all()


def test_zarr_v2_input_applies(tmp_path, results):
    adata = ad.concat(cohort(5), merge="same")
    with ad.settings.override(zarr_write_format=2):
        adata.write_zarr(tmp_path / "v2.zarr")
    path = str(tmp_path / "v2.zarr")
    run(results / "1", input=path)
    run(results / "2", input=path, decisions=accept_all(results / "1", results / "e.json"))
    run(results / "a", input=path, decisions=str(results / "2" / "qc_decisions.json"), mode="apply",
        target=str(tmp_path / "out.zarr"))
    keep = cell_qc(results / "a")["qc_pass"].to_numpy()
    assert (ad.read_zarr(tmp_path / "out.zarr").X != adata.X[keep]).nnz == 0


def test_failed_write_never_moves_the_curation_branch(tmp_path, results, monkeypatch):
    store = build_store(tmp_path / "ic", cohort(5), "icechunk")
    run(results / "1", input=store)
    run(results / "2", input=store, decisions=accept_all(results / "1", results / "e.json"))
    file = str(results / "2" / "qc_decisions.json")

    def broken(*args, **kwargs):
        raise OSError("disk gone")

    real = write.copy_cells
    monkeypatch.setattr(write, "copy_cells", broken)
    with pytest.raises(OSError):
        run(results / "fail1", input=store, decisions=file, mode="apply")
    assert "qc" not in Repo(store).branches()

    monkeypatch.setattr(write, "copy_cells", real)
    run(results / "ok", input=store, decisions=file, mode="apply")
    tip = Repo(store, branch="qc").log()[0].id
    monkeypatch.setattr(write, "copy_cells", broken)
    with pytest.raises(OSError):
        run(results / "fail2", input=store, decisions=file, mode="apply")
    assert Repo(store, branch="qc").log()[0].id == tip


def test_apply_follows_the_files_label_column(store, kind, results, tmp_path):
    run(results / "1", input=store, celltype_qc="true")
    d = decisions(results / "1")
    d["cell_types"] = {"T cell": {"min_genes": 10_000}}
    d["samples"].update({s: {} for s in d["pending"]})
    edited = write_json(results / "e.json", d)
    run(results / "2", input=store, decisions=edited, celltype_qc="true")
    file = str(results / "2" / "qc_decisions.json")
    assert decisions(results / "2")["label_column"] == LABEL
    run(results / "a", input=store, decisions=file, mode="apply",
        target=str(tmp_path / "o.zarr") if kind == "zarr" else "")
    pd.testing.assert_frame_equal(cell_qc(results / "2"), cell_qc(results / "a"))
    assert (results / "a" / "celltype_qc.csv").is_file()


def test_malformed_decisions_fail_with_one_line(store, results, caplog):
    path = write_json(results / "bad.json", {"version": 1, "input": None})
    assert exit_code(results / "r", input=store, decisions=path) == 1
    assert "'input' must be an object" in caplog.text
