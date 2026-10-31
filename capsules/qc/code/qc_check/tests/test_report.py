import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

import samples
from conftest import GENES, build_store, cohort
from helpers import accept_all, cell_qc, decisions, exit_code, run, summary, tree_digest, write_json


@pytest.fixture(autouse=True)
def small_cohort(monkeypatch):
    monkeypatch.setattr(samples, "MIN_CELLS", 50)


def test_report_writes_everything_and_never_touches_the_input(store, results):
    before = tree_digest(store)
    run(results, input=store)
    assert tree_digest(store) == before
    for name in ("qc_decisions.json", "cell_qc.parquet", "sample_qc.csv", "embedding.parquet", "report_summary.txt",
                 "run.json"):
        assert (results / name).is_file(), name
    assert not (results / "celltype_qc.csv").exists() and not (results / "apply_summary.csv").exists()
    for folder in ("samples", "groups", "embedding", "ambient"):
        assert any((results / "plots" / folder).glob("*.png")), folder

    table = pd.read_csv(results / "sample_qc.csv", index_col=0)
    assert "ambient:monocyte" in table.loc["SP0006", "flags"]
    assert {"ambient_monocyte_pct", "ambient_monocyte_p90", "ambient_monocyte_ratio", "fail_pct", "status"} <= set(table)
    d = decisions(results)
    flagged = set(table.index[table["flags"].notna()])
    assert set(d["pending"]) == flagged
    assert all(d["samples"][s] == {} for s in set(table.index) - flagged)

    cells = cell_qc(results)
    assert len(cells) == 720 and list(cells.columns) == ["sample", "qc_pass", "qc_fail_reason"]
    meta = pq.read_metadata(results / "cell_qc.parquet").metadata
    assert meta[b"previewed_sha256"].decode() == d["previewed_sha256"]
    assert meta[b"fingerprint"].decode() == d["input"]["fingerprint"]
    emb = pd.read_parquet(results / "embedding.parquet")
    assert list(emb.columns) == ["row", "sample_id", "umap_1", "umap_2", "leiden"]
    assert emb.groupby("sample_id").size().max() <= 80

    text = summary(results)
    assert text.startswith("Mode: report\napply would be refused: pending samples need a decision")
    assert "pool_id has a single value" in text


def test_preview_then_accepted(store, results):
    run(results / "1", input=store)
    edited = accept_all(results / "1", results / "edited.json", SP0002={"max_genes": 7000, "note": "deep"})
    run(results / "2", input=store, decisions=edited)
    d = decisions(results / "2")
    assert d["pending"] == [] and d["samples"]["SP0002"] == {"max_genes": 7000, "note": "deep"}
    assert "apply would be accepted" in summary(results / "2").splitlines()[1]
    assert pd.read_csv(results / "2" / "sample_qc.csv", index_col=0)["status"].eq("reviewed").all()


def test_decisions_drive_the_mask(store, results):
    run(results / "1", input=store)
    edited = accept_all(results / "1", results / "e.json", SP0003={"exclude": True})
    d = json.loads(Path(edited).read_text())
    d["defaults"]["min_genes"] = 280
    write_json(results / "e.json", d)
    run(results / "2", input=store, decisions=edited)
    cells = cell_qc(results / "2")
    assert not cells.loc[cells["sample"] == "SP0003", "qc_pass"].any()
    assert (cells.loc[cells["sample"] == "SP0003", "qc_fail_reason"] == "sample_excluded").all()
    assert (cells["qc_fail_reason"] == "low_genes").any()
    assert cells["qc_pass"].eq(cells["qc_fail_reason"].isna()).all()


def test_mito_warning(tmp_path, results):
    adatas = cohort(5)
    for a in adatas:
        a.var_names = [g.replace("MT-", "MITO") for g in GENES]
    run(results, input=build_store(tmp_path / "s.zarr", adatas, "zarr"))
    assert "var has no genes starting with MT-" in summary(results).split("Input:")[0]


def test_metrics_fallback(tmp_path, results):
    stored = cohort(5)
    bare = cohort(5, with_metrics=False)
    path = build_store(tmp_path / "bare.zarr", bare, "zarr")
    assert exit_code(results / "refused", input=path) == 1
    assert not (results / "refused" / "cell_qc.parquet").exists()
    run(results / "ok", input=path, compute_metrics="true")
    cells = cell_qc(results / "ok")
    expected = pd.concat([a.obs for a in stored])
    assert cells["n_genes_by_counts"].dtype == np.float32
    np.testing.assert_array_equal(cells["n_genes_by_counts"], expected["n_genes_by_counts"].astype(np.float32))
    np.testing.assert_allclose(cells["total_counts"], expected["total_counts"], rtol=1e-6)
    np.testing.assert_allclose(cells["pct_counts_mt"], expected["pct_counts_mt"], rtol=1e-5)
    assert "QC metrics computed from X" in summary(results / "ok")


def test_celltype_section(store, results):
    run(results / "off", input=store)
    assert not (results / "off" / "celltype_qc.csv").exists()
    run(results / "on", input=store, celltype_qc="true")
    table = pd.read_csv(results / "on" / "celltype_qc.csv", index_col=0)
    assert set(table.index) == {"T cell", "B cell", "NK cell", "Monocyte"}
    assert {"median_prob", "fail_pct_low_genes"} <= set(table)
    assert decisions(results / "on")["label_column"] == "pbmc_clean_AIFI_L1_2024_label"
    assert any((results / "on" / "plots" / "celltypes").glob("*.png"))
    run(results / "missing", input=store, celltype_qc="true", label_column="nope_label")
    assert "No `nope_label` column; per-cell-type QC skipped." in summary(results / "missing")
    assert not (results / "missing" / "celltype_qc.csv").exists()


def test_cell_types_ignored_when_off(store, results):
    run(results / "1", input=store)
    d = decisions(results / "1")
    d["cell_types"] = {"T cell": {"min_genes": 10_000}}
    run(results / "2", input=store, decisions=write_json(results / "e.json", d))
    assert "cell_types overrides are ignored" in summary(results / "2")
    assert cell_qc(results / "2")["qc_pass"].sum() == cell_qc(results / "1")["qc_pass"].sum()


def test_read_only_icechunk_without_origin_is_fine_for_a_report(tmp_path, results, monkeypatch):
    path = build_store(tmp_path / "ic", cohort(5), "icechunk")
    import annizarr.ic._repo as repo_module
    monkeypatch.setattr(repo_module, "is_readonly_path", lambda p: True)
    run(results, input=path)
    assert "the input is read-only; apply needs its writable origin" in summary(results)


def test_bad_inputs_fail_with_one_line(tmp_path, results):
    assert exit_code(results, input=str(tmp_path / "nowhere")) == 1
    assert exit_code(results, input=str(tmp_path / "nowhere"), decisions=str(tmp_path / "x.json")) == 1
