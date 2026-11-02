import anndata as ad
import numpy as np
import pytest
import zarr

import samples
from conftest import build_spatial_store, spatial_table
from helpers import accept_all, cell_qc, decisions, exit_code, run, summary


@pytest.fixture(autouse=True)
def small_cohort(monkeypatch):
    monkeypatch.setattr(samples, "MIN_CELLS", 50)


@pytest.fixture
def spatial(tmp_path):
    return build_spatial_store(tmp_path / "sp.zarr", "zarr")


def test_report_on_a_spatialdata_table(spatial, results):
    run(results, input=spatial)
    d = decisions(results)
    assert d["profile"] == "spatial" and d["defaults"] == {"min_counts": 20, "min_genes": 5, "min_nuclei": 1}
    cells = cell_qc(results)
    assert set(cells["sample"]) == {f"cells_SP000{i}" for i in range(1, 6)}
    nuclei = spatial_table().obs["nucleus_count"].to_numpy()
    assert (cells["qc_fail_reason"] == "no_nucleus").sum() == (nuclei == 0).sum()
    assert [c for c in cells.columns if c not in ("sample", "qc_pass", "qc_fail_reason")] == ["n_genes_by_counts"]
    assert any((results / "plots" / "spatial").glob("tissue_qc_p*.png"))
    assert not (results / "plots" / "ambient").exists()
    text = summary(results)
    assert "mito" not in text.split("Input:")[0] and "computed from X: n_genes_by_counts (" in text
    assert "samples are the table's regions (region)" in text


def test_apply_writes_the_curated_table(spatial, results, tmp_path):
    run(results / "1", input=spatial)
    run(results / "2", input=spatial, decisions=accept_all(results / "1", results / "e.json"))
    output = str(tmp_path / "curated.zarr")
    run(results / "a", input=spatial, decisions=str(results / "2" / "qc_decisions.json"), mode="apply", target=output)
    keep = cell_qc(results / "a")["qc_pass"].to_numpy()
    store = zarr.open_group(output, mode="r")
    table = ad.read_zarr(store["tables/table"])
    source = spatial_table()
    assert table.n_obs == keep.sum() and list(table.obs_names) == list(source.obs_names[keep])
    np.testing.assert_array_equal(table.obsm["spatial"], source.obsm["spatial"][keep])
    np.testing.assert_array_equal(table.obs["total_counts"], source.obs["total_counts"][keep])
    assert table.uns["spatialdata_attrs"]["region_key"] == "region" and "qc" in table.uns
    assert store.attrs["spatialdata_attrs"] == {"version": "0.2"} and "images" not in store
    assert ad.read_zarr(zarr.open_group(spatial, mode="r")["tables/table"]).n_obs == len(keep)


def test_spatialdata_icechunk_is_refused(tmp_path, results, caplog):
    path = build_spatial_store(tmp_path / "sp", "icechunk")
    assert exit_code(results, input=path) == 1
    assert "SpatialData is supported as a plain .zarr only" in caplog.text


def test_scrna_decisions_are_refused_for_spatial(spatial, results, caplog, tmp_path):
    from helpers import write_json
    import decisions as dec
    from profiles import SCRNA
    path = write_json(tmp_path / "d.json", dec.empty(SCRNA))
    assert exit_code(results, input=spatial, decisions=path) == 1
    assert "for a scrna store; this input is spatial" in caplog.text


def test_table_choice(tmp_path, results, caplog):
    path = build_spatial_store(tmp_path / "sp.zarr", "zarr")
    assert exit_code(results, input=path, table="nope") == 1
    assert "pick one of its tables with TABLE (table)" in caplog.text
