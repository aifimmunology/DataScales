import zipfile

import pytest

import conftest as fx
import inputs
from config import Member


def discover(tmp_path, *entries):
    return inputs.discover([str(e) for e in entries], tmp_path / "scratch")


def test_xenium_folder_is_one_sample_without_its_junk(tmp_path):
    folder = fx.xenium_bundle(tmp_path / "S1")
    fx.write(folder / ".DS_Store")
    fx.write(folder / "__MACOSX" / "._cells.parquet")
    found = discover(tmp_path, folder)
    (cand,) = found.candidates
    assert (cand.sid, cand.kind, cand.origin) == ("S1", "xenium", folder)
    assert {"cells.parquet", "cell_feature_matrix/matrix.mtx.gz", "morphology_focus/ch0000_dapi.ome.tif"} <= set(cand.files)
    assert not any("DS_Store" in k or "MACOSX" in k for k in cand.files)
    assert found.left_out == []


def test_zip_with_two_bundles_is_two_samples_read_from_the_archive(tmp_path):
    fx.xenium_bundle(tmp_path / "src" / "Breast")
    fx.xenium_bundle(tmp_path / "src" / "Lung")
    archive = fx.zip_folder(tmp_path / "src", tmp_path / "two_runs.zip")
    found = discover(tmp_path, archive)
    assert [c.sid for c in found.candidates] == ["Breast", "Lung"]
    breast = found.candidates[0]
    assert breast.origin == archive and breast.files["cells.parquet"].container == archive
    staged = inputs.fetch(breast, ["cells.parquet"], tmp_path / "stage")
    assert staged["cells.parquet"].read_bytes() == fx.content("cells.parquet")


def test_atera_bundle_is_found_with_or_without_its_marker(tmp_path):
    found = discover(tmp_path, fx.atera_bundle(tmp_path / "AteraKidney"),
                     fx.atera_bundle(tmp_path / "NoMarker", marker=False))
    assert [(c.sid, c.kind) for c in found.candidates] == [("AteraKidney", "atera"), ("NoMarker", "atera")]


def test_spatialdata_zarr_is_one_sample(tmp_path):
    store = fx.sd_v3(tmp_path / "atlas.zarr")
    (cand,) = discover(tmp_path, store).candidates
    assert (cand.sid, cand.kind, cand.zarr) == ("atlas", "spatialdata", Member(store, None, False, 0))


def test_per_file_downloads_are_grouped_one_sample_per_prefix(tmp_path):
    folder = tmp_path / "GSE_RAW"
    for prefix in ("GSM1_CtrlM1_", "GSM2_Tumor-"):
        for name in ("cell_feature_matrix.h5", "cells.parquet.gz", "transcripts.parquet.gz"):
            fx.write(folder / f"{prefix}{name}")
    found = discover(tmp_path, folder)
    assert sorted(c.sid for c in found.candidates) == ["GSM1_CtrlM1", "GSM2_Tumor"]
    cand = found.candidates[0]
    assert set(cand.files) == {"cell_feature_matrix.h5", "cells.parquet", "transcripts.parquet"}
    assert cand.files["cells.parquet"].gz


def test_mistyped_paths_are_reported_together_with_close_names(tmp_path):
    data = tmp_path / "data"
    fx.xenium_bundle(data / "Breast_2fov_outs")
    fx.xenium_bundle(data / "Lung_2fov_outs")
    with pytest.raises(inputs.InputError) as err:
        discover(tmp_path, data / "Breast_2fov_out", data / "Lung_2fov_outs", data / "Lugn_2fov_outs")
    message = str(err.value)
    assert message.startswith("2 inputs not found:")
    assert f"{data / 'Breast_2fov_out'}; closest in {data}: Breast_2fov_outs" in message
    assert f"{data / 'Lugn_2fov_outs'}; closest in {data}: Lung_2fov_outs" in message


def test_items_that_are_not_samples_are_left_out_with_a_reason(tmp_path):
    parent = tmp_path / "downloads"
    fx.xenium_bundle(parent / "S1")
    fx.write(parent / ".DS_Store")
    fx.write(parent / "notes.txt")
    fx.write(parent / "atlas.h5ad")
    with zipfile.ZipFile(parent / "qc_report.zip", "w") as zf:
        zf.writestr("report/index.html", "<html></html>")
    found = discover(tmp_path, parent)
    assert [c.sid for c in found.candidates] == ["S1"]
    left = {lo.path: lo.reason for lo in found.left_out}
    assert set(left) == {str(parent / n) for n in ("notes.txt", "atlas.h5ad", "qc_report.zip")}
    assert "anndata_ingest" in left[str(parent / "atlas.h5ad")]
    assert "no Xenium/Atera bundle" in left[str(parent / "qc_report.zip")]


def test_duplicate_sample_ids_are_refused(tmp_path):
    fx.xenium_bundle(tmp_path / "x1" / "S")
    fx.xenium_bundle(tmp_path / "x2" / "S")
    with pytest.raises(inputs.InputError, match=r"sample ID 'S'.*x1/S.*x2/S"):
        discover(tmp_path, tmp_path / "x1" / "S", tmp_path / "x2" / "S")


def test_inventory_lists_what_a_bundle_holds_and_what_it_lacks(tmp_path):
    folder = fx.xenium_full(tmp_path / "full")
    inv = inputs.inspect(fx.candidate(folder), tmp_path / "s")
    assert inv.missing_essential == [] and inv.n_cells == 30
    assert inv.chosen == {"matrix": "cell_feature_matrix.h5", "cells": "cells.parquet", "experiment": "experiment.xenium",
                          "cell_boundaries": "cell_boundaries.parquet",
                          "nucleus_boundaries": "nucleus_boundaries.parquet", "transcripts": "transcripts.parquet"}
    assert "morphology images" in inv.missing_optional
    (folder / "cell_feature_matrix.h5").unlink()
    inv = inputs.inspect(fx.candidate(folder), tmp_path / "s2")
    assert inv.missing_essential and inv.missing_essential[0].startswith("count matrix")
