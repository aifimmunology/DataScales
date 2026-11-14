"""Discovery over the messy layouts real downloads come in (synthetic look-alikes from fixtures_archives)."""
import tarfile
import zipfile

import pytest

import config
import discover
import stage
from contracts import Member
from fixtures_archives import (anndata_zarr, atera_bundle, content, geo_flat, macos_zip, mb_flat, mixed_parent,
                               per_file_10x, pipestance_tar, plain_zarr, sd_v2_old, sd_v3, tar_folder,
                               two_bundle_zip, write, xenium_bundle, zip_folder)


def run(tmp_path, *entries):
    return discover.discover([str(e) for e in entries], tmp_path / "scratch")


def by_sid(found):
    return {c.sid: c for c in found.candidates}


def reasons(found):
    return {lo.path: lo.reason for lo in found.left_out}


# ------------------------------------------------------------------------------------------ names

@pytest.mark.parametrize("canon,gz,key,flag", [
    ("cells.parquet", True, "cells.parquet", True),
    ("cells.csv", True, "cells.csv.gz", False),                  # 10x ships it gzipped: read as is
    ("transcripts.csv", True, "transcripts.csv.gz", False),
    ("experiment.xenium", True, "experiment.xenium", True),
    ("morphology.ome.tif", True, "morphology.ome.tif", True),
    ("cell_feature_matrix.h5", False, "cell_feature_matrix.h5", False),
    ("matrix.mtx", True, "cell_feature_matrix/matrix.mtx.gz", False),
    ("features.tsv", True, "cell_feature_matrix/features.tsv.gz", False),
    ("morphology_focus_0001.ome.tif", True, "morphology_focus/morphology_focus_0001.ome.tif", True),
])
def test_canonical_keys(canon, gz, key, flag):
    assert discover.canonical_key(canon, gz) == (key, flag)


@pytest.mark.parametrize("name,expected", [
    ("GSM9829813_CtrlM1_cells.parquet.gz", ("GSM9829813_CtrlM1", "GSM9829813_CtrlM1_", "cells.parquet", True)),
    ("GSM9223468_Br2039-cell_boundaries.csv.gz", ("GSM9223468_Br2039", "GSM9223468_Br2039-",
                                                  "cell_boundaries.csv.gz", False)),
    ("GSE335761_S33126__cells.zarr.zip", ("GSE335761_S33126", "GSE335761_S33126__", "cells.zarr.zip", False)),
    ("MB263-experiment.xenium", ("MB263", "MB263-", "experiment.xenium", False)),
    ("csc_cell_feature_matrix.zarr.zip", ("", "", "csc_cell_feature_matrix.zarr.zip", False)),  # Atera, no prefix
    ("x_reseg_transcripts.parquet", ("x_reseg", "x_reseg_", "transcripts.parquet", False)),
    ("mycells.parquet", None),                                   # no separator before the 10x name
    ("3p_count_filtered_feature_bc_matrix.h5", None),            # Chromium matrices never match
    ("cells (2).parquet", None),
])
def test_split_name(name, expected):
    assert discover.split_name(name) == expected


@pytest.mark.parametrize("names,sid", [
    (["outs", "PatientA_run1", "runs"], "PatientA_run1"),
    (["Xenium_V1_human_Breast_2fov_outs"], "Xenium_V1_human_Breast_2fov"),
    (["run7-outs"], "run7"),
    (["data.zarr", "section7"], "section7"),
    (["combined_spatialdata_zarr", "1417642334"], "1417642334"),
    (["blobs.sdata.zarr"], "blobs"),
    (["XENIUM_TRANSFER", "Files", "S-BIAD1093"], "S-BIAD1093"),
    (["my_spatialdata", "donor3"], "donor3"),
])
def test_name_id(names, sid):
    assert discover.name_id(names) == sid


def test_sanitize():
    assert discover.sanitize("Breast 2fov")[0] == "Breast_2fov"
    assert "became" in discover.sanitize("Breast 2fov")[1]
    assert discover.sanitize("ok_id-1.2") == ("ok_id-1.2", None)


# ------------------------------------------------------------------------------------------ folders

def test_plain_bundle_folder(tmp_path):
    folder = xenium_bundle(tmp_path / "S1")
    write(folder / "._experiment.xenium")
    write(folder / ".DS_Store")
    write(folder / "__MACOSX" / "._cells.parquet")
    found = run(tmp_path, folder)
    (cand,) = found.candidates
    assert (cand.sid, cand.kind, cand.origin) == ("S1", "xenium", folder)
    assert set(cand.files) == set(discover.bundle_files(discover.DiskNode(folder))[0])
    assert "cell_feature_matrix/matrix.mtx.gz" in cand.files
    assert "morphology_focus/ch0000_dapi.ome.tif" in cand.files
    assert not any("._" in k or "DS_Store" in k or "MACOSX" in k for k in cand.files)
    assert cand.files["cells.parquet"] == Member(folder / "cells.parquet", None, False, len(content("cells.parquet")))
    assert not any(m.gz for m in cand.files.values())
    assert found.left_out == []


def test_outs_folder_and_its_parent_give_the_same_sample(tmp_path):
    xenium_bundle(tmp_path / "PatientA_run1" / "outs")
    xenium_bundle(tmp_path / "Breast_outs")
    for entry, sid in [(tmp_path / "PatientA_run1", "PatientA_run1"),
                       (tmp_path / "PatientA_run1" / "outs", "PatientA_run1"),
                       (tmp_path / "Breast_outs", "Breast")]:
        assert [c.sid for c in run(tmp_path, entry).candidates] == [sid]


def test_bundles_deep_below_the_input_are_found(tmp_path):
    xenium_bundle(tmp_path / "donor" / "section1" / "xenium_output" / "output-XETG1")
    xenium_bundle(tmp_path / "donor" / "section2" / "xenium_output" / "output-XETG2")
    found = run(tmp_path, tmp_path / "donor")
    assert sorted(by_sid(found)) == ["output-XETG1", "output-XETG2"]


def test_mixed_parent_folder(tmp_path):
    parent = mixed_parent(tmp_path / "downloads")
    found = run(tmp_path, parent)
    kinds = {c.sid: c.kind for c in found.candidates}
    assert kinds == {"PatientA_run1": "xenium", "AteraKidney": "atera", "AteraNoMarker": "atera",
                     "atlas": "spatialdata"}
    assert any("inferred" in n for n in by_sid(found)["AteraNoMarker"].notes)
    left = reasons(found)
    assert set(left) == {str(parent / "ssam.zarr"), str(parent / "notes.txt"), str(parent / "qc_report.zip")}
    assert "without SpatialData elements" in left[str(parent / "ssam.zarr")]
    assert "no Xenium/Atera bundle" in left[str(parent / "qc_report.zip")]
    assert by_sid(found)["atlas"].zarr == Member(parent / "atlas.zarr", None, False, 0)


def test_xenium_kind_is_inferred_without_marker(tmp_path):
    folder = xenium_bundle(tmp_path / "NoMarker", marker=False)
    (cand,) = run(tmp_path, folder).candidates
    assert cand.kind == "xenium" and "experiment.xenium" not in cand.files
    assert any("inferred" in n for n in cand.notes)


def test_explorer_style_bundle_is_xenium(tmp_path):
    folder = xenium_bundle(tmp_path / "Breast_xe", files=("experiment.xenium", "cell_feature_matrix.zarr.zip",
                                                          "cells.zarr.zip", "analysis.zarr.zip"))
    (cand,) = run(tmp_path, folder).candidates
    assert cand.kind == "xenium" and "cells.zarr.zip" in cand.files


def test_lone_marker_above_bundles_uses_the_bundles(tmp_path):
    top = tmp_path / "S-BIAD3103"
    write(top / "experiment.xenium")
    write(top / "._experiment.xenium")
    inner = xenium_bundle(top / "Files" / "H1_run", marker=False)
    write(inner / "._experiment.xenium")                        # AppleDouble decoy, the real marker is missing
    found = run(tmp_path, top)
    assert [c.sid for c in found.candidates] == ["H1_run"]
    assert "lone marker" in reasons(found)[str(top / "experiment.xenium")]


def test_marker_without_data_is_left_out(tmp_path):
    folder = tmp_path / "Xenium_V1_human_Lung_2fov"
    write(folder / "experiment.xenium")
    write(folder / "gene_panel.json")
    with pytest.raises(discover.InputError, match="no data files"):
        run(tmp_path, folder)


def test_matrix_only_folder_is_left_out(tmp_path):
    write(tmp_path / "byid" / "1" / "cell_feature_matrix.h5")
    write(tmp_path / "byid" / "2" / "cells.parquet")
    with pytest.raises(discover.InputError) as err:
        run(tmp_path, tmp_path / "byid")
    assert "no cells table" in str(err.value) and "no count matrix" in str(err.value)


def test_partial_bundle_with_marker_is_a_candidate(tmp_path):
    folder = xenium_bundle(tmp_path / "partial", files=("experiment.xenium", "transcripts.parquet"))
    (cand,) = run(tmp_path, folder).candidates
    assert set(cand.files) == {"experiment.xenium", "transcripts.parquet"}     # inventory reports what's missing


# ------------------------------------------------------------------------------------------ prefixed folders

def test_10x_per_file_prefixed_folder(tmp_path):
    folder = per_file_10x(tmp_path / "asis")
    (cand,) = run(tmp_path, folder).candidates
    assert cand.sid == "Xenium_FFPE_Human_Breast_Cancer_Rep1" and cand.kind == "xenium"
    assert set(cand.files) == {"experiment.xenium", "cell_feature_matrix.h5", "cells.parquet", "cells.zarr.zip",
                               "cell_boundaries.csv.gz", "transcripts.parquet", "morphology_mip.ome.tif",
                               "analysis.zarr.zip", "cell_feature_matrix.zarr.zip"}
    assert cand.files["cells.zarr.zip"].container.name == "Xenium_FFPE_Human_Breast_Cancer_Rep1_cells.zarr.zip"
    assert any("prefix" in n for n in cand.notes)


def test_geo_flat_folder_two_gsms(tmp_path):
    folder = geo_flat(tmp_path / "GSE_RAW")
    found = run(tmp_path, folder)
    cands = by_sid(found)
    assert sorted(cands) == ["GSM9543314_EX-LN1", "GSM9829813_CtrlM1"]
    a = cands["GSM9829813_CtrlM1"]
    assert a.kind == "xenium"
    assert {k: m.gz for k, m in a.files.items()} == {
        "experiment.xenium": True, "cells.parquet": True, "cell_boundaries.parquet": True,
        "transcripts.parquet": True, "reseg_transcripts.parquet": True, "morphology.ome.tif": True,
        "cell_feature_matrix.h5": False, "cell_metadata.csv.gz": False}
    assert a.files["reseg_transcripts.parquet"].container.name == "GSM9829813_CtrlM1_reseg_transcripts.parquet.gz"
    b = cands["GSM9543314_EX-LN1"]
    assert {k: m.gz for k, m in b.files.items()} == {
        "cell_feature_matrix/matrix.mtx.gz": False, "cell_feature_matrix/features.tsv.gz": False,
        "cell_feature_matrix/barcodes.tsv.gz": False, "cells.csv.gz": False, "nucleus_boundaries.parquet": True}
    assert any("inferred" in n for n in b.notes)
    left = reasons(found)
    assert str(folder / "GSM7782697_3p_count_filtered_feature_bc_matrix.h5") in left
    assert str(folder / "GSE336196_README.txt") in left


def test_one_gsm_folder_given_directly(tmp_path):
    folder = tmp_path / "GSM10073485"
    for name in ("cells.parquet", "cell_boundaries.parquet", "transcripts.parquet"):
        write(folder / f"GSM10073485_P10_{name}.gz")
    write(folder / "GSM10073485_P10_cell_feature_matrix.h5")
    (cand,) = run(tmp_path, folder).candidates
    assert cand.sid == "GSM10073485_P10" and cand.files["cells.parquet"].gz


def test_series_prefix_with_double_separator(tmp_path):
    folder = tmp_path / "series"
    write(folder / "GSE335761_S33126__cells.zarr.zip")
    write(folder / "GSE335761_S33126_cells.parquet.gz")
    write(folder / "GSE335761_S33126_cell_feature_matrix.h5")
    write(folder / "GSE335761_S33126_experiment.xenium.gz")
    (cand,) = run(tmp_path, folder).candidates
    assert cand.sid == "GSE335761_S33126"
    assert set(cand.files) == {"cells.zarr.zip", "cells.parquet", "cell_feature_matrix.h5", "experiment.xenium"}


def test_mb_prefixed_folder_two_samples(tmp_path):
    top = mb_flat(tmp_path / "S-BIAD1093")
    found = run(tmp_path, top)
    assert sorted(by_sid(found)) == ["MB263", "MB266"]
    mb = by_sid(found)["MB263"]
    assert "analysis.zarr.zip" in mb.files and "cells.zarr.zip" in mb.files       # bundle files, not samples
    left = reasons(found)
    assert any(p.endswith("MB263-*") and "no data files" in r for p, r in left.items())
    assert any(p.endswith("Fused_405nm.tif") for p in left)


def test_unprefixed_bundle_next_to_prefixed_sample(tmp_path):
    folder = xenium_bundle(tmp_path / "mixed", files=("experiment.xenium", "cell_feature_matrix.h5", "cells.parquet"),
                           marker=False)
    write(folder / "B_cell_feature_matrix.h5")
    write(folder / "B_cells.parquet")
    found = by_sid(run(tmp_path, folder))
    assert set(found["mixed"].files) == {"cell_feature_matrix.h5", "cells.parquet"}
    assert set(found["B"].files) == {"cell_feature_matrix.h5", "cells.parquet"}


def test_browser_renamed_copies_are_left_out(tmp_path):
    folder = tmp_path / "flat_browser"
    for n in ("", " (1)", " (2)"):
        write(folder / f"cells{n}.parquet")
        write(folder / f"cell_feature_matrix{n}.h5")
    with pytest.raises(discover.InputError, match="browser-renamed"):
        run(tmp_path, folder)


# ------------------------------------------------------------------------------------------ archives

def test_two_bundle_zip_gives_two_samples(tmp_path):
    archive = two_bundle_zip(tmp_path)
    found = run(tmp_path, archive)
    assert [c.sid for c in found.candidates] == ["Xenium_V1_human_Breast_2fov", "Xenium_V1_human_Lung_2fov"]
    breast = found.candidates[0]
    assert breast.origin == archive
    assert breast.files["cells.parquet"] == Member(archive, "Xenium_V1_human_Breast_2fov/cells.parquet", False,
                                                   len(content("cells.parquet")))


def test_zip_with_bundle_at_root_and_outs_root(tmp_path):
    bundle = xenium_bundle(tmp_path / "src")
    at_root = zip_folder(bundle, tmp_path / "Breast_outs.zip", arcroot="")
    in_outs = zip_folder(bundle, tmp_path / "Rep1_outs.zip", arcroot="outs")
    assert [c.sid for c in run(tmp_path, at_root).candidates] == ["Breast"]
    assert [c.sid for c in run(tmp_path, in_outs).candidates] == ["Rep1"]


def test_pipestance_tar_gz(tmp_path):
    archive = pipestance_tar(tmp_path)
    (cand,) = run(tmp_path, archive).candidates
    assert cand.sid == "GSM7990532_output-XETG00048__0003392"
    assert not any("RANGER" in k or k.startswith("_") or "journal" in k for k in cand.files)
    member = cand.files["cell_feature_matrix.h5"]
    assert member.container == archive and member.inner.endswith("/outs/cell_feature_matrix.h5")
    staged = stage.fetch(cand, ["cell_feature_matrix.h5", "cell_feature_matrix/"], tmp_path / "stage")
    assert staged["cell_feature_matrix.h5"].read_bytes() == content("cell_feature_matrix.h5")
    assert (tmp_path / "stage" / "cell_feature_matrix" / "matrix.mtx.gz").exists()


def test_extracted_pipestance_folder_and_its_outs(tmp_path):
    archive = pipestance_tar(tmp_path)

    with tarfile.open(archive) as tf:
        tf.extractall(tmp_path / "x", filter="data")
    run_dir = next((tmp_path / "x").iterdir())
    assert [c.sid for c in run(tmp_path, run_dir).candidates] == [run_dir.name]
    assert [c.sid for c in run(tmp_path, run_dir / "outs").candidates] == [run_dir.name]
    assert run(tmp_path, run_dir).left_out == []                  # Ranger internals are silent junk


def test_macos_zip(tmp_path):
    archive = macos_zip(tmp_path)
    found = run(tmp_path, archive)
    (cand,) = found.candidates
    assert cand.sid == "Xenium_V1_human_Breast_2fov"
    assert not any("MACOSX" in m.inner or "/._" in m.inner or "DS_Store" in m.inner for m in cand.files.values())
    assert found.left_out == []


def test_folder_with_archives_inside(tmp_path):
    parent = tmp_path / "figshare"
    parent.mkdir()
    two_bundle_zip(tmp_path).rename(parent / "Methods manuscript.zip")
    xenium_bundle(tmp_path / "src")
    zip_folder(tmp_path / "src", parent / "output-XETG_1554HairFollicles.zip", arcroot="output-XETG_1554HF")
    write(parent / "gene_list.csv")
    found = run(tmp_path, parent)
    assert sorted(by_sid(found)) == ["Xenium_V1_human_Breast_2fov", "Xenium_V1_human_Lung_2fov", "output-XETG_1554HF"]
    assert list(reasons(found)) == [str(parent / "gene_list.csv")]


def test_zip_and_its_extracted_copy_count_once(tmp_path, caplog):
    parent = tmp_path / "dl"
    bundle = xenium_bundle(parent / "Breast_outs")
    zip_folder(bundle, parent / "Breast_outs.zip")
    with caplog.at_level("INFO", logger="discover"):
        found = run(tmp_path, parent)
    (cand,) = found.candidates
    assert cand.origin == bundle
    assert found.left_out == []                       # not an item "that isn't a sample": one INFO line instead
    assert f"using the folder {bundle}; the archive {parent / 'Breast_outs.zip'} holds the same bundle" in caplog.text


def test_stray_bundle_files_are_listed_once(tmp_path):
    parent = tmp_path / "flat"
    xenium_bundle(parent / "S1")
    zip_folder(xenium_bundle(tmp_path / "src" / "S2"), parent / "S2_outs.zip")
    write(parent / "analysis.zarr.zip")
    write(parent / "analysis_summary.html")
    write(parent / ".DS_Store")
    write(parent / "__MACOSX" / "._S1")
    found = run(tmp_path, parent)
    assert sorted(by_sid(found)) == ["S1", "S2"]
    text = " ".join(f"{lo.path} {lo.reason}" for lo in found.left_out)
    assert text.count("analysis.zarr.zip") == 1 and len(found.left_out) == 1


def test_geo_sidecars_beside_an_outs_zip_are_one_line(tmp_path):
    folder = tmp_path / "GSE243280"
    folder.mkdir()
    prefix = "GSM7780153_Xenium_FFPE_Human_Breast_Cancer_Rep1"
    zip_folder(xenium_bundle(tmp_path / "src" / "outs"), folder / f"{prefix}_outs.zip")
    for name in ("gene_panel.json.gz", "panel.tsv.gz", "gene_groups.csv.gz"):
        write(folder / f"{prefix}_{name}")
    write(folder / "GSM7782697_3p_count_filtered_feature_bc_matrix.h5")
    found = run(tmp_path, folder)
    assert [c.sid for c in found.candidates] == [prefix]
    left = reasons(found)
    assert left[f"{folder / prefix}*"].startswith(f"3 extra file(s) beside {prefix}_outs.zip")
    assert str(folder / "GSM7782697_3p_count_filtered_feature_bc_matrix.h5") in left
    assert len(left) == 2


def test_big_archives_are_announced_in_decimal_gb(tmp_path, caplog, monkeypatch):
    archive = two_bundle_zip(tmp_path)
    real = discover._size
    monkeypatch.setattr(discover, "_size", lambda p: 4_900_000_000 if p == archive else real(p))
    with caplog.at_level("INFO", logger="discover"):
        run(tmp_path, archive)
    assert "Listing the archive two_runs.zip (4.9 GB)" in caplog.text


def test_unsupported_and_broken_archives(tmp_path):
    xenium_bundle(tmp_path / "in" / "S1")
    write(tmp_path / "in" / "regions.rar")
    write(tmp_path / "in" / "empty.zip", b"")
    write(tmp_path / "in" / "broken.zip", b"PK\x03\x04 not really")
    write(tmp_path / "in" / "broken.tar.gz", b"\x1f\x8b garbage")
    left = reasons(run(tmp_path, tmp_path / "in"))
    assert "Extract it" in left[str(tmp_path / "in" / "regions.rar")]
    assert "empty file" in left[str(tmp_path / "in" / "empty.zip")]
    assert "not a valid .zip" in left[str(tmp_path / "in" / "broken.zip")]
    assert "can't be read as an archive" in left[str(tmp_path / "in" / "broken.tar.gz")]


def test_archive_inside_a_zip_is_left_out(tmp_path):
    inner = two_bundle_zip(tmp_path)
    outer = tmp_path / "outer.zip"
    with zipfile.ZipFile(outer, "w") as zf:
        zf.write(inner, "nested/two_runs.zip")
        zf.writestr("nested/readme.txt", "x")
    with pytest.raises(discover.InputError, match="archive inside an archive|no Xenium"):
        run(tmp_path, outer)


# ------------------------------------------------------------------------------------------ SpatialData

def test_spatialdata_folders(tmp_path):
    found = run(tmp_path, sd_v3(tmp_path / "new.zarr"), sd_v3(tmp_path / "old_v3.zarr", root_attrs=False),
                sd_v2_old(tmp_path / "visium.zarr"))
    assert [(c.sid, c.kind) for c in found.candidates] == [("new", "spatialdata"), ("old_v3", "spatialdata"),
                                                           ("visium", "spatialdata")]
    assert all(c.files == {} for c in found.candidates)


def test_found_lines(tmp_path, caplog):
    with caplog.at_level("INFO", logger="discover"):
        run(tmp_path, sd_v3(tmp_path / "atlas.zarr"), xenium_bundle(tmp_path / "S1"))
    assert f"Found SpatialData object atlas at {tmp_path / 'atlas.zarr'}\n" in caplog.text + "\n"
    assert "files)" not in caplog.text.split("Found Xenium")[0]
    assert f"Found Xenium bundle S1 at {tmp_path / 'S1'} (" in caplog.text


def test_spatialdata_folder_without_zarr_suffix(tmp_path):
    (cand,) = run(tmp_path, sd_v2_old(tmp_path / "mibitof_pre01")).candidates
    assert (cand.sid, cand.kind) == ("mibitof_pre01", "spatialdata")


def test_generic_zarr_names_use_the_parent(tmp_path):
    sd_v3(tmp_path / "1417642334" / "combined_spatialdata_zarr")
    sd_v3(tmp_path / "1417642340" / "combined_spatialdata_zarr")
    sd_v3(tmp_path / "sandbox" / "data.zarr")
    found = run(tmp_path, tmp_path / "1417642334", tmp_path / "1417642340", tmp_path / "sandbox" / "data.zarr")
    assert [c.sid for c in found.candidates] == ["1417642334", "1417642340", "sandbox"]


def test_non_spatialdata_zarrs_are_left_out(tmp_path):
    plain = plain_zarr(tmp_path / "ssam.zarr")
    adata = anndata_zarr(tmp_path / "table.zarr")
    with pytest.raises(discover.InputError) as err:
        run(tmp_path, plain, adata)
    assert "without SpatialData elements" in str(err.value) and "AnnData" in str(err.value)


def test_zarr_zip_at_root_and_with_macos_twin(tmp_path):
    store = sd_v2_old(tmp_path / "src" / "multiple_elements.zarr")
    at_root = zip_folder(store, tmp_path / "multiple_elements.zarr.zip", arcroot="")
    named = zip_folder(store, tmp_path / "workshop_subset.zarr.zip", arcroot="workshop_subset.zarr",
                       extra={"__MACOSX/._workshop_subset.zarr": b"x",
                              "__MACOSX/workshop_subset.zarr/._.zgroup": b"x"})
    sandbox = zip_folder(store, tmp_path / "merfish_spatialdata_0.7.1.zip", arcroot="data.zarr")
    found = run(tmp_path, at_root, named, sandbox)
    cands = by_sid(found)
    assert sorted(cands) == ["merfish_spatialdata_0.7.1", "multiple_elements", "workshop_subset"]
    assert cands["multiple_elements"].zarr == Member(at_root, None, False, 0)
    assert cands["workshop_subset"].zarr == Member(named, "workshop_subset.zarr", False, 0)
    root = stage.spatialdata(cands["workshop_subset"], tmp_path / "staged")
    assert (root / "table" / "table" / ".zattrs").exists()
    assert not any("MACOSX" in str(p) or p.name.startswith("._") for p in root.rglob("*"))
    root = stage.spatialdata(cands["multiple_elements"], tmp_path / "staged")
    assert root.name == "multiple_elements.zarr" and (root / ".zgroup").exists()


def test_plain_zarr_zip_is_left_out(tmp_path):
    store = plain_zarr(tmp_path / "src" / "ssamdataset-osmFISH.zarr")
    archive = zip_folder(store, tmp_path / "ssamdataset-osmFISH.zarr.zip")
    with pytest.raises(discover.InputError, match="without SpatialData elements"):
        run(tmp_path, archive)


def test_spatialdata_tar_gz(tmp_path):
    store = sd_v3(tmp_path / "src" / "cells.zarr")
    archive = tar_folder(store, tmp_path / "cells.tar.gz")
    (cand,) = run(tmp_path, archive).candidates
    assert (cand.sid, cand.zarr.inner) == ("cells", "cells.zarr")
    root = stage.spatialdata(cand, tmp_path / "staged")
    assert (root / "zarr.json").exists() and (root / "tables" / "zarr.json").exists()


# ------------------------------------------------------------------------------------------ entries

def test_file_inside_a_bundle_means_the_bundle(tmp_path):
    folder = xenium_bundle(tmp_path / "S1")
    for target in (folder / "experiment.xenium", folder / "cell_feature_matrix.h5",
                   folder / "morphology_focus" / "ch0000_dapi.ome.tif", folder / "cells.zarr.zip",
                   folder / "morphology_focus"):
        (cand,) = run(tmp_path, target).candidates
        assert cand.sid == "S1" and cand.origin == folder
        assert any("inside this sample" in n for n in cand.notes)


def test_prefixed_file_means_its_group_only(tmp_path):
    top = mb_flat(tmp_path / "S-BIAD1093")
    (cand,) = run(tmp_path, top / "Files" / "XENIUM_TRANSFER" / "MB266-cells.parquet").candidates
    assert cand.sid == "MB266"


def test_a_lone_file_is_refused_clearly(tmp_path):
    write(tmp_path / "somewhere" / "matrix.h5")
    with pytest.raises(discover.InputError, match="single file"):
        run(tmp_path, tmp_path / "somewhere" / "matrix.h5")


def test_missing_paths_globs_tilde_and_data_dir(tmp_path, monkeypatch):
    with pytest.raises(discover.InputError, match="Input not found"):
        run(tmp_path, tmp_path / "missing")
    data = tmp_path / "data"
    xenium_bundle(data / "asset" / "S1")
    xenium_bundle(data / "asset" / "S2")
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.chdir(tmp_path / "data" / "asset" / "S1")
    assert [c.sid for c in discover.discover(["asset/S2"], tmp_path).candidates] == ["S2"]
    assert [c.sid for c in discover.discover(["asset/S*"], tmp_path).candidates] == ["S1", "S2"]
    with pytest.raises(discover.InputError, match="also looked for"):
        discover.discover(["nothing_here"], tmp_path)
    with pytest.raises(discover.InputError, match="Nothing matches"):
        discover.discover([str(tmp_path / "zz*")], tmp_path)
    monkeypatch.setenv("HOME", str(data))
    assert [c.sid for c in discover.discover(["~/asset/S1"], tmp_path).candidates] == ["S1"]
    with pytest.raises(discover.InputError, match="No inputs"):
        discover.discover([], tmp_path)


def test_repeated_and_overlapping_entries_count_once(tmp_path):
    xenium_bundle(tmp_path / "runs" / "A")
    xenium_bundle(tmp_path / "runs" / "B")
    found = run(tmp_path, tmp_path / "runs", tmp_path / "runs" / "A", f"{tmp_path / 'runs' / 'A'}/")
    assert [c.sid for c in found.candidates] == ["A", "B"]


def test_duplicate_sample_ids_are_refused(tmp_path):
    xenium_bundle(tmp_path / "x1" / "S")
    xenium_bundle(tmp_path / "x2" / "S", files=("experiment.xenium", "cells.parquet"))
    with pytest.raises(discover.InputError, match=r"sample ID 'S'.*x1/S.*x2/S"):
        run(tmp_path, tmp_path / "x1" / "S", tmp_path / "x2" / "S")


def test_sample_ids_are_sanitized(tmp_path):
    (cand,) = run(tmp_path, xenium_bundle(tmp_path / "Breast 2fov (rep)")).candidates
    assert cand.sid == "Breast_2fov_rep"
    assert any("became" in n for n in cand.notes)


def test_nothing_usable_lists_why(tmp_path):
    write(tmp_path / "in" / "README.txt")
    with zipfile.ZipFile(tmp_path / "in" / "qc.zip", "w") as zf:
        zf.writestr("a.html", "x")
    with pytest.raises(discover.InputError) as err:
        run(tmp_path, tmp_path / "in")
    assert "README.txt" in str(err.value) and "qc.zip" in str(err.value)


def test_many_loose_files_are_summarised(tmp_path):
    xenium_bundle(tmp_path / "in" / "S1")
    for i in range(20):
        write(tmp_path / "in" / f"figure_{i}.png")
    found = run(tmp_path, tmp_path / "in")
    (lo,) = found.left_out
    assert lo.path == str(tmp_path / "in") and lo.reason.startswith("20 files")


def test_too_deep_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(discover, "MAX_DEPTH", 2)
    xenium_bundle(tmp_path / "a" / "b" / "c" / "d" / "S1")
    with pytest.raises(discover.InputError, match="folder levels"):
        run(tmp_path, tmp_path / "a")


def test_kinds_mix_freely_across_entries(tmp_path):
    found = run(tmp_path, xenium_bundle(tmp_path / "X1"), atera_bundle(tmp_path / "A1"),
                sd_v3(tmp_path / "S1.zarr"), macos_zip(tmp_path))
    assert [(c.sid, c.kind) for c in found.candidates] == [
        ("X1", "xenium"), ("A1", "atera"), ("S1", "spatialdata"), ("Xenium_V1_human_Breast_2fov", "xenium")]
