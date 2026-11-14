"""existing.plan / load_part: splitting objects into samples, old store layouts, element and table
problems, and the streamed per-sample table."""
import logging

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp
import zarr
from anndata.io import read_elem

import existing
import fixtures_sdata as F
from conftest import HUMAN, write_spatialdata


def elements(part):
    return sorted(name for _, name in part.elements)


def source_rows(path, table="table", folder="tables"):
    return read_elem(zarr.open_group(path / folder / table, mode="r", use_consolidated=False))


def streamed(loaded):
    return ad.read_zarr(loaded.table.path)


# ---------------------------------------------------------------- splitting (as before)

def test_single_sample_takes_the_folder_name(tmp_path):
    path = write_spatialdata(tmp_path / "one.zarr", {"x": {"genes": HUMAN, "n": 20, "seed": 0}})
    [part] = existing.plan(path, "one", "sample_id")
    assert part.sid == "one" and part.table == "table" and part.regions == ("cell_boundaries",)
    assert len(part.elements) == 5


def test_concatenated_samples_split_by_coordinate_system(tmp_path):
    path = write_spatialdata(tmp_path / "two.zarr", {"a": {"genes": HUMAN, "n": 20, "seed": 0},
                                                     "b": {"genes": HUMAN, "n": 30, "seed": 1}})
    parts = {p.sid: p for p in existing.plan(path, "two", "sample_id")}
    assert set(parts) == {"two_a", "two_b"}
    assert parts["two_a"].tags == ("a",)
    assert elements(parts["two_a"]) == sorted(f"{n}-a" for n in ("morphology_focus", "cell_labels", "cell_boundaries",
                                                             "nucleus_boundaries", "transcripts"))


def test_regions_sharing_a_coordinate_system_stay_one_sample(tmp_path):
    path = write_spatialdata(tmp_path / "shared.zarr", {"a": {"genes": HUMAN, "n": 20, "seed": 0},
                                                        "b": {"genes": HUMAN, "n": 30, "seed": 1}}, shared_system=True)
    [part] = existing.plan(path, "shared", "sample_id")
    assert part.sid == "shared" and len(part.regions) == 2 and len(part.elements) == 10 and part.tags == ()


def test_several_tables_in_one_coordinate_system_keep_one(tmp_path, caplog):
    path = write_spatialdata(tmp_path / "bins.zarr", {"a": {"genes": HUMAN, "n": 20, "seed": 0},
                                                      "b": {"genes": HUMAN, "n": 30, "seed": 1}},
                             shared_system=True, one_table=False)
    [part] = existing.plan(path, "bins", "sample_id")
    assert part.table == "table_a" and part.regions == ("cell_boundaries-a",)
    assert "leaving out table_b" in caplog.text


def test_sample_column_names_samples_and_tables_per_sample(tmp_path):
    path = write_spatialdata(tmp_path / "col.zarr", {"P1": {"genes": HUMAN, "n": 20, "seed": 0},
                                                     "P2": {"genes": HUMAN, "n": 30, "seed": 1}},
                             sample_column=True, one_table=False)
    parts = {p.sid: p for p in existing.plan(path, "col", "sample_id")}
    assert set(parts) == {"P1", "P2"} and parts["P2"].table == "table_P2"
    loaded = existing.load_part(path, parts["P2"], tmp_path / "scratch")
    assert loaded.table.n_obs == 30 and set(loaded.table.obs["sample_id"]) == {"P2"}


def test_region_with_several_samples_is_refused(tmp_path):
    import pandas as pd
    from anndata.io import write_elem

    path = write_spatialdata(tmp_path / "mixed.zarr", {"x": {"genes": HUMAN, "n": 20, "seed": 0}})
    obs = zarr.open_group(path / "tables" / "table" / "obs", mode="r+")
    write_elem(obs, "sample_id", pd.Categorical(["A"] * 10 + ["B"] * 10))
    with pytest.raises(existing.SplitError, match="several samples"):
        existing.plan(path, "mixed", "sample_id")


def test_tags():
    assert existing.tags(["cell_labels-a", "cell_labels-bb"]) == {"cell_labels-a": "a", "cell_labels-bb": "bb"}
    assert existing.tags(["s1_cell_boundaries", "s2_cell_boundaries"]) == {"s1_cell_boundaries": "s1",
                                                                          "s2_cell_boundaries": "s2"}


# ---------------------------------------------------------------- streamed rows

def test_load_part_streams_only_the_parts_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(existing, "NNZ_PER_CHUNK", 7)          # many tiny chunks
    path = F.write(tmp_path / "two.zarr", {"a": {"n": 10, "seed": 0}, "b": {"n": 12, "seed": 1}})
    parts = {p.sid: p for p in existing.plan(path, "two", "sample_id")}
    loaded = existing.load_part(path, parts["two_b"], tmp_path / "scratch")
    source = source_rows(path)
    rows = np.flatnonzero(source.obs["region"].astype(str) == "cells-b")
    out = streamed(loaded)
    assert out.n_obs == loaded.table.n_obs == 12 and list(out.var_names) == F.GENES
    assert np.array_equal(out.X.toarray(), source.X[rows].toarray())
    assert list(out.obs_names) == list(source.obs_names[rows]) and out.obsm["spatial"].shape == (12, 2)
    assert sp.issparse(out.X) and zarr.open_group(loaded.table.path / "X")["indptr"].dtype == np.int64
    assert set(loaded.elements["shapes"]) == {"cells-b"} and loaded.region_key == "region"
    assert loaded.regions == ["cells-b"] and loaded.tags == ("b",)


def test_dense_x_is_streamed(tmp_path):
    path = F.write(tmp_path / "dense.zarr", {"a": {"n": 9, "seed": 2}})
    t = source_rows(path)
    t.X = t.X.toarray()
    F.replace_table(path, t)
    [part] = existing.plan(path, "dense", "sample_id")
    out = streamed(existing.load_part(path, part, tmp_path / "scratch"))
    assert np.array_equal(out.X.toarray(), t.X)


# ---------------------------------------------------------------- old stores

def test_pre01_v2_store_with_table_table(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    path = F.pre01(tmp_path / "old.zarr", {"a": {"n": 10, "seed": 0}, "b": {"n": 8, "seed": 1}})
    assert (path / ".zgroup").exists() and (path / "table" / "table" / ".zgroup").exists()
    assert existing.is_spatialdata(path)
    parts = {p.sid: p for p in existing.plan(path, "old", "sample_id")}
    assert set(parts) == {"old_a", "old_b"} and "older SpatialData layout" in caplog.text
    loaded = existing.load_part(path, parts["old_a"], tmp_path / "scratch")
    source = source_rows(path, folder="table")
    rows = np.flatnonzero(source.obs["region"].astype(str) == "cells-a")
    assert np.array_equal(streamed(loaded).X.toarray(), source.X[rows].toarray())
    assert set(loaded.elements["images"]) == {"image-a"}
    assert (zarr.open_group(loaded.table.path).metadata.zarr_format == 3)


def test_v2_store_without_root_attributes(tmp_path):
    path = F.strip_root_attrs(F.write(tmp_path / "noattrs.zarr", {"a": {"n": 6, "seed": 0}}, v2=True))
    assert "spatialdata_attrs" not in existing.root_attrs(path) and existing.is_spatialdata(path)
    [part] = existing.plan(path, "noattrs", "sample_id")
    assert existing.load_part(path, part, tmp_path / "s").table.n_obs == 6


def test_plain_zarr_is_not_spatialdata(tmp_path):
    zarr.open_group(tmp_path / "plain.zarr", mode="w").create_array("x", shape=(3,), dtype="i4")
    assert not existing.is_spatialdata(tmp_path / "plain.zarr")


def test_image_without_coordinate_system_is_left_out_with_one_warning(tmp_path, caplog):
    path = F.unplace_image(F.write(tmp_path / "cs.zarr", {"a": {"n": 6, "seed": 0}}), "images", "image-a")
    with caplog.at_level(logging.WARNING):
        [part] = existing.plan(path, "cs", "sample_id")
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING and r.name == "existing"]
    assert len(warnings) == 1 and "images/image-a (in no coordinate system)" in warnings[0].getMessage()
    assert ("images", "image-a") not in part.elements and ("shapes", "cells-a") in part.elements
    loaded = existing.load_part(path, part, tmp_path / "s")
    assert loaded.elements["images"] == {} and set(loaded.elements["labels"]) == {"labels-a"}


def test_unreadable_object_is_a_split_error(tmp_path, monkeypatch):
    path = F.write(tmp_path / "bad.zarr", {"a": {"n": 6, "seed": 0}})
    monkeypatch.setattr(existing, "read_elements", lambda *a, **k: 1 / 0)
    with pytest.raises(existing.SplitError, match="can't be read as SpatialData .ZeroDivisionError"):
        existing.plan(path, "bad", "sample_id")


# ---------------------------------------------------------------- tables

def test_cell_ID_instance_key_is_renamed(tmp_path):
    path = F.write(tmp_path / "liver.zarr", {"a": {"n": 7, "seed": 0}}, instance_key="cell_ID")
    [part] = existing.plan(path, "liver", "sample_id")
    loaded = existing.load_part(path, part, tmp_path / "s")
    assert loaded.instance_key == "source_cell_ID" and "source_cell_ID" in loaded.table.obs
    assert "cell_ID" not in streamed(loaded).obs


def test_safe_names():
    assert existing.safe_names(["cell_ID", "cell_id", "a b", "A_B", "__x", "_index", "ok"]) == {
        "cell_ID": "source_cell_ID", "a b": "a_b", "A_B": "source_A_B", "__x": "source__x", "_index": "source_index"}
    assert existing.safe_names(["Sample_ID", "source_Sample_ID"]) == {"Sample_ID": "source_Sample_ID",
                                                                      "source_Sample_ID": "source_source_Sample_ID"}


def test_counts_in_raw_are_used(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    path = F.raw_counts(F.write(tmp_path / "hne.zarr", {"a": {"n": 9, "seed": 3}}))
    [part] = existing.plan(path, "hne", "sample_id")
    loaded = existing.load_part(path, part, tmp_path / "s")
    source = source_rows(path)
    assert np.array_equal(streamed(loaded).X.toarray(), source.raw.X.toarray()) and "using .raw" in caplog.text
    assert list(existing.table_var(path, part).index) == list(source.raw.var_names)


def test_counts_layer_is_used(tmp_path):
    path = F.write(tmp_path / "lay.zarr", {"a": {"n": 9, "seed": 4}})
    t = source_rows(path)
    t.layers["counts"] = t.X.copy()
    t.X = sp.csr_matrix(np.log1p(t.X.toarray()))
    F.replace_table(path, t)
    [part] = existing.plan(path, "lay", "sample_id")
    assert np.array_equal(streamed(existing.load_part(path, part, tmp_path / "s")).X.toarray(),
                          t.layers["counts"].toarray())


def test_object_without_table_is_refused(tmp_path):
    path = F.drop_tables(F.write(tmp_path / "toy.zarr", {"a": {"n": 6, "seed": 0}}))
    assert existing.is_spatialdata(path)
    with pytest.raises(existing.SplitError, match="has no table; spatial_ingest needs a table of cells x genes") as err:
        existing.plan(path, "toy", "sample_id")
    assert str(tmp_path) not in str(err.value)          # the caller names the user's input once


def test_protein_like_table_is_refused_with_the_reason(tmp_path):
    path = F.protein_like(F.write(tmp_path / "codex.zarr", {"a": {"n": 12, "seed": 0}}))
    with pytest.raises(existing.SplitError) as err:
        existing.plan(path, "codex", "sample_id")
    message = str(err.value)
    assert "doesn't hold RNA counts" in message and "negative values" in message
    assert "protein intensities" in message and "CD0" in message and str(tmp_path) not in message


def test_capsule_store_as_input_leaves_its_joint_table_out_quietly(tmp_path, caplog):
    import joint
    import store as stores

    path = F.write(tmp_path / "ours.zarr", {"a": {"n": 6, "seed": 0}}, table_name="a_table")
    zarr.open_group(path, mode="r+", use_consolidated=False).attrs[stores.MARKER] = {"version": 1}
    joint.add(path, "a", F.sample_table(tmp_path / "t.zarr", "a", F.GENES, 6, 0), tmp_path / "staging", 1)
    with caplog.at_level(logging.INFO, logger="existing"):
        [part] = existing.plan(path, "ours", "sample_id")
    assert part.table == "a_table"
    record = next(r for r in caplog.records if "joint table" in r.getMessage())
    assert record.levelno == logging.INFO


def test_no_left_out_warning_before_a_refusal(tmp_path, caplog):
    path = F.protein_like(F.unplace_image(F.write(tmp_path / "p.zarr", {"a": {"n": 6, "seed": 0}}),
                                          "images", "image-a"))
    with caplog.at_level(logging.WARNING, logger="existing"), pytest.raises(existing.SplitError):
        existing.plan(path, "p", "sample_id")
    assert not [r for r in caplog.records if r.name == "existing"]


def test_single_sample_object_has_no_tags_and_keeps_element_names(tmp_path):
    import layout

    path = write_spatialdata(tmp_path / "single.zarr", {"x": {"genes": HUMAN, "n": 20, "seed": 0}})
    [part] = existing.plan(path, "single", "sample_id")
    assert part.tags == ()
    assert layout.element_name("cell_boundaries", part.sid, part.tags) == "single_cell_boundaries"


def test_region_named_like_its_section_is_not_doubled(tmp_path):
    import layout

    path = F.write(tmp_path / "visium.zarr", {"a": {"n": 5, "seed": 0}, "b": {"n": 6, "seed": 1}})
    parts = {p.sid: p for p in existing.plan(path, "visium", "sample_id")}
    # the regions `cells-a`/`cells-b` are tagged `a`/`b`; an element named exactly by the tag becomes the sample ID
    assert layout.element_name("cells-a", "visium_a", parts["visium_a"].tags) == "visium_a_cells"
    assert layout.element_name("a", "visium_a", parts["visium_a"].tags) == "visium_a"
