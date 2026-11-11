import pytest

import existing
from conftest import HUMAN, write_spatialdata


def elements(part):
    return sorted(name for _, name in part.elements)


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


def test_shared_system_with_a_sample_column_splits(tmp_path):
    path = write_spatialdata(tmp_path / "col2.zarr", {"P1": {"genes": HUMAN, "n": 20, "seed": 0},
                                                      "P2": {"genes": HUMAN, "n": 30, "seed": 1}},
                             shared_system=True, sample_column=True)
    parts = {p.sid: p for p in existing.plan(path, "col2", "sample_id")}
    assert set(parts) == {"P1", "P2"} and all(n.endswith("-P2") for n in elements(parts["P2"]))


def test_sample_column_names_samples_and_tables_per_sample(tmp_path):
    path = write_spatialdata(tmp_path / "col.zarr", {"P1": {"genes": HUMAN, "n": 20, "seed": 0},
                                                     "P2": {"genes": HUMAN, "n": 30, "seed": 1}},
                             sample_column=True, one_table=False)
    parts = {p.sid: p for p in existing.plan(path, "col", "sample_id")}
    assert set(parts) == {"P1", "P2"} and parts["P2"].table == "table_P2"
    table = existing.read_table(path, parts["P2"])
    assert table.n_obs == 30 and set(table.obs["sample_id"]) == {"P2"}


def test_region_with_several_samples_is_refused(tmp_path):
    path = write_spatialdata(tmp_path / "mixed.zarr", {"x": {"genes": HUMAN, "n": 20, "seed": 0}})
    import zarr
    from anndata.io import write_elem
    import pandas as pd
    obs = zarr.open_group(path / "tables" / "table" / "obs", mode="r+")
    write_elem(obs, "sample_id", pd.Categorical(["A"] * 10 + ["B"] * 10))
    with pytest.raises(existing.SplitError, match="several samples"):
        existing.plan(path, "mixed", "sample_id")


def test_tags():
    assert existing.tags(["cell_labels-a", "cell_labels-bb"]) == {"cell_labels-a": "a", "cell_labels-bb": "bb"}
    assert existing.tags(["s1_cell_boundaries", "s2_cell_boundaries"]) == {"s1_cell_boundaries": "s1",
                                                                          "s2_cell_boundaries": "s2"}
