import pytest

from config import ConfigError, Settings, load_settings, panel_values, split_list


def test_full_panel_in_order():
    s = load_settings(["s3://b/store.zarr", "Atera bundle (folder or .zip)", "/data/a, /data/b", "/data/m.pkl"], {})
    assert (s.store, s.input_type, s.samples, s.celltypist_pkl) == ("s3://b/store.zarr", "atera", "/data/a, /data/b", "/data/m.pkl")
    assert s.remote


def test_dropped_optional_values_are_placed_by_shape():
    assert panel_values(["/results/x.zarr", "/data/samples"]) == {"store": "/results/x.zarr", "samples": "/data/samples"}
    assert panel_values(["/results/x.zarr", "Existing SpatialData (.zarr)", "/data/m.pkl"]) == {
        "store": "/results/x.zarr", "input_type": "Existing SpatialData (.zarr)", "celltypist_pkl": "/data/m.pkl"}


def test_a_samples_path_named_like_a_short_input_type_stays_a_path():
    s = load_settings(["/results/x.zarr", "Atera bundle (folder or .zip)", "xenium"], {})
    assert s.input_type == "atera" and s.samples == "xenium"


def test_empty_values_keep_env_then_default():
    s = load_settings(["", "", "", ""], {"STORE": "/scratch/s.zarr"})
    assert s.store == "/scratch/s.zarr" and s.input_type == "xenium" and s.samples == "/data/samples"
    assert load_settings([], {}).store == "/results/spatialdata.zarr"


def test_input_type_labels_and_short_names():
    assert Settings(input_type="xenium bundle folder or zip").input_type == "xenium"
    assert Settings(input_type="existing").input_type == "spatialdata"
    with pytest.raises(ConfigError):
        Settings(input_type="visium")


def test_flags_and_bad_values():
    assert load_settings(["--store", "/x.zarr", "--create", "true"], {}).create
    with pytest.raises(ConfigError):
        load_settings([], {"N_JOBS": "many"})
    with pytest.raises(ConfigError):
        panel_values(["a", "b", "c", "d", "e"])


def test_split_list_and_create_roots(tmp_path):
    assert split_list(" 'a', [b];c\n") == ["a", "b", "c"]
    s = Settings(create_roots=str(tmp_path))
    assert s.may_create(str(tmp_path / "new.zarr")) and not s.may_create(str(tmp_path))
    assert not s.may_create("/elsewhere/new.zarr") and s.may_create("s3://bucket/new.zarr")
