import logging
from pathlib import Path

import pytest

from config import ConfigError, Settings, load_settings, panel_values, split_list

OLD = "Atera bundle (folder or .zip)"


def test_panel_in_order():
    s = load_settings(["s3://b/store.zarr", "/data/a, /data/b", "/data/m.pkl"], {})
    assert (s.store, s.samples, s.celltypist_pkl) == ("s3://b/store.zarr", "/data/a, /data/b", "/data/m.pkl")
    assert s.remote and s.sample_entries() == ["/data/a", "/data/b"]


def test_dropped_optional_values_are_placed_by_shape():
    assert panel_values(["/results/x.zarr", "/data/samples"]) == {"store": "/results/x.zarr", "samples": "/data/samples"}
    assert panel_values(["/results/x.zarr", "/data/M.PKL"]) == {"store": "/results/x.zarr", "celltypist_pkl": "/data/M.PKL"}
    assert panel_values(["/results/x.zarr", "", "/data/m.pkl"]) == {"store": "/results/x.zarr", "samples": "",
                                                                    "celltypist_pkl": "/data/m.pkl"}


def test_old_four_value_calls_ignore_the_input_type(caplog):
    with caplog.at_level(logging.INFO):
        s = load_settings(["s3://b/store.zarr", OLD, "/data/a", "/data/m.pkl"], {})
    assert (s.store, s.samples, s.celltypist_pkl) == ("s3://b/store.zarr", "/data/a", "/data/m.pkl")
    assert "('Atera bundle (folder or .zip)')" in caplog.text and not [r for r in caplog.records if r.levelno >= logging.WARNING]
    # an old call with the .pkl dropped: three values, the label is still recognised
    assert panel_values(["/r/x.zarr", OLD, "/data/a"]) == {"store": "/r/x.zarr", "samples": "/data/a"}
    assert load_settings(["", "", "", ""], {"STORE": "/scratch/s.zarr"}).store == "/scratch/s.zarr"


def test_input_type_env_and_flag_are_ignored(caplog):
    with caplog.at_level(logging.INFO):
        s = load_settings(["--store", "/x.zarr", "--input-type", "atera"], {"INPUT_TYPE": "visium"})
    assert s.store == "/x.zarr" and caplog.text.count("no longer needed") == 2
    assert not hasattr(s, "input_type")


def test_a_samples_path_named_like_a_short_input_type_stays_a_path():
    assert load_settings(["/results/x.zarr", "xenium"], {}).samples == "xenium"
    assert load_settings(["/results/x.zarr", "xenium", ""], {}).samples == "xenium"


def test_empty_values_keep_env_then_default():
    s = load_settings(["", "", ""], {"STORE": "/scratch/s.zarr"})
    assert s.store == "/scratch/s.zarr" and s.samples == "/data/samples" and s.default_samples
    assert load_settings([], {}).store == "/results/spatialdata.zarr"


def test_tilde_and_pkl_case(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    s = Settings(store="~/s.zarr", celltypist_pkl="'~/models/Model.PKL'")
    assert s.store == str(tmp_path / "s.zarr")
    assert s.model_file() == tmp_path / "models" / "Model.PKL"
    with pytest.raises(ConfigError, match="isn't a .pkl"):
        Settings(celltypist_pkl="/data/model.pickle").model_file()
    assert Settings().model_file() is None


def test_relative_model_is_looked_up_under_data(monkeypatch, tmp_path):
    import config
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "m.pkl").write_bytes(b"")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.chdir(tmp_path / "models")
    assert Settings(celltypist_pkl="models/m.pkl").model_file() == tmp_path / "models" / "m.pkl"


def test_flags_and_bad_values():
    assert load_settings(["--store", "/x.zarr", "--create", "true"], {}).create
    with pytest.raises(ConfigError):
        load_settings([], {"N_JOBS": "many"})
    with pytest.raises(ConfigError, match="at most 3 .*older panel's 4"):
        panel_values(["a", "b", "c", "d", "e"])


def test_split_list_and_create_roots(tmp_path):
    assert split_list(" 'a', [b];c\n") == ["a", "b", "c"]
    s = Settings(create_roots=str(tmp_path))
    assert s.may_create(str(tmp_path / "new.zarr")) and not s.may_create(str(tmp_path))
    assert not s.may_create("/elsewhere/new.zarr") and s.may_create("s3://bucket/new.zarr")
    assert Path(str(tmp_path)).resolve() in s.create_root_list()


def test_brackets_and_quotes_that_belong_to_a_name_stay():
    assert split_list("/data/tiny_atera (1), /data/run (2)/outs") == ["/data/tiny_atera (1)", "/data/run (2)/outs"]
    assert split_list('["/data/a b (1)", "/data/c"]') == ["/data/a b (1)", "/data/c"]
    assert split_list("('/data/x', '/data/y (3)')") == ["/data/x", "/data/y (3)"]
    assert split_list("'/data/with space'\n/data/z") == ["/data/with space", "/data/z"]
    s = load_settings(["'/results/my store (1).zarr'", "/data/tiny_atera (1)", "/data/m (2).pkl"], {})
    assert (s.store, s.sample_entries(), s.celltypist_pkl) == ("/results/my store (1).zarr", ["/data/tiny_atera (1)"],
                                                               "/data/m (2).pkl")
