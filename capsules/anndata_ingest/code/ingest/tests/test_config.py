from pathlib import Path

import pytest

import config
from config import CELLTYPIST_PRESETS, ConfigError, Settings, load_settings, panel_values


def test_env_names_are_field_names_in_upper_case():
    s = load_settings([], {"STORE": "s3://b/p", "RUN_SCRUBLET": "no", "CHUNK_CELLS": "123"})
    assert s.store == "s3://b/p" and s.run_scrublet is False and s.chunk_cells == 123


def test_panel_flags_override_env_and_empty_values_are_ignored():
    env = {"STORE": "from-env", "CELLTYPIST_MODELS": "env_model", "CREATE": "false"}
    s = load_settings(["--store", "from-flag", "--celltypist-models", "", "--create", "true", "--origin", ""], env)
    assert s.store == "from-flag" and s.celltypist_models == "env_model" and s.create is True and s.origin == ""


def test_defaults():
    assert load_settings([], {}) == Settings()


def test_sample_files_from_a_directory_or_a_list(tmp_path):
    for name in ("a.h5", "b.h5ad", "notes.txt"):
        (tmp_path / name).touch()
    assert [p.name for p in Settings(samples=str(tmp_path)).sample_files()] == ["a.h5", "b.h5ad"]
    listed = Settings(samples=f" {tmp_path / 'a.h5'} , {tmp_path / 'x.h5'} ").sample_files()
    assert [p.name for p in listed] == ["a.h5", "x.h5"]


def test_values_that_are_not_their_type_are_refused():
    with pytest.raises(ConfigError, match="CREATE='maybe'"):
        load_settings(["--create", "maybe"], {})
    with pytest.raises(ConfigError, match="CHUNK_CELLS='lots' is not a valid int"):
        load_settings([], {"CHUNK_CELLS": "lots"})


def test_model_files_are_names_in_the_models_dir_or_paths(tmp_path):
    s = Settings(celltypist_models=f"a, {tmp_path / 'b.pkl'}", models_dir=str(tmp_path / "m"))
    assert s.model_files() == [tmp_path / "m" / "a.pkl", tmp_path / "b.pkl"]
    assert Settings(run_celltypist=False).model_files() == []


def test_the_default_models_are_the_bundled_aifi_models():
    assert [p.name for p in Settings().model_files()] == [
        "pbmc_clean_AIFI_L1_2024.pkl", "pbmc_clean_AIFI_L2_2024.pkl", "pbmc_clean_AIFI_L3_2024.pkl"]
    assert all(p.is_file() for p in Settings().model_files())


def test_panel_values_are_positional_and_survive_a_dropped_origin():
    full = panel_values(["/results/s", "s3://b/p", "/data/samples", "None"])
    assert full == {"store": "/results/s", "origin": "s3://b/p", "samples": "/data/samples", "celltypist_models": "None"}
    assert panel_values(["/results/s", "/data/samples", "Only AIFI L1 celltypes"]) == {
        "store": "/results/s", "samples": "/data/samples", "celltypist_models": "Only AIFI L1 celltypes"}
    assert panel_values(["/results/s", "s3://b/p", "/data/x.h5"]) == {
        "store": "/results/s", "origin": "s3://b/p", "samples": "/data/x.h5"}
    s = load_settings(["/results/s", "", "/data/samples", "AIFI L1, L2, L3 celltypes"], {})
    assert s.store == "/results/s" and s.origin == "" and s.model_names() == Settings().model_names()
    assert panel_values(["/results/s", "/data/samples", "None", "/data/m/a.pkl"])["celltypist_pkl"] == "/data/m/a.pkl"
    assert panel_values(["/results/s", "", "/data/samples", "None", ""]) == {
        "store": "/results/s", "origin": "", "samples": "/data/samples", "celltypist_models": "None", "celltypist_pkl": ""}
    with pytest.raises(ConfigError, match="at most 5"):
        panel_values(["a", "b", "c", "d", "e", "f"])


def test_a_custom_pkl_replaces_the_dropdown(tmp_path, monkeypatch):
    s = Settings(celltypist_models="None", celltypist_pkl=f"{tmp_path}/spatial.pkl, /x/b.pkl")
    assert s.model_files() == [tmp_path / "spatial.pkl", Path("/x/b.pkl")]
    assert Settings(celltypist_pkl="a.pkl", run_celltypist=False).model_files() == []
    monkeypatch.setattr("config.DATA_DIR", tmp_path)
    (tmp_path / "m.pkl").touch()
    assert Settings(celltypist_pkl="m.pkl").model_files() == [tmp_path / "m.pkl"]


def test_every_dropdown_option_names_bundled_models():
    expected = {
        "AIFI L1, L2, L3 celltypes": ["pbmc_clean_AIFI_L1_2024", "pbmc_clean_AIFI_L2_2024", "pbmc_clean_AIFI_L3_2024"],
        "Only AIFI L1 celltypes": ["pbmc_clean_AIFI_L1_2024"],
        "CellTypist Immune_All High and Low labels": ["Immune_All_High", "Immune_All_Low"],
        "High and low AIFI cell labels": ["Immune_All_High", "Immune_All_Low"],
        "None": [],
    }
    assert set(expected) == set(CELLTYPIST_PRESETS)
    for label, models in expected.items():
        for variant in (label, label.upper(), f"  {label.lower()} "):
            s = Settings(celltypist_models=variant)
            assert s.model_names() == models and all(p.is_file() for p in s.model_files())
    assert Settings(celltypist_models="none").model_files() == []


def test_sample_lists_in_any_shape(tmp_path, monkeypatch):
    d1, d2 = tmp_path / "d1", tmp_path / "d2"
    for d in (d1, d2):
        d.mkdir()
    for path in (d1 / "a.h5", d1 / "b.H5AD", d1 / ".hidden.h5", d1 / "notes.txt", d2 / "c.h5", tmp_path / "e.h5ad"):
        path.touch()
    (d1 / "sub").mkdir()
    (d1 / "sub" / "deep.h5").touch()
    want = [d1 / "a.h5", d1 / "b.H5AD", d2 / "c.h5", tmp_path / "e.h5ad"]
    for raw in (f"{d1}, {d2}/, {tmp_path / 'e.h5ad'}", f"['{d1}/', \"{d2}\", '{tmp_path / 'e.h5ad'}']",
                f"{d1}\n{d2};{tmp_path / 'e.h5ad'},,", f"{d1}/*, {d2}, {tmp_path}/*.h5ad, {d1 / 'a.h5'}"):
        assert [p for p in Settings(samples=raw).sample_files()] == want, raw
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    assert Settings(samples="d2").sample_files() == [d2 / "c.h5"]
    assert Settings(samples="d2/*.h5").sample_files() == [d2 / "c.h5"]
    with pytest.raises(ConfigError, match="not a .h5/.h5ad file"):
        Settings(samples=str(tmp_path / "missing_dir")).sample_files()
    (tmp_path / "empty").mkdir()
    with pytest.raises(ConfigError, match="holds no"):
        Settings(samples=str(tmp_path / "empty")).sample_files()
    with pytest.raises(ConfigError, match="match"):
        Settings(samples=str(tmp_path / "*.loom")).sample_files()


def test_store_and_origin_are_cleaned():
    s = Settings(store=" '/results/store/' ", origin=" s3://bucket/prefix/ ")
    assert s.store == "/results/store" and s.origin == "s3://bucket/prefix"


def test_new_stores_may_be_created_inside_the_create_roots_or_remotely():
    s = Settings()
    assert s.may_create("/results/store") and s.may_create("/scratch/a/b") and s.may_create("s3://b/p")
    assert not s.may_create("/results") and not s.may_create("/data/store")
    assert not s.may_create("/results/../data/store")
    assert Settings(create=True).may_create("s3://b/p")
