import pytest

from config import ConfigError, Settings, load_settings


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
