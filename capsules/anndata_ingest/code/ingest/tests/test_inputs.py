"""Settings, reading input files, and the precheck's unit pieces."""
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pytest
from annizarr.sources import detect_format

import config
import inputs
from config import CELLTYPIST_PRESETS, ConfigError, Settings, load_settings, panel_values
from conftest import GENES, write_hise_h5, write_spatialdata

APP_PANEL = Path(__file__).resolve().parents[3] / ".codeocean" / "app-panel.json"


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
    raw = "Raw samples (QC, CellTypist, Scrublet)"
    full = panel_values(["/results/s", "s3://b/p", raw, "/data/samples", "None", ""])
    assert full == {"store": "/results/s", "origin": "s3://b/p", "input_type": raw, "samples": "/data/samples",
                    "celltypist_models": "None", "celltypist_pkl": ""}
    assert panel_values(["/results/s", "/data/samples", "Only AIFI L1 celltypes"]) == {
        "store": "/results/s", "samples": "/data/samples", "celltypist_models": "Only AIFI L1 celltypes"}
    assert panel_values(["/results/s", "s3://b/p", "/data/x.h5"]) == {
        "store": "/results/s", "origin": "s3://b/p", "samples": "/data/x.h5"}
    s = load_settings(["/results/s", "", raw, "/data/samples", "AIFI L1, L2, L3 celltypes"], {})
    assert s.store == "/results/s" and s.origin == "" and s.model_names() == Settings().model_names()
    assert panel_values(["/results/s", "/data/samples", "None", "/data/m/a.pkl"])["celltypist_pkl"] == "/data/m/a.pkl"
    existing = load_settings(["s3://b/p", "", "Existing AnnData or SpatialData table (as is)", "/data/x.h5ad"], {})
    assert existing.existing and existing.samples == "/data/x.h5ad" and existing.origin == ""
    with pytest.raises(ConfigError, match="at most 6"):
        panel_values(["a", "b", "c", "d", "e", "f", "g"])
    with pytest.raises(ConfigError, match="Input type"):
        Settings(input_type="processed")


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


def test_the_app_panel_matches_the_positional_order():
    if not APP_PANEL.is_file():
        pytest.skip("no .codeocean/app-panel.json next to code/")
    params = json.loads(APP_PANEL.read_text())["parameters"]
    assert len(params) == len(config.PANEL)
    by_name = dict(zip(config.PANEL, params))
    assert set(by_name["input_type"]["extra_data"]) <= set(config.INPUT_TYPES)
    assert set(by_name["celltypist_models"]["extra_data"]) <= set(config.CELLTYPIST_PRESETS)
    assert by_name["store"]["default_value"] == config.Settings().store


def test_detects_and_loads_hise(tmp_path):
    path = write_hise_h5(tmp_path / "B092-P1_SP09999-01.h5", "SP09999", 50, seed=3)
    assert detect_format(path) == "hise"
    adata = inputs.read_sample(path)
    assert adata.shape == (50, 500)
    assert adata.X.format == "csr" and adata.X.dtype == np.uint16
    assert list(adata.var_names[:2]) == ["MT-0", "MT-1"]
    assert adata.obs["pbmc_sample_id"].iloc[0] == "SP09999"
    assert adata.obs["n_umis"].to_numpy().sum() == adata.X.sum()


@pytest.mark.parametrize("name,expected", [
    ("B092-P1_SP02425-01.h5", "SP02425"), ("SP02430.h5", "SP02430"), ("B092-P1_BMC01671-05.h5", "BMC01671"),
])
def test_sample_id(name, expected):
    assert inputs.sample_id(name) == expected


def test_hise_variable_length_strings_and_adt_features(tmp_path):
    path = write_hise_h5(tmp_path / "B092-P1_SP09998-01.h5", "SP09998", 30, seed=4)
    with h5py.File(path, "r+") as f:
        m = f["matrix"]
        for key in ("barcodes", "features/name", "features/feature_type", "observations/cell_name", "observations/batch_id"):
            values = [v.decode() for v in m[key][:]]
            if key == "features/feature_type":
                values[-1] = "Antibody Capture"
            del m[key]
            m[key] = np.array(values, dtype=h5py.string_dtype())
    adata = inputs.read_sample(path)
    assert adata.n_vars == 499 and "GENE494" not in adata.var_names
    assert adata.obs_names[0] == "BC00000-1"
    assert adata.obs["cell_name"].iloc[0] == "SP09998_0" and adata.obs["batch_id"].iloc[0] == "B092"


def test_plain_10x_and_h5ad_load_in_memory(tmp_path):
    path = write_hise_h5(tmp_path / "B092-P1_SP09997-01.h5", "SP09997", 40, seed=5)
    with h5py.File(path, "r+") as f:
        del f["matrix/observations"]
    assert detect_format(path) == "10x"
    tenx = inputs.read_sample(path)
    assert tenx.shape == (40, 500) and tenx.X.format == "csr"

    h5ad = tmp_path / "B092-P1_SP09996-01.h5ad"
    ad.AnnData(tenx.X, obs=tenx.obs, var=tenx.var).write_h5ad(h5ad)
    back = inputs.read_sample(h5ad)
    assert not back.isbacked and back.X.format == "csr"
    assert (back.X != tenx.X).nnz == 0


def test_new_pipeline_layout_maps_to_the_hise_columns(tmp_path):
    path = write_hise_h5(tmp_path / "B092-P1_SP09995-01.h5", "SP09995", 20, seed=6)
    renames = {"batch_id": "Batch", "pool_id": "Pool", "chip_id": "Chip", "cell_uuid": "uuid",
               "pbmc_sample_id": "sample_name", "n_mito_umis": "n_mitochondrial_counts", "hto_barcode": "hashtag"}
    with h5py.File(path, "r+") as f:
        obs = f["matrix/observations"]
        for old, new in renames.items():
            obs.move(old, new)
        obs["prop_mitochondrial_counts"] = np.zeros(20)
    adata = inputs.read_sample(path)
    assert {"batch_id", "pool_id", "chip_id", "cell_uuid", "pbmc_sample_id", "n_mito_umis", "hto_barcode"} <= set(adata.obs)
    assert not set(renames.values()) & set(adata.obs)
    assert adata.obs["batch_id"].iloc[0] == "B092" and adata.obs["pbmc_sample_id"].iloc[0] == "SP09995"


def test_a_spatialdata_table_is_read_without_its_control_features(tmp_path):
    path = write_spatialdata(tmp_path / "Breast_A.zarr", 40, seed=1)
    adata = inputs.read_sample(path)
    assert inputs.zarr_kind(path) == "spatialdata" and list(adata.var_names) == GENES == list(inputs.gene_names(path))
    assert inputs.sample_id(path) == "Breast_A"


def test_generic_10x_file_names_take_the_bundle_folder_as_sample_id():
    assert inputs.sample_id("/data/Xenium_Lung_2fov/extract/cell_feature_matrix.h5") == "Xenium_Lung_2fov"
    assert inputs.sample_id("/data/run1/outs/filtered_feature_bc_matrix.h5") == "run1"
    assert inputs.sample_id("/data/multi/outs/per_sample_outs/S1/count/sample_filtered_feature_bc_matrix.h5") == "S1"
    assert inputs.sample_id("/data/B092-P1_SP02425-01.h5") == "SP02425"


@pytest.mark.parametrize("name", ["CHUNK_CELLS", "WRITE_CPUS"])
@pytest.mark.parametrize("value", ["0", "-5"])
def test_chunk_and_cpu_settings_must_be_positive(name, value):
    with pytest.raises(ConfigError, match="1 or more"):
        load_settings([], {name: value})


def test_samples_by_glob_upper_case_suffix_and_trailing_slash(tmp_path):
    for name in ("B092-P1_SP1-01.H5", "B092-P1_SP2-01.h5", "other.h5ad"):
        (tmp_path / name).touch()
    assert len(Settings(samples=str(tmp_path / "B092-P1_SP*")).sample_files()) == 2
    assert len(Settings(samples=f"{tmp_path}/").sample_files()) == 3


def test_paths_with_spaces_and_non_ascii_names(tmp_path):
    folder = tmp_path / "my data"
    folder.mkdir()
    (folder / "Probe_Ä1.h5").touch()
    files = Settings(samples=str(folder)).sample_files()
    assert files == [folder / "Probe_Ä1.h5"] and inputs.sample_id(files[0]) == "Probe_Ä1"
