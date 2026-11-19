import hashlib
import importlib
import json
import logging
from pathlib import Path

import pytest
import zarr
from anndata.io import read_elem

import config
import main
from conftest import bundle
from config import ConfigError, Settings, load_settings, panel_values

OVARY, ATERA = "Xenium_V1_Human_Ovary_tiny", "tiny_atera_dataset"
SIDS = [OVARY, ATERA]
CAPSULE = Path(__file__).resolve().parents[3]


def snapshot(path: Path) -> dict[str, str]:
    return {str(p.relative_to(path)): hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(path.rglob("*"))
            if p.is_file()}


def var_of(store: Path, table: str):
    return read_elem(zarr.open_group(store / "tables" / table / "var", mode="r"))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    env = {"RESULTS_DIR": str(tmp / "results"), "SCRATCH_DIR": str(tmp / "scratch"), "CREATE_ROOTS": str(tmp),
           "WRITE_CPUS": "2"}
    store = tmp / "store.zarr"
    samples = f"{bundle(OVARY, zipped=True)}, {bundle(ATERA, zipped=True)}"
    report = main.run(config.load_settings([str(store), samples], env))
    return store, env, report


def run(built, samples):
    store, env, _ = built
    return main.run(config.load_settings([str(store), str(samples)], env))


def test_public_xenium_and_atera_bundles_go_in(built):
    _, _, report = built
    assert report["appended_samples"] == SIDS and report["status"] == "succeeded"
    assert report["kinds"] == {OVARY: "xenium", ATERA: "atera"}


def test_elements_are_named_by_sample_id(built):
    store = built[0]
    for kind in ("images", "labels", "shapes", "points", "tables"):
        names = [p.name for p in (store / kind).iterdir() if p.is_dir()]
        assert all(n == "table" or n.startswith((f"{OVARY}_", f"{ATERA}_")) for n in names), (kind, names)
    assert {f"{sid}_table" for sid in SIDS} <= {p.name for p in (store / "tables").iterdir()}


def test_joint_table_holds_every_cell_and_the_union_of_panels(built):
    store = built[0]
    joint = read_elem(zarr.open_group(store / "tables" / "table" / "obs", mode="r"))
    assert joint["sample_id"].value_counts().to_dict() == {OVARY: 632, ATERA: 13}
    panels = [set(var_of(store, f"{sid}_table").index) for sid in SIDS]
    assert len(var_of(store, "table")) == len(set().union(*panels)) and len(panels[1]) == 53


def test_atera_table_is_indexed_by_gene_symbol(built):
    var = var_of(built[0], f"{ATERA}_table")
    assert not var.index.str.startswith("ENSG").any() and var.loc["ATOX1", "gene_ids"] == "ENSG00000177556"
    assert (var["feature_types"] == "Gene Expression").all()


def test_only_2d_images_are_stored_not_the_3d_zstacks(built):
    images = sorted(p.name for p in (built[0] / "images").iterdir() if p.is_dir())
    assert images == [f"{OVARY}_morphology_focus", f"{ATERA}_morphology"]


def test_rerun_skips_present_samples_and_changes_nothing(built):
    before = snapshot(built[0])
    again = run(built, bundle(OVARY, zipped=True))
    assert again["appended_samples"] == [] and again["skipped_samples"] == [OVARY]
    assert snapshot(built[0]) == before


def test_same_id_from_another_source_is_skipped_with_a_warning(built, caplog):
    before = snapshot(built[0])
    folder = bundle(ATERA)
    with caplog.at_level(logging.WARNING):
        again = run(built, folder)
    assert again["appended_samples"] == [] and again["skipped_samples"] == [ATERA]
    assert any(r.getMessage().startswith(f"{ATERA} is already in the store (added from ")
               and f"{folder} is skipped" in r.getMessage() for r in caplog.records)
    assert snapshot(built[0]) == before


def test_no_samples_logs_usage_and_exits_0(tmp_path, monkeypatch, caplog):
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "DEFAULT_SAMPLES", str(tmp_path / "data" / "samples"))
    for name, value in {"RESULTS_DIR": tmp_path / "results", "SCRATCH_DIR": tmp_path / "scratch",
                        "STORE": tmp_path / "results" / "spatialdata.zarr", "CREATE_ROOTS": tmp_path,
                        "SAMPLES": config.DEFAULT_SAMPLES}.items():
        monkeypatch.setenv(name, str(value))
    with caplog.at_level("INFO"):
        assert main.main([]) == 0
    assert "No samples were found" in caplog.text and not (tmp_path / "results").exists()


def test_panel_values_are_positional_and_placed_by_shape_when_dropped():
    s = load_settings(["s3://b/store.zarr", "/data/a, /data/b", "/data/m.pkl"], {})
    assert (s.store, s.sample_entries(), s.celltypist_pkl) == ("s3://b/store.zarr", ["/data/a", "/data/b"], "/data/m.pkl")
    assert panel_values(["/results/x.zarr", "/data/M.PKL"]) == {"store": "/results/x.zarr", "celltypist_pkl": "/data/M.PKL"}


def test_flags_override_env_and_bad_values_are_refused():
    assert load_settings(["--store", "/x.zarr"], {"STORE": "/env.zarr", "CREATE": "true"}).store == "/x.zarr"
    with pytest.raises(ConfigError):
        load_settings([], {"WRITE_CPUS": "many"})


def test_app_panel_matches_config_panel():
    panel_file = CAPSULE / ".codeocean" / "app-panel.json"
    if not panel_file.exists():
        pytest.skip(f"{panel_file} isn't beside the code")
    fields = {"Storage Location": "store", "List of paths to samples": "samples",
              "Custom CellTypist model (.pkl, optional)": "celltypist_pkl"}
    params = json.loads(panel_file.read_text())["parameters"]
    assert [fields[p["name"]] for p in params] == list(config.PANEL)
    defaults = {fields[p["name"]]: p.get("default_value", "") for p in params}
    assert defaults == {"store": Settings().store, "samples": config.DEFAULT_SAMPLES, "celltypist_pkl": ""}


PRIVATE_APIS = {
    "spatialdata_io.readers.xenium": ["_get_images", "_get_labels", "_get_morphology_focus", "_XeniumCells",
                                      "_parse_version_of_xenium_analyzer", "cell_id_str_from_prefix_suffix_uint32",
                                      "imread"],
    "spatialdata_io.readers._utils._utils": ["_initialize_raster_models_kwargs", "_set_reader_metadata"],
    "spatialdata_io._docs": ["inject_docs"],
    "spatialdata_io._constants._enum": ["ModeEnum"],
    "spatialdata._io.io_points": ["_read_points"],
    "spatialdata._io.io_raster": ["_read_multiscale"],
    "spatialdata._io.io_shapes": ["_read_shapes"],
    "spatialdata._io.io_zarr": ["get_raster_format_for_read"],
    "spatialdata._core.validation": ["validate_table_attr_keys"],
    "annizarr._sources._readers": ["as_reader"],
    "zarr.storage._local": ["_move_with_retry"],
    "zarr.codecs._v2": ["V2Codec"],
}


def test_private_apis_of_pinned_libraries_resolve():
    missing = [f"{module}.{name}" for module, names in PRIVATE_APIS.items()
               for name in names if not hasattr(importlib.import_module(module), name)]
    assert missing == []
    importlib.import_module("vendor.atera")
