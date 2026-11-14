"""Real public 10x bundles (spec §3S.12); skipped when they aren't present (SPATIAL_TEST_DATA)."""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import zarr
from anndata.io import read_elem

import main
from conftest import bundle
from test_pipeline import readback, snapshot

ANNDATA_INGEST = next((p / "anndata_ingest" / "code" / "ingest" for p in Path(__file__).resolve().parents
                       if (p / "anndata_ingest" / "code" / "ingest").is_dir()), None)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Ovary (zip) and Breast (folder) Xenium, then Atera (zip), into one store."""
    import config

    tmp = tmp_path_factory.mktemp("bundles")
    env = {"RESULTS_DIR": str(tmp / "results"), "SCRATCH_DIR": str(tmp / "scratch"), "CREATE_ROOTS": str(tmp),
           "WRITE_CPUS": "2", "N_JOBS": "2"}
    store = tmp / "store.zarr"
    xenium = f"{bundle('Xenium_V1_Human_Ovary_tiny', zipped=True)}, {bundle('Xenium_V1_human_Breast_2fov')}"
    first = main.run(config.load_settings([str(store), xenium], env))
    second = main.run(config.load_settings([str(store), str(bundle("tiny_atera_dataset", zipped=True))], env))
    return store, env, first, second


def test_xenium_and_atera_bundles_into_one_store(built):
    store, _, first, second = built
    assert first["appended_samples"] == ["Xenium_V1_Human_Ovary_tiny", "Xenium_V1_human_Breast_2fov"]
    assert second["appended_samples"] == ["tiny_atera_dataset"]
    sids = ["Xenium_V1_Human_Ovary_tiny", "Xenium_V1_human_Breast_2fov", "tiny_atera_dataset"]
    back = readback(store, sids)
    assert back["systems"] == sorted(sids)
    for sid in sids:
        assert back[sid]["systems"] == [sid] and back[sid]["joined"][0] == back[sid]["joined"][1] > 0
    assert "tiny_atera_dataset_morphology_3d" in back["elements"]["images"]
    assert back["tables"]["Xenium_V1_human_Breast_2fov_table"] == [7275, 280]
    assert back["tables"]["tiny_atera_dataset_table"] == [13, 53]
    assert back["tables"]["table"][0] == 632 + 7275 + 13
    panels = [set(read_elem(zarr.open_group(store / "tables" / f"{sid}_table" / "var", mode="r")).index) for sid in sids]
    assert back["tables"]["table"][1] == len(set().union(*panels))
    assert second["kinds"] == {"tiny_atera_dataset": "atera"} and first["kinds"]["Xenium_V1_human_Breast_2fov"] == "xenium"
    assert not (Path(built[1]["SCRATCH_DIR"]) / "extract" / "tiny_atera_dataset").exists()


def test_every_table_is_symbol_indexed_gene_expression(built):
    store = built[0]
    for name in ("Xenium_V1_human_Breast_2fov_table", "tiny_atera_dataset_table", "table"):
        var = read_elem(zarr.open_group(store / "tables" / name / "var", mode="r"))
        assert list(var.columns)[:2] == ["gene_ids", "feature_types"] and (var["feature_types"] == "Gene Expression").all()
        assert not var.index.str.startswith("ENSG").any() and var.index.is_unique
    atera = read_elem(zarr.open_group(store / "tables" / "tiny_atera_dataset_table" / "var", mode="r"))
    assert "ATOX1" in atera.index and atera.loc["ATOX1", "gene_ids"] == "ENSG00000177556"


def test_rerun_is_byte_identical_and_mouse_is_refused(built):
    import config

    store, env, _, _ = built
    before = snapshot(store)
    again = main.run(config.load_settings([str(store), str(bundle("Xenium_V1_human_Breast_2fov"))], env))
    assert again["appended_samples"] == [] and snapshot(store) == before
    with pytest.raises(main.IngestError, match=r"Xenium_Prime_Mouse_Ileum_tiny \(mouse\); the store holds human genes"):
        main.run(config.load_settings([str(store), str(bundle("Xenium_Prime_Mouse_Ileum_tiny", zipped=True))], env))
    assert snapshot(store) == before


def test_joint_table_goes_into_anndata_ingest(built, tmp_path):
    if ANNDATA_INGEST is None:
        pytest.skip("anndata_ingest isn't beside this capsule")
    out = tmp_path / "joint.zarr"
    env = {"RESULTS_DIR": str(tmp_path), "STAGING_DIR": str(tmp_path / "staging"), "CREATE_ROOTS": str(tmp_path),
           "PATH": "/usr/bin:/bin"}
    done = subprocess.run([sys.executable, "main.py", str(out), "", "Existing AnnData or SpatialData table (as is)",
                           str(built[0]), "None"], cwd=ANNDATA_INGEST, env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-3000:]
    obs = read_elem(zarr.open_group(out / "obs", mode="r"))
    assert pd.Series(obs["sample_id"]).nunique() == 3 and len(obs) == 632 + 7275 + 13
