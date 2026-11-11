import zipfile
from pathlib import Path

import pytest

import inputs
from config import Settings


def xenium_folder(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "experiment.xenium").write_text("{}")
    return path


def zipped(folder: Path, out: Path, nested: bool) -> Path:
    with zipfile.ZipFile(out, "w") as zf:
        for f in folder.iterdir():
            zf.write(f, f"{folder.name}/{f.name}" if nested else f.name)
    return out


def test_bundle_ids():
    assert inputs.bundle_id(Path("/d/Xenium_V1_human_Breast_2fov_outs.zip")) == "Xenium_V1_human_Breast_2fov"
    assert inputs.bundle_id(Path("/d/run7/outs")) == "run7"
    assert inputs.zarr_id(Path("/d/my.zarr")) == "my"


@pytest.mark.parametrize("nested", [False, True])
def test_zip_bundles_are_recognized(tmp_path, nested):
    folder = xenium_folder(tmp_path / "S1")
    z = zipped(folder, tmp_path / "S1_outs.zip", nested)
    assert inputs.bundle_kind(z) == "xenium"
    assert inputs.zip_root(z) == ("S1/" if nested else "")


def test_folder_lists_children_and_bundle_is_one_sample(tmp_path, caplog):
    xenium_folder(tmp_path / "runs" / "A")
    xenium_folder(tmp_path / "runs" / "B")
    (tmp_path / "runs" / "notes.txt").write_text("x")
    found = inputs.resolve(Settings(samples=str(tmp_path / "runs"), input_type="xenium"))
    assert "ignored: notes.txt" in caplog.text
    assert [i.sid for i in found] == ["A", "B"]
    assert [i.sid for i in inputs.resolve(Settings(samples=str(tmp_path / "runs" / "A")))] == ["A"]


def test_wrong_type_bad_and_duplicate_ids_are_refused(tmp_path):
    atera = tmp_path / "at"
    atera.mkdir()
    (atera / "experiment.spatial").write_text("{}")
    with pytest.raises(inputs.InputError, match="Atera bundle"):
        inputs.resolve(Settings(samples=str(atera), input_type="xenium"))
    xenium_folder(tmp_path / "bad name")
    with pytest.raises(inputs.InputError, match="letters, digits"):
        inputs.resolve(Settings(samples=str(tmp_path / "bad name")))
    xenium_folder(tmp_path / "x1" / "S")
    xenium_folder(tmp_path / "x2" / "S")
    with pytest.raises(inputs.InputError, match="Two inputs"):
        inputs.resolve(Settings(samples=f"{tmp_path / 'x1' / 'S'}, {tmp_path / 'x2' / 'S'}"))
    with pytest.raises(inputs.InputError, match="not found"):
        inputs.resolve(Settings(samples=str(tmp_path / "missing")))
