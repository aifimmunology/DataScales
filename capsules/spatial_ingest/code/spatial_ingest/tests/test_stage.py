"""Staging what discovery found: symlinks for plain files, extraction from archives, gunzip of GEO files."""
import gzip
import zipfile

import pytest

import discover
import stage
from contracts import Candidate, Member
from fixtures_archives import content, geo_flat, pipestance_tar, sd_v3, write, xenium_bundle, zip_folder


def test_plain_files_are_symlinked(tmp_path):
    folder = xenium_bundle(tmp_path / "S1")
    (cand,) = discover.discover([str(folder)], tmp_path).candidates
    out = stage.fetch(cand, ["cells.parquet", "morphology_focus/", "not_there.h5"], tmp_path / "stage")
    assert set(out) == {"cells.parquet", "morphology_focus/ch0000_dapi.ome.tif"}
    assert out["cells.parquet"].is_symlink() and out["cells.parquet"].read_bytes() == content("cells.parquet")


def test_geo_gz_members_are_decompressed_under_canonical_names(tmp_path):
    folder = geo_flat(tmp_path / "GSE")
    cands = {c.sid: c for c in discover.discover([str(folder)], tmp_path).candidates}
    a = stage.bundle(cands["GSM9829813_CtrlM1"], tmp_path / "a", skip=["morphology.ome.tif"])
    assert (a / "cells.parquet").read_bytes() == content("cells.parquet")
    assert (a / "experiment.xenium").read_bytes() == content("experiment.xenium")
    assert (a / "reseg_transcripts.parquet").read_bytes() == content("reseg_transcripts.parquet")
    assert not (a / "morphology.ome.tif").exists()
    assert (a / "cell_metadata.csv.gz").is_symlink()                 # read gzipped as it is
    b = stage.bundle(cands["GSM9543314_EX-LN1"], tmp_path / "b")
    assert gzip.decompress((b / "cell_feature_matrix" / "matrix.mtx.gz").read_bytes()) == content("matrix.mtx")
    assert not list(b.rglob("*.partial"))


def test_members_come_out_of_zip_and_tar(tmp_path):
    archive = pipestance_tar(tmp_path)
    (cand,) = discover.discover([str(archive)], tmp_path).candidates
    out = stage.bundle(cand, tmp_path / "t")
    assert (out / "transcripts.parquet").read_bytes() == content("transcripts.parquet")
    folder = xenium_bundle(tmp_path / "Z")
    write(folder / "cells.parquet.gz", gzip.compress(b"gzipped parquet"))
    (cand,) = discover.discover([str(zip_folder(folder, tmp_path / "Z.zip"))], tmp_path).candidates
    assert cand.files["cells.parquet"].gz is False                    # the plain copy wins over the .gz one
    path = stage.fetch_member(Member(tmp_path / "Z.zip", "Z/cells.parquet.gz", True), tmp_path / "z" / "c.parquet")
    assert path.read_bytes() == b"gzipped parquet"


def test_a_tar_sample_is_read_in_one_pass(tmp_path, monkeypatch, caplog):
    """Inventory, precheck and readers fetch one file at a time; the tar is still streamed only once."""
    archive = pipestance_tar(tmp_path)
    (cand,) = discover.discover([str(archive)], tmp_path).candidates
    opened = []
    real_open = stage.tarfile.open
    monkeypatch.setattr(stage.tarfile, "open", lambda *a, **k: opened.append(a) or real_open(*a, **k))
    monkeypatch.setattr(stage, "BIG", 0)
    with caplog.at_level("INFO", logger="stage"):
        out = stage.fetch(cand, ["cells.parquet"], tmp_path / "s")
    assert len(opened) == 1 and out["cells.parquet"].read_bytes() == content("cells.parquet")
    expected = [k for k in cand.files if not k.startswith(stage.PREFETCH_SKIP)]
    assert f"Extracting {len(expected)} file(s) from" in caplog.text and " GB)" in caplog.text
    for key in ("transcripts.parquet", "cell_feature_matrix/", "experiment.xenium", "morphology_focus/"):
        for path in stage.fetch(cand, [key], tmp_path / "s").values():
            assert path.read_bytes() == content(path.name)
    assert len(opened) == 1                                          # every later fetch found its file staged
    assert not (tmp_path / "s" / "analysis").exists() and not (tmp_path / "s" / "analysis.zarr.zip").exists()
    assert stage.fetch(cand, ["analysis.zarr.zip"], tmp_path / "s")["analysis.zarr.zip"].exists()
    assert len(opened) == 2                                          # skipped files still come when asked for


def test_gb_is_decimal():
    assert stage.gb(4_900_000_000) == "4.9 GB"


def test_missing_tar_member_is_an_error(tmp_path):
    archive = pipestance_tar(tmp_path)
    with pytest.raises((KeyError, FileNotFoundError)):
        stage.fetch_member(Member(archive, "nope/cells.parquet"), tmp_path / "x")


def test_spatialdata_folder_is_used_in_place(tmp_path):
    store = sd_v3(tmp_path / "a.zarr")
    cand = Candidate("a", "spatialdata", store, zarr=Member(store))
    assert stage.spatialdata(cand, tmp_path / "s") == store
    with pytest.raises(ValueError):
        stage.spatialdata(Candidate("b", "xenium", store), tmp_path / "s")


def test_spatialdata_from_zip_extracts_only_the_store(tmp_path):
    archive = tmp_path / "bundle_and_store.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("s.zarr/zarr.json", '{"zarr_format": 3, "node_type": "group", "attributes": '
                                        '{"spatialdata_attrs": {}}}')
        for i in range(40):                                            # many small chunks: threaded extraction
            zf.writestr(f"s.zarr/images/img/0/c/{i}", bytes([i]) * 10)
        zf.writestr("s.zarr/images/zarr.json", "{}")
        zf.writestr("other/big.bin", b"x" * 1000)
    (cand,) = discover.discover([str(archive)], tmp_path).candidates
    root = stage.spatialdata(cand, tmp_path / "s")
    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    assert len(files) == 42 and not (tmp_path / "s" / "other").exists()
    assert (root / "images/img/0/c/7").read_bytes() == bytes([7]) * 10


def test_spatialdata_from_zip_skips_junk_and_unsafe_names(tmp_path):
    archive = tmp_path / "x.zarr.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("./x.zarr/zarr.json", '{"zarr_format": 3, "node_type": "group", "attributes": '
                                          '{"spatialdata_attrs": {}}}')
        zf.writestr("x.zarr/tables/zarr.json", "{}")
        zf.writestr("x.zarr/._zarr.json", "junk")
        zf.writestr("x.zarr/.DS_Store", "junk")
        zf.writestr("x.zarr/../../evil.txt", "nope")
        zf.writestr("__MACOSX/x.zarr/._zarr.json", "junk")
    (cand,) = discover.discover([str(archive)], tmp_path).candidates
    root = stage.spatialdata(cand, tmp_path / "s")
    assert sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()) == \
        ["tables/zarr.json", "zarr.json"]
    assert not (tmp_path / "evil.txt").exists()
    assert stage.spatialdata(cand, tmp_path / "s") == root            # second call reuses the extraction
    stage.cleanup(tmp_path / "s")
    assert not (tmp_path / "s").exists()


def test_plain_tar_ending_in_a_zip_member_is_read_as_tar(tmp_path):
    """zipfile.is_zipfile accepts a plain .tar whose last member is a .zip; staging and discovery must not."""
    import tarfile
    import zipfile

    import discover
    import stage
    from contracts import Candidate, Member

    inner = tmp_path / "cells.zarr.zip"
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("zarr.json", "{}")
    (tmp_path / "experiment.xenium").write_text("{}")
    archive = tmp_path / "run.tar"
    with tarfile.open(archive, "w") as tf:
        tf.add(tmp_path / "experiment.xenium", arcname="run/experiment.xenium")
        tf.add(inner, arcname="run/cells.zarr.zip")
    assert zipfile.is_zipfile(archive) and not stage.is_zip(archive) and stage.is_zip(inner)
    assert discover.ArchiveIndex(archive).zip is False
    c = Candidate("run", "xenium", archive, files={"experiment.xenium": Member(archive, "run/experiment.xenium")})
    assert stage.fetch(c, ["experiment.xenium"], tmp_path / "out")["experiment.xenium"].read_text() == "{}"
