"""copy_group (init's zarr import) and Repo.copy / `scizarr-ic copy` (repo clone)."""
from __future__ import annotations

import numpy as np
import pytest
import zarr

from scizarr_ic import Repo, ScizarrError
from scizarr_ic.cli import main
from scizarr_ic.copy import copy_group
from scizarr_ic.head import HEAD_FILE

from conftest import snapshot_tree


def test_copy_group_preserves_nested_layout_and_data(tmp_path):
    src = zarr.open_group(str(tmp_path / "s.zarr"), mode="w")
    src.attrs["root"] = "meta"
    a = src.create_array("X", shape=(20, 4), dtype="float64", chunks=(7, 4))
    a[...] = np.arange(80, dtype="float64").reshape(20, 4)
    a.attrs["encoding-type"] = "array"
    b = src.create_group("layers").create_array("counts", shape=(20, 4), dtype="int32", chunks=(5, 2))
    b[...] = np.ones((20, 4), dtype="int32")

    dst = zarr.open_group(str(tmp_path / "d.zarr"), mode="w")
    assert copy_group(src, dst, band_bytes=512) == 2  # tiny bands -> several chunk-aligned writes
    assert dst.attrs["root"] == "meta"
    assert dst["X"].chunks == (7, 4) and dst["X"].attrs["encoding-type"] == "array"
    np.testing.assert_array_equal(dst["X"][...], a[...])
    assert dst["layers"]["counts"].chunks == (5, 2)
    np.testing.assert_array_equal(dst["layers"]["counts"][...], b[...])


def test_copy_repo_is_faithful_and_independent(src_zarr, repo_path, tmp_path):
    path, _ = src_zarr
    a = Repo.init(path, repo_path)
    a.checkout("dev", create=True)
    a.writable().attrs["on_dev"] = True
    a.commit("dev work")
    before = snapshot_tree(repo_path)

    b = a.copy(tmp_path / "b.icechunk")

    assert b.branch == "dev" and set(b.branches()) == {"main", "dev"}
    for br in ("main", "dev"):
        assert [s.id for s in b.log(branch=br)] == [s.id for s in a.log(branch=br)]
    assert (tmp_path / "b.icechunk" / HEAD_FILE).is_file()
    b.checkout("feature", create=True)
    b.writable().attrs["mine"] = 1
    b.commit("edit the copy")
    assert "feature" not in a.branches()
    assert snapshot_tree(repo_path) == before


def test_copy_refuses_bad_destinations(src_zarr, repo_path, tmp_path, monkeypatch):
    path, _ = src_zarr
    repo = Repo.init(path, repo_path)
    with pytest.raises(ScizarrError, match="not empty"):
        repo.copy(repo_path)
    with pytest.raises(ScizarrError, match="s3:// only"):
        repo.copy("gs://bucket/prefix")
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with pytest.raises(ScizarrError, match="boto3"):
        repo.copy("s3://bucket/prefix")  # checked before any network I/O


def test_cli_copy(src_zarr, repo_path, tmp_path, capsys):
    path, _ = src_zarr
    main(["init", str(path), str(repo_path)])
    dest = tmp_path / "cli-copy.icechunk"
    assert main(["copy", "-C", str(repo_path), str(dest)]) == 0
    assert "Copied" in capsys.readouterr().out
    assert main(["checkout", "-C", str(dest), "-b", "work"]) == 0
    assert "work" in Repo(dest).branches() and "work" not in Repo(repo_path).branches()
