"""Repo.copy / `scizarr-ic copy`: a writable clone of a (read-only) repo, history intact."""
from __future__ import annotations

import os

import pytest

from scizarr_ic import Repo, ScizarrError
from scizarr_ic.cli import main
from scizarr_ic.head import HEAD_FILE

from conftest import snapshot_tree


def _ids(repo, branch):
    return [s.id for s in repo.log(branch=branch)]


def test_copy_mount_to_local_is_faithful_and_writable(mounted, tmp_path):
    mount, origin, before = mounted
    src = Repo(origin)
    src.checkout("dev", create=True)
    src.writable().attrs["on_dev"] = True
    src.commit("dev work")
    # refresh the read-only "mount" so it carries both branches, then re-freeze it
    import shutil
    shutil.rmtree(mount); shutil.copytree(origin, mount); (mount / HEAD_FILE).unlink()
    before = snapshot_tree(mount)

    dest = tmp_path / "out" / "copy.icechunk"
    copy = Repo(mount).copy(dest)

    assert copy.path == str(dest) and not copy.readonly_path and not copy.resolved
    assert copy.origin_url() == os.path.abspath(str(dest))          # re-stamped
    assert set(copy.branches()) == {"main", "dev"}
    for b in ("main", "dev"):
        assert _ids(copy, b) == _ids(Repo(origin), b)                # snapshot ids preserved
    assert copy.root(snapshot_id=_ids(copy, "dev")[0]).attrs["on_dev"] is True
    assert copy.branch == "main"                                     # carried from the source
    assert (dest / HEAD_FILE).is_file()

    copy.checkout("feature", create=True)
    copy.writable().attrs["mine"] = 1
    copy.commit("edit the copy")
    assert "feature" not in Repo(origin).branches()                  # origin untouched
    assert snapshot_tree(mount) == before                            # mount untouched


def test_copy_carries_current_branch(mounted, tmp_path):
    mount, origin, _ = mounted
    Repo(mount).checkout("dev", create=True)          # lands at origin; HEAD sidecar = dev
    copy = Repo(mount).copy(tmp_path / "c.icechunk")  # dev only exists at the origin...
    # ...so the stale mount copy cannot carry it: fall back to main rather than fail
    assert copy.branch == "main"


def test_copy_refuses_bad_destinations(mounted, tmp_path, monkeypatch):
    mount, origin, before = mounted
    repo = Repo(mount)
    with pytest.raises(ScizarrError, match="read-only"):
        repo.copy(mount.parent / "other.icechunk")
    with pytest.raises(ScizarrError, match="not empty"):
        repo.copy(origin)
    with pytest.raises(ScizarrError, match="s3:// only"):
        repo.copy("gs://bucket/prefix")
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with pytest.raises(ScizarrError, match="boto3"):
        repo.copy("s3://bucket/prefix")  # checked before any network I/O
    assert snapshot_tree(mount) == before


def test_copy_of_writable_repo_is_independent(src_zarr, repo_path, tmp_path):
    path, _ = src_zarr
    a = Repo.init(path, repo_path)
    b = a.copy(tmp_path / "b.icechunk")
    b.writable().attrs["b_only"] = 1
    b.commit("b")
    assert "b_only" not in a.root().attrs
    assert a.log()[0].id == b.log()[1].id


def test_cli_copy(mounted, tmp_path, capsys):
    mount, origin, before = mounted
    dest = tmp_path / "cli-copy.icechunk"
    assert main(["copy", "-C", str(mount), str(dest)]) == 0
    assert "Copied" in capsys.readouterr().out
    assert main(["checkout", "-C", str(dest), "-b", "work"]) == 0
    assert "work" in Repo(dest).branches()
    assert snapshot_tree(mount) == before
