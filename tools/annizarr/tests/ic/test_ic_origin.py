"""Read from one location, write to another: ``origin=`` / ``--origin`` / ANNIZARR_ORIGIN.

The typical case is a read-only local mirror of an s3:// prefix; here the mirror is a
stale local copy and the origin a local dir, which exercises the same code paths.
"""

from __future__ import annotations

import os

import pytest
from _ic_helpers import snapshot_tree

from annizarr._cli import main
from annizarr._ic import Repo
from annizarr._ic._head import HEAD_FILE
from annizarr.errors import RepoError


def test_reads_stay_on_path_writes_land_at_origin(mirror, tmp_path, monkeypatch):
    mirror_path, origin, before = mirror
    import icechunk

    real_open = icechunk.Repository.open

    def guarded(storage, *a, **kw):  # reads must never open the origin
        assert str(origin) not in repr(storage), "read opened the origin"
        return real_open(storage, *a, **kw)

    monkeypatch.setattr(icechunk.Repository, "open", staticmethod(guarded))
    repo = Repo(mirror_path, origin=str(origin))
    assert repo.resolved and not repo.readonly_path
    assert repo.log()[0].message == "import" and repo.open_zarr("r")["X"].shape == (12, 5)
    monkeypatch.setattr(icechunk.Repository, "open", staticmethod(real_open))

    repo.checkout("dev", create=True)
    repo.open_zarr("w").attrs["note"] = "via mirror"
    snap = repo.commit("edit")

    origin_repo = Repo(origin, branch="dev")
    assert origin_repo.log()[0].id == snap and origin_repo.open_zarr("r").attrs["note"] == "via mirror"
    assert repo.open_zarr("r").attrs["note"] == "via mirror"  # origin is authoritative once opened
    assert snapshot_tree(mirror_path) == before  # nothing written into the mirror
    assert not (mirror_path / HEAD_FILE).exists()  # HEAD went to the sidecar
    assert repo._head.is_sidecar and Repo(mirror_path, origin=str(origin)).branch == "dev"


def test_origin_from_env_and_cli_flag(mirror, capsys, monkeypatch):
    mirror_path, origin, before = mirror
    monkeypatch.setenv("ANNIZARR_ORIGIN", str(origin))
    Repo(mirror_path).checkout("via-env", create=True)
    assert "via-env" in Repo(origin).branches()

    monkeypatch.delenv("ANNIZARR_ORIGIN")
    assert main(["ic", "checkout", str(mirror_path), "--origin", str(origin), "-b", "via-flag"]) == 0
    captured = capsys.readouterr()
    assert "Created and switched to branch 'via-flag'" in captured.err
    assert f"note: writing to {origin}" in captured.err
    assert "via-flag" in Repo(origin).branches()
    assert snapshot_tree(mirror_path) == before


def test_stale_mirror_finds_branch_at_origin(mirror):
    mirror_path, origin, _ = mirror
    Repo(origin).checkout("feature", create=True)
    repo = Repo(mirror_path, origin=str(origin))
    assert "feature" not in repo._reader.list_branches()  # the copy never saw it
    repo.checkout("feature")  # found at the origin
    assert repo.branch == "feature" and "feature" in repo.tree()


def test_readonly_path_without_origin_reads_only(mirror, monkeypatch):
    mirror_path, _origin, before = mirror
    real_access = os.access
    monkeypatch.setattr(os, "access", lambda p, m: False if str(p).startswith(str(mirror_path)) else real_access(p, m))
    repo = Repo(mirror_path)
    assert repo.readonly_path and repo.origin is None and not repo.resolved
    assert repo.log() and repo.open_zarr("r")["X"].shape == (12, 5)
    for op in (
        lambda: repo.open_zarr("w"),
        lambda: repo.checkout("x", create=True),
        lambda: repo.cherrypick(repo.log()[0].id),
    ):
        with pytest.raises(RepoError, match=r"--origin.*annizarr ic copy"):
            op()
    assert snapshot_tree(mirror_path) == before


def test_unreachable_origin_explains(mirror, tmp_path):
    mirror_path, _origin, _ = mirror
    repo = Repo(mirror_path, origin=str(tmp_path / "gone.icechunk"))
    assert repo.log()  # reads unaffected
    with pytest.raises(RepoError, match="Cannot open the writable origin"):
        repo.open_zarr("w")
