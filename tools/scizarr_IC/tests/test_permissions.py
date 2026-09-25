"""Read vs write permissions: what each kind of location may do, and how failures read.

Locations: a writable local repo; a read-only mount with a reachable origin (Code Ocean
*linked* S3 asset); a read-only mount whose stamped origin is unreachable (an *internal*
EFS asset, or a linked asset without the AWS secret); a remote URI.
"""
from __future__ import annotations

import os

import pytest

from scizarr_ic import Repo, ScizarrError
from scizarr_ic.head import HEAD_FILE, HeadStore
from scizarr_ic.storage import canonical_location, is_readonly_path, storage_for


# -- read-only detection -------------------------------------------------------

def test_nonexistent_child_of_readonly_dir_is_readonly(tmp_path, monkeypatch):
    """A path that doesn't exist yet is judged by the directory it would be created in."""
    ro = tmp_path / "ro"
    ro.mkdir()
    real_access = os.access
    monkeypatch.setattr(os, "access", lambda p, mode: False if str(p) == str(ro) else real_access(p, mode))
    assert is_readonly_path(ro / "new.icechunk")
    assert is_readonly_path(ro / "deeper" / "new.icechunk")
    assert not is_readonly_path(tmp_path / "elsewhere" / "new.icechunk")


def test_init_into_readonly_dir_is_a_clean_error(src_zarr, tmp_path, monkeypatch):
    path, _ = src_zarr
    ro = tmp_path / "ro"
    ro.mkdir()
    monkeypatch.setenv("SCIZARR_IC_READONLY_PREFIXES", str(ro))
    with pytest.raises(ScizarrError, match="read-only"):
        Repo.init(path, ro / "sub" / "new.icechunk")
    assert not (ro / "sub").exists()


def test_remote_uris_are_never_readonly_and_need_no_env():
    assert not is_readonly_path("s3://bucket/prefix")
    assert not is_readonly_path("gs://bucket/prefix")
    assert canonical_location("s3://bucket/prefix") == "s3://bucket/prefix"
    with pytest.raises(ScizarrError, match="bucket"):
        storage_for("s3://")


# -- writable local repo ---------------------------------------------------------

def test_writable_local_repo_is_its_own_origin(src_zarr, repo_path):
    path, _ = src_zarr
    repo = Repo.init(path, repo_path)
    assert not repo.readonly_path and not repo.resolved and not repo.read_only
    assert repo.origin == str(repo_path)
    assert (repo_path / HEAD_FILE).read_text().strip() == "main"
    assert not repo._head.is_sidecar
    repo.checkout("dev", create=True)
    assert (repo_path / HEAD_FILE).read_text().strip() == "dev"


# -- read-only mount, origin unreachable -----------------------------------------

@pytest.fixture
def mount_with_dead_origin(mounted, monkeypatch):
    """Mount whose stamped origin can't be opened (no such path / no credentials)."""
    mount, origin, before = mounted
    monkeypatch.setenv("SCIZARR_IC_ORIGIN", str(origin.parent / "gone.icechunk"))
    return mount, before


def test_dead_origin_reads_fine_but_writes_explain(mount_with_dead_origin):
    mount, before = mount_with_dead_origin
    repo = Repo(mount)
    assert repo.readonly_path and repo.resolved and not repo.read_only  # an origin *is* known
    assert repo.log()[0].message == "import"
    assert repo.root()["X"].shape == (12, 5)
    for op in (lambda: repo.writable(), lambda: repo.checkout("x", create=True),
               lambda: repo.cherrypick(repo.log()[0].id)):
        with pytest.raises(ScizarrError, match="Cannot open the writable origin"):
            op()
    from conftest import snapshot_tree
    assert snapshot_tree(mount) == before


def test_dead_origin_head_falls_back_to_main(mount_with_dead_origin, tmp_path, capsys):
    """HEAD names a branch the mount doesn't have and the origin can't be asked: fall back."""
    mount, _ = mount_with_dead_origin
    HeadStore(key=canonical_location(os.environ["SCIZARR_IC_ORIGIN"])).save("ghost")
    repo = Repo(mount)
    assert repo.branch == "main"
    assert "falling back" in capsys.readouterr().err


def test_explicit_missing_branch_on_dead_origin_errors(mount_with_dead_origin):
    mount, _ = mount_with_dead_origin
    with pytest.raises(ScizarrError, match="does not exist"):
        Repo(mount, branch="nope")


# -- read-only mount, reachable origin (already covered in test_mount; edge cases here) --

def test_mount_reads_never_touch_origin_credentials(mounted, monkeypatch):
    """log/tree/root on a mount must not open the origin — that is what lets a capsule
    without the AWS secret read a linked asset."""
    mount, origin, _ = mounted
    import icechunk

    monkeypatch.setattr(icechunk.Repository, "open",
                        _guard_open(icechunk.Repository.open, forbidden=str(origin)))
    repo = Repo(mount)
    repo.log(); repo.tree(); repo.root(); repo.branches(); repo.origin_url()
    assert repo._writer is None


def _guard_open(real_open, *, forbidden: str):
    @classmethod
    def guarded(cls, storage, *a, **kw):
        # icechunk.Storage has no public path accessor; the repr carries the prefix
        if forbidden in repr(storage):
            raise AssertionError(f"reads must not open the origin {forbidden}")
        return real_open(storage, *a, **kw)
    return guarded


def test_write_on_mount_then_reads_come_from_origin(mounted):
    mount, origin, _ = mounted
    repo = Repo(mount)
    repo.writable().attrs["k"] = 1
    repo.commit("c")
    assert repo._writer is not None
    assert repo.root().attrs["k"] == 1  # served by the origin; the stale mount lacks it
    assert "k" not in Repo(origin).root().attrs or Repo(origin).log()[0].message == "c"
