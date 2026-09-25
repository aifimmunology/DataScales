"""Read-only mount -> stamped origin: reads stay on the mount, writes land at the origin.

Simulates a Code Ocean data asset hermetically: the origin repo is a writable local
dir, the "mount" is a plain copy of it that SCIZARR_IC_READONLY_PREFIXES marks as
read-only (tests run as root, so chmod can't). Because the copy never sees the origin's
later writes, it also doubles as a *stale* mount for the fallback tests.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from scizarr_ic import Repo, ScizarrError
from scizarr_ic.cli import main
from scizarr_ic.storage import ORIGIN_KEY


def _snapshot_tree(root: Path) -> dict[str, int]:
    return {
        str(p.relative_to(root)): p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()
    }


@pytest.fixture
def mounted(src_zarr, tmp_path, monkeypatch):
    """(mount_path, origin_path, mount_tree_before) with HEAD sidecars in tmp_path/home."""
    path, _ = src_zarr
    origin = tmp_path / "origin.icechunk"
    Repo.init(path, origin, message="import")
    (origin / "scizarr_head").unlink()  # the copy must not carry the origin's HEAD file

    mount_root = tmp_path / "mount"
    mount = mount_root / "store"
    shutil.copytree(origin, mount)
    monkeypatch.setenv("SCIZARR_IC_READONLY_PREFIXES", str(mount_root))
    monkeypatch.setenv("SCIZARR_IC_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("SCIZARR_IC_ORIGIN", raising=False)
    return mount, origin, _snapshot_tree(mount)


def test_init_stamps_origin(src_zarr, repo_path):
    path, _ = src_zarr
    repo = Repo.init(path, repo_path)
    assert repo.origin_url() == os.path.abspath(str(repo_path))
    assert repo.origin == str(repo_path)
    assert not repo.resolved and not repo.readonly_path


def test_init_rejects_readonly_out(src_zarr, tmp_path, monkeypatch):
    path, _ = src_zarr
    ro = tmp_path / "ro"
    ro.mkdir()
    monkeypatch.setenv("SCIZARR_IC_READONLY_PREFIXES", str(ro))
    with pytest.raises(ScizarrError, match="read-only"):
        Repo.init(path, ro / "new.icechunk")


def test_mount_resolves_origin_and_reads_in_place(mounted):
    mount, origin, before = mounted
    repo = Repo(mount)
    assert repo.readonly_path
    assert repo.origin == os.path.abspath(str(origin))
    assert repo.resolved and not repo.read_only
    assert [s.message for s in repo.log()][0] == "import"
    assert repo.root().attrs["encoding-type"] == "anndata"
    assert repo._writer is None  # reads never opened the origin
    assert _snapshot_tree(mount) == before


def test_checkout_create_on_mount_writes_origin_only(mounted, tmp_path):
    mount, origin, before = mounted
    repo = Repo(mount)
    repo.checkout("dev", create=True)
    assert repo.branch == "dev"

    assert "dev" in Repo(origin).branches()  # landed at the origin
    assert _snapshot_tree(mount) == before  # nothing written into the mount
    assert not (mount / "scizarr_head").exists()
    heads = list((tmp_path / "home" / "heads").iterdir())
    assert any(p.read_text().strip() == "dev" for p in heads if p.suffix == "")


def test_commit_via_mount_lands_in_origin(mounted):
    mount, origin, before = mounted
    repo = Repo(mount)
    repo.writable().attrs["note"] = "from mount"
    snap = repo.commit("edit via mount")

    reader = Repo(origin)
    assert reader.log()[0].id == snap
    assert reader.root().attrs["note"] == "from mount"
    assert _snapshot_tree(mount) == before


def test_head_sidecar_and_stale_mount_fallback(mounted):
    mount, origin, _ = mounted
    Repo(mount).checkout("dev", create=True)

    # New process: the mount copy still lacks 'dev' (stale view) but HEAD says 'dev'.
    reopened = Repo(mount)
    assert reopened.branch == "dev"
    assert "dev" in reopened.tree()
    assert reopened._writer is not None  # the origin was consulted for the missing branch


def test_checkout_existing_branch_only_at_origin(mounted):
    mount, origin, _ = mounted
    Repo(origin).checkout("feature", create=True)
    Repo(origin).checkout("main")
    repo = Repo(mount)
    assert repo.branch == "main"
    repo.checkout("feature")  # not in the stale mount; found at the origin
    assert repo.branch == "feature"


def test_mount_without_origin_reads_but_cannot_write(mounted, monkeypatch):
    mount, origin, _ = mounted
    # strip the stamp from the mount copy only
    import icechunk

    monkeypatch.delenv("SCIZARR_IC_READONLY_PREFIXES")
    icechunk.Repository.open(icechunk.local_filesystem_storage(str(mount))).set_metadata({})
    monkeypatch.setenv("SCIZARR_IC_READONLY_PREFIXES", str(mount.parent))

    repo = Repo(mount)
    assert repo.origin is None and repo.read_only
    assert repo.log()  # reads still fine
    with pytest.raises(ScizarrError, match=ORIGIN_KEY):
        repo.checkout("x", create=True)
    with pytest.raises(ScizarrError, match=ORIGIN_KEY):
        repo.writable()


def test_explicit_origin_override(mounted, tmp_path, src_zarr, monkeypatch):
    mount, origin, _ = mounted
    path, _ = src_zarr
    other = tmp_path / "other.icechunk"
    Repo.init(path, other)

    repo = Repo(mount, origin=str(other))
    repo.checkout("via-arg", create=True)
    assert "via-arg" in Repo(other).branches()
    assert "via-arg" not in Repo(origin).branches()

    monkeypatch.setenv("SCIZARR_IC_ORIGIN", str(other))
    Repo(mount).checkout("via-env", create=True)
    assert "via-env" in Repo(other).branches()


def test_cli_origin_show_and_checkout_on_mount(mounted, capsys):
    mount, origin, before = mounted
    assert main(["origin", "-C", str(mount)]) == 0
    out = capsys.readouterr().out
    assert "read-only" in out and str(origin) in out and "(resolved)" in out

    assert main(["checkout", "-C", str(mount), "-b", "cli-dev"]) == 0
    captured = capsys.readouterr()
    assert "Created and switched to branch 'cli-dev'" in captured.out
    assert "note: writing to" in captured.err

    assert main(["tree", "-C", str(mount)]) == 0
    assert "* cli-dev" in capsys.readouterr().out
    assert _snapshot_tree(mount) == before


def test_cli_origin_set(src_zarr, repo_path, capsys):
    path, _ = src_zarr
    Repo.init(path, repo_path)
    assert main(["origin", "-C", str(repo_path), "s3://bucket/prefix"]) == 0
    assert "Stamped" in capsys.readouterr().out
    assert Repo(repo_path).origin_url() == "s3://bucket/prefix"
