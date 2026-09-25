"""Frozen mounts (Code Ocean internal EFS assets): a read-only COPY of the source.

The stamped origin is disconnected from such a mount, so it is ignored: reads work,
writes are refused with a pointer to ``copy``, and ``--origin`` can still force one.
Frozenness comes from the mount's filesystem type, which the hermetic tests fake.
"""
from __future__ import annotations

import pytest

from scizarr_ic import Repo, ScizarrError
from scizarr_ic import storage
from scizarr_ic.cli import main
from scizarr_ic.storage import is_frozen_mount, mount_fstype

from conftest import snapshot_tree


@pytest.fixture
def frozen(mounted, monkeypatch):
    mount, origin, before = mounted
    monkeypatch.setattr(storage, "mount_fstype",
                        lambda p: "nfs4" if str(p).startswith(str(mount)) else "xfs")
    # repo.py imported the name too — patch both binding sites
    import scizarr_ic.repo as repo_mod
    monkeypatch.setattr(repo_mod, "mount_fstype", storage.mount_fstype)
    return mount, origin, before


def test_mount_fstype_reads_proc_mounts(tmp_path):
    fs = mount_fstype(tmp_path)
    assert fs is None or isinstance(fs, str)
    assert not is_frozen_mount(tmp_path)  # writable, whatever the fs


def test_frozen_needs_readonly_too(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "mount_fstype", lambda p: "nfs4")
    assert not is_frozen_mount(tmp_path)  # writable nfs is just a filesystem


def test_frozen_mount_ignores_stamped_origin(frozen):
    mount, origin, before = frozen
    repo = Repo(mount)
    assert repo.frozen and repo.readonly_path and repo.read_only
    assert repo.origin is None and repo.origin_url() is not None  # stamp present, unused
    assert [s.message for s in repo.log()] == ["import", "Repository initialized"]
    assert repo.root()["X"].shape == (12, 5)
    for op in (repo.writable, lambda: repo.checkout("x", create=True),
               lambda: repo.cherrypick(repo.log()[0].id)):
        with pytest.raises(ScizarrError, match="frozen copy.*scizarr-ic copy"):
            op()
    assert "x" not in Repo(origin).branches()
    assert snapshot_tree(mount) == before


def test_frozen_mount_copy_is_the_way_out(frozen, tmp_path):
    mount, origin, before = frozen
    copy = Repo(mount).copy(tmp_path / "mine.icechunk")
    assert not copy.frozen and not copy.read_only
    copy.checkout("work", create=True)
    copy.writable().attrs["k"] = 1
    copy.commit("edit")
    assert copy.log()[1].id == Repo(mount).log()[0].id
    assert snapshot_tree(mount) == before


def test_frozen_mount_explicit_origin_still_wins(frozen, monkeypatch):
    mount, origin, _ = frozen
    Repo(mount, origin=str(origin)).checkout("forced", create=True)
    assert "forced" in Repo(origin).branches()
    monkeypatch.setenv("SCIZARR_IC_ORIGIN", str(origin))
    assert Repo(mount).origin == str(origin) and not Repo(mount).read_only


def test_cli_origin_reports_frozen(frozen, capsys):
    mount, origin, _ = frozen
    assert main(["origin", "-C", str(mount)]) == 0
    out = capsys.readouterr().out
    assert "frozen copy" in out and "nfs4" in out and "scizarr-ic copy" in out
    assert main(["checkout", "-C", str(mount), "-b", "nope"]) == 1
    assert "frozen copy" in capsys.readouterr().err
