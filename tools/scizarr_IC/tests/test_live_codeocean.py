"""Live checks against a real read-only mount (Code Ocean data asset). Opt-in:

    SCIZARR_IC_LIVE_REPO=/data/<asset> pytest tests/test_live_codeocean.py
    SCIZARR_IC_LIVE_WRITE=1   additionally creates + deletes a scratch branch at the
                              s3:// origin (needs the capsule's AWS role)

Skipped entirely when SCIZARR_IC_LIVE_REPO is unset, so the hermetic suite stays offline.
"""
from __future__ import annotations

import os
import time

import pytest

from scizarr_ic import Repo
from scizarr_ic.storage import is_readonly_path, is_remote, storage_for

LIVE = os.environ.get("SCIZARR_IC_LIVE_REPO")
pytestmark = pytest.mark.skipif(not LIVE, reason="SCIZARR_IC_LIVE_REPO not set")


@pytest.fixture
def live(tmp_path, monkeypatch):
    monkeypatch.setenv("SCIZARR_IC_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("SCIZARR_IC_ORIGIN", raising=False)
    return Repo(LIVE)


def test_mount_is_detected_readonly_without_overrides(live, monkeypatch):
    monkeypatch.delenv("SCIZARR_IC_READONLY_PREFIXES", raising=False)
    assert not os.access(LIVE, os.W_OK)
    assert is_readonly_path(LIVE) and live.readonly_path
    assert is_readonly_path(os.path.join(LIVE, "new.icechunk"))  # nothing can be created inside


def test_reads_work_without_opening_the_origin(live):
    assert live.log() and live.tree() and live.branches()
    assert live._writer is None
    assert live.origin_url() and is_remote(live.origin_url())
    assert live.resolved and not live.read_only


def test_copy_to_scratch(live, tmp_path):
    copy = live.copy(tmp_path / "copy.icechunk")
    assert [s.id for s in copy.log()] == [s.id for s in live.log()]
    copy.checkout("scratch", create=True)
    assert "scratch" not in live.branches()


@pytest.mark.skipif(not os.environ.get("SCIZARR_IC_LIVE_WRITE"), reason="SCIZARR_IC_LIVE_WRITE not set")
def test_branch_roundtrip_at_origin(live):
    import icechunk

    name = f"scz-test-{int(time.time())}"
    tip = live.checkout(name, create=True)
    try:
        origin = Repo(live.origin)
        assert name in origin.branches() and origin.log(branch=name)[0].id == tip
        assert live.branch == name and Repo(LIVE).branch == name  # HEAD shared via sidecar
    finally:
        live.checkout("main")
        icechunk.Repository.open(storage_for(live.origin)).delete_branch(name)
    assert name not in Repo(live.origin).branches()
