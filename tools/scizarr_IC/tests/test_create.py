"""Repo.create / Repo.exists / Repo.session — the pieces an ingest pipeline builds on."""
from __future__ import annotations

import pytest

from scizarr_ic import Repo, ScizarrError


def test_create_empty_repo_and_exists(tmp_path):
    out = tmp_path / "empty.icechunk"
    assert not Repo.exists(out)
    repo = Repo.create(out)
    assert Repo.exists(out) and repo.branch == "main"
    assert [s.message for s in repo.log()] == ["Repository initialized"]
    assert repo.origin_url() == str(out.resolve())
    with pytest.raises(ScizarrError, match="not empty"):
        Repo.create(out)


def test_exists_never_creates(tmp_path):
    p = tmp_path / "nothing.icechunk"
    assert not Repo.exists(p) and not p.exists()


def test_session_shares_writable_state(src_zarr, repo_path):
    path, _ = src_zarr
    repo = Repo.init(path, repo_path)
    ws = repo.session(writable=True)
    assert ws is repo.session(writable=True) and not ws.read_only
    repo.writable().attrs["via_group"] = 1
    assert ws.has_uncommitted_changes
    repo.commit("one commit for both")
    assert repo.session().read_only
    assert repo.root().attrs["via_group"] == 1
