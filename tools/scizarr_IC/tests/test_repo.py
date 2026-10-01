"""Python-API coverage for scizarr_ic.Repo."""
from __future__ import annotations

import numpy as np
import pytest

from scizarr_ic import Repo, ScizarrError


def test_init_imports_store_faithfully(src_zarr, repo_path):
    path, x = src_zarr
    repo = Repo.init(path, repo_path, message="import")

    assert repo.branch == "main"
    root = repo.open_zarr("r")
    assert root.attrs["encoding-type"] == "anndata"
    xa = root["X"]
    np.testing.assert_array_equal(xa[...], x)
    # layout carried over from the source
    assert xa.chunks == (6, 5)
    assert xa.attrs["encoding-type"] == "array"
    assert set(root["obs"].array_keys()) == {"_index"}


def test_init_rejects_nonempty_output(src_zarr, repo_path):
    path, _ = src_zarr
    Repo.init(path, repo_path)
    with pytest.raises(ScizarrError):
        Repo.init(path, repo_path)


def test_open_missing_repo_errors(tmp_path):
    with pytest.raises(ScizarrError):
        Repo(tmp_path / "nope.icechunk")


def test_commit_records_snapshot(src_zarr, repo_path):
    path, _ = src_zarr
    repo = Repo.init(path, repo_path)

    g = repo.open_zarr("w")
    g.attrs["note"] = "edited"
    snapshot_id = repo.commit("add note")

    assert isinstance(snapshot_id, str) and snapshot_id
    log = repo.log()
    assert log[0].id == snapshot_id
    assert log[0].message == "add note"
    assert repo.open_zarr("r").attrs["note"] == "edited"


def test_commit_without_session_errors(src_zarr, repo_path):
    path, _ = src_zarr
    repo = Repo.init(path, repo_path)
    with pytest.raises(ScizarrError):
        repo.commit("nothing staged")


def test_log_newest_first(src_zarr, repo_path):
    path, _ = src_zarr
    repo = Repo.init(path, repo_path)
    for i in range(3):
        repo.open_zarr("w").attrs[f"k{i}"] = i
        repo.commit(f"edit {i}")
    messages = [s.message for s in repo.log()]
    assert messages[:3] == ["edit 2", "edit 1", "edit 0"]


def test_checkout_create_branches_and_isolates_commits(src_zarr, repo_path):
    path, _ = src_zarr
    repo = Repo.init(path, repo_path)
    repo.open_zarr("w").attrs["on_main"] = True
    repo.commit("main work")

    repo.checkout("experiment", create=True)
    assert repo.branch == "experiment"
    repo.open_zarr("w").attrs["on_exp"] = True
    repo.commit("exp work")

    assert "on_exp" in repo.open_zarr("r").attrs
    repo.checkout("main")
    assert "on_exp" not in repo.open_zarr("r").attrs
    assert "on_main" in repo.open_zarr("r").attrs
    assert set(repo.branches()) == {"main", "experiment"}


def test_checkout_missing_branch_errors(src_zarr, repo_path):
    path, _ = src_zarr
    repo = Repo.init(path, repo_path)
    with pytest.raises(ScizarrError):
        repo.checkout("ghost")


def test_reopen_lands_on_main_unless_branch_given(src_zarr, repo_path):
    path, _ = src_zarr
    Repo.init(path, repo_path).checkout("dev", create=True)   # checkout is in-process only

    assert Repo(repo_path).branch == "main"
    assert Repo(repo_path, branch="dev").branch == "dev"
    with pytest.raises(ScizarrError, match="No branch 'ghost'"):
        Repo(repo_path, branch="ghost")


def test_cherrypick_resets_branch_state(src_zarr, repo_path):
    path, _ = src_zarr
    repo = Repo.init(path, repo_path)
    first = repo.log()[0].id

    repo.open_zarr("w").attrs["v"] = 2
    repo.commit("second")
    assert repo.open_zarr("r").attrs.get("v") == 2

    repo.cherrypick(first)
    assert "v" not in repo.open_zarr("r").attrs
    assert repo.log()[0].id == first


def test_cherrypick_unknown_snapshot_errors(src_zarr, repo_path):
    path, _ = src_zarr
    repo = Repo.init(path, repo_path)
    with pytest.raises(ScizarrError):
        repo.cherrypick("deadbeef")


def test_uncommitted_changes_block_checkout(src_zarr, repo_path):
    path, _ = src_zarr
    repo = Repo.init(path, repo_path)
    repo.open_zarr("w").attrs["dirty"] = True
    with pytest.raises(ScizarrError):
        repo.checkout("other", create=True)
    repo.discard()
    repo.checkout("other", create=True)  # clean now


def test_tree_lists_all_branches(src_zarr, repo_path):
    path, _ = src_zarr
    repo = Repo.init(path, repo_path)
    repo.checkout("b1", create=True)
    repo.checkout("main")
    repo.checkout("b2", create=True)

    assert set(repo.branches()) == {"main", "b1", "b2"}
    shown = repr(repo.tree())
    assert shown.startswith("On branch b2\n") and all(b in shown for b in ("main", "b1", "b2"))


def test_create_empty_repo_and_exists(tmp_path):
    out = tmp_path / "empty.icechunk"
    assert not Repo.exists(out) and not out.exists()
    repo = Repo.create(out)
    assert Repo.exists(out) and repo.branch == "main"
    assert [s.message for s in repo.log()] == ["Repository initialized"]
    with pytest.raises(ScizarrError, match="not empty"):
        Repo.create(out)


def test_writable_reuses_one_session_until_commit(src_zarr, repo_path):
    path, _ = src_zarr
    repo = Repo.init(path, repo_path)
    repo.open_zarr("w").attrs["k"] = 1
    repo.open_zarr("w").attrs["j"] = 2          # same staged session, not a second one
    assert repo._session.has_uncommitted_changes
    repo.commit("one commit for both")
    assert repo._session is None and repo.open_zarr("r").attrs["k"] == 1 and repo.open_zarr("r").attrs["j"] == 2


def test_open_zarr_modes_and_snapshot_rules(src_zarr, repo_path):
    path, _ = src_zarr
    repo = Repo.init(path, repo_path)
    first = repo.log()[0].id
    repo.open_zarr("w").attrs["v"] = 2
    repo.commit("second")

    # "r" time-travels; "w" always writes the tip, so snapshot_id is rejected there
    assert "v" not in repo.open_zarr("r", snapshot_id=first).attrs
    assert repo.open_zarr("r").attrs["v"] == 2
    with pytest.raises(ScizarrError, match="snapshot_id is read-only"):
        repo.open_zarr("w", snapshot_id=first)
    with pytest.raises(ScizarrError, match="mode must be 'r' or 'w'"):
        repo.open_zarr("a")


def test_reprs_read_git_like(src_zarr, repo_path):
    path, _ = src_zarr
    repo = Repo.init(path, repo_path, message="import")
    repo.checkout("dev", create=True)
    repo.open_zarr("w").attrs["k"] = 1
    snap = repo.commit("edit k")

    shown = repr(repo)
    assert shown.splitlines()[0] == f"Repo({str(repo_path)!r})"
    assert "  branch: dev" in shown and f"  tip:    {snap[:12]}" in shown and shown.endswith("edit k")
    assert repr(repo.branches()).splitlines() == ["* dev", "  main"]
    log = repr(repo.log()).splitlines()
    assert len(log) == 3 and log[0].startswith(snap[:12]) and log[0].endswith("  edit k")
    assert log[-1].endswith("Repository initialized")

    root = repo.open_zarr("r")
    head, *members = repr(root).splitlines()
    assert head == f"<Group '/' at {repo_path}  branch dev  snapshot {snap[:12]}  read-only>"
    assert members == ["  arrays: X", "  groups: obs, var"]
    assert "detached" in repr(repo.open_zarr("r", snapshot_id=snap))

    repo.open_zarr("w").attrs["j"] = 2
    assert "writable, uncommitted changes>" in repr(repo.open_zarr("w")).splitlines()[0]
    assert repr(repo).endswith("  staged: uncommitted changes")
    repo.discard()
    assert "staged" not in repr(repo)
