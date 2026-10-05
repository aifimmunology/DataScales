from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from annizarr.errors import StorageError


def _hide_icechunk(monkeypatch: pytest.MonkeyPatch) -> None:
    # None in sys.modules makes any subsequent `import icechunk` raise ImportError
    # immediately, whether or not the real module was already imported this session
    monkeypatch.setitem(sys.modules, "icechunk", None)


def test_repo_without_icechunk_raises_storage_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _hide_icechunk(monkeypatch)
    import annizarr

    with pytest.raises(StorageError, match=r"annizarr\[icechunk\]"):
        annizarr.Repo(str(tmp_path / "nonexistent"))


def test_import_annizarr_ic_without_icechunk(monkeypatch: pytest.MonkeyPatch) -> None:
    _hide_icechunk(monkeypatch)
    monkeypatch.delitem(sys.modules, "annizarr.ic", raising=False)

    ic = importlib.import_module("annizarr.ic")

    assert hasattr(ic, "Repo")
    assert sys.modules["icechunk"] is None  # importing the module never touched icechunk


def test_repo_copy_to_s3_without_boto3_raises_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    # check_copyable is Repo.copy's pre-flight: no I/O, so no repo needed
    from annizarr.errors import RepoError
    from annizarr.ic._copy import check_copyable

    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with pytest.raises(RepoError, match=r"annizarr\[icechunk\]"):
        check_copyable("some/local/repo", "s3://bucket/prefix")
