from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from annizarr._cli import main
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


def test_cli_ic_log_without_icechunk_reports_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _hide_icechunk(monkeypatch)

    exit_code = main(["ic", "log", str(tmp_path / "nonexistent")])
    err = capsys.readouterr().err

    assert exit_code == 1
    assert "error:" in err
    assert "annizarr[icechunk]" in err


def test_import_annizarr_ic_without_icechunk(monkeypatch: pytest.MonkeyPatch) -> None:
    _hide_icechunk(monkeypatch)
    for mod in ("annizarr.ic", "annizarr.ic"):
        monkeypatch.delitem(sys.modules, mod, raising=False)

    ic = importlib.import_module("annizarr.ic")

    assert hasattr(ic, "Repo")
    assert sys.modules["icechunk"] is None  # importing the module never touched icechunk


def test_repo_copy_to_s3_without_boto3_raises_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    # check_copyable is Repo.copy's/`ic copy`'s pre-flight: no I/O, so no repo needed
    from annizarr.errors import RepoError
    from annizarr.ic._copy import check_copyable

    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with pytest.raises(RepoError, match=r"annizarr\[icechunk\]"):
        check_copyable("some/local/repo", "s3://bucket/prefix")


def test_cli_ic_copy_to_s3_without_boto3_reports_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("icechunk")  # a real repo is needed to reach the s3 destination check
    import zarr

    from annizarr.ic import Repo

    src = tmp_path / "src.zarr"
    zarr.open_group(str(src), mode="w").create_array("X", shape=(2, 2), dtype="float32", chunks=(2, 2))
    repo_path = tmp_path / "repo.icechunk"
    Repo.init(src, repo_path)

    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    exit_code = main(["ic", "copy", str(repo_path), "s3://bucket/prefix"])
    err = capsys.readouterr().err

    assert exit_code == 1
    assert "annizarr[icechunk]" in err
