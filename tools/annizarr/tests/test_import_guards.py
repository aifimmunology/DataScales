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
    for mod in ("annizarr.ic", "annizarr._ic"):
        monkeypatch.delitem(sys.modules, mod, raising=False)

    ic = importlib.import_module("annizarr.ic")

    assert hasattr(ic, "Repo")
    assert sys.modules["icechunk"] is None  # importing the module never touched icechunk
