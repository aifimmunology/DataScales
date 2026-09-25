"""End-to-end CLI coverage (init/log/tree/checkout/cherrypick)."""
from __future__ import annotations

import pytest

from scizarr_ic import Repo
from scizarr_ic.cli import main


def test_cli_init_and_log(src_zarr, repo_path, capsys):
    path, _ = src_zarr
    assert main(["init", str(path), str(repo_path), "-m", "first import"]) == 0
    assert "Initialized icechunk repo" in capsys.readouterr().out

    assert main(["log", "-C", str(repo_path), "--oneline"]) == 0
    out = capsys.readouterr().out
    assert "first import" in out


def test_cli_checkout_and_tree(src_zarr, repo_path, capsys):
    path, _ = src_zarr
    main(["init", str(path), str(repo_path)])
    capsys.readouterr()

    assert main(["checkout", "-C", str(repo_path), "-b", "dev"]) == 0
    assert "dev" in capsys.readouterr().out

    assert main(["tree", "-C", str(repo_path)]) == 0
    tree_out = capsys.readouterr().out
    assert "* dev" in tree_out  # current branch marked
    assert "main" in tree_out


def test_cli_cherrypick(src_zarr, repo_path, capsys):
    path, _ = src_zarr
    repo = Repo.init(path, repo_path)
    base = repo.log()[0].id
    repo.writable().attrs["x"] = 1
    repo.commit("second")
    capsys.readouterr()

    assert main(["cherrypick", "-C", str(repo_path), base]) == 0
    assert base[:12] in capsys.readouterr().out
    assert "x" not in Repo(repo_path).root().attrs


def test_cli_error_exit_code(tmp_path, capsys):
    rc = main(["log", "-C", str(tmp_path / "missing.icechunk")])
    assert rc == 1
    assert "error:" in capsys.readouterr().err
