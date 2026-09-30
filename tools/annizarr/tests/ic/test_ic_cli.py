"""End-to-end CLI coverage (init/log/tree/checkout/cherrypick) through the flat CLI."""

from __future__ import annotations

from annizarr._cli import main
from annizarr.ic import Repo


def test_cli_init_log_checkout_tree(src_zarr, repo_path, capsys):
    path, _ = src_zarr
    assert main(["ic", "init", str(path), str(repo_path), "-m", "first import"]) == 0
    assert "Initialized icechunk repo" in capsys.readouterr().err

    assert main(["ic", "log", str(repo_path), "--oneline"]) == 0
    assert "first import" in capsys.readouterr().out

    assert main(["ic", "checkout", str(repo_path), "-b", "dev"]) == 0
    assert "dev" in capsys.readouterr().err

    assert main(["ic", "tree", str(repo_path)]) == 0
    tree_out = capsys.readouterr().out
    assert "* dev" in tree_out  # current branch marked
    assert "main" in tree_out


def test_cli_cherrypick(src_zarr, repo_path, capsys):
    path, _ = src_zarr
    repo = Repo.init(path, repo_path)
    base = repo.log()[0].id
    repo.open_zarr("w").attrs["x"] = 1
    repo.commit("second")
    capsys.readouterr()

    assert main(["ic", "cherrypick", str(repo_path), base]) == 0
    assert base[:12] in capsys.readouterr().err
    assert "x" not in Repo(repo_path).open_zarr("r").attrs


def test_cli_error_exit_code(tmp_path, capsys):
    rc = main(["ic", "log", str(tmp_path / "missing.icechunk")])
    assert rc == 1
    assert "error:" in capsys.readouterr().err
