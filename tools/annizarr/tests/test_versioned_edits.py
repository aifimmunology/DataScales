"""Versioned editing through Repo: branch/message wiring, one commit per op, time-travel."""

from __future__ import annotations

from pathlib import Path

import pytest

from _builders import make_adata, make_h5ad
from annizarr._ic import Repo
from annizarr._ops import add_expr, append, convert_h5ad, sort
from annizarr._storage import open_input_group
from annizarr.config import AppConfig, ChunkConfig, IOConfig

_CHUNKS = ChunkConfig(x_row_chunk=16, x_col_chunk=4, sparse_flat_chunk=64)


def _ic_cfg(**io) -> AppConfig:
    return AppConfig(io=IOConfig(backend="icechunk", **io), chunks=_CHUNKS)


def _plain_cfg(**io) -> AppConfig:
    return AppConfig(io=IOConfig(**io), chunks=_CHUNKS)


def test_icechunk_lifecycle_one_commit_per_op(tmp_path: Path) -> None:
    pytest.importorskip("icechunk")
    repo_path = tmp_path / "repo.icechunk"

    h5_a = make_h5ad(tmp_path, "a.h5ad", adata=make_adata(n_obs=20, seed=0))
    r_convert = convert_h5ad(str(h5_a), output=str(repo_path), cfg=_ic_cfg())

    r_add_expr = add_expr(str(repo_path), fmt="csr", cfg=_ic_cfg())

    h5_b = make_h5ad(tmp_path, "b.h5ad", adata=make_adata(n_obs=8, seed=1))
    cells_store = tmp_path / "b.zarr"
    convert_h5ad(str(h5_b), output=str(cells_store), cfg=_plain_cfg())
    r_append = append(str(repo_path), cells=str(cells_store), drop_derived=True, cfg=_ic_cfg())

    sorted_path = tmp_path / "sorted.icechunk"
    r_sort = sort(str(repo_path), output=str(sorted_path), by=("cell_type",), cfg=_ic_cfg())

    repo = Repo(str(repo_path))
    log = repo.log(branch="main")
    # newest first: append, add-expr, convert, then icechunk's own "Repository initialized"
    assert [s.id for s in log[:3]] == [r_append.snapshot_id, r_add_expr.snapshot_id, r_convert.snapshot_id]
    assert log[-1].message == "Repository initialized"
    assert len(log) == 4  # exactly one commit per op, nothing extra

    assert "annizarr append" in log[0].message
    assert "annizarr add-expr csr" in log[1].message
    assert "annizarr convert" in log[2].message

    sorted_repo = Repo(str(sorted_path))
    sorted_log = sorted_repo.log(branch="main")
    assert len(sorted_log) == 2  # sort's own new repo: one commit + the init snapshot
    assert sorted_log[0].id == r_sort.snapshot_id
    assert "annizarr sort by cell_type" in sorted_log[0].message


def test_branch_dev_created_on_existing_repo_leaves_main_unchanged(tmp_path: Path) -> None:
    pytest.importorskip("icechunk")
    repo_path = tmp_path / "repo.icechunk"
    h5 = make_h5ad(tmp_path, adata=make_adata(n_obs=20, seed=0))
    r_convert = convert_h5ad(str(h5), output=str(repo_path), cfg=_ic_cfg())

    r_dev = add_expr(str(repo_path), fmt="csr", cfg=_plain_cfg(), branch="dev")
    assert r_dev.snapshot_id is not None

    main_log = Repo(str(repo_path), branch="main").log(branch="main")
    assert main_log[0].id == r_convert.snapshot_id  # main's tip untouched by the dev commit

    dev_log = Repo(str(repo_path), branch="dev").log(branch="dev")
    assert dev_log[0].id == r_dev.snapshot_id
    assert r_convert.snapshot_id in {s.id for s in dev_log}  # dev branched off main's tip


def test_custom_message_appears_verbatim(tmp_path: Path) -> None:
    pytest.importorskip("icechunk")
    repo_path = tmp_path / "repo.icechunk"
    h5 = make_h5ad(tmp_path, adata=make_adata(n_obs=10, seed=0))
    convert_h5ad(str(h5), output=str(repo_path), cfg=_ic_cfg(), message="a custom message")

    repo = Repo(str(repo_path))
    assert repo.log(branch="main")[0].message == "a custom message"


def test_open_input_group_snapshot_id_reads_pre_add_expr_state(tmp_path: Path) -> None:
    pytest.importorskip("icechunk")
    repo_path = tmp_path / "repo.icechunk"
    h5 = make_h5ad(tmp_path, adata=make_adata(n_obs=10, seed=0))
    r_convert = convert_h5ad(str(h5), output=str(repo_path), cfg=_ic_cfg())
    add_expr(str(repo_path), fmt="csr", cfg=_ic_cfg())

    old = open_input_group(str(repo_path), snapshot_id=r_convert.snapshot_id)
    assert "gexp" not in (list(old["layers"]) if "layers" in old else [])

    current = open_input_group(str(repo_path))
    assert "gexp" in list(current["layers"])


def test_in_place_op_auto_detects_icechunk_without_backend_flag(tmp_path: Path) -> None:
    pytest.importorskip("icechunk")
    repo_path = tmp_path / "repo.icechunk"
    h5 = make_h5ad(tmp_path, adata=make_adata(n_obs=10, seed=0))
    convert_h5ad(str(h5), output=str(repo_path), cfg=_ic_cfg())

    # cfg.io.backend defaults to "zarr" here — the icechunk repo layout is auto-detected
    result = add_expr(str(repo_path), fmt="csr", cfg=_plain_cfg())
    assert result.snapshot_id is not None
