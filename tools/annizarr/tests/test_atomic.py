"""Atomic plain-zarr outputs: temp-dir writes, verify-then-swap, and cleanup on failure."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import pytest

from _builders import make_adata, make_h5ad, make_store
from annizarr._ops import convert_h5ad, sort
from annizarr.config import AppConfig, ChunkConfig, IOConfig

_CHUNKS = ChunkConfig(x_row_chunk=16, x_col_chunk=4, sparse_flat_chunk=64)


def _cfg(**io) -> AppConfig:
    return AppConfig(io=IOConfig(**io), chunks=_CHUNKS)


def _boom(*_a: object, **_kw: object) -> None:
    raise RuntimeError("boom")


def test_convert_failure_leaves_no_target_and_no_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import annizarr._ops._convert as _convert_mod

    monkeypatch.setattr(_convert_mod, "write_adata", _boom)

    h5 = make_h5ad(tmp_path, adata=make_adata(n_obs=10, seed=0))
    out = tmp_path / "out.zarr"
    with pytest.raises(RuntimeError, match="boom"):
        convert_h5ad(str(h5), output=str(out), cfg=_cfg())

    assert not out.exists()
    assert not list(tmp_path.glob("out.zarr.tmp-*"))


def test_convert_keyboard_interrupt_leaves_no_target_and_no_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import annizarr._ops._convert as _convert_mod

    def _interrupt(*_a: object, **_kw: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(_convert_mod, "write_adata", _interrupt)

    h5 = make_h5ad(tmp_path, adata=make_adata(n_obs=10, seed=0))
    out = tmp_path / "out.zarr"
    with pytest.raises(KeyboardInterrupt):
        convert_h5ad(str(h5), output=str(out), cfg=_cfg())

    assert not out.exists()
    assert not list(tmp_path.glob("out.zarr.tmp-*"))


def test_convert_overwrite_failing_rerun_leaves_old_store_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h5 = make_h5ad(tmp_path, adata=make_adata(n_obs=10, seed=0))
    out = tmp_path / "out.zarr"
    convert_h5ad(str(h5), output=str(out), cfg=_cfg())
    old_n_obs = ad.read_zarr(str(out)).n_obs

    import annizarr._ops._convert as _convert_mod

    monkeypatch.setattr(_convert_mod, "write_adata", _boom)
    with pytest.raises(RuntimeError, match="boom"):
        convert_h5ad(str(h5), output=str(out), cfg=_cfg(overwrite=True))

    assert out.exists()
    assert not list(tmp_path.glob("out.zarr.tmp-*"))
    assert ad.read_zarr(str(out)).n_obs == old_n_obs  # old store intact and still readable


def test_convert_overwrite_successful_rerun_replaces_store(tmp_path: Path) -> None:
    out = tmp_path / "out.zarr"
    h5_a = make_h5ad(tmp_path, "a.h5ad", adata=make_adata(n_obs=10, seed=0))
    convert_h5ad(str(h5_a), output=str(out), cfg=_cfg())
    h5_b = make_h5ad(tmp_path, "b.h5ad", adata=make_adata(n_obs=7, seed=1))
    convert_h5ad(str(h5_b), output=str(out), cfg=_cfg(overwrite=True))

    assert ad.read_zarr(str(out)).n_obs == 7
    assert not list(tmp_path.glob("out.zarr.tmp-*"))


def test_convert_success_leaves_only_the_target(tmp_path: Path) -> None:
    h5 = make_h5ad(tmp_path, adata=make_adata(n_obs=10, seed=0))
    out = tmp_path / "out.zarr"
    convert_h5ad(str(h5), output=str(out), cfg=_cfg())
    assert {p.name for p in tmp_path.iterdir()} == {h5.name, "out.zarr"}


def test_sort_temp_buckets_cleaned_up_on_success_and_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = make_store(tmp_path, adata=make_adata(n_obs=20, seed=0), cfg=_cfg())

    sorted_out = tmp_path / "sorted.zarr"
    sort(str(store), output=str(sorted_out), by=("cell_type",), cfg=_cfg())
    assert sorted_out.exists()
    assert not list(tmp_path.glob("annizarr_sort_*"))
    assert not list(tmp_path.glob("sorted.zarr.tmp-*"))

    import annizarr._sorting as _sorting_mod

    monkeypatch.setattr(_sorting_mod, "_write_concatenated_csr", _boom)
    sorted_out2 = tmp_path / "sorted2.zarr"
    with pytest.raises(RuntimeError, match="boom"):
        sort(str(store), output=str(sorted_out2), by=("cell_type",), cfg=_cfg())

    assert not sorted_out2.exists()
    assert not list(tmp_path.glob("annizarr_sort_*"))
    assert not list(tmp_path.glob("sorted2.zarr.tmp-*"))


def test_convert_icechunk_failure_leaves_no_extra_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("icechunk")
    import annizarr._ops._convert as _convert_mod
    from annizarr._ic import Repo

    monkeypatch.setattr(_convert_mod, "write_adata", _boom)

    h5 = make_h5ad(tmp_path, adata=make_adata(n_obs=10, seed=0))
    repo_path = tmp_path / "repo.icechunk"
    with pytest.raises(RuntimeError, match="boom"):
        convert_h5ad(str(h5), output=str(repo_path), cfg=_cfg(backend="icechunk"))

    repo = Repo(str(repo_path))
    assert [s.message for s in repo.log(branch="main")] == ["Repository initialized"]  # no commit landed
