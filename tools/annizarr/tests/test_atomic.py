"""Atomic plain-zarr outputs: temp-dir writes, verify-then-swap, and cleanup on failure."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import anndata as ad
import pytest

from _builders import make_adata, make_h5ad, make_store
from annizarr.config import AppConfig, ChunkConfig, IOConfig
from annizarr.errors import ConversionError, StorageError
from annizarr.ops import convert_h5ad, sort

_CHUNKS = ChunkConfig(x_row_chunk=16, x_col_chunk=4, sparse_flat_chunk=64)


def _cfg(**io) -> AppConfig:
    return AppConfig(io=IOConfig(**io), chunks=_CHUNKS)


def _boom(*_a: object, **_kw: object) -> None:
    raise RuntimeError("boom")


def _fail_open_store_replace_on_call(monkeypatch: pytest.MonkeyPatch, n: int) -> None:
    # zarr's own LocalStore writes go through os.replace per chunk (atomic .partial-file
    # renames), so patching the os module directly would count those too; instead swap out
    # just the `os` name inside annizarr._storage._open, where the atomic-swap calls happen,
    # so only THOSE calls are counted/faked (the n-th one raises without performing the
    # rename; every other call goes through the real os.replace).
    import types

    import annizarr._storage._open as _open_mod

    real_replace = os.replace
    calls = {"count": 0}

    def _fake(src: object, dst: object) -> None:
        calls["count"] += 1
        if calls["count"] == n:
            raise OSError("boom-replace")
        real_replace(src, dst)

    monkeypatch.setattr(_open_mod, "os", types.SimpleNamespace(replace=_fake))


def test_convert_failure_or_interrupt_leaves_no_target_and_no_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # annizarr.ops._convert (the name) is shadowed by the convert function; fetch the module via sys.modules.
    _convert_mod = importlib.import_module("annizarr.ops._convert")

    def _interrupt(*_a: object, **_kw: object) -> None:
        raise KeyboardInterrupt

    h5 = make_h5ad(tmp_path, adata=make_adata(n_obs=10, seed=0))

    monkeypatch.setattr(_convert_mod, "write_adata", _boom)
    out = tmp_path / "out.zarr"
    with pytest.raises(RuntimeError, match="boom"):
        convert_h5ad(str(h5), output=str(out), cfg=_cfg())
    assert not out.exists()
    assert not list(tmp_path.glob("out.zarr.tmp-*"))

    # KeyboardInterrupt is a BaseException, not an Exception -- cleanup must still run
    monkeypatch.setattr(_convert_mod, "write_adata", _interrupt)
    out2 = tmp_path / "out2.zarr"
    with pytest.raises(KeyboardInterrupt):
        convert_h5ad(str(h5), output=str(out2), cfg=_cfg())
    assert not out2.exists()
    assert not list(tmp_path.glob("out2.zarr.tmp-*"))


def test_convert_overwrite_failing_rerun_keeps_old_store_then_successful_rerun_replaces_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h5 = make_h5ad(tmp_path, adata=make_adata(n_obs=10, seed=0))
    out = tmp_path / "out.zarr"
    convert_h5ad(str(h5), output=str(out), cfg=_cfg())
    old_n_obs = ad.read_zarr(str(out)).n_obs

    # annizarr.ops._convert (the name) is shadowed by the convert function; fetch the module via sys.modules.
    _convert_mod = importlib.import_module("annizarr.ops._convert")

    monkeypatch.setattr(_convert_mod, "write_adata", _boom)
    with pytest.raises(RuntimeError, match="boom"):
        convert_h5ad(str(h5), output=str(out), cfg=_cfg(overwrite=True))
    assert out.exists()
    assert not list(tmp_path.glob("out.zarr.tmp-*"))
    assert ad.read_zarr(str(out)).n_obs == old_n_obs  # old store intact and still readable
    monkeypatch.undo()

    h5_b = make_h5ad(tmp_path, "b.h5ad", adata=make_adata(n_obs=7, seed=1))
    convert_h5ad(str(h5_b), output=str(out), cfg=_cfg(overwrite=True))
    assert ad.read_zarr(str(out)).n_obs == 7
    assert not list(tmp_path.glob("out.zarr.tmp-*"))


def test_convert_overwrite_os_replace_failure_on_either_call_restores_old_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1st os.replace is target -> aside; 2nd is tmp -> target (the 1st moved the old store
    # aside). Failing on the 1st means the swap never began (old store untouched, never even
    # moved); failing on the 2nd means the old store must be put back at target. Either way:
    # no .tmp-*/.old-* siblings left behind.
    h5_a = make_h5ad(tmp_path, "a.h5ad", adata=make_adata(n_obs=10, seed=0))
    expected_names = {"a.h5ad"}
    for call_n in (1, 2):
        out = tmp_path / f"out{call_n}.zarr"
        convert_h5ad(str(h5_a), output=str(out), cfg=_cfg())
        old_n_obs = ad.read_zarr(str(out)).n_obs

        h5_b = make_h5ad(tmp_path, f"b{call_n}.h5ad", adata=make_adata(n_obs=7, seed=1))
        _fail_open_store_replace_on_call(monkeypatch, call_n)
        with pytest.raises(ConversionError, match="boom-replace"):
            convert_h5ad(str(h5_b), output=str(out), cfg=_cfg(overwrite=True))
        monkeypatch.undo()

        assert out.exists()
        assert ad.read_zarr(str(out)).n_obs == old_n_obs
        expected_names |= {f"b{call_n}.h5ad", f"out{call_n}.zarr"}
        assert {p.name for p in tmp_path.iterdir()} == expected_names  # no .tmp-*/.old-* siblings


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

    import annizarr._core._sorting as _sorting_mod

    monkeypatch.setattr(_sorting_mod, "write_matrix", _boom)
    sorted_out2 = tmp_path / "sorted2.zarr"
    with pytest.raises(RuntimeError, match="boom"):
        sort(str(store), output=str(sorted_out2), by=("cell_type",), cfg=_cfg())

    assert not sorted_out2.exists()
    assert not list(tmp_path.glob("annizarr_sort_*"))
    assert not list(tmp_path.glob("sorted2.zarr.tmp-*"))


def test_convert_icechunk_failure_leaves_no_extra_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("icechunk")
    # annizarr.ops._convert (the name) is shadowed by the convert function; fetch the module via sys.modules.
    _convert_mod = importlib.import_module("annizarr.ops._convert")
    from annizarr.ic import Repo

    monkeypatch.setattr(_convert_mod, "write_adata", _boom)

    h5 = make_h5ad(tmp_path, adata=make_adata(n_obs=10, seed=0))
    repo_path = tmp_path / "repo.icechunk"
    with pytest.raises(RuntimeError, match="boom"):
        convert_h5ad(str(h5), output=str(repo_path), cfg=_cfg(backend="icechunk"))

    repo = Repo(str(repo_path))
    assert [s.message for s in repo.log(branch="main")] == ["Repository initialized"]  # no commit landed


def _write_wrong_shape_x(root, shape: tuple[int, int]) -> None:
    root.attrs["encoding-type"] = "anndata"
    root.attrs["encoding-version"] = "0.1.0"
    root.require_array("X", shape=shape, dtype="float32", chunks=shape)


def test_finalize_rejects_x_shape_mismatch_and_leaves_no_target(tmp_path: Path) -> None:
    from annizarr._storage import open_output_store

    out_path = tmp_path / "bad.zarr"
    out = open_output_store(out_path, _cfg(), expected_shape=(5, 5))
    _write_wrong_shape_x(out.root, (2, 2))
    with pytest.raises(StorageError, match="shape mismatch"):
        out.finalize()
    out.abort()
    assert not out_path.exists()
    assert not list(tmp_path.glob("bad.zarr.tmp-*"))


def test_finalize_shape_mismatch_leaves_existing_overwrite_target_intact(tmp_path: Path) -> None:
    from annizarr._storage import open_output_store

    store = make_store(tmp_path, "existing.zarr", adata=make_adata(n_obs=5, n_vars=5, seed=0), cfg=_cfg())
    old_n_obs = ad.read_zarr(str(store)).n_obs

    out = open_output_store(store, _cfg(overwrite=True), expected_shape=(5, 5))
    _write_wrong_shape_x(out.root, (3, 3))
    with pytest.raises(StorageError, match="shape mismatch"):
        out.finalize()
    out.abort()

    assert store.exists()
    assert ad.read_zarr(str(store)).n_obs == old_n_obs
