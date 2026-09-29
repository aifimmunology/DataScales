from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from annizarr._storage import (
    bucket_prefix,
    canonical_location,
    is_icechunk_repo,
    is_readonly_path,
    open_output_store,
    require_icechunk,
)
from annizarr.config import AppConfig, IOConfig
from annizarr.errors import StorageError

# scheme/is_remote/bucket_prefix(happy path)/store_name are covered more thoroughly by the
# hypothesis property tests in test_layout_props.py; bucket_prefix's error path has no
# hypothesis coverage (its generator always draws a non-empty bucket segment), so it keeps
# a fixed-example test here.


def test_bucket_prefix_no_bucket_raises() -> None:
    with pytest.raises(StorageError, match="Missing bucket"):
        bucket_prefix("s3:///prefix")


def test_is_readonly_path_and_canonical_location(tmp_path: Path) -> None:
    writable = tmp_path / "writable"
    writable.mkdir()
    assert is_readonly_path(str(writable)) is False
    assert canonical_location(str(writable)) == os.path.realpath(str(writable))
    # remote URIs are never read-only, and their canonical location is the URI as given
    assert is_readonly_path("s3://bucket/prefix") is False
    assert canonical_location("s3://bucket/prefix") == "s3://bucket/prefix"


def test_is_icechunk_repo(tmp_path: Path) -> None:
    assert is_icechunk_repo(tmp_path / "does-not-exist") is False  # missing dir

    repo_dir = tmp_path / "repo.icechunk"
    repo_dir.mkdir()
    assert is_icechunk_repo(repo_dir) is False  # no snapshots/, no repo

    (repo_dir / "snapshots").mkdir()
    (repo_dir / "repo").write_text("x")
    assert is_icechunk_repo(repo_dir) is True

    (repo_dir / "zarr.json").write_text("{}")
    assert is_icechunk_repo(repo_dir) is False  # zarr.json present -> plain zarr, not icechunk


def test_require_icechunk_returns_module_or_raises_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("icechunk")
    icechunk = require_icechunk()
    assert icechunk.__name__ == "icechunk"

    monkeypatch.setitem(sys.modules, "icechunk", None)
    with pytest.raises(StorageError) as exc_info:
        require_icechunk()
    assert "annizarr[icechunk]" in str(exc_info.value)


def test_open_output_store_finalize_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "out.zarr"
    out = open_output_store(target, AppConfig(io=IOConfig(consolidate_metadata=True)))
    out.root.attrs["encoding-type"] = "anndata"
    out.root.attrs["encoding-version"] = "0.1.0"
    out.root.require_array("X", shape=(1, 1), dtype="float32")
    assert out.finalize() is None
    assert out.finalize() is None  # second call is a no-op, not a re-consolidate
    assert target.exists()
    assert not list(tmp_path.glob("*.tmp-*"))  # temp dir renamed away, none left behind

    pytest.importorskip("icechunk")
    out_ic = open_output_store(tmp_path / "repo.icechunk", AppConfig(io=IOConfig(backend="icechunk")))
    out_ic.root.attrs["marker"] = 1
    first = out_ic.finalize()
    second = out_ic.finalize()
    assert first is not None
    assert second == first  # cached, not a second (empty) commit


def test_open_output_store_remote_zarr_backend_errors_without_importing_icechunk() -> None:
    # icechunk may already be imported by an earlier test in this session; hide it so
    # this test can verify the remote+zarr-backend error path never imports it.
    saved = sys.modules.pop("icechunk", None)
    try:
        cfg = AppConfig()  # default backend is "zarr"
        with pytest.raises(StorageError, match="--ic"):
            open_output_store("s3://bucket/store.zarr", cfg)
        assert "icechunk" not in sys.modules
    finally:
        if saved is not None:
            sys.modules["icechunk"] = saved
