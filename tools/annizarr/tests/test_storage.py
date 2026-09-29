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
    is_remote,
    open_output_store,
    require_icechunk,
    scheme,
    store_name,
)
from annizarr.config import AppConfig, IOConfig
from annizarr.errors import StorageError


def test_scheme() -> None:
    assert scheme("s3://bucket/prefix") == "s3"
    assert scheme("gs://bucket") == "gs"
    assert scheme("gcs://bucket/prefix") == "gcs"
    assert scheme("/local/path") == ""
    assert scheme("relative/path") == ""


def test_is_remote() -> None:
    assert is_remote("s3://bucket/prefix") is True
    assert is_remote("gs://bucket") is True
    assert is_remote("gcs://bucket/prefix") is True
    assert is_remote("/local/path") is False
    assert is_remote("relative/path") is False


def test_bucket_prefix() -> None:
    assert bucket_prefix("s3://bucket/prefix") == ("bucket", "prefix")
    assert bucket_prefix("gs://bucket") == ("bucket", None)
    assert bucket_prefix("gcs://bucket/a/b") == ("bucket", "a/b")


def test_bucket_prefix_no_bucket_raises() -> None:
    with pytest.raises(StorageError, match="Missing bucket"):
        bucket_prefix("s3:///prefix")


def test_store_name() -> None:
    assert store_name("/a/b/c.zarr/") == "c.zarr"
    assert store_name("/a/b/c.zarr") == "c.zarr"
    assert store_name("s3://bucket/prefix/store.zarr") == "store.zarr"


def test_is_readonly_path_and_canonical_location(tmp_path: Path) -> None:
    writable = tmp_path / "writable"
    writable.mkdir()
    assert is_readonly_path(str(writable)) is False
    assert canonical_location(str(writable)) == os.path.realpath(str(writable))
    # remote URIs are never read-only, and their canonical location is the URI as given
    assert is_readonly_path("s3://bucket/prefix") is False
    assert canonical_location("s3://bucket/prefix") == "s3://bucket/prefix"


def test_is_icechunk_repo(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo.icechunk"
    repo_dir.mkdir()
    assert is_icechunk_repo(repo_dir) is False  # no snapshots/, no repo

    (repo_dir / "snapshots").mkdir()
    (repo_dir / "repo").write_text("x")
    assert is_icechunk_repo(repo_dir) is True

    (repo_dir / "zarr.json").write_text("{}")
    assert is_icechunk_repo(repo_dir) is False  # zarr.json present -> plain zarr, not icechunk


def test_is_icechunk_repo_missing_dir(tmp_path: Path) -> None:
    assert is_icechunk_repo(tmp_path / "does-not-exist") is False


def test_require_icechunk_returns_module() -> None:
    pytest.importorskip("icechunk")
    icechunk = require_icechunk()
    assert icechunk.__name__ == "icechunk"


def test_require_icechunk_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "icechunk", None)
    with pytest.raises(StorageError) as exc_info:
        require_icechunk()
    assert "annizarr[icechunk]" in str(exc_info.value)


def test_open_output_store_finalize_is_idempotent_plain_zarr(tmp_path: Path) -> None:
    root, finalize = open_output_store(tmp_path / "out.zarr", AppConfig(io=IOConfig(consolidate_metadata=True)))
    root.attrs["marker"] = 1
    assert finalize() is None
    assert finalize() is None  # second call is a no-op, not a re-consolidate


def test_open_output_store_finalize_is_idempotent_icechunk(tmp_path: Path) -> None:
    pytest.importorskip("icechunk")
    root, finalize = open_output_store(tmp_path / "repo.icechunk", AppConfig(io=IOConfig(backend="icechunk")))
    root.attrs["marker"] = 1
    first = finalize()
    second = finalize()
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
