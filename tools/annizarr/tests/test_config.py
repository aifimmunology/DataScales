import os
from pathlib import Path

import pytest

from annizarr._core._config import resolve_backend_cfg
from annizarr.config import AppConfig, ChunkConfig, IOConfig, apply_cli_overrides, load_config


def test_defaults() -> None:
    cfg = load_config(None)
    assert isinstance(cfg, AppConfig)
    assert cfg.chunks.x_row_chunk == 2048
    assert cfg.io.x_storage == "csr"
    assert cfg.chunks.auto_shard is False
    assert cfg.io.lazy is True
    assert cfg.chunks.cpus == (os.cpu_count() or 1)


def test_resolve_backend_cfg_leaves_lazy_untouched_for_every_backend() -> None:
    # icechunk no longer forces eager or rejects lazy input: a lazy, not-thread-safe reader
    # into an icechunk session runs the read-ahead pipeline at write time instead (see
    # writer_parallel_mode / test_features.py's icechunk-lazy roundtrip).
    for backend in ("zarr", "icechunk"):
        for lazy in (True, False):
            resolved = resolve_backend_cfg(AppConfig(io=IOConfig(backend=backend, lazy=lazy)))
            assert resolved.io.lazy == lazy


def test_resolve_backend_cfg_warns_on_shard_factor_with_sparse_storage(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("WARNING", logger="annizarr")
    resolve_backend_cfg(AppConfig(chunks=ChunkConfig(x_shard_factor=2), io=IOConfig(x_storage="csr")))
    assert any("only applies to dense X" in r.message for r in caplog.records)


def test_toml_load_and_section_guards(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        """
[io]
overwrite = true
x_storage = "dense"

[chunks]
x_row_chunk = 1024
x_col_chunk = 512
""".strip(),
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_file))
    assert cfg.io.overwrite is True
    assert cfg.io.x_storage == "dense"
    assert cfg.chunks.x_row_chunk == 1024
    assert cfg.chunks.x_col_chunk == 512

    bad_storage = tmp_path / "bad_storage.toml"
    bad_storage.write_text('[io]\nx_storage = "not-a-mode"\n')
    with pytest.raises(ValueError):
        load_config(str(bad_storage))

    bad_section = tmp_path / "bad_section.toml"
    bad_section.write_text("[chunk]\nx_row_chunk = 1000\n")
    with pytest.raises(ValueError, match="chunk"):
        load_config(str(bad_section))


def test_cli_overrides() -> None:
    cfg = load_config(None)
    cfg2 = apply_cli_overrides(cfg, x_row_chunk=128, overwrite=True, x_storage="csr")
    assert cfg2.chunks.x_row_chunk == 128
    assert cfg2.io.overwrite is True
    assert cfg2.io.x_storage == "csr"

    assert cfg.chunks.auto_shard is False
    cfg3 = apply_cli_overrides(cfg, auto_shard=True)
    assert cfg3.chunks.auto_shard is True
    # None means "don't touch it" (a caller other than the CLI's own --auto-shard, which
    # always passes a bool)
    cfg4 = apply_cli_overrides(cfg3, auto_shard=None)
    assert cfg4.chunks.auto_shard is True
