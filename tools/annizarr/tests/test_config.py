from pathlib import Path

import pytest

from annizarr._core._config import resolve_backend_cfg
from annizarr.config import AppConfig, IOConfig, apply_cli_overrides, load_config
from annizarr.errors import ConversionError


def test_defaults() -> None:
    cfg = load_config(None)
    assert isinstance(cfg, AppConfig)
    assert cfg.chunks.x_row_chunk == 2048
    assert cfg.io.x_storage == "csr"
    assert cfg.chunks.auto_shard is False
    assert cfg.io.backed is None  # auto-select
    assert cfg.io.eager_max_bytes == 2 * 1024**3


def test_eager_max_bytes_toml_validation(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("[io]\neager_max_bytes = -1\n")
    with pytest.raises(ValueError, match="eager_max_bytes"):
        load_config(str(bad))

    good = tmp_path / "config.toml"
    good.write_text("[io]\neager_max_bytes = 1000\n")
    assert load_config(str(good)).io.eager_max_bytes == 1000


def test_resolve_backend_cfg_branches(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger="annizarr")
    resolved = resolve_backend_cfg(AppConfig(io=IOConfig(backend="icechunk", backed=None)))
    assert resolved.io.backed is False  # auto-selects eager
    assert any("auto-selecting eager" in r.message for r in caplog.records)

    with pytest.raises(ConversionError, match="does not support --backed"):
        resolve_backend_cfg(AppConfig(io=IOConfig(backend="icechunk", backed=True)))

    resolved = resolve_backend_cfg(AppConfig(io=IOConfig(backend="zarr", backed=None)))
    assert resolved.io.backed is None  # zarr backend: left untouched


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
