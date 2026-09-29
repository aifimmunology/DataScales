from pathlib import Path

import pytest

from annizarr._config import resolve_backend_cfg
from annizarr.config import AppConfig, IOConfig, apply_cli_overrides, load_config
from annizarr.errors import ConversionError


def test_default_config() -> None:
    cfg = load_config(None)
    assert isinstance(cfg, AppConfig)
    assert cfg.chunks.x_row_chunk == 2048
    assert cfg.io.x_storage == "csr"
    assert cfg.chunks.auto_shard is False


def test_default_backed_is_auto() -> None:
    cfg = load_config(None)
    assert cfg.io.backed is None
    assert cfg.io.eager_max_bytes == 2 * 1024**3


def test_negative_eager_max_bytes_rejected(tmp_path: Path) -> None:
    cfg_file = tmp_path / "bad.toml"
    cfg_file.write_text("[io]\neager_max_bytes = -1\n")
    with pytest.raises(ValueError, match="eager_max_bytes"):
        load_config(str(cfg_file))


def test_eager_max_bytes_configurable_via_toml(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text("[io]\neager_max_bytes = 1000\n")
    cfg = load_config(str(cfg_file))
    assert cfg.io.eager_max_bytes == 1000


def test_resolve_backend_cfg_icechunk_backed_none_resolves_to_eager(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger="annizarr")
    cfg = AppConfig(io=IOConfig(backend="icechunk", backed=None))
    resolved = resolve_backend_cfg(cfg)
    assert resolved.io.backed is False
    assert any("auto-selecting eager" in r.message for r in caplog.records)


def test_resolve_backend_cfg_icechunk_backed_true_still_rejected() -> None:
    cfg = AppConfig(io=IOConfig(backend="icechunk", backed=True))
    with pytest.raises(ConversionError, match="does not support --backed"):
        resolve_backend_cfg(cfg)


def test_resolve_backend_cfg_zarr_backend_leaves_backed_none() -> None:
    cfg = AppConfig(io=IOConfig(backend="zarr", backed=None))
    resolved = resolve_backend_cfg(cfg)
    assert resolved.io.backed is None


def test_toml_load(tmp_path: Path) -> None:
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


def test_cli_override() -> None:
    cfg = load_config(None)
    cfg2 = apply_cli_overrides(cfg, x_row_chunk=128, overwrite=True, x_storage="csr")

    assert cfg2.chunks.x_row_chunk == 128
    assert cfg2.io.overwrite is True
    assert cfg2.io.x_storage == "csr"


def test_cli_override_auto_shard() -> None:
    cfg = load_config(None)
    assert cfg.chunks.auto_shard is False
    cfg2 = apply_cli_overrides(cfg, auto_shard=True)
    assert cfg2.chunks.auto_shard is True
    # None means "don't touch it" (neither --auto-shard nor --no-auto-shard given)
    cfg3 = apply_cli_overrides(cfg2, auto_shard=None)
    assert cfg3.chunks.auto_shard is True


def test_invalid_x_storage_rejected(tmp_path: Path) -> None:
    cfg_file = tmp_path / "bad.toml"
    cfg_file.write_text(
        """
[io]
x_storage = "not-a-mode"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_config(str(cfg_file))


def test_csc_x_storage_valid(tmp_path: Path) -> None:
    cfg_file = tmp_path / "csc.toml"
    cfg_file.write_text(
        """
[io]
x_storage = "csc"
""".strip(),
        encoding="utf-8",
    )

    cfg = load_config(str(cfg_file))
    assert cfg.io.x_storage == "csc"


def test_unknown_top_level_section_rejected(tmp_path: Path) -> None:
    cfg_file = tmp_path / "bad.toml"
    cfg_file.write_text("[chunk]\nx_row_chunk = 1000\n")

    with pytest.raises(ValueError, match="chunk"):
        load_config(str(cfg_file))
