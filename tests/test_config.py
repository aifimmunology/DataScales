from pathlib import Path

from datascale.config import AppConfig, apply_cli_overrides, load_config


def test_default_config() -> None:
    cfg = load_config(None)
    assert isinstance(cfg, AppConfig)
    assert cfg.chunks.x_row_chunk == 2048
    assert cfg.io.x_storage == "auto"


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
    cfg2 = apply_cli_overrides(cfg, x_row_chunk=128, overwrite=True, x_storage="sparse-csr")

    assert cfg2.chunks.x_row_chunk == 128
    assert cfg2.io.overwrite is True
    assert cfg2.io.x_storage == "sparse-csr"


def test_invalid_x_storage_rejected(tmp_path: Path) -> None:
    cfg_file = tmp_path / "bad.toml"
    cfg_file.write_text(
        """
[io]
x_storage = "not-a-mode"
""".strip(),
        encoding="utf-8",
    )

    try:
        load_config(str(cfg_file))
        raised = False
    except ValueError:
        raised = True

    assert raised


def test_csc_x_storage_valid(tmp_path: Path) -> None:
    cfg_file = tmp_path / "csc.toml"
    cfg_file.write_text(
        """
[io]
x_storage = "sparse-csc"
""".strip(),
        encoding="utf-8",
    )

    cfg = load_config(str(cfg_file))
    assert cfg.io.x_storage == "sparse-csc"

