from pathlib import Path

from datascale.config import AppConfig, apply_cli_overrides, load_config


def test_default_config() -> None:
    cfg = load_config(None)
    assert isinstance(cfg, AppConfig)
    assert cfg.chunks.x_row_chunk == 2048


def test_toml_load(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        """
[io]
overwrite = true

[chunks]
x_row_chunk = 1024
x_col_chunk = 512
""".strip(),
        encoding="utf-8",
    )

    cfg = load_config(str(cfg_file))

    assert cfg.io.overwrite is True
    assert cfg.chunks.x_row_chunk == 1024
    assert cfg.chunks.x_col_chunk == 512


def test_cli_override() -> None:
    cfg = load_config(None)
    cfg2 = apply_cli_overrides(cfg, x_row_chunk=128, overwrite=True)

    assert cfg2.chunks.x_row_chunk == 128
    assert cfg2.io.overwrite is True
