from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import tomllib

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


XStorageMode = Literal["auto", "sparse-csr", "sparse-csc", "dense"]


@dataclass(frozen=True)
class IOConfig:
    overwrite: bool = False
    consolidate_metadata: bool = False
    x_storage: XStorageMode = "auto"


@dataclass(frozen=True)
class ChunkConfig:
    x_row_chunk: int = 2048
    x_col_chunk: int = 2048
    sparse_flat_chunk: int = 4096


@dataclass(frozen=True)
class ValidationConfig:
    reject_spatial: bool = True
    require_non_empty: bool = True
    min_obs: int = 1
    min_vars: int = 1


@dataclass(frozen=True)
class AppConfig:
    io: IOConfig = IOConfig()
    chunks: ChunkConfig = ChunkConfig()
    validation: ValidationConfig = ValidationConfig()


def _normalize_x_storage(value: str) -> XStorageMode:
    mode = value.lower().strip()
    allowed = {"auto", "sparse-csr", "sparse-csc", "dense"}
    if mode not in allowed:
        allowed_list = ", ".join(sorted(allowed))
        raise ValueError(f"Invalid io.x_storage '{value}'. Expected one of: {allowed_list}")
    return mode  # type: ignore[return-value]


def _validate_config(config: AppConfig) -> AppConfig:
    return replace(config, io=replace(config.io, x_storage=_normalize_x_storage(config.io.x_storage)))


def _read_config_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix == ".toml":
        return tomllib.loads(text)

    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("YAML config requested but PyYAML is not installed.")
        data = yaml.safe_load(text)
        return data or {}

    raise ValueError(f"Unsupported config file extension: {suffix}")


def _merge_dataclass(base: Any, patch: dict[str, Any]) -> Any:
    valid_fields = {k for k in base.__dataclass_fields__.keys()}
    unknown = set(patch.keys()) - valid_fields
    if unknown:
        bad = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown config keys for {type(base).__name__}: {bad}")
    return replace(base, **patch)


def load_config(config_path: str | None = None) -> AppConfig:
    config = AppConfig()

    if not config_path:
        return config

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    data = _read_config_file(path)

    io_patch = data.get("io", {})
    chunks_patch = data.get("chunks", {})
    validation_patch = data.get("validation", {})

    if not isinstance(io_patch, dict) or not isinstance(chunks_patch, dict) or not isinstance(validation_patch, dict):
        raise ValueError("Config sections [io], [chunks], and [validation] must be maps/objects.")

    config = replace(
        config,
        io=_merge_dataclass(config.io, io_patch),
        chunks=_merge_dataclass(config.chunks, chunks_patch),
        validation=_merge_dataclass(config.validation, validation_patch),
    )

    return _validate_config(config)


def apply_cli_overrides(
    config: AppConfig,
    overwrite: bool | None = None,
    consolidate_metadata: bool | None = None,
    x_storage: str | None = None,
    x_row_chunk: int | None = None,
    x_col_chunk: int | None = None,
    sparse_flat_chunk: int | None = None,
) -> AppConfig:
    io_cfg = config.io
    chunk_cfg = config.chunks

    if overwrite is not None:
        io_cfg = replace(io_cfg, overwrite=overwrite)
    if consolidate_metadata is not None:
        io_cfg = replace(io_cfg, consolidate_metadata=consolidate_metadata)
    if x_storage is not None:
        io_cfg = replace(io_cfg, x_storage=_normalize_x_storage(x_storage))
    if x_row_chunk is not None:
        chunk_cfg = replace(chunk_cfg, x_row_chunk=x_row_chunk)
    if x_col_chunk is not None:
        chunk_cfg = replace(chunk_cfg, x_col_chunk=x_col_chunk)
    if sparse_flat_chunk is not None:
        chunk_cfg = replace(chunk_cfg, sparse_flat_chunk=sparse_flat_chunk)

    return _validate_config(replace(config, io=io_cfg, chunks=chunk_cfg))
