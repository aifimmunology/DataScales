from __future__ import annotations

from dataclasses import dataclass, replace
from importlib.resources import path
from pathlib import Path
from typing import Any, Literal

import tomllib

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


XStorageMode = Literal["sparse-csr", "sparse-csc", "dense"]
BackendMode = Literal["zarr", "icechunk"]
IcechunkStorageMode = Literal["local", "gcs"]


@dataclass(frozen=True)
class IOConfig:
    overwrite: bool = False
    consolidate_metadata: bool = False
    x_storage: XStorageMode = "sparse-csr"
    backed: bool = False  # load h5ad in backed (HDF5-streamed) mode; opt-in only
    # Storage backend for the output store. "zarr" writes a plain on-disk zarr; "icechunk"
    # writes through a transactional, versioned Icechunk repository (one commit per convert).
    backend: BackendMode = "zarr"
    icechunk_storage: IcechunkStorageMode = "local"
    # GCS scaffolding — not wired into a working path yet (local only for now).
    gcs_bucket: str | None = None
    gcs_prefix: str = ""


@dataclass(frozen=True)
class ChunkConfig:
    x_row_chunk: int = 2048
    x_col_chunk: int = 2048
    sparse_flat_chunk: int = 1_000_000
    cpus: int = 1  # workers for parallel matrix chunk writes; threads in-memory, processes when backed; raise on HPC
    # Pack dense X inner chunks into shards of (x_row_chunk, x_col_chunk) * factor.
    # 1 = no sharding. Dense X only (sparse output ignores it). See converter._dense_shards.
    x_shard_factor: int = 1


@dataclass(frozen=True)
class ValidationConfig:
    reject_spatial: bool = True
    require_non_empty: bool = True
    min_obs: int = 1
    min_vars: int = 1


@dataclass(frozen=True)
class GroupingConfig:
    """Sort + partition X by one or more obs columns (Feature B).

    When enabled, rows are physically sorted by ``sort_by`` (primary key first), so each
    distinct key tuple is a contiguous row range recorded in ``uns/datascale_sort_index``.
    Subsets can then be read with stock anndata/zarr by slicing ``X[start:end]`` over the
    matching range(s) — no datascale dependency required. All obs-aligned arrays are
    reordered consistently so the store stays a valid AnnData. convert-h5ad only, with
    sparse-csr or dense X.
    """
    enabled: bool = False
    sort_by: tuple[str, ...] = ()  # obs column names, primary sort key first


@dataclass(frozen=True)
class AppConfig:
    io: IOConfig = IOConfig()
    chunks: ChunkConfig = ChunkConfig()
    validation: ValidationConfig = ValidationConfig()
    grouping: GroupingConfig = GroupingConfig()


def _normalize_x_storage(value: str) -> XStorageMode:
    mode = value.lower().strip()
    allowed = {"sparse-csr", "sparse-csc", "dense"}
    if mode not in allowed:
        allowed_list = ", ".join(sorted(allowed))
        raise ValueError(f"Invalid io.x_storage '{value}'. Expected one of: {allowed_list}")
    return mode  # type: ignore[return-value]


def _normalize_backend(value: str) -> BackendMode:
    mode = value.lower().strip()
    allowed = {"zarr", "icechunk"}
    if mode not in allowed:
        allowed_list = ", ".join(sorted(allowed))
        raise ValueError(f"Invalid io.backend '{value}'. Expected one of: {allowed_list}")
    return mode  # type: ignore[return-value]


def _validate_config(config: AppConfig) -> AppConfig:
    io = replace(
        config.io,
        x_storage=_normalize_x_storage(config.io.x_storage),
        backend=_normalize_backend(config.io.backend),
    )
    # sort_by may arrive from TOML/YAML as a list; freeze it to a tuple. A bare string
    # ("AIFI_L1") is treated as a single key.
    sort_by = config.grouping.sort_by
    if isinstance(sort_by, str):
        sort_by = (sort_by,)
    grouping = replace(config.grouping, sort_by=tuple(sort_by))
    if grouping.enabled and not grouping.sort_by:
        raise ValueError("grouping.enabled is true but grouping.sort_by is empty.")
    if config.chunks.x_shard_factor < 1:
        raise ValueError(
            f"chunks.x_shard_factor must be >= 1 (1 = no sharding); got {config.chunks.x_shard_factor}."
        )
    return replace(config, io=io, grouping=grouping)


def _read_config_file(path: Path) -> dict[str, Any]:
    """Physically reads the config file, and parses it according to the file extension."""
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix == ".toml":
        return tomllib.loads(text)

    if suffix in {".yaml", ".yml"}:
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
    """
    Loads a config file, and uses the valid config values to override the defaults.
    If no path is provided, returns the default config.
    """
    
    config = AppConfig()

    if not config_path:
        return config

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    data = _read_config_file(path)

    known_sections = {"io", "chunks", "validation", "grouping"}
    unknown_sections = set(data.keys()) - known_sections
    if unknown_sections:
        bad = ", ".join(sorted(unknown_sections))
        raise ValueError(
            f"Unknown top-level config sections: {bad}. "
            f"Expected only: {', '.join(sorted(known_sections))}"
        )

    io_patch = data.get("io", {})
    chunks_patch = data.get("chunks", {})
    validation_patch = data.get("validation", {})
    grouping_patch = data.get("grouping", {})

    if not all(isinstance(p, dict) for p in (io_patch, chunks_patch, validation_patch, grouping_patch)):
        raise ValueError("Config sections [io], [chunks], [validation], [grouping] must be maps/objects.")

    config = replace(
        config,
        io=_merge_dataclass(config.io, io_patch),
        chunks=_merge_dataclass(config.chunks, chunks_patch),
        validation=_merge_dataclass(config.validation, validation_patch),
        grouping=_merge_dataclass(config.grouping, grouping_patch),
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
    x_shard_factor: int | None = None,
    cpus: int | None = None,
    backed: bool | None = None,
    backend: str | None = None,
    sort_by: list[str] | None = None,
) -> AppConfig:
    io_cfg = config.io
    chunk_cfg = config.chunks
    grouping_cfg = config.grouping

    if overwrite is not None:
        io_cfg = replace(io_cfg, overwrite=overwrite)
    if consolidate_metadata is not None:
        io_cfg = replace(io_cfg, consolidate_metadata=consolidate_metadata)
    if x_storage is not None:
        io_cfg = replace(io_cfg, x_storage=_normalize_x_storage(x_storage))
    if backed is not None:
        io_cfg = replace(io_cfg, backed=backed)
    if backend is not None:
        io_cfg = replace(io_cfg, backend=_normalize_backend(backend))
    if x_row_chunk is not None:
        chunk_cfg = replace(chunk_cfg, x_row_chunk=x_row_chunk)
    if x_col_chunk is not None:
        chunk_cfg = replace(chunk_cfg, x_col_chunk=x_col_chunk)
    if sparse_flat_chunk is not None:
        chunk_cfg = replace(chunk_cfg, sparse_flat_chunk=sparse_flat_chunk)
    if x_shard_factor is not None:
        chunk_cfg = replace(chunk_cfg, x_shard_factor=x_shard_factor)
    if cpus is not None:
        chunk_cfg = replace(chunk_cfg, cpus=cpus)
    if sort_by is not None:
        grouping_cfg = replace(grouping_cfg, enabled=True, sort_by=tuple(sort_by))

    return _validate_config(
        replace(config, io=io_cfg, chunks=chunk_cfg, grouping=grouping_cfg)
    )
