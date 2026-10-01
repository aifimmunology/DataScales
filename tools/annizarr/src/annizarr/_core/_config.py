from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from annizarr.errors import ValidationError

if TYPE_CHECKING:
    from annizarr.typing import XStorage

yaml: Any
try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

logger = logging.getLogger(__name__)

BackendMode = Literal["zarr", "icechunk"]

__all__ = [
    "AppConfig",
    "ChunkConfig",
    "ConcatConfig",
    "GroupingConfig",
    "IOConfig",
    "ValidationConfig",
    "apply_cli_overrides",
    "load_config",
]


@dataclass(frozen=True, slots=True)
class IOConfig:
    """Input/output toggles shared by every op.

    Parameters
    ----------
    overwrite
        Replace an existing output path/store instead of erroring.
    consolidate_metadata
        Consolidate zarr metadata into one object after writing (plain zarr only).
    x_storage
        On-disk layout for X (and layers): ``"csr"``, ``"csc"``, or ``"dense"``.
    backed
        Load h5ad input in backed (HDF5-streamed) mode instead of eagerly. ``None``
        (the default) auto-selects per input: :func:`annizarr._sources._h5ad.load_h5ad`
        peeks the on-disk size of ``X`` (no data read) and picks backed when it exceeds
        ``eager_max_bytes``, else eager. Ignored for in-memory/10x input (always eager).
    eager_max_bytes
        Auto-select threshold, in bytes, used when ``backed`` is ``None``: an h5ad whose
        on-disk ``X`` (``data``+``indices``+``indptr`` for sparse, the raw dataset for
        dense) exceeds this loads backed instead of eagerly. Ignored when ``backed`` is
        set explicitly.
    backend
        ``"zarr"`` writes a plain on-disk store; ``"icechunk"`` writes through a
        transactional, versioned Icechunk repository (one commit per op). Icechunk
        targets are a local path or an ``s3://bucket/prefix`` URL (env credentials).
    """

    overwrite: bool = False
    consolidate_metadata: bool = False
    x_storage: XStorage = "csr"
    backed: bool | None = None
    eager_max_bytes: int = 2 * 1024**3
    backend: BackendMode = "zarr"


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    """Chunk/shard sizing and worker count for matrix writes.

    Parameters
    ----------
    x_row_chunk
        Row chunk size for X (dense row axis; CSR major axis granularity).
    x_col_chunk
        Column chunk size for dense X.
    sparse_flat_chunk
        Flat chunk size for sparse X ``data``/``indices``.
    cpus
        Workers for parallel matrix chunk writes: threads for in-memory input,
        processes when backed (h5py is not thread-safe); raise on HPC.
    x_shard_factor
        Pack this many chunks per shard along each axis of dense X (``1`` = no
        sharding; sparse output ignores it). See :func:`annizarr._core._layout.dense_shards`.
    auto_shard
        Shard our own 1-D sparse arrays (``data``/``indices`` of X, layers, raw.X, and the
        add-expr/rechunk/sort-created ones) with zarr's ``shards="auto"``, and set
        ``ad.settings.auto_shard_zarr_v3`` around every ``write_elem`` call this op makes
        (obs/var columns, obsm, uns, …), so anndata's own writes are auto-sharded too.
        Cuts object/file count on remote or many-small-chunk stores at a small write-time
        cost (see :func:`annizarr._writers._encoding.sparse_shards`). Dense X is unaffected
        — it keeps the explicit ``x_shard_factor`` above, never ``shards="auto"``. Default
        chosen from a read-latency benchmark (see
        ``benchmarking_results/autoshard/README.md``); ``False`` reproduces the pre-autoshard
        on-disk layout exactly.
    """

    x_row_chunk: int = 2048
    x_col_chunk: int = 2048
    sparse_flat_chunk: int = 1_000_000
    cpus: int = 1
    x_shard_factor: int = 1
    auto_shard: bool = False


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """Thresholds for single-cell AnnData validation."""

    reject_spatial: bool = True
    require_non_empty: bool = True
    min_obs: int = 1
    min_vars: int = 1


@dataclass(frozen=True, slots=True)
class GroupingConfig:
    """Sort + partition X by one or more obs columns, for ``convert --sort-by``."""

    enabled: bool = False
    sort_by: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ConcatConfig:
    """obs-column policy for multi-file ``convert``/``concat``."""

    obs_columns: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AppConfig:
    """The resolved configuration passed to every op.

    Parameters
    ----------
    io
        Input/output toggles.
    chunks
        Chunk/shard sizing and worker count.
    validation
        Single-cell AnnData validation thresholds.
    grouping
        ``convert --sort-by`` sort/partition settings.
    concat
        Multi-file concat obs-column policy.
    """

    io: IOConfig = IOConfig()
    chunks: ChunkConfig = ChunkConfig()
    validation: ValidationConfig = ValidationConfig()
    grouping: GroupingConfig = GroupingConfig()
    concat: ConcatConfig = ConcatConfig()


def _normalize_x_storage(value: str) -> XStorage:
    mode = value.lower().strip()
    allowed = {"csr", "csc", "dense"}
    if mode not in allowed:
        allowed_list = ", ".join(sorted(allowed))
        raise ValidationError(f"Invalid io.x_storage '{value}'. Expected one of: {allowed_list}")
    return mode  # type: ignore[return-value]  # mode is in `allowed`, a subset of XStorage's literals


def _normalize_backend(value: str) -> BackendMode:
    mode = value.lower().strip()
    allowed = {"zarr", "icechunk"}
    if mode not in allowed:
        allowed_list = ", ".join(sorted(allowed))
        raise ValidationError(f"Invalid io.backend '{value}'. Expected one of: {allowed_list}")
    return mode  # type: ignore[return-value]  # mode is in `allowed`, a subset of BackendMode's literals


def _validate_config(config: AppConfig) -> AppConfig:
    io = replace(
        config.io,
        x_storage=_normalize_x_storage(config.io.x_storage),
        backend=_normalize_backend(config.io.backend),
    )
    sort_by = config.grouping.sort_by  # may arrive as a list or bare string; freeze to a tuple
    if isinstance(sort_by, str):
        sort_by = (sort_by,)
    grouping = replace(config.grouping, sort_by=tuple(sort_by))
    if grouping.enabled and not grouping.sort_by:
        raise ValidationError("grouping.enabled is true but grouping.sort_by is empty.")
    obs_columns = config.concat.obs_columns  # same TOML/YAML list-or-string freeze as sort_by
    if isinstance(obs_columns, str):
        obs_columns = (obs_columns,)
    concat = replace(config.concat, obs_columns=tuple(obs_columns))
    if config.chunks.x_shard_factor < 1:
        raise ValidationError(
            f"chunks.x_shard_factor must be >= 1 (1 = no sharding); got {config.chunks.x_shard_factor}."
        )
    if io.eager_max_bytes < 0:
        raise ValidationError(f"io.eager_max_bytes must be >= 0; got {io.eager_max_bytes}.")
    return replace(config, io=io, grouping=grouping, concat=concat)


def _read_config_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix == ".toml":
        return tomllib.loads(text)

    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise ValidationError("YAML config support requires PyYAML.")
        data = yaml.safe_load(text)
        return data or {}

    raise ValidationError(f"Unsupported config file extension: {suffix}")


def _merge_dataclass(base: Any, patch: dict[str, Any]) -> Any:
    valid_fields = set(base.__dataclass_fields__.keys())
    unknown = set(patch.keys()) - valid_fields
    if unknown:
        bad = ", ".join(sorted(unknown))
        raise ValidationError(f"Unknown config keys for {type(base).__name__}: {bad}")
    return replace(base, **patch)


def load_config(config_path: str | None = None) -> AppConfig:
    """Load a ``.toml``/``.yaml``/``.yml`` config file and merge it over the defaults."""
    config = AppConfig()

    if not config_path:
        return config

    path = Path(config_path)
    if not path.exists():
        raise ValidationError(f"Config file not found: {path}")

    data = _read_config_file(path)

    known_sections = {"io", "chunks", "validation", "grouping", "concat"}
    unknown_sections = set(data.keys()) - known_sections
    if unknown_sections:
        bad = ", ".join(sorted(unknown_sections))
        raise ValidationError(
            f"Unknown top-level config sections: {bad}. Expected only: {', '.join(sorted(known_sections))}"
        )

    io_patch = data.get("io", {})
    chunks_patch = data.get("chunks", {})
    validation_patch = data.get("validation", {})
    grouping_patch = data.get("grouping", {})
    concat_patch = data.get("concat", {})

    patches = (io_patch, chunks_patch, validation_patch, grouping_patch, concat_patch)
    if not all(isinstance(p, dict) for p in patches):
        raise ValidationError(
            "Config sections [io], [chunks], [validation], [grouping], [concat] must be maps/objects."
        )

    config = replace(
        config,
        io=_merge_dataclass(config.io, io_patch),
        chunks=_merge_dataclass(config.chunks, chunks_patch),
        validation=_merge_dataclass(config.validation, validation_patch),
        grouping=_merge_dataclass(config.grouping, grouping_patch),
        concat=_merge_dataclass(config.concat, concat_patch),
    )

    return _validate_config(config)


def apply_cli_overrides(
    config: AppConfig,
    *,
    overwrite: bool | None = None,
    consolidate_metadata: bool | None = None,
    x_storage: str | None = None,
    x_row_chunk: int | None = None,
    x_col_chunk: int | None = None,
    sparse_flat_chunk: int | None = None,
    x_shard_factor: int | None = None,
    auto_shard: bool | None = None,
    cpus: int | None = None,
    backed: bool | None = None,
    backend: str | None = None,
    sort_by: list[str] | None = None,
    obs_columns: list[str] | None = None,
) -> AppConfig:
    """Apply CLI flag overrides (``None`` = unset) onto a loaded config."""
    io_cfg = config.io
    chunk_cfg = config.chunks
    grouping_cfg = config.grouping
    concat_cfg = config.concat

    if overwrite is not None:
        io_cfg = replace(io_cfg, overwrite=overwrite)
    if consolidate_metadata is not None:
        io_cfg = replace(io_cfg, consolidate_metadata=consolidate_metadata)
    if x_storage is not None:
        io_cfg = replace(io_cfg, x_storage=_normalize_x_storage(x_storage))
    if backed is not None:  # None means "don't touch it" (IOConfig.backed is itself tri-state), not "reset to auto"
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
    if auto_shard is not None:
        chunk_cfg = replace(chunk_cfg, auto_shard=auto_shard)
    if cpus is not None:
        chunk_cfg = replace(chunk_cfg, cpus=cpus)
    if sort_by is not None:
        grouping_cfg = replace(grouping_cfg, enabled=True, sort_by=tuple(sort_by))
    if obs_columns is not None:
        concat_cfg = replace(concat_cfg, obs_columns=tuple(obs_columns))

    return _validate_config(replace(config, io=io_cfg, chunks=chunk_cfg, grouping=grouping_cfg, concat=concat_cfg))


def resolve_backend_cfg(cfg: AppConfig) -> AppConfig:
    from annizarr.errors import ConversionError

    if cfg.chunks.x_shard_factor > 1 and cfg.io.x_storage != "dense":
        logger.warning(
            f"x_shard_factor={cfg.chunks.x_shard_factor} only applies to dense X; "
            f"x_storage={cfg.io.x_storage!r} is sparse, so sharding is ignored."
        )
    if cfg.io.backend == "icechunk" and cfg.io.backed is None:
        logger.info("backend='icechunk' does not support backed input; auto-selecting eager (backed=False).")
        cfg = replace(cfg, io=replace(cfg.io, backed=False))
    # icechunk's backed writers would need Session.fork() (not available) to reopen by path
    if cfg.io.backend == "icechunk" and cfg.io.backed:
        raise ConversionError(
            "backend='icechunk' does not support --backed input yet (backed writers use "
            "worker processes that reopen the store by path; the icechunk session is "
            "in-process only). Convert eagerly (omit --backed), or use backend='zarr'."
        )
    return cfg
