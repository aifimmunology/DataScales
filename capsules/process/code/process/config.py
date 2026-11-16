"""Every setting of a run: a field below, overridden by an upper-case environment variable, then by `--flag`."""
from __future__ import annotations

import argparse
import os
import re
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

METHODS = {"pca": "pca", "scvi": "scvi", "scvi (train)": "scvi"}
RMM_MODES = ("managed", "pool", "none")
PATHS = ("input", "results_dir", "scvi_model_dir")
OUTPUT_PLACEHOLDER = "s3://bucket/prefix"
FORMATS = ("auto", "zarr", "icechunk")
PCA_KEY, HARMONY_KEY, SCVI_KEY = "X_pca", "X_pca_harmony", "X_scVI"
EMBEDDING_KEYS = (PCA_KEY, HARMONY_KEY, SCVI_KEY)
UMAP_DEFAULT, LEIDEN_DEFAULT = "X_umap", "leiden"
NAME = re.compile(r"^[A-Za-z0-9_.\-]+$")
_BOOLS = {"1": True, "true": True, "yes": True, "on": True, "0": False, "false": False, "no": False, "off": False}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Config:
    # App Panel: input data
    input: str = ""                    # AnnData .zarr, SpatialData .zarr, or icechunk repo; "" prints usage and exits
    table: str = "table"               # SpatialData: tables/<table>
    branch: str = "main"               # icechunk: the branch read and the results are committed on
    subset_column: str = ""            # obs column; "" runs on every cell
    subset_values: str = ""            # comma-separated values of subset_column to keep
    # App Panel: embedding
    method: str = "pca"                # pca | scvi
    batch_key: str = ""                # pca: harmony on these obs columns; scvi: batch (first) and covariates (rest)
    n_top_genes: int = 2000
    n_comps: int = 50
    # App Panel: neighbors, UMAP & clustering
    n_neighbors: int = 20
    umap_min_dist: float = 0.45
    leiden_resolutions: str = "1.1"    # comma-separated; one gives obs/<leiden_key>, several give obs/<leiden_key>_<res>
    umap_key: str = UMAP_DEFAULT           # obsm name of the UMAP coordinates
    leiden_key: str = LEIDEN_DEFAULT         # obs column name (prefix when there are several resolutions)
    # App Panel: UMAP plots
    color_keys: str = "batch_id,chip_id,pbmc_clean_AIFI_L1_2024_label,leiden"  # obs columns; leiden = the clusters
    # App Panel: output (store.py): "" saves a copy in results_dir
    output: str = ""                   # s3://, a path, or a name; a name or a path under results_dir is a copy there
    output_format: str = "auto"        # auto | zarr | icechunk; auto: the input's format (copy) or what is there (else)

    results_dir: str = "/results"
    sample_key: str = "sample_id"      # counted per cluster when the store has it
    chunk_rows: int = 24_000
    target_sum: float = 1e4
    hvg_flavor: str = "seurat"
    hvg_batch_key: str = ""
    scale_max_value: float = 10.0
    n_pcs: int = 30
    neighbors_algorithm: str = "ivfflat"
    umap_n_components: int = 2
    leiden_iterations: int = 100
    harmony_max_iter: int = 30
    harmony_theta: float = 2.0
    harmony_alpha: float = 0.2
    random_seed: int = 0
    plots: bool = True

    gpus: str = ""                     # e.g. "0,1"; "" uses every visible GPU
    rmm_mode: str = "managed"          # managed (spills to host RAM) | pool (rmm_pool_size of each GPU) | none
    rmm_pool_size: float = 0.8
    threads_per_worker: int = 1
    enable_cudf_spill: bool = True
    zarr_concurrency: int = 12
    zarr_max_workers: int = 12
    sample_interval_s: float = 0.1

    scvi_model_dir: str = ""           # "" is <results_dir>/scvi_model
    scvi_n_latent: int = 30
    scvi_n_layers: int = 2
    scvi_n_hidden: int = 128
    scvi_dropout_rate: float = 0.1
    scvi_gene_likelihood: str = "nb"   # nb | zinb | poisson
    scvi_max_epochs: int = 0           # 0: scvi-tools' own choice from the number of cells
    scvi_batch_size: int = 128
    scvi_lr: float = 1e-3
    scvi_train_size: float = 0.9
    scvi_early_stopping: bool = False
    scvi_early_stopping_patience: int = 45
    scvi_accelerator: str = "auto"

    @property
    def batch_keys(self) -> list[str]:
        return split(self.batch_key)

    @property
    def subset(self) -> list[str]:
        return split(self.subset_values) if self.subset_column else []

    @property
    def color_columns(self) -> list[str]:
        return split(self.color_keys)

    @property
    def resolutions(self) -> list[float]:
        return [float(r) for r in split(self.leiden_resolutions)]

    @property
    def leiden_keys(self) -> dict[str, float]:
        res = self.resolutions
        return {self.leiden_key: res[0]} if len(res) == 1 else {f"{self.leiden_key}_{r}": r for r in res}

    @property
    def gpu_ids(self) -> list[int]:
        return [int(g) for g in split(self.gpus)]

    @property
    def model_dir(self) -> str:
        return self.scvi_model_dir or f"{self.results_dir.rstrip('/')}/scvi_model"

    def _validate_names(self) -> None:
        for name in ("umap_key", "leiden_key"):
            if not NAME.match(getattr(self, name)):
                raise ConfigError(f"{name} may only have letters, digits, _ . - (no spaces), got {getattr(self, name)!r}")
        if self.umap_key in EMBEDDING_KEYS:
            raise ConfigError(f"umap_key {self.umap_key!r} is already the name of the PCA/scVI embedding; choose another, e.g. X_umap")
        taken = {self.sample_key, self.subset_column, self.hvg_batch_key, *self.batch_keys}
        clash = taken & {self.leiden_key, *self.leiden_keys}
        if clash:
            raise ConfigError(f"leiden_key {sorted(clash)[0]!r} is also a batch, subset or sample column and would overwrite it; choose another")

    def changed(self) -> dict:
        base = Config()
        return {k: v for k, v in asdict(self).items() if v != getattr(base, k)}

    def validate(self) -> Config:
        if self.rmm_mode not in RMM_MODES:
            raise ConfigError(f"rmm_mode must be one of {', '.join(RMM_MODES)}, got {self.rmm_mode!r}")
        try:
            resolutions = self.resolutions
        except ValueError:
            raise ConfigError(f"leiden_resolutions must be numbers separated by commas, got {self.leiden_resolutions!r}") from None
        if not resolutions or not all(0 < r < float("inf") for r in resolutions):
            raise ConfigError("leiden_resolutions needs at least one positive number")
        try:
            self.gpu_ids
        except ValueError:
            raise ConfigError(f"gpus must be numbers separated by commas, got {self.gpus!r}") from None
        if not 0 < self.rmm_pool_size <= 1:
            raise ConfigError(f"rmm_pool_size must be above 0 and at most 1, got {self.rmm_pool_size}")
        if ".." in Path(self.output).parts and not self.output.startswith("/") and "://" not in self.output:
            raise ConfigError("output can't contain '..'; a name is saved in /results, or give a full path")
        for name, low, high in (("n_top_genes", 50, None), ("n_comps", 2, None), ("n_neighbors", 2, None),
                                ("umap_min_dist", 0, 1)):
            value = getattr(self, name)
            if value < low or (high is not None and value > high):
                limit = f"at least {low}" if high is None else f"between {low} and {high}"
                raise ConfigError(f"{name} must be {limit}, got {value}")
        self._validate_names()
        if self.output_format not in FORMATS:
            raise ConfigError(f"output_format must be one of {', '.join(FORMATS)}, got {self.output_format!r}")
        if "://" in self.results_dir:
            raise ConfigError("results_dir is a local path; for s3:// use output")
        if self.subset_column and not self.subset:
            raise ConfigError(f"subset_column {self.subset_column!r} is set but subset_values is empty: add the values to keep, or clear the column")
        return self


def split(text: str) -> list[str]:
    return [p.strip() for p in str(text).split(",") if p.strip()]


_TYPES = {f.name: type(f.default) for f in fields(Config)}


def _cast(name: str, raw: str):
    kind, raw = _TYPES[name], raw.strip()
    try:
        if name == "method":
            return METHODS[raw.lower()]
        if name.endswith("_format"):
            return raw.lower()
        return _BOOLS[raw.lower()] if kind is bool else kind(raw)
    except (KeyError, ValueError):
        expected = " or ".join(sorted(set(METHODS.values()))) if name == "method" else kind.__name__
        raise ConfigError(f"{name}={raw!r} is not a valid {expected}") from None


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run", description="Optional GPU pipeline: normalize, HVG, PCA (+harmony) or "
                                "scVI, neighbors, UMAP, leiden. Each flag is also an upper-case environment variable.")
    for f in fields(Config):
        flags = dict.fromkeys([f"--{f.name}", f"--{f.name.replace('_', '-')}"])
        p.add_argument(*flags, dest=f.name, nargs="?", const="", metavar=_TYPES[f.name].__name__.upper(),
                       help=f"default: {f.default!r}")
    return p


def _absolute(path: str, cwd: str) -> str:
    if not path or "://" in path:
        return path
    return str((Path(cwd) / path).resolve())


def load_config(argv: list[str] | None = None, env: dict[str, str] | None = None) -> Config:
    """Defaults, then environment variables, then flags; an empty value or a bare flag leaves the setting as it was.

    Relative paths are resolved from PROCESS_CWD (the folder `run` was started in), else the current folder.
    """
    env = os.environ if env is None else env
    from_env = {n: _cast(n, env[n.upper()]) for n in _TYPES if env.get(n.upper(), "").strip()}
    from_args = {n: _cast(n, v) for n, v in vars(parser().parse_args(argv)).items() if v is not None and v.strip()}
    cfg = replace(Config(), **{**from_env, **from_args})
    if cfg.output.strip().rstrip("/") == OUTPUT_PLACEHOLDER:
        cfg = replace(cfg, output="")
    cwd = env.get("PROCESS_CWD") or os.getcwd()
    return replace(cfg, **{n: _absolute(getattr(cfg, n), cwd) for n in PATHS}).validate()
