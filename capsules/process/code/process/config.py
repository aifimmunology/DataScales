"""Every setting of a run: a field below, overridden by an upper-case environment variable, then by `--flag`."""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

METHODS = {"pca": "pca", "scvi": "scvi", "scvi (train)": "scvi"}
RMM_MODES = ("managed", "pool", "none")
PATHS = ("input", "origin", "output", "results_dir", "scvi_model_dir")
_BOOLS = {"1": True, "true": True, "yes": True, "on": True, "0": False, "false": False, "no": False, "off": False}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Config:
    # App Panel: input data
    input: str = ""                    # AnnData .zarr, SpatialData .zarr, or icechunk repo; "" prints usage and exits
    table: str = "table"               # SpatialData: tables/<table>
    branch: str = "main"               # icechunk branch the results are committed on
    origin: str = ""                   # icechunk: writable location behind a read-only data asset
    subset_column: str = ""            # obs column; "" runs on every cell
    subset_values: str = ""            # comma-separated values of subset_column to keep
    # App Panel: embedding
    method: str = "pca"                # pca | scvi
    batch_key: str = ""                # pca: harmony on these obs columns; scvi: batch (first) and covariates (rest)
    n_top_genes: int = 2000
    n_comps: int = 50
    # App Panel: neighbors, UMAP, clustering, output
    n_neighbors: int = 20
    umap_min_dist: float = 0.45
    leiden_resolutions: str = "1.1"    # comma-separated; one gives obs/leiden, several give obs/leiden_<res>
    output: str = ""                   # zarr input: the new copy; "" is <results_dir>/<stem>.processed.zarr

    results_dir: str = "/results"
    sample_key: str = "sample_id"      # plotted and counted per cluster when the store has it
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
    def resolutions(self) -> list[float]:
        return [float(r) for r in split(self.leiden_resolutions)]

    @property
    def leiden_keys(self) -> dict[str, float]:
        res = self.resolutions
        return {"leiden": res[0]} if len(res) == 1 else {f"leiden_{r}": r for r in res}

    @property
    def gpu_ids(self) -> list[int]:
        return [int(g) for g in split(self.gpus)]

    @property
    def model_dir(self) -> str:
        return self.scvi_model_dir or f"{self.results_dir.rstrip('/')}/scvi_model"

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
        if not resolutions or min(resolutions) <= 0:
            raise ConfigError("leiden_resolutions needs at least one positive number")
        if self.subset_column and not self.subset:
            raise ConfigError(f"subset_column {self.subset_column!r} needs subset_values")
        return self


def split(text: str) -> list[str]:
    return [p.strip() for p in str(text).split(",") if p.strip()]


_TYPES = {f.name: type(f.default) for f in fields(Config)}


def _cast(name: str, raw: str):
    kind, raw = _TYPES[name], raw.strip()
    try:
        if name == "method":
            return METHODS[raw.lower()]
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
    cwd = env.get("PROCESS_CWD") or os.getcwd()
    return replace(cfg, **{n: _absolute(getattr(cfg, n), cwd) for n in PATHS}).validate()
